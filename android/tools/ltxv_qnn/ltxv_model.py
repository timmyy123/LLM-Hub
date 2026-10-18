"""Static-shape LTX-Video 0.9.x transformer for ONNX/QNN export.

Mirrors the stable-diffusion.cpp LTXV runner (src/model/diffusion/ltxv.hpp) and the reference
Lightricks Transformer3DModel, but drops every dynamic code path so each piece exports as a fixed
graph for the Hexagon HTP:

* batch is 1 and folded away: hidden states are [tokens, dim]
* the sinusoidal timestep projection (256 features) and the RoPE cos/sin tables are computed on the
  host and passed in, because fp16 sin/cos of t*1000 is not accurate enough on the NPU
* the transformer is cut into chunks; the first chunk also runs the prelude (patchify, adaLN-single,
  caption projection) and the last chunk runs the output head
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PREFIX = "model.diffusion_model."
MASK_NEG = -10000.0


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class RMSNormW(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return rms_norm(x, self.eps) * self.weight


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved rotary embedding over the full inner dim. x: [T, D], cos/sin: [T, D/2]."""
    t, d = x.shape
    pairs = x.reshape(t, d // 2, 2)
    x1 = pairs[..., 0]
    x2 = pairs[..., 1]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.stack((out1, out2), dim=-1).reshape(t, d)


class Attention(nn.Module):
    def __init__(self, dim: int, context_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(context_dim, dim)
        self.to_v = nn.Linear(context_dim, dim)
        self.q_norm = RMSNormW(dim)
        self.k_norm = RMSNormW(dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim)])

    def forward(self, x, context=None, cos=None, sin=None, mask=None):
        context = x if context is None else context
        q = self.q_norm(self.to_q(x))
        k = self.k_norm(self.to_k(context))
        v = self.to_v(context)
        if cos is not None:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        tq, tk = q.shape[0], k.shape[0]
        q = q.reshape(tq, self.heads, self.head_dim).transpose(0, 1)
        k = k.reshape(tk, self.heads, self.head_dim).transpose(0, 1)
        v = v.reshape(tk, self.heads, self.head_dim).transpose(0, 1)
        scores = torch.matmul(q, k.transpose(1, 2)) * (1.0 / math.sqrt(self.head_dim))
        if mask is not None:
            scores = scores + mask
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(0, 1).reshape(tq, -1)
        return self.to_out[0](out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.proj_in = nn.Linear(dim, dim * mult)
        self.proj_out = nn.Linear(dim * mult, dim)

    def forward(self, x):
        return self.proj_out(F.gelu(self.proj_in(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.attn1 = Attention(dim, dim, heads)
        self.attn2 = Attention(dim, dim, heads)
        self.ff = FeedForward(dim)
        self.scale_shift_table = nn.Parameter(torch.zeros(6, dim))

    def forward(self, x, context, tmod, cos, sin, mask):
        # tmod: [T, 6*dim] from adaLN-single
        mods = self.scale_shift_table.unsqueeze(0) + tmod.reshape(tmod.shape[0], 6, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods.unbind(1)
        h = rms_norm(x) * (1 + scale_msa) + shift_msa
        x = x + self.attn1(h, cos=cos, sin=sin) * gate_msa
        x = x + self.attn2(x, context=context, mask=mask)
        h = rms_norm(x) * (1 + scale_mlp) + shift_mlp
        x = x + self.ff(h) * gate_mlp
        return x


class Prelude(nn.Module):
    def __init__(self, in_channels: int, dim: int, caption_channels: int):
        super().__init__()
        self.patchify_proj = nn.Linear(in_channels, dim)
        self.t_linear_1 = nn.Linear(256, dim)
        self.t_linear_2 = nn.Linear(dim, dim)
        self.adaln_linear = nn.Linear(dim, 6 * dim)
        self.cap_linear_1 = nn.Linear(caption_channels, dim)
        self.cap_linear_2 = nn.Linear(dim, dim)

    def forward(self, latent, t_freq, caption):
        x = self.patchify_proj(latent)
        emb = self.t_linear_2(F.silu(self.t_linear_1(t_freq)))
        tmod = self.adaln_linear(F.silu(emb))
        context = self.cap_linear_2(F.gelu(self.cap_linear_1(caption), approximate="tanh"))
        return x, tmod, emb, context


class Head(nn.Module):
    def __init__(self, dim: int, out_channels: int):
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(2, dim))
        self.proj_out = nn.Linear(dim, out_channels)

    def forward(self, x, emb):
        mods = self.scale_shift_table.unsqueeze(0) + emb.unsqueeze(1)
        shift, scale = mods.unbind(1)
        x = F.layer_norm(x, (x.shape[-1],), eps=1e-6)
        return self.proj_out(x * (1 + scale) + shift)


class Chunk(nn.Module):
    """Transformer blocks [start, end); optionally with the prelude and/or the output head."""

    def __init__(self, cfg: dict, start: int, end: int, with_prelude: bool, with_head: bool):
        super().__init__()
        self.start, self.end = start, end
        dim, heads = cfg["dim"], cfg["heads"]
        self.prelude = Prelude(cfg["in_channels"], dim, cfg["caption_channels"]) if with_prelude else None
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(start, end)])
        self.head = Head(dim, cfg["out_channels"]) if with_head else None

    def forward(self, *args):
        if self.prelude is not None:
            latent, t_freq, caption, mask, cos, sin = args
            x, tmod, emb, context = self.prelude(latent, t_freq, caption)
        elif self.head is not None:
            x, tmod, emb, context, mask, cos, sin = args
        else:
            x, tmod, context, mask, cos, sin = args
            emb = None
        for block in self.blocks:
            x = block(x, context, tmod, cos, sin, mask)
        if self.head is not None:
            return self.head(x, emb)
        if self.prelude is not None:
            return x, tmod, emb, context
        return x

    def load_from(self, sd: dict):
        def take(name):
            return sd[PREFIX + name].float()

        own = {}
        if self.prelude is not None:
            p = "prelude."
            own[p + "patchify_proj.weight"] = take("patchify_proj.weight")
            own[p + "patchify_proj.bias"] = take("patchify_proj.bias")
            own[p + "t_linear_1.weight"] = take("adaln_single.emb.timestep_embedder.linear_1.weight")
            own[p + "t_linear_1.bias"] = take("adaln_single.emb.timestep_embedder.linear_1.bias")
            own[p + "t_linear_2.weight"] = take("adaln_single.emb.timestep_embedder.linear_2.weight")
            own[p + "t_linear_2.bias"] = take("adaln_single.emb.timestep_embedder.linear_2.bias")
            own[p + "adaln_linear.weight"] = take("adaln_single.linear.weight")
            own[p + "adaln_linear.bias"] = take("adaln_single.linear.bias")
            own[p + "cap_linear_1.weight"] = take("caption_projection.linear_1.weight")
            own[p + "cap_linear_1.bias"] = take("caption_projection.linear_1.bias")
            own[p + "cap_linear_2.weight"] = take("caption_projection.linear_2.weight")
            own[p + "cap_linear_2.bias"] = take("caption_projection.linear_2.bias")
        for local, idx in enumerate(range(self.start, self.end)):
            src = f"transformer_blocks.{idx}."
            dst = f"blocks.{local}."
            for attn in ("attn1", "attn2"):
                for part in ("to_q", "to_k", "to_v", "to_out.0"):
                    own[f"{dst}{attn}.{part}.weight"] = take(f"{src}{attn}.{part}.weight")
                    own[f"{dst}{attn}.{part}.bias"] = take(f"{src}{attn}.{part}.bias")
                own[f"{dst}{attn}.q_norm.weight"] = take(f"{src}{attn}.q_norm.weight")
                own[f"{dst}{attn}.k_norm.weight"] = take(f"{src}{attn}.k_norm.weight")
            own[f"{dst}ff.proj_in.weight"] = take(f"{src}ff.net.0.proj.weight")
            own[f"{dst}ff.proj_in.bias"] = take(f"{src}ff.net.0.proj.bias")
            own[f"{dst}ff.proj_out.weight"] = take(f"{src}ff.net.2.weight")
            own[f"{dst}ff.proj_out.bias"] = take(f"{src}ff.net.2.bias")
            own[f"{dst}scale_shift_table"] = take(f"{src}scale_shift_table")
        if self.head is not None:
            own["head.scale_shift_table"] = take("scale_shift_table")
            own["head.proj_out.weight"] = take("proj_out.weight")
            own["head.proj_out.bias"] = take("proj_out.bias")
        missing, unexpected = self.load_state_dict(own, strict=True), None
        return missing


def config_from_state_dict(sd: dict) -> dict:
    dim = sd[PREFIX + "patchify_proj.weight"].shape[0]
    layers = 1 + max(int(k[len(PREFIX):].split(".")[1]) for k in sd if k.startswith(PREFIX + "transformer_blocks."))
    return {
        "in_channels": sd[PREFIX + "patchify_proj.weight"].shape[1],
        "out_channels": sd[PREFIX + "proj_out.weight"].shape[0],
        "dim": dim,
        "heads": 32,
        "caption_channels": sd[PREFIX + "caption_projection.linear_1.weight"].shape[1],
        "layers": layers,
    }


# ---------------------------------------------------------------------------------------------
# Host-side helpers (also implemented in C++ on the device; keep both in sync).
# ---------------------------------------------------------------------------------------------

def timestep_freq(timesteps: torch.Tensor, dim: int = 256, scale: float = 1000.0) -> torch.Tensor:
    """diffusers Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0) of t*scale."""
    half = dim // 2
    exponent = -math.log(10000.0) * torch.arange(half, dtype=torch.float64) / half
    args = (timesteps.double() * scale)[:, None] * torch.exp(exponent)[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).float()


def rope_tables(width: int, height: int, frames: int, dim: int, fps: float,
                theta: float = 10000.0, max_pos=(20, 2048, 2048), scale=(8, 32, 32)):
    """cos/sin [T, dim/2] for interleaved RoPE over causal pixel coordinates (LTX-Video 0.9)."""
    n = dim // 6
    indices = torch.exp(math.log(theta) * torch.linspace(0, 1, n, dtype=torch.float64)) * (math.pi / 2)
    pad = dim // 2 - 3 * n
    coords = []
    for t in range(frames):
        pt = max(0.0, t * scale[0] + 1 - scale[0]) / fps
        for h in range(height):
            for w in range(width):
                coords.append((pt / max_pos[0], h * scale[1] / max_pos[1], w * scale[2] / max_pos[2]))
    c = torch.tensor(coords, dtype=torch.float64)  # [T, 3]
    freqs = indices[None, :, None] * (c[:, None, :] * 2 - 1)  # [T, n, 3]
    freqs = freqs.reshape(c.shape[0], 3 * n)
    freqs = torch.cat([torch.zeros(c.shape[0], pad, dtype=torch.float64), freqs], dim=1)
    return torch.cos(freqs).float(), torch.sin(freqs).float()
