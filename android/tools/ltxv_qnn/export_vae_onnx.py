"""Export the LTX-Video 0.9.5+ VAE decoder to a static-shape ONNX graph for QAIRT.

Inputs:  z [1, 128, T, H, W] already un-normalized (latent * std + mean) and noised on the host,
         temb [1] decode timestep (0.05).
Output:  video [1, 3, F, H*32, W*32] in [-1, 1].

Usage: python export_vae_onnx.py --ckpt ltxv-2b-0.9.8-distilled.safetensors --out onnx/ltxv_vae_512x320x25.onnx \
           --width 512 --height 320 --frames 25
"""

import argparse
import os

import onnx
import torch
from diffusers import AutoencoderKLLTXVideo


_movedim = torch.Tensor.movedim


def _movedim_explicit(self, source, destination):
    # diffusers' LTX VAE normalizes channels with x.movedim(1, -1); the ONNX exporter turns that
    # into a Transpose with a negative perm entry, which is invalid. Emit an explicit permute.
    if isinstance(source, int) and isinstance(destination, int):
        n = self.dim()
        src, dst = source % n, destination % n
        order = [d for d in range(n) if d != src]
        order.insert(dst, src)
        return self.permute(*order)
    return _movedim(self, source, destination)


torch.Tensor.movedim = _movedim_explicit


# The HTP supports tensors up to rank 5, but diffusers shuffles with rank-8 reshape/permute. These
# helpers produce identical results (batch 1) using only rank-5 intermediates.
def depth_to_space_3d(x, st, sh, sw):
    """[1, C*st*sh*sw, F, H, W] -> [1, C, F*st, H*sh, W*sw]; channel = c*st*sh*sw + pt*sh*sw + ph*sw + pw."""
    _, cs, f, h, w = x.shape
    c = cs // (st * sh * sw)
    x = x.reshape(c, st, sh * sw, f, h * w).permute(0, 3, 1, 2, 4)  # [c, f, pt, ph*sw+pw, hw]
    x = x.reshape(c * f * st, sh, sw, h, w).permute(0, 3, 1, 4, 2)  # [c f pt, h, ph, w, pw]
    return x.reshape(1, c, f * st, h * sh, w * sw)


def unpatchify(x, p):
    """LTX unpatchify (patch_size_t 1): channel = c*p*p + a*p + b, a -> width offset, b -> height offset."""
    _, cs, f, h, w = x.shape
    c = cs // (p * p)
    x = x.reshape(c, p, p, f, h * w).permute(0, 3, 1, 2, 4)  # [c, f, a, b, hw]
    x = x.reshape(c * f, p, p, h, w).permute(0, 3, 2, 4, 1)  # [c f, h, b, w, a]
    return x.reshape(1, c, f, h * p, w * p)


def make_rms_norm_forward(original):
    # HTP RmsNorm is limited to rank 4; the VAE normalizes rank-5 channels-last tensors.
    def forward(self, hidden_states):
        if hidden_states.dim() == 5:
            b, f, h, w, c = hidden_states.shape
            return original(self, hidden_states.reshape(b * f * h, w, c)).reshape(b, f, h, w, c)
        return original(self, hidden_states)
    return forward


def channel_mods(temb, table):
    """temb [1, k*C, 1, 1, 1] + table [k, C] -> k tensors of shape [1, C, 1, 1, 1] (no rank-6 broadcast)."""
    k, c = table.shape
    mods = temb.reshape(k, c) + table
    return [mods[i].reshape(1, c, 1, 1, 1) for i in range(k)]


def resnet_forward(self, inputs, temb=None, generator=None):
    assert self.per_channel_scale1 is None and self.per_channel_scale2 is None, "noise injection is not exported"
    hidden_states = self.norm1(inputs.movedim(1, -1)).movedim(-1, 1)
    if self.scale_shift_table is not None:
        shift_1, scale_1, shift_2, scale_2 = channel_mods(temb, self.scale_shift_table)
        hidden_states = hidden_states * (1 + scale_1) + shift_1
    hidden_states = self.conv1(self.nonlinearity(hidden_states))
    hidden_states = self.norm2(hidden_states.movedim(1, -1)).movedim(-1, 1)
    if self.scale_shift_table is not None:
        hidden_states = hidden_states * (1 + scale_2) + shift_2
    hidden_states = self.conv2(self.dropout(self.nonlinearity(hidden_states)))
    if self.norm3 is not None:
        inputs = self.norm3(inputs.movedim(1, -1)).movedim(-1, 1)
    if self.conv_shortcut is not None:
        inputs = self.conv_shortcut(inputs)
    return hidden_states + inputs


def upsampler_forward(self, hidden_states):
    st, sh, sw = self.stride
    if self.residual:
        residual = depth_to_space_3d(hidden_states, st, sh, sw)
        repeats = (st * sh * sw) // self.upscale_factor
        residual = residual.repeat(1, repeats, 1, 1, 1)[:, :, st - 1:]
    hidden_states = depth_to_space_3d(self.conv(hidden_states), st, sh, sw)[:, :, st - 1:]
    if self.residual:
        hidden_states = hidden_states + residual
    return hidden_states


def decoder_forward(self, hidden_states, temb=None):
    hidden_states = self.conv_in(hidden_states)
    if self.timestep_scale_multiplier is not None:
        temb = temb * self.timestep_scale_multiplier
    hidden_states = self.mid_block(hidden_states, temb)
    for up_block in self.up_blocks:
        hidden_states = up_block(hidden_states, temb)
    hidden_states = self.norm_out(hidden_states.movedim(1, -1)).movedim(-1, 1)
    if self.time_embedder is not None:
        temb = self.time_embedder(timestep=temb.flatten(), resolution=None, aspect_ratio=None,
                                  batch_size=hidden_states.size(0), hidden_dtype=hidden_states.dtype)
        shift, scale = channel_mods(temb.view(1, -1, 1, 1, 1), self.scale_shift_table)
        hidden_states = hidden_states * (1 + scale) + shift
    hidden_states = self.conv_out(self.conv_act(hidden_states))
    assert self.patch_size_t == 1
    return unpatchify(hidden_states, self.patch_size)


class DecoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.decoder = vae.decoder

    def forward(self, z, temb):
        return self.decoder(z, temb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--check", action="store_true", help="compare the exported graph against PyTorch")
    args = ap.parse_args()

    vae = AutoencoderKLLTXVideo.from_single_file(args.ckpt, torch_dtype=torch.float32).eval()
    model = DecoderWrapper(vae).eval()
    t = (args.frames - 1) // 8 + 1
    z = torch.randn(1, 128, t, args.height // 32, args.width // 32)
    temb = torch.tensor([0.05])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with torch.no_grad():
        ref = model(z, temb)  # original diffusers code
        import diffusers.models.autoencoders.autoencoder_kl_ltx as ltx
        ltx.LTXVideoUpsampler3d.forward = upsampler_forward
        ltx.LTXVideoResnetBlock3d.forward = resnet_forward
        type(vae.decoder).forward = decoder_forward
        from diffusers.models.normalization import RMSNorm
        RMSNorm.forward = make_rms_norm_forward(RMSNorm.forward)
        patched = model(z, temb)
        print("rank-5 rewrite vs diffusers max abs diff", (patched - ref).abs().max().item())
        torch.onnx.export(model, (z, temb), args.out, input_names=["z", "temb"], output_names=["video"],
                          opset_version=17, do_constant_folding=True, dynamo=False)
    m = onnx.load(args.out, load_external_data=True)
    onnx.save_model(m, args.out, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=os.path.basename(args.out) + ".data", size_threshold=1024)
    print("exported", args.out, "output", tuple(ref.shape))
    if args.check:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"z": z.numpy(), "temb": temb.numpy()})[0]
        err = abs(out - ref.numpy()).mean()
        print("onnxruntime vs torch mean abs err", err)


if __name__ == "__main__":
    main()
