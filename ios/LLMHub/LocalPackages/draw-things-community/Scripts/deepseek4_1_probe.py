#!/usr/bin/env python3
"""Probe downloaded V4.1 weights without loading a whole layer's expert bank.

Runs the checkpoint's selected model.py classes unchanged, with CPU replacements
for CUDA kernels and lazy, bounded weight loading. This is an architectural probe,
not a claim of bit-exact TileLang GEMM parity. Requires torch, numpy, tokenizers,
and sympy. All checkpoint access is read-only; outputs go to --output-dir.
"""

import argparse
import ast
from collections import Counter, OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import types
from typing import Literal

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tokenizers import Tokenizer


class Checkpoint:
    def __init__(self, root, cache_bytes):
        self.root = root
        self.index = json.loads((root / "model.safetensors.index.json").read_text())
        self.infos = {}
        self.files = {}
        self.cache = OrderedDict()
        self.cache_bytes = cache_bytes
        self.resident_bytes = 0
        self.bytes_read = 0
        self.row_reads = []
        for filename in sorted(set(self.index["weight_map"].values())):
            path = root / filename
            fd = os.open(path, os.O_RDONLY)
            self.files[filename] = fd
            length = int.from_bytes(os.pread(fd, 8, 0), "little")
            header = json.loads(os.pread(fd, length, 8))
            payload_end = 0
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                assert name not in self.infos, name
                assert self.index["weight_map"][name] == filename, name
                self.infos[name] = dict(info, filename=filename, base=8 + length)
                payload_end = max(payload_end, info["data_offsets"][1])
            assert path.stat().st_size == 8 + length + payload_end, filename
        assert set(self.infos) == set(self.index["weight_map"])

    def read(self, name, start=0, length=None):
        info = self.infos[name]
        lo, hi = info["data_offsets"]
        length = hi - lo - start if length is None else length
        assert start >= 0 and 0 <= length <= hi - lo - start
        data = os.pread(self.files[info["filename"]], length, info["base"] + lo + start)
        assert len(data) == length, name
        self.bytes_read += length
        return data

    @staticmethod
    def decode(data, dtype):
        a = np.frombuffer(data, dtype=np.uint8).copy()
        if dtype == "F8_E4M3":
            return torch.from_numpy(a).view(torch.float8_e4m3fn).float()
        if dtype == "F8_E8M0":
            # E8M0 255 is NaN, not a valid scale.
            assert not np.any(a == 255)
            return torch.from_numpy(np.exp2(a.astype(np.float32) - 127))
        if dtype == "BF16":
            return torch.from_numpy(a.view(np.uint16).astype(np.uint32) << 16).view(torch.float32)
        if dtype == "F32":
            return torch.from_numpy(a.view(np.float32))
        if dtype == "I8":
            return torch.from_numpy(a)
        raise ValueError(dtype)

    def tensor(self, name):
        info = self.infos[name]
        return self.decode(self.read(name), info["dtype"]).reshape(info["shape"])

    def matrix(self, name):
        if name in self.cache:
            self.cache.move_to_end(name)
            return self.cache[name]
        info = self.infos[name]
        dtype = info["dtype"]
        size = math.prod(info["shape"]) * 4 * (2 if dtype == "I8" else 1)
        while self.cache and self.resident_bytes + size > self.cache_bytes:
            _, old = self.cache.popitem(last=False)
            self.resident_bytes -= old.numel() * old.element_size()
        w = self.tensor(name)
        if dtype in ("F8_E4M3", "I8"):
            scales = self.tensor(name.removesuffix("weight") + "scale")
            if dtype == "I8":
                table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)
                w = torch.stack((table[(w & 15).long()], table[(w >> 4).long()]), -1).flatten(-2)
                assert scales.shape == (w.shape[0], w.shape[1] // 32)
                w.view(w.shape[0], -1, 32).mul_(scales[..., None])
            else:
                assert scales.shape == (w.shape[0] // 32, w.shape[1] // 32)
                w.view(w.shape[0] // 32, 32, w.shape[1] // 32, 32).mul_(scales[:, None, :, None])
        if size <= self.cache_bytes:
            self.cache[name] = w
            self.resident_bytes += size
        return w

    def rows(self, name, ids):
        info = self.infos[name]
        width = info["shape"][1]
        item_bytes = 2 if info["dtype"] == "BF16" else 1
        start = time.perf_counter()
        unique, inverse = torch.unique(ids.flatten(), sorted=True, return_inverse=True)
        data = []
        for row in unique.tolist():
            assert 0 <= row < info["shape"][0]
            value = self.decode(self.read(name, row * width * item_bytes, width * item_bytes), info["dtype"])
            if info["dtype"] == "F8_E4M3":
                scale_name = name.removesuffix("weight") + "scale"
                scales = self.decode(self.read(scale_name, row * (width // 32), width // 32), "F8_E8M0")
                value.view(-1, 32).mul_(scales[:, None])
            data.append(value)
        out = torch.stack(data)[inverse].reshape(*ids.shape, width).bfloat16()
        self.row_reads.append(dict(name=name, requested=ids.numel(), unique=unique.numel(), seconds=time.perf_counter() - start))
        return out


def conform(x, block=32, fp4=False, e4m3_scale=False):
    groups = x.float().reshape(*x.shape[:-1], -1, block)
    amax = groups.abs().amax(-1, keepdim=True)
    if fp4 and e4m3_scale:
        scale = (amax.clamp_min(6 * 2**-9) / 6).to(torch.float8_e4m3fn).float()
    else:
        minimum = 6 * 2**-126 if fp4 else 1e-4
        unrounded = amax.clamp_min(minimum) * (1.0 / (6 if fp4 else 448))
        bits = unrounded.contiguous().view(torch.int32)
        scale = ((bits & 0x7f800000) + ((bits & 0x007fffff) != 0).int() * 0x00800000).view(torch.float32)
    z = (groups / scale).clamp(-6 if fp4 else -448, 6 if fp4 else 448)
    if fp4:
        levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
        mids = (levels[:-1] + levels[1:]) / 2
        idx = torch.bucketize(z.abs().contiguous(), mids)
        # Round ties to the even E2M1 mantissa encoding.
        tie_up = (idx < 7) & (idx % 2 == 1) & (z.abs() == mids[idx.clamp_max(6)])
        idx = idx + tie_up
        z = torch.copysign(levels[idx], z)
    else:
        z = z.to(torch.float8_e4m3fn).float()
    return (z * scale).reshape(x.shape).to(x.dtype)


def hc_split(mix, scale, base, hc, iterations, eps):
    pre = torch.sigmoid(mix[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mix[..., hc:2*hc] * scale[1] + base[hc:2*hc])
    comb = (mix[..., 2*hc:] * scale[2] + base[2*hc:]).unflatten(-1, (hc, hc)).softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(1, iterations):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def sparse_attention(q, kv, sinks, indices, scale, compute_dtype=torch.float32):
    # Small probe sequences only. Preserve duplicates and the zero-valued sink.
    selected = kv[:, indices[0].clamp_min(0)]
    scores = torch.einsum("bshd,bskd->bshk", q.to(compute_dtype), selected.to(compute_dtype)) * scale
    scores.masked_fill_(indices[:, :, None, :] < 0, -torch.inf)
    scores = torch.cat((scores, sinks.float()[None, None, :, None].expand(*scores.shape[:-1], 1)), -1)
    probs = scores.softmax(-1)[..., :-1]
    return torch.einsum("bshk,bskd->bshd", probs, selected.to(compute_dtype)).to(q.dtype)


def load_reference(root, store, gemm_dtype=torch.float32):
    """Use upstream class bodies; substitute only storage and backend operations."""
    spec = importlib.util.spec_from_file_location("deepseek4_1_engram_reference", root / "inference/engram.py")
    engram = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = engram
    spec.loader.exec_module(engram)

    class LazyLinear(nn.Module):
        def __init__(self, in_features, out_features, bias=False, dtype=None):
            super().__init__()
            assert not bias
            self.dtype = dtype or torch.bfloat16
            self.path = None

        @property
        def weight(self):
            return store.matrix(self.path + ".weight").to(self.dtype)

        def forward(self, x):
            name = self.path + ".weight"
            kind = store.infos[name]["dtype"]
            # Explicit fp32 compressor projections are not activation-quantized.
            if kind in ("F8_E4M3", "I8") and self.dtype != torch.float32:
                x = conform(x)
            w = store.matrix(name)
            return F.linear(x.to(gemm_dtype), w.to(gemm_dtype)).to(x.dtype)

    class RowEmbedding(nn.Module):
        def __init__(self, num_embeddings, dim):
            super().__init__()
            self.path = None

        def forward(self, indices):
            return store.rows(self.path + ".weight", indices)

    def fp8_roundtrip(x, block_size, *unused, **kwargs):
        assert unused[-1] is True or kwargs.get("inplace") is True
        x.copy_(conform(x, block_size))
        return x

    def fp4_roundtrip(x, block_size, inplace, scale_dtype=None):
        assert inplace
        x.copy_(conform(x, block_size, fp4=True, e4m3_scale=scale_dtype == torch.float8_e4m3fn))
        return x

    names = {
        "set_dtype", "ModelArgs", "RMSNorm", "Engram", "precompute_freqs_cis", "apply_rotary_emb",
        "get_window_topk_idxs", "Compressor", "Indexer", "select_candidate_blocks", "Attention",
        "Gate", "Expert", "MoE", "Block", "ParallelHead", "make_identity_pre_mix", "SharedAttentionRuntime",
    }
    tree = ast.parse((root / "inference/model.py").read_text())
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    module = types.ModuleType("deepseek4_1_model_reference")
    sys.modules[module.__name__] = module
    diagnostic_f = types.SimpleNamespace(**{name: getattr(F, name) for name in dir(F)})
    diagnostic_f.linear = lambda x, w, bias=None: F.linear(x.to(gemm_dtype), w.to(gemm_dtype), None if bias is None else bias.to(gemm_dtype)).to(x.dtype)
    diagnostic_attention = lambda q, kv, sinks, indices, scale: sparse_attention(q, kv, sinks, indices, scale, compute_dtype=gemm_dtype)
    module.__dict__.update(dict(
        torch=torch, nn=nn, F=diagnostic_f, math=math, contextmanager=contextmanager,
        dataclass=dataclass, lru_cache=lru_cache, Literal=Literal,
        world_size=1, rank=0, default_dtype=torch.bfloat16,
        Linear=LazyLinear, ColumnParallelLinear=LazyLinear, RowParallelLinear=LazyLinear,
        ParallelEngramEmbedding=RowEmbedding, EngramLayout=engram.EngramLayout,
        fp8_block_size=32, fp4_block_size=32, scale_fmt="ue8m0", scale_dtype=torch.float32,
        act_quant=fp8_roundtrip, fp4_act_quant=fp4_roundtrip,
        sparse_attn=diagnostic_attention, hc_split_sinkhorn=hc_split,
        linear=diagnostic_f.linear,
    ))
    exec(compile(tree, str(root / "inference/model.py"), "exec"), module.__dict__)
    module.shared_attn = module.SharedAttentionRuntime()

    def bind(model, prefix):
        for suffix, child in model.named_modules():
            path = prefix + ("." + suffix if suffix else "")
            if isinstance(child, (LazyLinear, RowEmbedding)):
                child.path = path
            for key, value in child.named_parameters(recurse=False):
                name = path + "." + key
                source = store.tensor(name)
                assert source.shape == value.shape, (name, source.shape, value.shape)
                value.data = source.to(value.dtype)
        return model.eval()

    return module, engram, bind


def stats(x):
    x = x.float()
    assert torch.isfinite(x).all()
    return dict(shape=list(x.shape), rms=float(x.square().mean().sqrt()), max_abs=float(x.abs().max()))


def comparison(x, y):
    # Float64 reductions keep cosine within its meaningful numerical range even
    # for long vectors with strongly unequal token magnitudes.
    x, y = x.double().flatten(), y.double().flatten()
    return dict(max_abs=float((x-y).abs().max()), relative_l2=float((x-y).norm() / y.norm().clamp_min(1e-30)), cosine=float(F.cosine_similarity(x[None], y[None])))


class Fixtures:
    def __init__(self, root):
        self.root = root
        self.entries = {}

    def save(self, name, value):
        value = value.detach().cpu().contiguous()
        kind = "int32" if value.dtype in (torch.int32, torch.int64) else "float32"
        a = value.to(torch.int32 if kind == "int32" else torch.float32).numpy()
        filename = name + ".bin"
        a.tofile(self.root / filename)
        self.entries[name] = dict(shape=list(a.shape), dtype=kind, file=filename)

    def finish(self):
        (self.root / "fixtures.json").write_text(json.dumps(self.entries, indent=2) + "\n")


def output_head_fixtures(store, fixtures, ref, token_ids):
    """Use bounded vocabulary slices and the official head/collapse methods."""
    def read(name):
        entry = fixtures.entries[name]
        return torch.from_numpy(np.fromfile(fixtures.root / entry['file'], dtype=np.float32).reshape(entry['shape']))

    residual = read('composed_layer2_prefill').bfloat16()[None]
    pre = read('composed_layer2_prefill_pre')[None]
    collapsed = ref.Block.hc_pre(None, residual, pre)
    norm = ref.RMSNorm(5120, 1e-20)
    norm.weight = nn.Parameter(store.tensor('norm.weight').bfloat16())
    normalized = norm(collapsed)
    fixtures.save('output_head_normalized', normalized[0])
    outputs = []
    info = store.infos['head.weight']
    for start in range(0, info['shape'][0], 2048):
        rows = min(2048, info['shape'][0] - start)
        data = store.read('head.weight', start * 5120 * 2, rows * 5120 * 2)
        weights = store.decode(data, 'BF16').reshape(rows, 5120)
        outputs.append(ref.ParallelHead.forward(types.SimpleNamespace(weight=weights), normalized, full_logits=True))
    fixtures.save('output_head_logits', torch.cat(outputs, -1)[0])
    embedding = store.rows('embed.weight', token_ids).bfloat16()
    fixtures.save('embedding_residual', embedding.unsqueeze(2).repeat(1, 1, 4, 1)[0])


def operation_fixtures(fixtures):
    generator = torch.Generator().manual_seed(4101)
    x = torch.randn(17, 512, generator=generator, dtype=torch.float32) * 13
    # FP4 midpoint ties, zero blocks, scale rounding ties, and distinct block maxima.
    x[0] = 0
    x[1] = torch.tensor([6, .25, .75, 1.25, 1.75, 2.5, 3.5, 5, -6, -.25, -.75, -1.25, -1.75, -2.5, -3.5, -5], dtype=torch.float32).repeat(32)
    x[2] = torch.tensor([448, 1.0625, 1.1875, -1.0625, -1.1875, .0009765625, .0029296875, 0], dtype=torch.float32).repeat(64)
    for i, scale in enumerate((2**-20, 2**-10, .1328125, .1484375, 1, 2, 16), 3):
        x[i] = x[1] * scale
    x[10] = torch.linspace(-1e-25, 1e-25, 512, dtype=torch.float32)
    fixtures.save('conform_input', x)
    for dtype, suffix in ((torch.float32, 'f32'), (torch.bfloat16, 'bf16'), (torch.float16, 'f16')):
        # Supply the same already-rounded input to every backend. ccv's generic
        # CPU float conversion truncates, whereas PyTorch and Metal round nearest.
        fixtures.save(f'conform_input_{suffix}', x.to(dtype))
    for name, block, fp4, e4m3_scale, tail in (
        ('fp8_64_tail', 64, False, False, 64),
        ('fp8_32', 32, False, False, 0),
        ('fp4_32_e8m0', 32, True, False, 0),
        ('fp4_16_e4m3', 16, True, True, 0),
        ('fp8_128', 128, False, False, 0),
    ):
        for dtype, suffix in ((torch.float32, 'f32'), (torch.bfloat16, 'bf16'), (torch.float16, 'f16')):
            values = x.to(dtype)
            prefix = values[:, :512-tail].contiguous() if tail else values
            expected = conform(prefix, block, fp4, e4m3_scale)
            if tail:
                expected = torch.cat((expected, values[:, -tail:]), -1)
            fixtures.save(f'conform_{name}_{suffix}', expected)
    dots = torch.tensor([0.0, -0.0, 1e-30, -1e-30, 1e-8, -1e-8, 1e-6, -1e-6, 1, -1, 100, -100], dtype=torch.float32)
    dots = torch.cat((dots, torch.randn(1028, generator=generator, dtype=torch.float32) * 10)).reshape(13, 80)
    fixtures.save('signed_sqrt_input', dots)
    fixtures.save('signed_sqrt_output', torch.copysign(dots.abs().clamp_min(1e-6).sqrt(), dots))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix-layers", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=5)
    parser.add_argument("--gemm-precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--weight-cache-mib", type=int, default=768)
    args = parser.parse_args()
    assert 3 <= args.tokens <= 16 and 3 <= args.prefix_layers <= 4
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(41)
    started = time.perf_counter()
    store = Checkpoint(args.models_dir, args.weight_cache_mib * 2**20)
    report = dict(method=f"Official class bodies; CPU {args.gemm_precision} GEMMs with explicit BF16 output and FP8/FP4 activation/cache roundtrips; no CUDA kernel parity claim", torch=torch.__version__, checkpoint=str(args.models_dir), header_count=len(store.infos), shard_count=len(store.files))
    report["source_sha256"] = {p: hashlib.sha256((args.models_dir / p).read_bytes()).hexdigest() for p in ("config.json", "inference/model.py", "inference/engram.py", "tokenizer.json")}
    report["dtype_counts"] = dict(Counter(i["dtype"] for i in store.infos.values()))
    report["source_layers"] = {part: sorted({int(n.split('.')[1]) for n in store.infos if n.startswith('layers.') and part in n}) for part in (".compressor.", ".indexer.", ".engram.")}
    print(json.dumps({k: report[k] for k in ("header_count", "shard_count", "dtype_counts", "source_layers")}), flush=True)
    report["gemm_precision"] = args.gemm_precision
    ref, engram, bind = load_reference(args.models_dir, store, torch.float64 if args.gemm_precision == "float64" else torch.float32)
    config = json.loads((args.models_dir / "inference/config.json").read_text())
    config.update(max_batch_size=1, max_seq_len=256, vision_n_layers=0, dspark_block_size=0, expert_dtype=None)
    cfg = ref.ModelArgs(**config)
    fixtures = Fixtures(args.output_dir)
    operation_fixtures(fixtures)
    positions = torch.tensor([0, 127, 128, 65535, 65536, 262143, 1048575], dtype=torch.long)
    fixtures.save('rotary_long_positions', positions)
    for compressed in (False, True):
        freqs = ref.precompute_freqs_cis(64, int(positions[-1]) + 1, cfg.original_seq_len if compressed else 0,
            cfg.compress_rope_theta if compressed else cfg.rope_theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow)
        fixtures.save('rotary_long_compressed' if compressed else 'rotary_long_raw', torch.view_as_real(freqs[positions]).flatten(-2))

    class TokenizerAdapter:
        backend_tokenizer = Tokenizer.from_file(str(args.models_dir / "tokenizer.json"))

        def __len__(self):
            return self.backend_tokenizer.get_vocab_size(with_added_tokens=True)

    tokenizer = TokenizerAdapter()
    prompt = "<｜begin▁of▁sentence｜><｜User｜>Explain why the sky is blue.<｜Assistant｜></think>"
    ids = torch.tensor(tokenizer.backend_tokenizer.encode(prompt, add_special_tokens=False).ids[:args.tokens], dtype=torch.int64)[None]
    report["token_ids"] = ids[0].tolist()
    layout = engram.EngramLayout.from_args(cfg)
    hash_state = engram.NgramHashState(cfg, layout, tokenizer)
    hash_full = hash_state(ids, 0)
    hash_chunks = torch.cat([hash_state(ids[:, :2], 0)] + [hash_state(ids[:, i:i+1], i) for i in range(2, args.tokens)], 1)
    assert torch.equal(hash_full, hash_chunks)
    report["hash_chunk_exact"] = True
    report["compressed_vocabulary"] = int(hash_state.token_map.max()) + 1
    (args.output_dir / "engram_hash_constants.json").write_text(json.dumps({k: getattr(hash_state, k).tolist() for k in ("token_map", "multipliers", "primes", "offsets")}, separators=(",", ":")) + "\n")
    fixtures.save("hash_ids", hash_full)
    tables = {}
    for layer_id in layout.layer_ids:
        value = store.infos[f'layers.{layer_id}.engram.embed.weight']
        scale = store.infos[f'layers.{layer_id}.engram.embed.scale']
        tables[str(layer_id)] = dict(rows=value['shape'][0], width=value['shape'][1],
            values=dict(file=value['filename'], offset=value['base'] + value['data_offsets'][0]),
            scales=dict(file=scale['filename'], offset=scale['base'] + scale['data_offsets'][0]))
    (args.output_dir / 'engram_tables.json').write_text(json.dumps(tables, indent=2) + '\n')
    print(f'Engram hash map agrees for a two-token prefill followed by {args.tokens - 2} single-token steps.', flush=True)
    layers = [bind(ref.Block(i, cfg, layout), f"layers.{i}") for i in range(args.prefix_layers)]
    embedding = store.rows("embed.weight", ids)

    def prefix(segments, capture=False):
        outputs = [[] for _ in layers]
        routes = [[] for _ in layers]
        traces = {}
        pos = 0
        for length in segments:
            h = embedding[:, pos:pos+length, None].repeat(1, 1, 4, 1)
            pre = ref.make_identity_pre_mix(h, 4)
            hashes = hash_state(ids[:, pos:pos+length], pos)
            for i, layer in enumerate(layers):
                t = time.perf_counter()
                handles = []
                if i == 0:
                    for name, module in layer.named_modules():
                        if name in ('attn_norm', 'attn.wq_a', 'attn.q_norm', 'attn.wq_b', 'attn.wkv', 'attn.kv_norm', 'attn', 'ffn_norm', 'ffn.shared_experts', 'ffn'):
                            def trace(module, inputs, result, stage=name):
                                traces.setdefault(stage, []).append(result.detach().clone().reshape(length, -1))
                            handles.append(module.register_forward_hook(trace))
                if layer.engram is not None:
                    before = h.clone()
                    h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index])
                    if capture:
                        report["engram_layer1_effect"] = comparison(h, before)
                if capture and i == 2:
                    flat = h.flatten(2).float()
                    mix = F.linear(flat, layer.hc_attn_fn) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + cfg.norm_eps)
                    a, p, c = layer.hc_mixes(h, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base)
                    for name, x in dict(hc_residual=h[0], hc_fn=layer.hc_attn_fn, hc_mix=mix[0], hc_scale=layer.hc_attn_scale, hc_base=layer.hc_attn_base, hc_pre=a[0], hc_post=p[0], hc_comb=c[0], hc_incoming_pre=pre[0], hc_weighted=(h.float()*pre[..., None]).sum(2)[0]).items():
                        fixtures.save(name, x)
                    report["shifted_pre_mix_effect"] = comparison(layer.hc_pre(h, pre), layer.hc_pre(h, a))
                    fixtures.save("hc_weighted_rounded", layer.hc_pre(h, pre)[0])
                    def capture_hc_branch(module, inputs, result):
                        fixtures.save("hc_branch", result[0])
                        fixtures.save("hc_expanded", layer.hc_post(result, h, p, c)[0])
                    handles.append(layer.attn.register_forward_hook(capture_hc_branch))
                hook = layer.ffn.gate.register_forward_hook(
                    lambda module, inputs, result, layer_index=i: routes[layer_index].append(result[1].detach().clone()))
                h, pre = layer(h, pos, pre, None)
                hook.remove()
                for handle in handles:
                    handle.remove()
                outputs[i].append(h.detach().clone())
                print(json.dumps(dict(phase='prefill' if capture else 'continuation', layer=i, position=pos, tokens=length, seconds=round(time.perf_counter()-t,3), output=stats(h))), flush=True)
            pos += length
        return [torch.cat(x, 1) for x in outputs], [torch.cat(x, 0) for x in routes], {k: torch.cat(v, 0) for k, v in traces.items()}

    with torch.inference_mode():
        full, full_routes, full_trace = prefix([args.tokens], capture=True)
        chunked, chunked_routes, chunked_trace = prefix([2] + [1] * (args.tokens - 2))
        report["prefix_parity"] = {str(i): comparison(chunked[i], full[i]) for i in range(len(layers))}
        report["prefix_per_token_parity"] = {str(i): [comparison(chunked[i][:, j], full[i][:, j]) for j in range(args.tokens)] for i in range(len(layers))}
        report["prefix_router_agreement"] = {str(i): (chunked_routes[i].sort(-1).values == full_routes[i].sort(-1).values).all(-1).tolist() for i in range(len(layers))}
        report['layer0_stage_parity'] = {k: [comparison(chunked_trace[k][j], v[j]) for j in range(args.tokens)] for k, v in full_trace.items()}
        report["prefix_outputs"] = {str(i): stats(x) for i, x in enumerate(full)}
        fixtures.save("prefix_layer2", full[2])
        print('Prefix parity: ' + json.dumps(report['prefix_parity']), flush=True)

        # Probe learned source/reindex projections on synthetic normalized inputs.
        # These are isolated components, not later activations of the true model.
        x = torch.randn(1, 5, cfg.dim, dtype=torch.float32).bfloat16()
        report["isolated_attention"] = {}
        for i in (2, 8, 14, 20, 21, 24):
            attention = bind(ref.Attention(i, cfg), f"layers.{i}.attn")
            t = time.perf_counter()
            source_kv = ref.shared_attn.compress_kv
            source_indices = ref.shared_attn.topk_idxs
            captured = {}
            reference_attention = ref.sparse_attn
            if i == 20:
                def capture_attention(q, kv, sinks, indices, scale):
                    y = reference_attention(q, kv, sinks, indices, scale)
                    captured.update(q=q.clone(), kv=kv.clone(), sinks=sinks.clone(), indices=indices.clone(), output=y.clone())
                    return y
                ref.sparse_attn = capture_attention
            y = attention(x, 0)
            ref.sparse_attn = reference_attention
            entry = dict(output=stats(y), seconds=time.perf_counter()-t)
            if i == 21:
                entry['kv_reused'] = ref.shared_attn.compress_kv is source_kv
                entry['indices_reused'] = ref.shared_attn.topk_idxs is source_indices
                assert entry['kv_reused'] and entry['indices_reused']
            if i == 24:
                entry['kv_reused'] = ref.shared_attn.compress_kv is source_kv
                assert entry['kv_reused']
            if i in (2, 20):
                comp = attention.compressor
                complete = comp(x, 0)
                pieces = [comp(x[:, :1], 0)] + [comp(x[:, j:j+1], j) for j in range(1, 5)]
                joined = torch.cat([p for p in pieces if p is not None], 1)
                entry['compressor_continuation'] = comparison(joined, complete)
                fixtures.save(f'compressor{i}_input', x[0].float())
                fixtures.save(f'compressor{i}_wkv', store.matrix(f'layers.{i}.attn.compressor.wkv.weight'))
                fixtures.save(f'compressor{i}_norm', comp.norm.weight)
                if i == 2:
                    fixtures.save(f'compressor{i}_wgate', store.matrix(f'layers.{i}.attn.compressor.wgate.weight'))
                fixtures.save(f'compressor{i}_output', complete[0].float())
            if i == 20:
                for name, value in dict(
                    attention_q=captured['q'],
                    attention_raw=captured['kv'][:, :5, None],
                    attention_global=captured['kv'][:, 5:, None],
                    attention_sinks=captured['sinks'].reshape(1, 1, 64, 1),
                    attention_indices=torch.where(ref.shared_attn.topk_idxs[0] >= 0, ref.shared_attn.topk_idxs[0] - 5, -1),
                    attention_output=captured['output'],
                ).items():
                    fixtures.save(name, value)
                normalized_q = (captured['q'].float() * torch.rsqrt(captured['q'].float().square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
                wrong = sparse_attention(normalized_q, captured['kv'], captured['sinks'], captured['indices'], 512**-0.5)
                entry['old_query_norm_effect'] = comparison(wrong, captured['output'])
                qr = attention.q_norm(attention.wq_a(x))
                q = attention.wq_b(qr).unflatten(-1, (64, 512))
                entry['query_before_rope'] = stats(q)
                fixtures.save('projection_input', x[0])
                fixtures.save('projection_query', captured['q'])
                fixtures.save('projection_raw', captured['kv'][:, :5, None])
                fixtures.save('projection_rank', qr[0])
                fixtures.save('projection_output', y[0])
                unrotated = captured['output'].clone()
                ref.apply_rotary_emb(unrotated[..., -64:], attention.freqs_cis[:5], True)
                low = torch.einsum('bsgd,grd->bsgr', unrotated.reshape(1, 5, 8, 4096), attention.wo_a.weight.reshape(8, 1024, 4096)).flatten(2)
                fixtures.save('projection_unrotated', unrotated)
                fixtures.save('projection_low', low[0])
                fixtures.save('projection_low_quantized', conform(low)[0])
                for width, label in ((512, 'projection_rotary'), (128, 'indexer_rotary')):
                    rotary = torch.zeros(5, width // 2, 2, dtype=torch.float32)
                    rotary[..., 0] = 1
                    rotary[:, -32:] = torch.view_as_real(attention.freqs_cis[:5])
                    fixtures.save(label, rotary.reshape(1, 5, 1, width))
                for suffix in ('wq_a.weight', 'wq_b.weight', 'wkv.weight', 'q_norm.weight', 'kv_norm.weight', 'wo_b.weight', 'indexer.wq_b.weight', 'indexer.wk.weight', 'indexer.k_norm.weight', 'indexer.weights_proj.weight'):
                    fixtures.save('projection_' + suffix.replace('.', '_'), store.matrix('layers.20.attn.' + suffix))
                # convert.py promotes the stored FP8 wo_a to BF16 before grouped GEMM.
                fixtures.save('projection_wo_a_weight', attention.wo_a.weight.reshape(8, 1024, 4096))
                indexer = attention.indexer
                index_q = indexer.wq_b(qr).unflatten(-1, (32, 128))
                ref.apply_rotary_emb(index_q[..., -64:], attention.freqs_cis[:5])
                fixtures.save('projection_index_q', conform(index_q, 32, fp4=True)[0])
                fixtures.save('projection_index_weights', (indexer.weights_proj(x) * (128**-.5 * 32**-.5))[0])
                latent = attention.compressor(x, 0)
                fixtures.save('projection_latent', latent[0])
                index_k = indexer.k_norm(indexer.wk(latent))
                ref.apply_rotary_emb(index_k[..., -64:], attention.freqs_cis[:5])
                fixtures.save('projection_index_k', conform(index_k, 32, fp4=True)[0])
            if i in (2, 20):
                long_x = torch.randn(1, 129, cfg.dim, dtype=torch.float32).bfloat16()
                whole = attention(long_x, 0)
                continued = torch.cat([attention(long_x[:, :127], 0), attention(long_x[:, 127:128], 127), attention(long_x[:, 128:129], 128)], 1)
                entry['window_127_1_1_parity'] = comparison(continued, whole)
                entry['window_last_token_parity'] = comparison(continued[:, -1], whole[:, -1])
                # Restore the five-token source state before the Reuse/Reindex
                # layers below consume it in the same-query ownership probe.
                attention(x, 0)
            report['isolated_attention'][str(i)] = entry
            print('Isolated attention: ' + json.dumps({str(i): entry}), flush=True)

        # Engram at the second site: real row reads and projection, synthetic residual.
        module = bind(ref.Engram(cfg, 14, layout), 'layers.14.engram')
        residual = torch.randn(1, args.tokens, 4, cfg.dim, dtype=torch.float32).bfloat16()
        output = module(residual, hash_full[:, :, 1])
        report['engram_layer14_effect'] = comparison(output, residual)
        embedding_rows = module.embed(hash_full[:, :, 1]).flatten(-2)
        fixtures.save('engram_embeddings', embedding_rows[0])
        fixtures.save('engram_wkv', store.matrix('layers.14.engram.wkv.weight'))
        kv = module.wkv(embedding_rows)
        fixtures.save('engram_residual', residual[0])
        fixtures.save('engram_kv', kv[0])
        fixtures.save('engram_q_weight', module.q_weight)
        fixtures.save('engram_k_weight', module.k_weight)
        fixtures.save('engram_output', output[0])
        key = kv[..., :4*cfg.dim].float().unflatten(-1, (4, cfg.dim))
        h = residual.float()
        rstd = torch.rsqrt(h.square().mean(-1) + cfg.norm_eps) * torch.rsqrt(key.square().mean(-1) + cfg.norm_eps)
        dot = (h * module.q_weight.float() * module.k_weight.float() * key).sum(-1) * rstd * cfg.dim**-.5
        fixtures.save('engram_dot', dot[0, :, :, None])

        # Real 384-expert router weights on a representative normalized residual.
        router = bind(ref.Gate(2, cfg), 'layers.2.ffn.gate')
        rx = layers[2].ffn_norm(full[2][:, -1].mean(1))
        weights, selected = router(rx)
        logits = F.linear(rx.float(), router.weight.float())
        fixtures.save('router_logits', logits)
        fixtures.save('router_bias', router.bias)
        fixtures.save('router_activation', rx)
        fixtures.save('router_weights', weights)
        fixtures.save('router_selected', selected)
        report['router_selected'] = selected.tolist()

        # Exercise learned FP8 shared and packed-FP4 routed experts, including
        # saturated gate/up activations and nontrivial routing weights.
        expert_x = torch.cat((full_trace['ffn_norm'][:5], torch.randn(3, cfg.dim, dtype=torch.float32).bfloat16() * 20))
        expert_weights = torch.linspace(.05, .7, expert_x.shape[0], dtype=torch.float32)[:, None]
        fixtures.save('expert_input', expert_x)
        fixtures.save('expert_route_weights', expert_weights)
        for label, path, weighted in (
            ('shared', 'layers.0.ffn.shared_experts', False),
            ('routed', 'layers.0.ffn.experts.0', True),
        ):
            expert = bind(ref.Expert(cfg.dim, cfg.moe_inter_dim, swiglu_limit=10), path)
            output = expert(expert_x, expert_weights if weighted else None)
            fixtures.save(f'expert_{label}_output', output)
            for projection in ('w1', 'w2', 'w3'):
                fixtures.save(f'expert_{label}_{projection}', store.matrix(f'{path}.{projection}.weight'))

        # A bounded three-expert bank uses the checkpoint's full-width matrices
        # and first three gate rows, so both resident and streamed MoE paths can
        # be checked without materializing a complete 384-expert layer.
        gate_weight = layers[0].ffn.gate.weight[:3].float()
        gate_bias = layers[0].ffn.gate.bias[:3]
        scores = F.softplus(F.linear(expert_x.float(), gate_weight)).sqrt()
        routes = (scores + gate_bias).topk(2, dim=-1).indices
        route_weights = scores.gather(1, routes)
        route_weights = route_weights / (route_weights.sum(-1, keepdim=True) + 1e-20) * 1.5
        total = torch.zeros_like(expert_x, dtype=torch.float32)
        for expert_id in range(3):
            expert = bind(ref.Expert(cfg.dim, cfg.moe_inter_dim, swiglu_limit=10), f'layers.0.ffn.experts.{expert_id}')
            token, slot = torch.where(routes == expert_id)
            if token.numel():
                total[token] += expert(expert_x[token], route_weights[token, slot, None])
        total += layers[0].ffn.shared_experts(expert_x)
        fixtures.save('expert_moe_output', total.bfloat16())
        fixtures.save('expert_moe_gate', gate_weight)
        fixtures.save('expert_moe_bias', gate_bias)
        for projection in ('w1', 'w2', 'w3'):
            bank = torch.stack([store.matrix(f'layers.0.ffn.experts.{i}.{projection}.weight') for i in range(3)])
            fixtures.save(f'expert_moe_{projection}', bank)

        # H=32, D=128 and 1025 keys ensures actual scoring rather than the
        # current operator's <=512-keys enumeration shortcut.
        iq = torch.randn(1, 32, 128, dtype=torch.float32)
        ik = torch.randn(1025, 128, dtype=torch.float32)
        iw = torch.randn(1, 32, dtype=torch.float32) / math.sqrt(128 * 32)
        score = (torch.einsum('thd,cd->thc', iq, ik).relu() * iw[..., None]).sum(1)
        selected = score.topk(512, -1).indices.sort(-1).values.int()
        for name, value in dict(index_q=iq, index_k=ik, index_weights=iw, index_selected=selected).items():
            fixtures.save(name, value)

        # V4.1 scores round at all three BF16 boundaries. Retain scores as the
        # oracle because top-k may choose different but equally ranked rows at
        # a BF16 tie; token-id equality is not a portable reference contract.
        for label, width, tokens, offset, ratio in (
            ('indexer_bf16', 1025, 3, 2047, 2),
            ('indexer_early', 5, 11, 0, 2),
            ('indexer_candidates', 16393, 3, 16390, 1),
            ('indexer_candidates_early', 9, 9, 0, 1),
        ):
            iq = conform(torch.randn(tokens, 32, 128, dtype=torch.float32).bfloat16(), fp4=True)
            ik = conform(torch.randn(width, 128, dtype=torch.float32).bfloat16(), fp4=True)
            iw = (torch.randn(tokens, 32, dtype=torch.float32) / math.sqrt(128 * 32)).bfloat16()
            # FP4 powers-of-two make the FP32 dot exact at these dimensions.
            dots = torch.einsum('thd,cd->thc', iq.float(), ik.float()).bfloat16()
            scores = (dots.relu() * iw[..., None]).sum(1)
            visible = (torch.arange(tokens) + offset + 1) // ratio
            scores = scores.float().masked_fill(torch.arange(width) >= visible[:, None], -torch.inf)
            for suffix, value in dict(q=iq, k=ik, weights=iw, scores=scores, visible=visible).items():
                fixtures.save(f'{label}_{suffix}', value)
            if 'candidates' in label:
                # Make the partial newest block's score low to prove it is pinned.
                # Keys for it are zero, so this does not bypass the actual scorer.
                ik[-1].zero_()
                dots = torch.einsum('thd,cd->thc', iq.float(), ik.float()).bfloat16()
                scores = (dots.relu() * iw[..., None]).sum(1).float()
                scores.masked_fill_(torch.arange(width) >= visible[:, None], -torch.inf)
                fixtures.save(f'{label}_k', ik)
                fixtures.save(f'{label}_scores', scores)
                mask = ref.select_candidate_blocks(scores, visible[:, None], 2048, 8)
                fixtures.save(f'{label}_mask', mask.int())
                # A later source uses independently trained query/head weights.
                later_q = conform(torch.randn(tokens, 32, 128, dtype=torch.float32).bfloat16(), fp4=True)
                later_w = (torch.randn(tokens, 32, dtype=torch.float32) / math.sqrt(128 * 32)).bfloat16()
                later_dots = torch.einsum('thd,cd->thc', later_q.float(), ik.float()).bfloat16()
                later_scores = (later_dots.relu() * later_w[..., None]).sum(1).float()
                later_scores.masked_fill_(torch.arange(width) >= visible[:, None], -torch.inf)
                for suffix, value in dict(later_q=later_q, later_weights=later_w, later_scores=later_scores).items():
                    fixtures.save(f'{label}_{suffix}', value)

        width = 16393  # More than 2048 blocks, including a partial newest block.
        score = torch.randn(1, 1, width, dtype=torch.float32)
        score[..., -1] = -100
        candidates = ref.select_candidate_blocks(score, width, 2048, 8)
        assert candidates[..., -1].all()
        assert candidates.sum() <= 2048 * 8
        later = torch.randn_like(score)
        dense = later.masked_fill(~candidates, -torch.inf).topk(512).indices.sort().values
        positions = candidates[0, 0].nonzero().flatten()
        compact = positions[later[0, 0, positions].topk(512).indices].sort().values
        assert torch.equal(dense[0, 0], compact)
        report['candidate_probe'] = dict(width=width, retained=int(candidates.sum()), newest_partial_block_retained=True, dense_mask_vs_compact_exact=True)

    # Compose complete attention with the actual checkpoint projections and
    # source-owned caches. These inputs are normalized synthetic sublayer inputs;
    # they are not a substitute for complete transformer-layer parity.
    with torch.inference_mode():
        attention_layers = (0, 2, 20, 21, 24)
        attentions = {i: bind(ref.Attention(i, cfg), f'layers.{i}.attn') for i in attention_layers}
        generator = torch.Generator().manual_seed(4102)
        attention_inputs = {i: torch.randn(1, 129, cfg.dim, generator=generator, dtype=torch.float32).bfloat16() for i in attention_layers}
        report['attention_composition'] = {}
        for i, attention in attentions.items():
            label = f'composed_attention{i}'
            fixtures.save(f'{label}_input', attention_inputs[i][0])
            prefix = f'layers.{i}.attn.'
            for name in sorted(store.infos):
                if not name.startswith(prefix) or name.endswith('.scale'):
                    continue
                suffix = name[len(prefix):]
                if suffix.endswith('.weight') or suffix == 'attn_sink':
                    value = store.matrix(name)
                    if suffix == 'wo_a.weight':
                        value = value.bfloat16().reshape(cfg.o_groups, cfg.o_lora_rank, -1)
                    fixtures.save(f'{label}_' + suffix.replace('.', '_'), value)
        for label, segments in (('prefill129', [129]), ('window127_1_1', [127, 1, 1]), ('prefill9', [9]), ('decode9', [1] * 9)):
            position = 0
            gathered = {i: [] for i in attention_layers}
            for length in segments:
                for i, attention in attentions.items():
                    original_attention = ref.sparse_attn
                    if label in ('prefill129', 'prefill9'):
                        def capture_heads(q, kv, sinks, indices, scale):
                            heads = original_attention(q, kv, sinks, indices, scale)
                            prefix = f'attention_heads{i}_{label}'
                            fixtures.save(prefix + '_q', q)
                            fixtures.save(prefix + '_raw', kv[:, :length, None])
                            fixtures.save(prefix + '_sinks', sinks)
                            fixtures.save(prefix + '_output', heads)
                            if attention.compress_ratio:
                                fixtures.save(prefix + '_global', kv[:, length:, None])
                                selected = indices[0, :, min(length, cfg.window_size):]
                                fixtures.save(prefix + '_indices', torch.where(selected >= 0, selected - length, -1))
                            return heads
                        ref.sparse_attn = capture_heads
                    y = attention(attention_inputs[i][:, position:position+length], position)
                    ref.sparse_attn = original_attention
                    gathered[i].append(y[0])
                position += length
            for i, attention in attentions.items():
                fixture = f'composed_attention{i}_{label}'
                value = torch.cat(gathered[i], 0)
                fixtures.save(fixture, value)
                # Convert the reference ring into chronological order; cache
                # layout differs but the values and token positions must agree.
                raw_ids = torch.arange(max(0, position - cfg.window_size), position) % cfg.window_size
                fixtures.save(fixture + '_raw', attention.window_kv_cache[0, raw_ids])
                if attention.is_kv_source:
                    rows = position // attention.compress_ratio
                    fixtures.save(fixture + '_global', attention.compress_kv_cache[0, :rows])
                    fixtures.save(fixture + '_index', attention.indexer.k_cache[0, :rows])
                if label in ('prefill129', 'prefill9'):
                    report['attention_composition'][f'{i}_{label}'] = stats(value)
            print(f'Composed attention reference {label} complete.', flush=True)

    # Complete HC/attention/FFN composition with real projection matrices and a
    # bounded three-expert bank. Keep the full 5120/2304 dimensions; only routing
    # is narrowed to the first three checkpoint experts, with top-2 selection.
    with torch.inference_mode():
        generator = torch.Generator().manual_seed(4103)
        count = ids.size(1)
        report['layer_composition'] = dict(experts=3, top_k=2, layers={})
        for i in (0, 1, 2, 20):
            layer = bind(ref.Block(i, cfg, layout), f'layers.{i}')
            layer.ffn.gate.weight.data = layer.ffn.gate.weight[:3].clone()
            layer.ffn.gate.bias.data = layer.ffn.gate.bias[:3].clone()
            layer.ffn.gate.topk = 2
            layer.ffn.experts = layer.ffn.experts[:3]
            layer.ffn.n_routed_experts = layer.ffn.n_local_experts = layer.ffn.experts_end_idx = 3
            layer.ffn.n_activated_experts = 2
            label = f'composed_layer{i}'
            h = torch.randn(1, count, cfg.hc_mult, cfg.dim, generator=generator, dtype=torch.float32).bfloat16()
            incoming = torch.rand(1, count, cfg.hc_mult, generator=generator, dtype=torch.float32)
            incoming /= incoming.sum(-1, keepdim=True)
            fixtures.save(label + '_input', h[0])
            fixtures.save(label + '_incoming_pre', incoming[0])
            if layer.engram is not None:
                hashes = hash_full[:, :, layer.engram.layer_hash_index]
                embeddings = store.rows(f'layers.{i}.engram.embed.weight', hashes).flatten(2)
                fixtures.save(label + '_engram_embeddings', embeddings[0])
            for name in sorted(store.infos):
                if not name.startswith(f'layers.{i}.'):
                    continue
                suffix = name[len(f'layers.{i}.'):]
                if suffix.endswith('.scale') or suffix.startswith(('ffn.experts.', 'engram.embed.')) or suffix.endswith('bias_vl'):
                    continue
                parameter = suffix.removesuffix('.weight')
                destination = label + '_param_' + parameter.replace('.', '_')
                attention_fixture = f'composed_attention{i}_' + suffix.removesuffix('.weight').replace('.', '_').removeprefix('attn_')
                attention_fixture += '' if suffix == 'attn.attn_sink' else '_weight'
                if suffix.startswith('attn.') and attention_fixture in fixtures.entries:
                    fixtures.entries[destination] = fixtures.entries[attention_fixture]
                    continue
                value = store.matrix(name)
                if suffix == 'attn.wo_a.weight':
                    value = value.bfloat16().reshape(cfg.o_groups, cfg.o_lora_rank, -1)
                if suffix in ('ffn.gate.weight', 'ffn.gate.bias'):
                    value = value[:3]
                fixtures.save(destination, value)
            for projection in ('w1', 'w2', 'w3'):
                destination = label + '_param_ffn_experts_' + projection
                if i == 0:
                    fixtures.entries[destination] = fixtures.entries['expert_moe_' + projection]
                else:
                    fixtures.save(destination, torch.stack([
                        store.matrix(f'layers.{i}.ffn.experts.{j}.{projection}.weight') for j in range(3)]))
            for mode, segments in (('prefill', [count]), ('decode', [1] * count)):
                position = 0
                outputs, mixes = [], []
                for length in segments:
                    x = h[:, position:position+length]
                    if layer.engram is not None:
                        x = layer.engram(x, hashes[:, position:position+length])
                    result, mix = layer(x, position, incoming[:, position:position+length], None)
                    outputs.append(result[0])
                    mixes.append(mix[0])
                    position += length
                fixtures.save(label + '_' + mode, torch.cat(outputs, 0))
                fixtures.save(label + '_' + mode + '_pre', torch.cat(mixes, 0))
                report['layer_composition']['layers'][f'{i}_{mode}'] = stats(torch.cat(outputs, 0))
            print(f'Composed layer {i} reference complete (three experts, top-2).', flush=True)

    output_head_fixtures(store, fixtures, ref, ids)
    report['row_reads'] = store.row_reads
    report['bytes_read'] = store.bytes_read
    report['elapsed_seconds'] = time.perf_counter() - started
    fixtures.finish()
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Wrote ' + str(args.output_dir / 'report.json'), flush=True)


if __name__ == '__main__':
    main()
