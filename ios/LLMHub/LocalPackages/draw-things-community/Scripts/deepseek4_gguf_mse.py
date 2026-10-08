#!/usr/bin/env python3
import argparse
import json
import mmap
import os
import struct
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

LLAMA_GGUF_PY = "/Users/liu/workspace/llama.cpp/gguf-py"
if os.path.isdir(LLAMA_GGUF_PY) and LLAMA_GGUF_PY not in sys.path:
    sys.path.insert(0, LLAMA_GGUF_PY)

from gguf.gguf_reader import GGUFReader
from gguf.quants import dequantize


FP4_E2M1 = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=np.float32,
)


@dataclass
class TensorInfo:
    dtype: str
    shape: Tuple[int, ...]
    start: int
    end: int


class SafeTensorShard:
    def __init__(self, path: str):
        self.path = path
        self.file = open(path, "rb")
        self.data = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        header_size = struct.unpack("<Q", self.data[:8])[0]
        self.buffer_start = 8 + header_size
        header = json.loads(self.data[8 : self.buffer_start])
        self.states: Dict[str, TensorInfo] = {}
        for name, value in header.items():
            if name == "__metadata__":
                continue
            shape = tuple(int(x) for x in value["shape"])
            if not shape:
                shape = (1,)
            start, end = value["data_offsets"]
            self.states[name] = TensorInfo(value["dtype"].upper(), shape, int(start), int(end))

    def info(self, name: str) -> TensorInfo:
        return self.states[name]

    def raw(self, name: str, dtype: np.dtype) -> np.ndarray:
        info = self.info(name)
        count = int(np.prod(info.shape))
        offset = self.buffer_start + info.start
        return np.frombuffer(self.data, dtype=dtype, count=count, offset=offset).reshape(info.shape)

    def dense(self, name: str) -> np.ndarray:
        info = self.info(name)
        if info.dtype == "F32":
            return self.raw(name, np.dtype("<f4")).astype(np.float32, copy=False)
        if info.dtype == "F16":
            return self.raw(name, np.dtype("<f2")).astype(np.float32)
        if info.dtype == "BF16":
            raw = self.raw(name, np.dtype("<u2")).astype(np.uint32)
            return (raw << 16).view(np.float32)
        if info.dtype == "F8_E4M3":
            return f8_e4m3(self.raw(name, np.dtype("u1")))
        if info.dtype == "F8_E8M0":
            return f8_e8m0(self.raw(name, np.dtype("u1")))
        raise ValueError(f"unsupported dtype {info.dtype} for {name}")

    def fp8_block_dense(self, weight_name: str, scale_name: str) -> np.ndarray:
        weight_info = self.info(weight_name)
        scale_info = self.info(scale_name)
        if weight_info.dtype != "F8_E4M3" or len(weight_info.shape) != 2:
            raise ValueError(f"invalid FP8 weight tensor {weight_name}")
        if scale_info.dtype != "F8_E8M0" or len(scale_info.shape) != 2:
            raise ValueError(f"invalid FP8 scale tensor {scale_name}")
        weight = self.raw(weight_name, np.dtype("u1"))
        scales = f8_e8m0(self.raw(scale_name, np.dtype("u1")))
        rows, columns = weight.shape
        row_scale = np.arange(rows) // 128
        column_scale = np.arange(columns) // 128
        return f8_e4m3(weight) * scales[row_scale[:, None], column_scale[None, :]]

    def fp4_dense(self, weight_name: str, scale_name: str) -> np.ndarray:
        weight_info = self.info(weight_name)
        scale_info = self.info(scale_name)
        if weight_info.dtype != "I8" or len(weight_info.shape) != 2:
            raise ValueError(f"invalid FP4 weight tensor {weight_name}")
        if scale_info.dtype != "F8_E8M0" or len(scale_info.shape) != 2:
            raise ValueError(f"invalid FP4 scale tensor {scale_name}")
        weight = self.raw(weight_name, np.dtype("u1"))
        scales = f8_e8m0(self.raw(scale_name, np.dtype("u1")))
        rows, packed_columns = weight.shape
        out = np.empty((rows, packed_columns * 2), dtype=np.float32)
        scale_for_packed = scales[:, (np.arange(packed_columns) * 2) // 32]
        out[:, 0::2] = FP4_E2M1[weight & 0x0F] * scale_for_packed
        out[:, 1::2] = FP4_E2M1[weight >> 4] * scale_for_packed
        return out


class SafeTensorCheckpoint:
    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
            index = json.load(f)
        self.weight_map = index["weight_map"]
        self.shards: Dict[str, SafeTensorShard] = {}

    def shard(self, name: str) -> SafeTensorShard:
        shard_name = self.weight_map[name]
        if shard_name not in self.shards:
            self.shards[shard_name] = SafeTensorShard(os.path.join(self.path, shard_name))
        return self.shards[shard_name]

    def dense(self, name: str) -> np.ndarray:
        return self.shard(name).dense(name)

    def fp8_block_dense(self, prefix: str) -> np.ndarray:
        weight = f"{prefix}.weight"
        return self.shard(weight).fp8_block_dense(weight, f"{prefix}.scale")

    def fp4_dense(self, prefix: str) -> np.ndarray:
        weight = f"{prefix}.weight"
        return self.shard(weight).fp4_dense(weight, f"{prefix}.scale")


@dataclass
class Stats:
    count: int = 0
    finite_count: int = 0
    nonfinite_count: int = 0
    sum_squared: float = 0.0
    sum_abs: float = 0.0
    max_abs: float = 0.0
    max_index: int = -1

    def add_arrays(self, reference: np.ndarray, candidate: np.ndarray, base_index: int = 0) -> None:
        if reference.shape != candidate.shape:
            raise ValueError(f"shape mismatch: reference {reference.shape}, candidate {candidate.shape}")
        reference = reference.astype(np.float32, copy=False)
        candidate = candidate.astype(np.float32, copy=False)
        diff = candidate - reference
        finite = np.isfinite(diff)
        flat_count = int(diff.size)
        self.count += flat_count
        finite_count = int(np.count_nonzero(finite))
        self.finite_count += finite_count
        self.nonfinite_count += flat_count - finite_count
        if finite_count == 0:
            return
        finite_diff = diff if finite_count == flat_count else diff[finite]
        abs_diff = np.abs(finite_diff)
        self.sum_squared += float(np.sum(finite_diff * finite_diff, dtype=np.float64))
        self.sum_abs += float(np.sum(abs_diff, dtype=np.float64))
        local_max_pos = int(np.argmax(abs_diff))
        local_max = float(abs_diff.reshape(-1)[local_max_pos])
        if local_max > self.max_abs:
            self.max_abs = local_max
            if finite_count == flat_count:
                self.max_index = base_index + local_max_pos
            else:
                self.max_index = base_index + int(np.flatnonzero(finite.reshape(-1))[local_max_pos])

    @property
    def mse(self) -> float:
        return self.sum_squared / self.finite_count if self.finite_count else float("nan")

    @property
    def rmse(self) -> float:
        return float(np.sqrt(self.mse))

    @property
    def mean_abs(self) -> float:
        return self.sum_abs / self.finite_count if self.finite_count else float("nan")


def f8_e8m0(values: np.ndarray) -> np.ndarray:
    return np.exp2(values.astype(np.int16) - 127).astype(np.float32)


def f8_e4m3(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.uint8, copy=False)
    sign = np.where((values & 0x80) == 0, 1.0, -1.0).astype(np.float32)
    exponent = ((values >> 3) & 0x0F).astype(np.int16)
    mantissa = (values & 0x07).astype(np.float32)
    subnormal = sign * mantissa * np.float32(0.001953125)
    normal = sign * (1.0 + mantissa * 0.125) * np.exp2(exponent - 7).astype(np.float32)
    out = np.where(exponent == 0, subnormal, normal).astype(np.float32)
    out[(exponent == 15) & ((values & 0x07) == 7)] = np.nan
    return out


def dense_specs(layer: int):
    prefix = f"layers.{layer}"
    return [
        (f"blk.{layer}.attn_q_a.weight", f"{prefix}.attn.wq_a", "fp8", "Q8_0 GGUF from FP8 source"),
        (f"blk.{layer}.attn_q_b.weight", f"{prefix}.attn.wq_b", "fp8", "Q8_0 GGUF from FP8 source"),
        (f"blk.{layer}.attn_kv.weight", f"{prefix}.attn.wkv", "fp8", "Q8_0 GGUF from FP8 source"),
        (f"blk.{layer}.ffn_gate_inp.weight", f"{prefix}.ffn.gate.weight", "direct", "F16 GGUF from direct source"),
        (f"blk.{layer}.ffn_gate_shexp.weight", f"{prefix}.ffn.shared_experts.w1", "fp8", "Q8_0 GGUF from FP8 source"),
        (f"blk.{layer}.ffn_up_shexp.weight", f"{prefix}.ffn.shared_experts.w3", "fp8", "Q8_0 GGUF from FP8 source"),
        (f"blk.{layer}.ffn_down_shexp.weight", f"{prefix}.ffn.shared_experts.w2", "fp8", "Q8_0 GGUF from FP8 source"),
    ]


def expert_specs(layer: int):
    prefix = f"layers.{layer}.ffn"
    return [
        (f"blk.{layer}.ffn_gate_exps.weight", "w1", 2048, 4096, f"{prefix}.experts.w1.weight", "IQ2_XXS GGUF from FP4 expert source"),
        (f"blk.{layer}.ffn_up_exps.weight", "w3", 2048, 4096, f"{prefix}.experts.w3.weight", "IQ2_XXS GGUF from FP4 expert source"),
        (f"blk.{layer}.ffn_down_exps.weight", "w2", 4096, 2048, f"{prefix}.experts.w2.weight", "Q2_K GGUF from FP4 expert source"),
    ]


def print_result(comparison: str, layer: int, tensor: str, note: str, shape: Iterable[int], stats: Stats) -> None:
    shape_text = "x".join(str(int(x)) for x in shape)
    print(
        f"{comparison},{layer},{tensor},{note},{shape_text},{stats.count},{stats.mse},"
        f"{stats.rmse},{stats.mean_abs},{stats.max_abs},{stats.max_index},{stats.nonfinite_count}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/Users/liu/workspace/ds4/DeepSeek-V4-Flash")
    parser.add_argument("--gguf", default="/Users/liu/workspace/ds4/ds4flash.gguf")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--skip-experts", action="store_true")
    parser.add_argument("--only-experts", action="store_true")
    parser.add_argument("--expert-limit", type=int, default=256)
    args = parser.parse_args()

    checkpoint = SafeTensorCheckpoint(args.checkpoint)
    reader = GGUFReader(args.gguf)
    tensors = {tensor.name: tensor for tensor in reader.tensors}

    print("comparison,layer,tensor,note,shape,count,mse,rmse,mean_abs,max_abs,max_index,nonfinite", flush=True)
    if not args.only_experts:
        for gguf_name, source, source_kind, note in dense_specs(args.layer):
            gguf_tensor = tensors[gguf_name]
            candidate = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
            reference = checkpoint.dense(source) if source_kind == "direct" else checkpoint.fp8_block_dense(source)
            stats = Stats()
            stats.add_arrays(reference, candidate)
            print_result("gguf_vs_safetensors", args.layer, gguf_name, note, candidate.shape, stats)

    if not args.skip_experts:
        for gguf_name, suffix, rows, columns, store_like_name, note in expert_specs(args.layer):
            gguf_tensor = tensors[gguf_name]
            stats = Stats()
            expert_count = min(args.expert_limit, int(gguf_tensor.data.shape[0]))
            for expert in range(expert_count):
                if expert % 32 == 0:
                    print(f"compare {suffix} expert {expert}/{expert_count}", file=sys.stderr, flush=True)
                candidate = dequantize(gguf_tensor.data[expert], gguf_tensor.tensor_type)
                reference = checkpoint.fp4_dense(f"layers.{args.layer}.ffn.experts.{expert}.{suffix}")
                stats.add_arrays(reference, candidate, base_index=expert * rows * columns)
            print_result(
                "gguf_vs_safetensors", args.layer, gguf_name, note,
                (expert_count, rows, columns), stats)


if __name__ == "__main__":
    main()
