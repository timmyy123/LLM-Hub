#!/usr/bin/env python3
import argparse
import base64
import csv
import os
import re
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np


GGUF_TYPE_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    31: "TQ1_0",
    32: "TQ2_0",
}

DATATYPE_NAMES = {
    16384: "F32",
    131072: "F16",
    262144: "F64",
    524288: "BF16",
    4: "I32",
    8: "I64",
}


def text_key(name):
    return f"__text_model__[t-{name}-0-0]"


def vision_key(name, slot=0):
    return f"__vision_model__[t-{name}-0-{slot}]"


def load_gguf_reader(path, gguf_py):
    sys.path.insert(0, gguf_py)
    from gguf import GGUFReader

    return GGUFReader(path)


def load_imatrix(path, gguf_py):
    reader = load_gguf_reader(path, gguf_py)
    sums = {}
    counts = {}
    for tensor in reader.tensors:
        name = tensor.name
        if name.endswith(".in_sum2"):
            sums[name[: -len(".in_sum2")]] = np.asarray(tensor.data, dtype=np.float32)
        elif name.endswith(".counts"):
            counts[name[: -len(".counts")]] = np.asarray(tensor.data, dtype=np.float32)
    imatrix = {}
    for name, values in sums.items():
        count = counts.get(name)
        if count is None or count.size == 0 or float(count.reshape(-1)[0]) <= 0:
            normalized = np.ones(values.shape, dtype=np.float32)
        else:
            normalized = values / float(count.reshape(-1)[0])
        imatrix[name] = base64.b64encode(
            np.asarray(normalized, dtype="<f4").tobytes()
        ).decode("ascii")
    return imatrix


def tensor_type_name(tensor):
    return GGUF_TYPE_NAMES.get(int(tensor.tensor_type), str(int(tensor.tensor_type)))


def shape_text(tensor):
    return "x".join(str(int(x)) for x in tensor.shape)


def source_name_for_text_key(store_key):
    body = store_key
    prefix = "__text_model__[t-"
    if body.startswith(prefix) and body.endswith("]"):
        body = body[len(prefix) : -1]
    body = re.sub(r"-0-\d+$", "", body)
    return body


def read_store_keys(path):
    if path is None:
        return {}
    store_path = Path(path).expanduser()
    if not store_path.exists():
        return {}
    keys = {}
    connection = sqlite3.connect(store_path)
    try:
        for name, datatype, dim in connection.execute(
            "select name, datatype, dim from tensors order by name"
        ):
            dims = []
            if dim:
                dims = list(struct.unpack("<" + "i" * (len(dim) // 4), dim))
                dims = [x for x in dims if x > 0]
            keys[name] = {
                "datatype": DATATYPE_NAMES.get(datatype, str(datatype)),
                "shape": "x".join(str(x) for x in dims),
            }
    finally:
        connection.close()
    return keys


def map_text_tensor(name):
    match = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
    if name == "token_embd.weight":
        return [(text_key("model.language_model.embed_tokens"), "model.language_model.embed_tokens.weight", "")]
    if name == "output.weight":
        return [(text_key("lm_head"), "lm_head.weight", "")]
    if name == "output_norm.weight":
        return [(text_key("model.language_model.norm"), "model.language_model.norm.weight", "rmsnorm_offset")]
    if match is None:
        return []
    layer = int(match.group(1))
    suffix = match.group(2)
    base = f"model.language_model.layers.{layer}"
    mappings = {
        "attn_gate.weight": ("linear_attn.in_proj_z", "linear_attn.in_proj_z.weight", "linear_attention_gate"),
        "attn_qkv.weight": ("linear_attn.in_proj_qkv", "linear_attn.in_proj_qkv.weight", "linear_attention_qkv"),
        "ssm_a": ("linear_attn.A_log", "linear_attn.A_log", "exp_to_decay"),
        "ssm_alpha.weight": ("linear_attn.in_proj_a", "linear_attn.in_proj_a.weight", ""),
        "ssm_beta.weight": ("linear_attn.in_proj_b", "linear_attn.in_proj_b.weight", ""),
        "ssm_conv1d.weight": ("linear_attn.conv1d.weight", "linear_attn.conv1d.weight", "reshape_conv1d_oihw"),
        "ssm_dt.bias": ("linear_attn.dt_bias", "linear_attn.dt_bias", ""),
        "ssm_norm.weight": ("linear_attn.norm", "linear_attn.norm.weight", ""),
        "ssm_out.weight": ("linear_attn.out_proj", "linear_attn.out_proj.weight", ""),
        "attn_k.weight": ("self_attn.k_proj", "self_attn.k_proj.weight", "interleave_rope"),
        "attn_v.weight": ("self_attn.v_proj", "self_attn.v_proj.weight", ""),
        "attn_output.weight": ("self_attn.o_proj", "self_attn.o_proj.weight", ""),
        "attn_q_norm.weight": ("self_attn.q_norm", "self_attn.q_norm.weight", "rmsnorm_offset_interleave_rope"),
        "attn_k_norm.weight": ("self_attn.k_norm", "self_attn.k_norm.weight", "rmsnorm_offset_interleave_rope"),
        "ffn_down.weight": ("mlp.down_proj", "mlp.down_proj.weight", ""),
        "ffn_gate.weight": ("mlp.gate_proj", "mlp.gate_proj.weight", ""),
        "ffn_up.weight": ("mlp.up_proj", "mlp.up_proj.weight", ""),
        "attn_norm.weight": ("input_layernorm", "input_layernorm.weight", "rmsnorm_offset"),
        "post_attention_norm.weight": ("post_attention_layernorm", "post_attention_layernorm.weight", "rmsnorm_offset"),
    }
    if suffix == "attn_q.weight":
        return [
            (text_key(f"{base}.self_attn.q_proj"), f"{base}.self_attn.q_proj.weight", "split_q_interleave_rope"),
            (text_key(f"{base}.self_attn.q_gate_proj"), f"{base}.self_attn.q_proj.weight", "split_q_gate"),
        ]
    if suffix not in mappings:
        return []
    target, hf_suffix, transform = mappings[suffix]
    return [(text_key(f"{base}.{target}"), f"{base}.{hf_suffix}", transform)]


def map_vision_tensor(name):
    match = re.fullmatch(r"v\.blk\.(\d+)\.(.+)", name)
    if name == "v.patch_embd.weight" or name == "v.patch_embd.weight.1":
        return [
            (
                vision_key("model.visual.patch_embed.proj"),
                "model.visual.patch_embed.proj.weight",
                "temporal_patch_slice_reshape",
            )
        ]
    if name == "v.patch_embd.bias":
        return [(vision_key("model.visual.patch_embed.proj", 1), "model.visual.patch_embed.proj.bias", "")]
    if name == "v.position_embd.weight":
        return [("model.visual.pos_embed.weight", "model.visual.pos_embed.weight", "standalone_tensor")]
    if name == "v.post_ln.weight":
        return [(vision_key("model.visual.merger.norm"), "model.visual.merger.norm.weight", "")]
    if name == "v.post_ln.bias":
        return [(vision_key("model.visual.merger.norm", 1), "model.visual.merger.norm.bias", "")]
    if name == "mm.0.weight":
        return [(vision_key("model.visual.merger.linear_fc1"), "model.visual.merger.linear_fc1.weight", "")]
    if name == "mm.0.bias":
        return [(vision_key("model.visual.merger.linear_fc1", 1), "model.visual.merger.linear_fc1.bias", "")]
    if name == "mm.2.weight":
        return [(vision_key("model.visual.merger.linear_fc2"), "model.visual.merger.linear_fc2.weight", "")]
    if name == "mm.2.bias":
        return [(vision_key("model.visual.merger.linear_fc2", 1), "model.visual.merger.linear_fc2.bias", "")]
    if match is None:
        return []
    block = int(match.group(1))
    suffix = match.group(2)
    base = f"model.visual.blocks.{block}"
    simple = {
        "attn_out.weight": ("attn.proj", 0),
        "attn_out.bias": ("attn.proj", 1),
        "ffn_up.weight": ("mlp.linear_fc1", 0),
        "ffn_up.bias": ("mlp.linear_fc1", 1),
        "ffn_down.weight": ("mlp.linear_fc2", 0),
        "ffn_down.bias": ("mlp.linear_fc2", 1),
        "ln1.weight": ("norm1", 0),
        "ln1.bias": ("norm1", 1),
        "ln2.weight": ("norm2", 0),
        "ln2.bias": ("norm2", 1),
    }
    if suffix == "attn_qkv.weight":
        return [
            (vision_key(f"{base}.attn.q_proj"), f"{base}.attn.q_proj.weight", "split_qkv_interleave_rope"),
            (vision_key(f"{base}.attn.k_proj"), f"{base}.attn.k_proj.weight", "split_qkv_interleave_rope"),
            (vision_key(f"{base}.attn.v_proj"), f"{base}.attn.v_proj.weight", "split_qkv"),
        ]
    if suffix == "attn_qkv.bias":
        return [
            (vision_key(f"{base}.attn.q_proj", 1), f"{base}.attn.q_proj.bias", "split_qkv_interleave_rope"),
            (vision_key(f"{base}.attn.k_proj", 1), f"{base}.attn.k_proj.bias", "split_qkv_interleave_rope"),
            (vision_key(f"{base}.attn.v_proj", 1), f"{base}.attn.v_proj.bias", "split_qkv"),
        ]
    if suffix not in simple:
        return []
    target, slot = simple[suffix]
    return [(vision_key(f"{base}.{target}", slot), f"{base}.{target}.{ 'bias' if slot else 'weight' }", "")]


def append_rows(rows, compact_rows, tensor, component, mappings, imatrix, source_file):
    gguf_name = tensor.name
    gguf_type = tensor_type_name(tensor)
    imatrix_payload = imatrix.get(gguf_name, "")
    for store_key, hf_key, transform in mappings:
        row = {
            "component": component,
            "hf_key": hf_key,
            "store_key": store_key,
            "gguf_key": gguf_name,
            "gguf_type": gguf_type,
            "target_format": gguf_type,
            "shape": shape_text(tensor),
            "transform": transform,
            "imatrix_key": gguf_name if imatrix_payload else "",
            "imatrix_available": "true" if imatrix_payload else "false",
            "source_file": source_file,
        }
        rows.append(row)
        if component == "text" and gguf_name not in compact_rows:
            compact_rows[gguf_name] = (gguf_name, gguf_type, imatrix_payload)


def add_reference_only_rows(rows, compact_rows, reference_keys, already_mapped):
    for store_key, info in sorted(reference_keys.items()):
        if store_key in already_mapped:
            continue
        if "mtp" not in store_key:
            continue
        source_name = source_name_for_text_key(store_key)
        datatype = info.get("datatype", "")
        rows.append(
            {
                "component": "mtp",
                "hf_key": source_name.replace("mtp.", "mtp.") + ".weight",
                "store_key": store_key,
                "gguf_key": "",
                "gguf_type": "",
                "target_format": datatype,
                "shape": info.get("shape", ""),
                "transform": "official_safetensors_only",
                "imatrix_key": "",
                "imatrix_available": "false",
                "source_file": "official_safetensors",
            }
        )


def write_compact(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "format", "imatrix"])
        for row in rows:
            writer.writerow(row)


def write_map(path, rows):
    fields = [
        "component",
        "hf_key",
        "store_key",
        "gguf_key",
        "gguf_type",
        "target_format",
        "shape",
        "transform",
        "imatrix_key",
        "imatrix_available",
        "source_file",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_audit(path, reference_keys, mapped_keys):
    if not reference_keys:
        return
    missing = sorted(set(reference_keys) - mapped_keys)
    extra = sorted(mapped_keys - set(reference_keys))
    with open(path, "w") as f:
        f.write(f"reference_keys: {len(reference_keys)}\n")
        f.write(f"mapped_keys: {len(mapped_keys)}\n")
        f.write(f"missing_from_map: {len(missing)}\n")
        for key in missing:
            f.write(f"missing {key}\n")
        f.write(f"extra_not_in_reference: {len(extra)}\n")
        for key in extra:
            f.write(f"extra {key}\n")


def generate_for_model(gguf_path, mmproj_path, imatrix, reference_keys, output_prefix, gguf_py):
    rows = []
    compact_rows = {}
    main_reader = load_gguf_reader(str(gguf_path), gguf_py)
    for tensor in main_reader.tensors:
        mappings = map_text_tensor(tensor.name)
        append_rows(rows, compact_rows, tensor, "text", mappings, imatrix, gguf_path.name)

    if mmproj_path is not None and mmproj_path.exists():
        mmproj_reader = load_gguf_reader(str(mmproj_path), gguf_py)
        for tensor in mmproj_reader.tensors:
            mappings = map_vision_tensor(tensor.name)
            append_rows(rows, compact_rows, tensor, "vision", mappings, {}, mmproj_path.name)

    mapped_keys = {row["store_key"] for row in rows}
    add_reference_only_rows(rows, compact_rows, reference_keys, mapped_keys)
    mapped_keys = {row["store_key"] for row in rows}
    write_compact(f"{output_prefix}_quantization.csv", compact_rows.values())
    write_map(f"{output_prefix}_tensor_map.csv", rows)
    write_audit(f"{output_prefix}_store_key_audit.txt", reference_keys, mapped_keys)
    return rows, compact_rows, mapped_keys


def main():
    parser = argparse.ArgumentParser(
        description="Generate Qwen3.5 GGUF quantization CSVs and tensor maps."
    )
    parser.add_argument("--gguf-py", default="/Users/liu/workspace/llama.cpp/gguf-py")
    parser.add_argument("--gguf-dir", default="Models/Qwen3.5-9B-GGUF")
    parser.add_argument("--q5-gguf", default="Qwen3.5-9B-UD-Q5_K_XL.gguf")
    parser.add_argument("--q4-gguf", default="Qwen3.5-9B-UD-Q4_K_XL.gguf")
    parser.add_argument("--mmproj", default="mmproj-BF16.gguf")
    parser.add_argument("--imatrix", default="imatrix_unsloth.gguf_file")
    parser.add_argument("--reference-store", default="/tmp/qwen_3.5_9b_f16.ckpt")
    parser.add_argument("--output-dir", default="Models/Qwen3.5-9B-GGUF")
    args = parser.parse_args()

    gguf_dir = Path(args.gguf_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    imatrix = load_imatrix(str(gguf_dir / args.imatrix), args.gguf_py)
    reference_keys = read_store_keys(args.reference_store)

    specs = [
        (gguf_dir / args.q5_gguf, output_dir / "qwen_3.5_9b_ud_q5_k_xl"),
        (gguf_dir / args.q4_gguf, output_dir / "qwen_3.5_9b_ud_q4_k_xl"),
    ]
    mmproj_path = gguf_dir / args.mmproj
    for gguf_path, output_prefix in specs:
        rows, compact_rows, mapped_keys = generate_for_model(
            gguf_path, mmproj_path, imatrix, reference_keys, output_prefix, args.gguf_py
        )
        print(
            f"{output_prefix}: map_rows={len(rows)} compact_rows={len(compact_rows)} "
            f"mapped_keys={len(mapped_keys)} reference_keys={len(reference_keys)}"
        )


if __name__ == "__main__":
    main()
