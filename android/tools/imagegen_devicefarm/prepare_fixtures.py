"""Download pinned weights and derive controlled encoder override fixtures.

Runs on the Device Farm host, with only the Python standard library and curl.
Altered encoders are test fixtures, not models offered to app users.

Usage: prepare_fixtures.py sdxl|sd3|embedgemma|embedgemma_fallback|litert_sd|litert_sd_app|soundgen|soundgen_long|soundgen_gpu|soundgen_app|llama_gpu DIRECTORY [BUILD]
"""
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

SDXL_BASE = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/462165984030d82259a11f4367a4eed129e94a7b/"
# Ungated SD3.5 Medium GGUF conversion. Its CLIP and T5 safetensors match Stability's official hashes.
SD3_BASE = "https://huggingface.co/second-state/stable-diffusion-3.5-medium-GGUF/resolve/58b78c305a43ddfcffe1ab54d7022995f61667ac/"
WEIGHTS = {
    "sdxl": {
        "sdxl.safetensors": (SDXL_BASE + "sd_xl_base_1.0.safetensors", "31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b"),
        "lora.safetensors": (SDXL_BASE + "sd_xl_offset_example-lora_1.0.safetensors", "4852686128f953d0277d0793e2f0335352f96a919c9c16a09787d77f55cbdf6f"),
    },
    "sd3": {
        "sd3.gguf": (SD3_BASE + "sd3.5_medium-Q4_0.gguf", "3bb8c5e9ab0a841117089ed4ed81d885bb85161df2a766b812f829bc55b31adf"),
        "clip-l.safetensors": (SD3_BASE + "clip_l.safetensors", "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd"),
        "clip-g.safetensors": (SD3_BASE + "clip_g.safetensors", "ec310df2af79c318e24d20511b601a591ca8cd4f1fce1d8dff822a356bcdb1f4"),
        "t5.gguf": (SD3_BASE + "t5xxl-Q4_0.gguf", "987ba47c158b890c274f78fd35324419f50941e846a49789f0977e9fe9d97ab7"),
    },
    # Google's MediaTek Dimensity 9400 EmbeddingGemma 2 builds, at the commits ModelData pins.
    "embedgemma": {
        "embeddinggemma-2-text-270m_MediaTek_MT6991.litertlm": (
            "https://huggingface.co/litert-community/embeddinggemma-2-text-270m-litert-lm/resolve/9be6e8b90982095dc05c2bd162e4b954ee4dbac7/embeddinggemma-2-text-270m_MediaTek_MT6991.litertlm",
            "7acc8e334d882216207c23b7a477179d5af530d67fd8e2cb50ab42d3457f90c0"),
        "embeddinggemma-2-740m_Google_Tensor_G5.litertlm": "5b53d76c2c7c66d2dc6230ff158ea0033ac5939ac7cfcf827c3a22cdfc742a5e",
    "embeddinggemma-2-740m_Google_Tensor_G6.litertlm": "ed0e4f413e705946eda544078d8bf5acf10fe5aee2946451c81919a1f59665e7",
    "embeddinggemma-2-740m_MediaTek_MT6991.litertlm": (
            "https://huggingface.co/litert-community/embeddinggemma-2-740m-litert-lm/resolve/24d962e906c7d332c6428e71c9676855024569e2/embeddinggemma-2-740m_MediaTek_MT6991.litertlm",
            "99e6a35b824b4f7ad4f7c7f20c59ba380aa90e0b4442aadfee3674230c6c7cca"),
    },
}
# Every SoundGen file, at the commits ModelData pins (names are unique across the three models).
WEIGHTS["soundgen"] = {
    "dit_model.tflite": ("https://huggingface.co/jegly/audio/resolve/e50bf9dff759c8622371a9c43ede989ea2d04b32/dit_model.tflite",
        "119fe0020f2fafd88366a2869fdbf2f1aaa3933f8daf64d2116dcc115348ee83"),
    "autoencoder_model_undilated_1g.tflite": ("https://huggingface.co/timmyboy26/soundgen-quick-litert-gpu/resolve/610d2ad5b7058f53da8465b584d042b6ee8ed6f9/autoencoder_model_undilated_1g.tflite",
        "15c9c8236a48c64075124cad0737e5d8cb35c154f78cb40756db5d882076ec47"),
    "conditioners_float32.tflite": ("https://huggingface.co/jegly/audio/resolve/e50bf9dff759c8622371a9c43ede989ea2d04b32/conditioners_float32.tflite",
        "c1139b09ea66a944966dfce4722075401f9eb723c1f59bfd5bdd797690bcce91"),
    "spiece.model": ("https://huggingface.co/jegly/audio/resolve/e50bf9dff759c8622371a9c43ede989ea2d04b32/spiece.model",
        "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86"),
    "dit_L256_int8.tflite": ("https://huggingface.co/jegly/noise/resolve/43a3e8771f62bfc9257276c7e2adee9c76d601dd/dit_L256_int8.tflite",
        "252da2091c8ab83ba8405c923a0f0e9fbd87402ae15d372c5b7075430fe0293c"),
    "ae_dec_L256_int8.tflite": ("https://huggingface.co/jegly/noise/resolve/43a3e8771f62bfc9257276c7e2adee9c76d601dd/ae_dec_L256_int8.tflite",
        "8d35f984242999ff6d7ce3975b3c5632e1c3d35c0a222e7869918202ee3f815c"),
    "t5gemma_enc_int8.tflite": ("https://huggingface.co/jegly/noise/resolve/43a3e8771f62bfc9257276c7e2adee9c76d601dd/t5gemma_enc_int8.tflite",
        "a418899d084e7141d508e4fb865fbbf177e5a689b1a3c57bcc90a9505c39566e"),
    "tokenizer.model": ("https://huggingface.co/jegly/noise/resolve/43a3e8771f62bfc9257276c7e2adee9c76d601dd/tokenizer.model",
        "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2"),
    "dit_L2048_int8.tflite": ("https://huggingface.co/jegly/noise/resolve/43a3e8771f62bfc9257276c7e2adee9c76d601dd/dit_L2048_int8.tflite",
        "cbab297f07650140b604ad5ba1e1a12d2f9ed320114b5831ad5aa383092b4cf7"),
    "ae_dec_L2048_int8.tflite": ("https://huggingface.co/jegly/noise/resolve/43a3e8771f62bfc9257276c7e2adee9c76d601dd/ae_dec_L2048_int8.tflite",
        "a8db29d280c2ff0e6576ce1db0f7850a982492c51e8f71e739baa90c83db75c4"),
}
WEIGHTS["soundgen_gpu"] = {}  # Downloaded on the phone.
WEIGHTS["soundgen_app"] = {}  # Downloaded on the phone.
WEIGHTS["llama_gpu"] = {}  # Downloaded on the phone.
WEIGHTS["soundgen_long"] = {name: WEIGHTS["soundgen"][name] for name in
                            ("dit_L2048_int8.tflite", "ae_dec_L2048_int8.tflite", "t5gemma_enc_int8.tflite", "tokenizer.model")}

EMBEDGEMMA_740M = "https://huggingface.co/litert-community/embeddinggemma-2-740m-litert-lm/resolve/24d962e906c7d332c6428e71c9676855024569e2/"
EMBEDGEMMA_740M_HASHES = {
    "embeddinggemma-2-740m.litertlm": "e7a8a2204b91e0f96e92960e84a09a89212e1633dcb7575a9bf3378b4df77f4c",
    "embeddinggemma-2-740m_Google_Tensor_G5.litertlm": "5b53d76c2c7c66d2dc6230ff158ea0033ac5939ac7cfcf827c3a22cdfc742a5e",
    "embeddinggemma-2-740m_Google_Tensor_G6.litertlm": "ed0e4f413e705946eda544078d8bf5acf10fe5aee2946451c81919a1f59665e7",
    "embeddinggemma-2-740m_MediaTek_MT6991.litertlm": "99e6a35b824b4f7ad4f7c7f20c59ba380aa90e0b4442aadfee3674230c6c7cca",
    "embeddinggemma-2-740m_Qualcomm_SM8650.litertlm": "3b8a18c6d043b8132c4eda45611296e1ae455ada82cda8b591ec0ca141e041c5",
    "embeddinggemma-2-740m_Qualcomm_SM8750.litertlm": "ac1973c0195dc6e28a621651f2f5e04d35fdc7a7b11cdbbe94f29f5dccb57996",
}


def embedgemma_fallback_weights(tag):
    """Video Moment Finder with TAG's NPU build and the standard build it falls back to when the NPU can't run it."""
    names = [f"embeddinggemma-2-740m_{tag or 'MediaTek_MT6991'}.litertlm", "embeddinggemma-2-740m.litertlm"]
    return {name: (EMBEDGEMMA_740M + name, EMBEDGEMMA_740M_HASHES[name]) for name in names}


# SD1.5 LiteRT NPU builds; BUILD names the AOT folder (tensor_g5).
SD_LITERT_BASE = "https://huggingface.co/timmyboy26/stable-diffusion-v1-5-LiteRT-NPU/resolve/f71417cd334e9cee78571412782b91a0be1b78e1/"
SD_LITERT_HASHES = {
    "tensor_g5/clip.tflite": "bf086c6b1ca79c62d2679b569542f1106beb580a8ee5ef520328d7dd313b0c21",
    "tensor_g5/decoder.tflite": "d9dfdefb6c23c663c4fbbc440cfe4235b61096743dadf6e4478f6968bac86897",
    "tensor_g5/diffusion.tflite": "4a730296a9405a308ae6882bd83a257125b9070aa75e7a55d0ea7cf082eb5ec0",
    "tensor_g6/clip.tflite": "4dfb6c80efa1149244a8bb4aaf8df4886cafe30d7db57dc1eef4aae49107fe89",
    "tensor_g6/decoder.tflite": "cad3ceb4cc319319225808d50d10bb84fd4f9261f0c5e38afaace2963a9dbcf5",
    "tensor_g6/diffusion.tflite": "cdaac432868808328c3eba53e85358b132ff07fd023954bf28c67f8aeb615f3d",
    "source/clip.tflite": "e14a305dde98b063ff49591dedc43e416a1620f450be6d697fc2884d44e2287e",
    "source/decoder.tflite": "c71756885f8873bcaf5299fb37a63eb7755bfd908fb34b0a41da95de07b05549",
    "source/diffusion.tflite": "93d1639e7e965318c0ff06003c20b4bc1ae800cf695bdbe9489be7787de9f4b1",
    "tokenizer/merges.txt": "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a",
    "tokenizer/vocab.json": "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349",
}


def litert_sd_weights(build):
    """BUILD's AOT models and the uncompiled sources, each folder with the CLIP tokenizer files."""
    files = {f"aot/{n}": f"{build}/{n}" for n in ("diffusion.tflite", "decoder.tflite", "clip.tflite")}
    files.update({f"source/{n}": f"source/{n}" for n in ("diffusion.tflite", "decoder.tflite", "clip.tflite")})
    for directory in ("aot", "source"):
        files.update({f"{directory}/{n}": f"tokenizer/{n}" for n in ("vocab.json", "merges.txt")})
    return {local: (SD_LITERT_BASE + path, SD_LITERT_HASHES[path]) for local, path in files.items()}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(directory, suite):
    for destination, (url, expected) in WEIGHTS[suite].items():
        path = directory / destination
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or sha256(path) != expected:
            subprocess.run(["curl", "-fL", "--retry", "3", "--max-time", "1800", url + "?download=true", "-o", str(path)], check=True)
        if sha256(path) != expected:
            raise ValueError(f"SHA-256 mismatch: {destination}")
        print(f"Verified {destination}: {expected}", flush=True)


def clip_g_name(name):
    """Use the same canonical tensor names as stable-diffusion.cpp's OpenCLIP converter."""
    exact = {
        "ln_final.bias": "text_model.final_layer_norm.bias",
        "ln_final.weight": "text_model.final_layer_norm.weight",
        "positional_embedding": "text_model.embeddings.position_embedding.weight",
        "token_embedding.weight": "text_model.embeddings.token_embedding.weight",
        "text_projection": "text_model.text_projection",
    }
    if name in exact:
        return exact[name]
    match = re.fullmatch(r"transformer\.resblocks\.(\d+)\.(.+)", name)
    if match:
        suffixes = {
            "attn.in_proj_bias": "self_attn.in_proj.bias", "attn.in_proj_weight": "self_attn.in_proj.weight",
            "attn.out_proj.bias": "self_attn.out_proj.bias", "attn.out_proj.weight": "self_attn.out_proj.weight",
            "ln_1.bias": "layer_norm1.bias", "ln_1.weight": "layer_norm1.weight",
            "ln_2.bias": "layer_norm2.bias", "ln_2.weight": "layer_norm2.weight",
            "mlp.c_fc.bias": "mlp.fc1.bias", "mlp.c_fc.weight": "mlp.fc1.weight",
            "mlp.c_proj.bias": "mlp.fc2.bias", "mlp.c_proj.weight": "mlp.fc2.weight",
        }
        return f"text_model.encoder.layers.{match[1]}.{suffixes[match[2]]}"
    if name == "logit_scale":
        return None  # Used by contrastive training, not image generation.
    raise ValueError(f"Unexpected CLIP-G tensor: {name}")


def prepare_sdxl(directory):
    source = directory / "sdxl.safetensors"
    with source.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        tensors = json.loads(stream.read(size))
    prefix = "conditioner.embedders.1.model."
    selected = {}
    for key, value in tensors.items():
        if key.startswith(prefix):
            name = clip_g_name(key[len(prefix):])
            if name:
                selected[name] = value
    if len(selected) != 389:
        raise ValueError(f"Unexpected CLIP-G tensor count: {len(selected)}")
    header, offset = {}, 0
    for name, value in selected.items():
        begin, end = value["data_offsets"]
        header[name] = dict(value, data_offsets=[offset, offset + end - begin])
        offset += end - begin
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    encoder = directory / "clip-g.safetensors"
    with source.open("rb") as stream, encoder.open("wb") as out:
        out.write(struct.pack("<Q", len(encoded)))
        out.write(encoded)
        for value in selected.values():
            begin, end = value["data_offsets"]
            stream.seek(8 + size + begin)
            remaining = end - begin
            while remaining:
                block = stream.read(min(remaining, 8 * 1024 * 1024))
                if not block:
                    raise EOFError("Truncated checkpoint")
                out.write(block)
                remaining -= len(block)
    altered = directory / "clip-g-zero-projection.safetensors"
    shutil.copyfile(encoder, altered)
    begin, end = header["text_model.text_projection"]["data_offsets"]
    with altered.open("r+b") as stream:
        stream.seek(8 + len(encoded) + begin)
        stream.write(bytes(end - begin))
    return [encoder, altered]


def gguf_tensor(path, wanted):
    """Return (type, absolute data offset, element count) for one tensor in a GGUF v3 file."""
    with path.open("rb") as stream:
        def read(fmt):
            return struct.unpack("<" + fmt, stream.read(struct.calcsize("<" + fmt)))[0]

        def string():
            return stream.read(read("Q")).decode()

        scalars = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:
                item, count = read("I"), read("Q")
                return [value(item) for _ in range(count)]
            return read(scalars[kind])

        if stream.read(4) != b"GGUF" or read("I") != 3:
            raise ValueError("Expected a GGUF v3 file")
        tensor_count, kv_count = read("Q"), read("Q")
        metadata = {}
        for _ in range(kv_count):
            key = string()
            metadata[key] = value(read("I"))
        found = None
        for _ in range(tensor_count):
            name = string()
            dims = [read("Q") for _ in range(read("I"))]
            kind, offset = read("I"), read("Q")
            if name == wanted:
                elements = 1
                for dim in dims:
                    elements *= dim
                found = (kind, offset, elements)
        alignment = metadata.get("general.alignment", 32)
        data_start = stream.tell() + (-stream.tell() % alignment)
    if not found:
        raise ValueError(f"Missing GGUF tensor: {wanted}")
    return found[0], data_start + found[1], found[2]


def prepare_sd3(directory):
    # Negating T5's final RMSNorm scale negates every encoder output without changing the file layout.
    altered = directory / "t5-negated-norm.gguf"
    shutil.copyfile(directory / "t5.gguf", altered)
    kind, offset, elements = gguf_tensor(altered, "encoder.final_layer_norm.weight")
    if kind != 2 or elements % 32:
        raise ValueError(f"Expected a Q4_0 T5 norm, found type {kind}")
    with altered.open("r+b") as stream:
        # Q4_0 blocks are an F16 scale followed by 16 bytes of 4-bit values; flip each scale's sign bit.
        for block in range(elements // 32):
            stream.seek(offset + block * 18 + 1)
            high = stream.read(1)[0]
            stream.seek(-1, 1)
            stream.write(bytes([high ^ 0x80]))
    return [altered]


def prepare(suite, directory, build=""):
    directory.mkdir(parents=True, exist_ok=True)
    if suite == "litert_sd":
        WEIGHTS[suite] = litert_sd_weights(build)
    if suite == "litert_sd_app":
        WEIGHTS[suite] = {"app/" + local[4:]: value for local, value in litert_sd_weights(build).items() if local.startswith("aot/")}
    if suite == "embedgemma_fallback":
        WEIGHTS[suite] = embedgemma_fallback_weights(build)
    download(directory, suite)
    derived = {"sdxl": prepare_sdxl, "sd3": prepare_sd3}.get(suite, lambda _: [])(directory)
    manifest = {str(path.relative_to(directory)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in [*(directory / name for name in WEIGHTS[suite]), *derived]}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    prepare(sys.argv[1], Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "")
