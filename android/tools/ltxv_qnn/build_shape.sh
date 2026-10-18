#!/usr/bin/env bash
# Build the NPU transformer (INT8 weights) and VAE decoder (FP16) for one video shape and Hexagon arch.
# Usage: build_shape.sh <arch> <soc_model> <width> <height> <frames>   e.g. build_shape.sh v73 43 768 512 25
ARCH=$1; SOC=$2; W=$3; H=$4; F=$5
cd ~/ltxqnn
source ~/ltxqnn/qairt_env.sh
set -euo pipefail
TOKENS=$(( (W / 32) * (H / 32) * ((F - 1) / 8 + 1) ))
TAG=${W}x${H}x${F}
mkdir -p onnx_$TAG dlc_$TAG/raw ctx_$TAG/$ARCH overrides
for i in 0 1 2 3; do
  [ -f onnx_$TAG/ltxv_chunk$i.onnx ] || python export_onnx.py --ckpt models/ltxv-2b-0.9.8-distilled.safetensors --out onnx_$TAG --tokens $TOKENS --only $i
  [ -f overrides/ltxv_chunk$i.json ] || python make_overrides.py onnx_$TAG/ltxv_chunk$i.onnx overrides/ltxv_chunk$i.json
  [ -f dlc_$TAG/raw/ltxv_chunk$i.dlc ] || qairt-converter --input_network onnx_$TAG/ltxv_chunk$i.onnx --output_path dlc_$TAG/raw/ltxv_chunk$i.dlc \
      --quantization_overrides overrides/ltxv_chunk$i.json --float_bitwidth 16 --float_bias_bitwidth 16 > dlc_$TAG/conv$i.log 2>&1
  [ -f dlc_$TAG/ltxv_chunk$i.dlc ] || qairt-quantizer -i dlc_$TAG/raw/ltxv_chunk$i.dlc -o dlc_$TAG/ltxv_chunk$i.dlc \
      --enable_float_fallback --float_bitwidth 16 --float_bias_bitwidth 16 > dlc_$TAG/quant$i.log 2>&1
done
VAE=ltxv_vae_$TAG
[ -f onnx_$TAG/$VAE.onnx ] || python export_vae_onnx.py --ckpt models/ltxv-2b-0.9.8-distilled.safetensors --out onnx_$TAG/$VAE.onnx --width $W --height $H --frames $F
# Conv3d has no INT8-weight/FP16-activation HTP kernel, so the VAE decoder stays FP16.
[ -f dlc_$TAG/$VAE.dlc ] || qairt-converter --input_network onnx_$TAG/$VAE.onnx --output_path dlc_$TAG/$VAE.dlc --float_bitwidth 16 --float_bias_bitwidth 16 > dlc_$TAG/conv_vae.log 2>&1
for G in ltxv_chunk0 ltxv_chunk1 ltxv_chunk2 ltxv_chunk3 $VAE; do
  cat > ctx_$TAG/htp_${ARCH}_$G.json <<JSON
{"graphs": [{"graph_names": ["$G"], "vtcm_mb": 8, "O": 3, "fp16_relaxed_precision": 1}],
 "devices": [{"soc_model": $SOC, "dsp_arch": "$ARCH"}]}
JSON
  cat > ctx_$TAG/ext_${ARCH}_$G.json <<JSON
{"backend_extensions": {"shared_library_path": "$QAIRT_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so",
 "config_file_path": "$HOME/ltxqnn/ctx_$TAG/htp_${ARCH}_$G.json"}}
JSON
  [ -f ctx_$TAG/$ARCH/$G.bin ] || qnn-context-binary-generator --backend $QAIRT_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so \
      --dlc_path dlc_$TAG/$G.dlc --binary_file $G --output_dir ctx_$TAG/$ARCH --config_file ctx_$TAG/ext_${ARCH}_$G.json > ctx_$TAG/gen_${ARCH}_$G.log 2>&1
  echo "$G: $(stat -c %s ctx_$TAG/$ARCH/$G.bin)"
done
echo "BUILD_SHAPE_DONE $ARCH $TAG"
