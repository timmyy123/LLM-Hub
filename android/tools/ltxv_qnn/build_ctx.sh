#!/usr/bin/env bash
# Export -> convert (fp16) -> compile QNN context binaries for one Hexagon arch.
# Usage: build_ctx.sh <dsp_arch> <soc_model>   e.g. build_ctx.sh v73 43
ARCH=$1
SOC=$2
cd ~/ltxqnn
# QAIRT's envsetup.sh references unset variables, so enable strict mode only after sourcing it.
source ~/ltxqnn/qairt_env.sh
set -euo pipefail
CHUNKS="0 1 2 3"
# Weight-only INT8 (per-row symmetric) with FP16 activations: an FP16 transformer (3.9 GB) does not
# fit the Hexagon's 32-bit address space, INT8 weights (~2 GB) do and run ~3x faster.
mkdir -p overrides dlc/raw
for i in $CHUNKS; do
  [ -f onnx/ltxv_chunk$i.onnx ] || python export_onnx.py --ckpt models/ltxv-2b-0.9.8-distilled.safetensors --out onnx --only $i
  [ -f overrides/ltxv_chunk$i.json ] || python make_overrides.py onnx/ltxv_chunk$i.onnx overrides/ltxv_chunk$i.json
  # Same file name in both steps: QNN names the graph after the converter output file.
  [ -f dlc/raw/ltxv_chunk$i.dlc ] || qairt-converter --input_network onnx/ltxv_chunk$i.onnx --output_path dlc/raw/ltxv_chunk$i.dlc \
      --quantization_overrides overrides/ltxv_chunk$i.json --float_bitwidth 16 --float_bias_bitwidth 16 > dlc/conv$i.log 2>&1
  [ -f dlc/ltxv_chunk$i.dlc ] || qairt-quantizer -i dlc/raw/ltxv_chunk$i.dlc -o dlc/ltxv_chunk$i.dlc \
      --enable_float_fallback --float_bitwidth 16 --float_bias_bitwidth 16 > dlc/quant$i.log 2>&1
done
mkdir -p ctx/$ARCH
for i in $CHUNKS; do
  cat > ctx/htp_${ARCH}_$i.json <<JSON
{"graphs": [{"graph_names": ["ltxv_chunk$i"], "vtcm_mb": 8, "O": 3, "fp16_relaxed_precision": 1}],
 "devices": [{"soc_model": $SOC, "dsp_arch": "$ARCH"}]}
JSON
  cat > ctx/ext_${ARCH}_$i.json <<JSON
{"backend_extensions": {"shared_library_path": "$QAIRT_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so",
 "config_file_path": "$HOME/ltxqnn/ctx/htp_${ARCH}_$i.json"}}
JSON
  [ -f ctx/$ARCH/ltxv_chunk$i.bin ] || qnn-context-binary-generator --backend $QAIRT_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so \
      --dlc_path dlc/ltxv_chunk$i.dlc --binary_file ltxv_chunk$i --output_dir ctx/$ARCH --config_file ctx/ext_${ARCH}_$i.json > ctx/gen_${ARCH}_$i.log 2>&1
  echo "chunk $i done: $(ls -la ctx/$ARCH/ltxv_chunk$i.bin)"
done
echo "BUILD_CTX_DONE $ARCH"
