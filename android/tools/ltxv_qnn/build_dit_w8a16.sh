#!/usr/bin/env bash
# Fully integer transformer chunks (INT8 weights, INT16 activations, no FP16 anywhere) for Hexagon NPUs
# without FP16 such as the Snapdragon 8s Gen 4 (SM8735, soc_model 85).
# Usage: build_dit_w8a16.sh <calib_dir> "<arch soc_model> [<arch soc_model> ...]"
#   e.g. build_dit_w8a16.sh calib_dit_512 "v75 57 v75 85"
# Needs onnx/ltxv_chunk*.onnx (export_onnx.py, 640 tokens), overrides/ltxv_chunk*.json (make_overrides.py)
# and the calibration inputs from make_dit_calib.py.
CALIB=$1; TARGETS=$2
cd ~/ltxqnn
ulimit -s unlimited  # qairt-converter recurses deeply in pattern matching and can overflow the default stack
source ~/ltxqnn/qairt_env.sh
set -euo pipefail
OUT=w8dit
mkdir -p $OUT/raw
for i in 0 1 2 3; do
  G=ltxv_chunk$i
  # 16-bit activation ranges from onnxruntime (multi-threaded; qairt-quantizer calibrates on one thread).
  [ -f $OUT/act$i.json ] || python make_vae_encodings.py onnx/$G.onnx $CALIB/chunk$i/input_list.txt $OUT/act$i.json
  [ -f $OUT/enc$i.json ] || python make_vae_encodings.py --merge $OUT/act$i.json overrides/$G.json $OUT/enc$i.json onnx/$G.onnx
  # Attention projections' inputs become extra outputs (see add_attn_outputs.py).
  [ -f onnx/${G}_w8.onnx ] || python add_attn_outputs.py onnx/$G.onnx onnx/${G}_w8.onnx
  [ -f $OUT/raw/$G.dlc ] || qairt-converter --input_network onnx/${G}_w8.onnx --output_path $OUT/raw/$G.dlc \
      --quantization_overrides $OUT/enc$i.json > $OUT/conv$i.log 2>&1 || { tail -5 $OUT/conv$i.log; exit 1; }
  # No --enable_float_fallback: tensors without an override are calibrated on one sample, so nothing is
  # left in floating point. IO stays quantized (UFIXED_POINT_16); the runtime converts it.
  head -1 $CALIB/chunk$i/input_list.txt > $OUT/one$i.txt
  [ -f $OUT/$G.dlc ] || qairt-quantizer -i $OUT/raw/$G.dlc -o $OUT/$G.dlc --input_list $OUT/one$i.txt \
      --act_bitwidth 16 --weights_bitwidth 8 --bias_bitwidth 32 > $OUT/quant$i.log 2>&1 || { tail -5 $OUT/quant$i.log; exit 1; }
  echo "$G quantized: $(stat -c %s $OUT/$G.dlc)"
done
set -- $TARGETS
while [ $# -ge 2 ]; do
  ARCH=$1; SOC=$2; shift 2
  D=$OUT/${ARCH}_$SOC
  mkdir -p $D
  for i in 0 1 2 3; do
    G=ltxv_chunk$i
    cat > $D/htp_$G.json <<JSON
{"graphs": [{"graph_names": ["$G"], "vtcm_mb": 8, "O": 3}], "devices": [{"soc_model": $SOC, "dsp_arch": "$ARCH"}]}
JSON
    cat > $D/ext_$G.json <<JSON
{"backend_extensions": {"shared_library_path": "$QAIRT_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so", "config_file_path": "$HOME/ltxqnn/$D/htp_$G.json"}}
JSON
    [ -f $D/$G.bin ] || qnn-context-binary-generator --backend $QAIRT_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so \
        --dlc_path $OUT/$G.dlc --binary_file $G --output_dir $D --config_file $D/ext_$G.json > $D/gen_$G.log 2>&1 \
        || { echo "FAILED $ARCH soc $SOC $G: $(grep -m3 ERROR $D/gen_$G.log | sed 's/.*<E> //' | tr '\n' ' ')"; continue; }
    echo "$ARCH soc $SOC $G: $(stat -c %s $D/$G.bin)"
  done
done
echo DIT_W8A16_DONE
