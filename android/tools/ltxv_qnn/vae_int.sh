#!/usr/bin/env bash
# Integer-only W8A16 VAE decoder (no FP16 ops) for Hexagon NPUs without FP16, e.g. Snapdragon 8s Gen 4.
# Every tensor gets an encoding (make_vae_encodings.py --complete), so qairt-quantizer's float fallback
# has nothing to fall back on and no slow single-threaded calibration is needed.
# Needs w8a16b/act.json (make_vae_encodings.py) and w8a16b/params.json (make_overrides.py).
cd ~/ltxqnn
ulimit -s unlimited
source ~/ltxqnn/qairt_env.sh
set -euo pipefail
N=ltxv_vae_512x320x25; O=vaeint
mkdir -p $O/raw
# Unfused RMS norms: see unfuse_rms_norm.py.
[ -f onnx/${N}_int.onnx ] || python unfuse_rms_norm.py onnx/$N.onnx onnx/${N}_int.onnx
[ -f $O/enc.json ] || python make_vae_encodings.py --merge w8a16b/act.json w8a16b/params.json $O/enc.json onnx/${N}_int.onnx --complete
[ -f $O/raw/$N.dlc ] || qairt-converter --input_network onnx/${N}_int.onnx --output_path $O/raw/$N.dlc \
    --quantization_overrides $O/enc.json > $O/conv.log 2>&1 || { tail -5 $O/conv.log; exit 1; }
[ -f $O/$N.dlc ] || qairt-quantizer -i $O/raw/$N.dlc -o $O/$N.dlc --enable_float_fallback \
    --act_bitwidth 16 --weights_bitwidth 8 --bias_bitwidth 32 > $O/quant.log 2>&1 || { tail -5 $O/quant.log; exit 1; }
# arch soc_model vtcm_mb: v73/68/4 MB serves the 8s Gen 4, 8s Gen 3, 7+ Gen 3 and 7 Gen 4 (the smallest
# VTCM of those chips); v75/57 is for checking the graph on a Snapdragon 8 Gen 3.
for P in "v73 68 4" "v75 57 8"; do set -- $P; D=$O/${1}_$2; mkdir -p $D
  cat > $D/htp.json <<J
{"graphs": [{"graph_names": ["$N"], "vtcm_mb": $3, "O": 3}], "devices": [{"soc_model": $2, "dsp_arch": "$1"}]}
J
  cat > $D/ext.json <<J
{"backend_extensions": {"shared_library_path": "$QAIRT_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so", "config_file_path": "$HOME/ltxqnn/$D/htp.json"}}
J
  qnn-context-binary-generator --backend $QAIRT_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so --dlc_path $O/$N.dlc --binary_file $N --output_dir $D --config_file $D/ext.json > $D/gen.log 2>&1 \
    && echo "VAE $1 soc $2: $(stat -c %s $D/$N.bin)" || echo "VAE $1 soc $2: FAIL $(grep -m2 ERROR $D/gen.log | sed "s/.*<E> //" | tr "\n" " " | cut -c1-200)"
done
echo VAE_INT_DONE
