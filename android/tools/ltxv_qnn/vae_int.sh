#!/usr/bin/env bash
# Integer-only W8A16 VAE decoder (no FP16 ops) for Hexagon NPUs without FP16, e.g. Snapdragon 8s Gen 4.
cd ~/ltxqnn
ulimit -s unlimited
source ~/ltxqnn/qairt_env.sh
set -euo pipefail
N=ltxv_vae_512x320x25; O=vaeint
mkdir -p $O
head -1 calib_512/input_list.txt > $O/one.txt
[ -f $O/$N.dlc ] || qairt-quantizer -i w8a16b/raw/$N.dlc -o $O/$N.dlc --input_list $O/one.txt \
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
