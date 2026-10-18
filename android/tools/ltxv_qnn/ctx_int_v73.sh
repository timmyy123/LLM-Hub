#!/usr/bin/env bash
# Integer transformer chunks for Hexagon v73 chips without FP16 (8s Gen 4, 8s Gen 3, 7+ Gen 3, 7 Gen 4).
# soc_model 68 (8s Gen 3) and 4 MB VTCM: the smallest VTCM of those chips; larger VTCM runs it too.
cd ~/ltxqnn; ulimit -s unlimited; source ~/ltxqnn/qairt_env.sh >/dev/null 2>&1
D=w8dit/v73_68; mkdir -p $D
for i in 0 1 2 3; do G=ltxv_chunk$i
  cat > $D/htp_$G.json <<J
{"graphs": [{"graph_names": ["$G"], "vtcm_mb": 4, "O": 3}], "devices": [{"soc_model": 68, "dsp_arch": "v73"}]}
J
  cat > $D/ext_$G.json <<J
{"backend_extensions": {"shared_library_path": "$QAIRT_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so", "config_file_path": "$HOME/ltxqnn/$D/htp_$G.json"}}
J
  [ -f $D/$G.bin ] || qnn-context-binary-generator --backend $QAIRT_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so --dlc_path w8dit/$G.dlc --binary_file $G --output_dir $D --config_file $D/ext_$G.json > $D/gen_$G.log 2>&1 \
    && echo "$G: $(stat -c %s $D/$G.bin)" || echo "FAILED $G: $(grep -m2 ERROR $D/gen_$G.log | sed "s/.*<E> //" | tr "\n" " ")"
done
echo CTX_V73_DONE
