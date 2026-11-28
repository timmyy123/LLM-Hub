"""AOT-compile the float16-safe Parakeet model for NPUs with LiteRT (Linux x86_64 only).

Usage: compile_npu.py MODEL.tflite OUTPUT_DIR SOC [SOC ...] [flag=value ...]   e.g. MT6991 SM8850 TENSOR_G5

flag=value arguments go to the compiler for every SoC, e.g. qualcomm_optimization_level=O2: Qualcomm's default O3
crashes (segfault at 96%) on Parakeet Ultra's 10 s float16 encoder for SM8750, and O2 compiles it.
Google Tensor needs the ai-edge-litert-sdk-google-tensor package, installed with GOOGLE_TENSOR_SDK_BETA pointing at
Google's litert_plugin_compiler.tar.gz.

MODEL.tflite is the output of make_fp16_safe.py. MediaTek chips need the `ai-edge-litert-sdk-mediatek`
package; Qualcomm chips need the QAIRT SDK's lib/x86_64-linux-clang on LD_LIBRARY_PATH (2.50 matches
the runtime the app ships), plus a libc++.so.1 and libunwind.so.1 (an LLVM 18 lib folder) for the compiler plugin. One chip at a time: two compiles in one go have left an empty output.
"""
import pathlib
import sys
import time

from ai_edge_litert.aot import aot_compile
from ai_edge_litert.aot.vendors.google_tensor import target as google_tensor
from ai_edge_litert.aot.vendors.mediatek import target as mediatek
from ai_edge_litert.aot.vendors.qualcomm import target as qualcomm

model, out = sys.argv[1], pathlib.Path(sys.argv[2])
flags = dict(arg.split("=", 1) for arg in sys.argv[3:] if "=" in arg)
for soc in [arg for arg in sys.argv[3:] if "=" not in arg]:
    vendor = mediatek if soc.startswith("MT") else google_tensor if soc.startswith("TENSOR") else qualcomm
    dest = out / soc
    dest.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        aot_compile.aot_compile(model, output_dir=dest, target=vendor.Target(getattr(vendor.SocModel, soc)), keep_going=False, **flags)
        print("DONE", soc, round(time.time() - started), "s", flush=True)
    except Exception as error:  # the plugin reports its own details in a temp file it names
        print("FAILED", soc, repr(error)[:800], flush=True)
    for path in sorted(dest.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(out)} {path.stat().st_size} bytes", flush=True)
