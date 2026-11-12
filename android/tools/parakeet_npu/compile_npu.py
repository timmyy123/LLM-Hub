"""AOT-compile the float16-safe Parakeet model for NPUs with LiteRT (Linux x86_64 only).

Usage: compile_npu.py MODEL.tflite OUTPUT_DIR SOC [SOC ...]      e.g. MT6991 MT6993 SM8750 SM8850

MODEL.tflite is the output of make_fp16_safe.py. MediaTek chips need the `ai-edge-litert-sdk-mediatek`
package; Qualcomm chips need the QAIRT SDK's lib/x86_64-linux-clang on LD_LIBRARY_PATH (2.50 matches
the runtime the app ships). One chip at a time: two compiles in one go have left an empty output.
"""
import pathlib
import sys
import time

from ai_edge_litert.aot import aot_compile
from ai_edge_litert.aot.vendors.mediatek import target as mediatek
from ai_edge_litert.aot.vendors.qualcomm import target as qualcomm

model, out = sys.argv[1], pathlib.Path(sys.argv[2])
for soc in sys.argv[3:]:
    vendor = mediatek if soc.startswith("MT") else qualcomm
    dest = out / soc
    dest.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        aot_compile.aot_compile(model, output_dir=dest, target=vendor.Target(getattr(vendor.SocModel, soc)), keep_going=False)
        print("DONE", soc, round(time.time() - started), "s", flush=True)
    except Exception as error:  # the plugin reports its own details in a temp file it names
        print("FAILED", soc, repr(error)[:800], flush=True)
    for path in sorted(dest.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(out)} {path.stat().st_size} bytes", flush=True)
