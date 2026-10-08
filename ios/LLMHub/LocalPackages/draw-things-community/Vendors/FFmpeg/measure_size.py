#!/usr/bin/env python3
"""Link the same two-entry-point fixture for every codec selection."""
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

sdk, arch, minimum, archive, source, header, binary, report = sys.argv[1:]
sdkroot = subprocess.check_output(["xcrun", "--sdk", sdk, "--show-sdk-path"], text=True).strip()
target = f"{arch}-apple-" + (f"macos{minimum}" if sdk == "macosx" else f"ios{minimum}")
if sdk == "iphonesimulator":
    target += "-simulator"
args = ["xcrun", "clang", "-target", target, "-isysroot", sdkroot, "-O2", "-I" + str(Path(header).parent),
        source, archive, "-Wl,-dead_strip", "-Wl,-no_uuid", "-lc++", "-lz", "-liconv", "-lbz2", "-o", binary]
for framework in ["AudioToolbox", "CoreFoundation", "CoreMedia", "CoreVideo", "Security", "VideoToolbox"]:
    args += ["-framework", framework]
subprocess.run(args, check=True)
subprocess.run(["xcrun", "strip", "-S", "-x", binary], check=True)
subprocess.run(["codesign", "--remove-signature", binary], check=True)
data = Path(binary).read_bytes()
Path(report).write_text(json.dumps(dict(archive_bytes=os.path.getsize(archive), linked_bytes=len(data),
    gzip_bytes=len(gzip.compress(data, compresslevel=9, mtime=0)), arch=arch, sdk=sdk, minimum_os=minimum), indent=2) + "\n")
