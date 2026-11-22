"""Write an AWS Device Farm instrumentation spec for LlamaCppAionDeviceTest (the Aion Q4_0 GGUF on llama.cpp).

Usage: make_gguf_test_spec.py OUTPUT.yml

The Device Farm host downloads the catalog's public GGUF (pinned commit), checks its SHA-256 and copies it into the
app's models folder, where the test finds it and skips its own download. Each backend test runs in its own
instrumentation process (a native crash ends the process) and every test runs even when an earlier one fails; the
NPU test skips itself on chips llama.cpp's Hexagon backend does not support. Device Farm wraps each command
separately, so every command is a single shell line.
"""
import json
from pathlib import Path
import sys

URL = ("https://huggingface.co/timmyboy26/aion-1.0-instruct-2.3b-q4_0-gguf/resolve/"
       "b783c0eec446e1c4f02b60d2ca7a6e60e0533d7f/Aion-1.0-Instruct-2.3B-Q4_0.gguf")
SHA256 = "82a71c2311e3803a166aa11da844d4b547839ed2d941c1384e812e29070519be"
NAME = "Aion-1.0-Instruct-2.3B-Q4_0.gguf"

output = Path(sys.argv[1])
device = 'adb -s "$DEVICEFARM_DEVICE_UDID"'
log = '"$DEVICEFARM_LOG_DIR'
package = "com.llmhub.llmhub"
staged = f"/data/local/tmp/{NAME}"
tests = ["textCpu", "textGpu", "textNpu"]
runner = "$DEVICEFARM_TEST_PACKAGE_NAME/$DEVICEFARM_TEST_PACKAGE_RUNNER"
spec = {
    "version": 0.1,
    "android_test_host": "amazon_linux_2",
    "phases": {
        "pre_test": {"commands": [
            f'{device} shell getprop > {log}/device-properties.txt"',
            f'{device} shell getprop ro.soc.model | tee {log}/soc.txt"',
            f'{device} shell cat /proc/meminfo | head -3 > {log}/memory.txt"',
            f"curl -fsSL --retry 3 -o /tmp/{NAME} '{URL}'",
            f'echo "{SHA256}  /tmp/{NAME}" | sha256sum -c - | tee {log}/model-sha256.txt"',
            f"{device} push /tmp/{NAME} {staged}",
            f"{device} shell run-as {package} mkdir -p files/models",
            f"{device} shell run-as {package} cp {staged} files/models/{NAME}",
            f"{device} shell rm -f {staged}",
            f'{device} logcat -c',
        ]},
        "test": {"commands": [
            *[f'{device} shell am instrument -r -w --no-window-animation '
              f'-e class com.llmhub.llmhub.inference.LlamaCppAionDeviceTest#{t} {runner} 2>&1 | '
              f'sed "s/^/[{t}] /" | tee -a {log}/instrument.log"' for t in tests],
            f'{device} logcat -d -s LlamaCppAion:I LlamaCppFallback:I > {log}/aion-results.txt"',
            f'! grep -Eq "FAILURES|INSTRUMENTATION_FAILED|Process crashed" {log}/instrument.log"',
        ]},
        "post_test": {"commands": [
            f'{device} logcat -d -v threadtime > {log}/logcat.txt"',
            f'{device} shell run-as {package} rm -rf files/models',
        ]},
    },
    "artifacts": ["$DEVICEFARM_LOG_DIR"],
}
output.write_text(json.dumps(spec, indent=2))  # JSON is valid YAML.
