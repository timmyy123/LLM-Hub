"""Write an AWS Device Farm instrumentation spec for AionLiteRtLmDeviceTest.

Usage: make_test_spec.py MODEL_URL SHA256 OUTPUT.yml

MODEL_URL is a private, time-limited link (e.g. an S3 presigned URL) to a local Aion .litertlm conversion; the
Device Farm host downloads it, checks its SHA-256 and pushes it to /data/local/tmp. Each test runs in its own
instrumentation process (a native crash ends the process) and every test runs even when an earlier one fails.
Device Farm wraps each command separately, so every command is a single shell line.
"""
import json
from pathlib import Path
import sys

url, sha256, output = sys.argv[1], sys.argv[2], Path(sys.argv[3])
device = 'adb -s "$DEVICEFARM_DEVICE_UDID"'
log = '"$DEVICEFARM_LOG_DIR'
model = "/data/local/tmp/aion.litertlm"
tests = ["capitalCpu", "capitalGpu", "longPromptRecallCpu", "longPromptRecallGpu", "germanCpu"]
runner = "$DEVICEFARM_TEST_PACKAGE_NAME/$DEVICEFARM_TEST_PACKAGE_RUNNER"
spec = {
    "version": 0.1,
    "android_test_host": "amazon_linux_2",
    "phases": {
        "pre_test": {"commands": [
            f'{device} shell getprop > {log}/device-properties.txt"',
            f'{device} shell getprop ro.soc.model | tee {log}/soc.txt"',
            f'{device} shell df -h /data > {log}/device-disk.txt"',
            f"curl -fsSL --retry 3 -o /tmp/aion.litertlm '{url}'",
            f'echo "{sha256}  /tmp/aion.litertlm" | sha256sum -c - | tee {log}/model-sha256.txt"',
            f"{device} push /tmp/aion.litertlm {model}",
            f'{device} logcat -c',
        ]},
        "test": {"commands": [
            *[f'{device} shell am instrument -r -w --no-window-animation -e aionModel {model} '
              f'-e class com.llmhub.llmhub.inference.AionLiteRtLmDeviceTest#{t} {runner} 2>&1 | '
              f'sed "s/^/[{t}] /" | tee -a {log}/instrument.log"' for t in tests],
            f'{device} logcat -d -s AionLiteRtLm:I LiteRtLmInference:D > {log}/aion-results.txt"',
            f'! grep -Eq "FAILURES|INSTRUMENTATION_FAILED|Process crashed" {log}/instrument.log"',
        ]},
        "post_test": {"commands": [
            f'{device} logcat -d -v threadtime > {log}/logcat.txt"',
            f'{device} shell rm -f {model}',
            f'{device} shell run-as com.llmhub.llmhub rm -rf files/models',
        ]},
    },
    "artifacts": ["$DEVICEFARM_LOG_DIR"],
}
output.write_text(json.dumps(spec, indent=2))  # JSON is valid YAML.
