"""Write an AWS Device Farm instrumentation spec for LlamaCppFlashStreamingDeviceTest#lfm25Accelerators.

Usage: make_test_spec.py OUTPUT.yml [MODES]   (MODES default: vulkan,cpu)

The phone downloads LFM2.5-24B-A2B Q4_0 (13.6 GB) itself, inside the app, from the catalog's pinned URL into the
app's models folder, checked against its SHA-256 (the test's model_sha256 argument); nothing is pushed from the
host. Each mode's prompt and decode speed is in the runtime's "Turn:" log line, kept in flash-results.txt.
Device Farm wraps each command separately, so every command is a single shell line.
"""
import json
from pathlib import Path
import sys

SHA256 = "d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5"
output = Path(sys.argv[1])
modes = sys.argv[2] if len(sys.argv) > 2 else "vulkan,cpu"
device = 'adb -s "$DEVICEFARM_DEVICE_UDID"'
log = '"$DEVICEFARM_LOG_DIR'
package = "com.llmhub.llmhub.preview"  # the debug build tested here
runner = "$DEVICEFARM_TEST_PACKAGE_NAME/$DEVICEFARM_TEST_PACKAGE_RUNNER"
test = "com.llmhub.llmhub.inference.LlamaCppFlashStreamingDeviceTest#lfm25Accelerators"
spec = {
    "version": 0.1,
    "android_test_host": "amazon_linux_2",
    "phases": {
        "pre_test": {"commands": [
            f'{device} shell getprop ro.soc.model | tee {log}/soc.txt"',
            f'{device} shell cat /proc/meminfo | head -3 | tee {log}/memory.txt"',
            f'{device} logcat -G 16M',
            f'{device} logcat -c',
        ]},
        "test": {"commands": [
            f'{device} shell am instrument -r -w --no-window-animation -e class {test} '
            f'-e model_sha256 {SHA256} -e modes {modes} {runner} 2>&1 | tee {log}/instrument.log"',
            f'{device} logcat -d -s LlamaCppStream:I LlamaCppFlashStream:I LlamaCppFallback:I OnDeviceFixtures:I '
            f'LlamaCppInference:I > {log}/flash-results.txt"',
            f'grep -E "Turn:|load failed|Downloaded" {log}/flash-results.txt" | cut -c1-300',
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
