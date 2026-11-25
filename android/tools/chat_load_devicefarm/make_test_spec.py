"""Write an AWS Device Farm instrumentation spec for ChatModelLoadDeviceTest.

Usage: make_test_spec.py OUTPUT.yml [TEST_CLASS LOG_TAG TEST,TEST,...]
       e.g. make_test_spec.py race.yml LiteRtLmCloseRaceDeviceTest LiteRtLmCloseRace unloadWhileGeneratingGpu,stopThenUnloadGpu

The phone downloads each pinned model itself (the test checks its SHA-256). Each test runs in its own
instrumentation process (a native crash ends the process) and every test runs even when an earlier one fails.
Device Farm wraps each command separately, so every command is a single shell line.
"""
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
device = 'adb -s "$DEVICEFARM_DEVICE_UDID"'
log = '"$DEVICEFARM_LOG_DIR'
package = "com.llmhub.llmhub"
test_class, tag, tests = "ChatModelLoadDeviceTest", "ChatModelLoad", [
    "gemma3_1bGpu", "gemma3_1bCpu", "gemma4E2bGpu", "gemma4E2bCpu", "lfmCpu", "lfmGpu", "lfmNpu"]
if len(sys.argv) > 2:
    test_class, tag, tests = sys.argv[2], sys.argv[3], sys.argv[4].split(",")
runner = "$DEVICEFARM_TEST_PACKAGE_NAME/$DEVICEFARM_TEST_PACKAGE_RUNNER"
spec = {
    "version": 0.1,
    "android_test_host": "amazon_linux_2",
    "phases": {
        "pre_test": {"commands": [
            f'{device} shell getprop > {log}/device-properties.txt"',
            f'{device} shell getprop ro.soc.model | tee {log}/soc.txt"',
            f'{device} shell cat /proc/meminfo | head -3 > {log}/memory.txt"',
            f'{device} logcat -c',
        ]},
        "test": {"commands": [
            *[f'{device} shell am instrument -r -w --no-window-animation '
              f'-e class com.llmhub.llmhub.inference.{test_class}#{t} {runner} 2>&1 | '
              f'sed "s/^/[{t}] /" | tee -a {log}/instrument.log"' for t in tests],
            f'{device} logcat -d -s {tag}:* > {log}/results.txt"',
            f'{device} logcat -d -b crash > {log}/crash-buffer.txt"',
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
