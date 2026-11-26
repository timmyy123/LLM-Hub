"""Write an AWS Device Farm instrumentation spec for StableAudio3DeviceTest.

Usage: make_test_spec.py OUTPUT.yml [MODEL [SECONDS]]
       e.g. make_test_spec.py sa3.yml "Stable Audio 3 Small Music" 10,30

The phone downloads the model itself through the app's ModelDownloader (pinned Hugging Face commit), then
generates each length and logs timings, peak memory and output levels under the StableAudio3DeviceTest tag.
Device Farm wraps each command separately, so every command is a single shell line.
"""
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
model = sys.argv[2] if len(sys.argv) > 2 else "Stable Audio 3 Small Music"
seconds = sys.argv[3] if len(sys.argv) > 3 else "10,30"
device = 'adb -s "$DEVICEFARM_DEVICE_UDID"'
log = '"$DEVICEFARM_LOG_DIR'
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
            f"{device} shell \"am instrument -r -w --no-window-animation "
            f"-e class com.llmhub.llmhub.soundgen.StableAudio3DeviceTest "
            f"-e sa3Model '{model}' -e sa3Seconds {seconds} {runner}\" 2>&1 | tee {log}/instrument.log\"",
            f'{device} logcat -d -s StableAudio3DeviceTest:* StableAudio3:* MusicGeneratorBackend:* > {log}/results.txt"',
            f'{device} logcat -d -b crash > {log}/crash-buffer.txt"',
            f'! grep -Eq "FAILURES|INSTRUMENTATION_FAILED|Process crashed" {log}/instrument.log"',
        ]},
        "post_test": {"commands": [
            f'{device} logcat -d -v threadtime > {log}/logcat.txt"',
        ]},
    },
    "artifacts": ["$DEVICEFARM_LOG_DIR"],
}
output.write_text(json.dumps(spec, indent=2))  # JSON is valid YAML.
