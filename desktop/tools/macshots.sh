#!/bin/sh
# Real Liquid Glass screenshots of the Mac app: ./macshots.sh <out-dir> [pages] [wait-seconds] [extra app args split by |]
# e.g. desktop/tools/macshots.sh /tmp/shots design,studio,storage 4
set -e
OUT=${1:?out dir}; PAGES=${2:-chat,agent,design,studio}; WAIT=${3:-4}; ARGS=${4:-}
cd "$(dirname "$0")/../macos"
DD=${DERIVED_DATA:-build-tests}
rm -rf "$DD/shots.xcresult"; mkdir -p "$OUT"
TEST_RUNNER_SHOT_PAGES="$PAGES" TEST_RUNNER_SHOT_WAIT="$WAIT" TEST_RUNNER_SHOT_ARGS="$ARGS" \
  xcodebuild test -project LLMHub.xcodeproj -scheme LLMHub -derivedDataPath "$DD" -resultBundlePath "$DD/shots.xcresult" \
  -only-testing:LLMHubUITests/PageShots/testPageShots -quiet 2>&1 | grep -E "error|Test.*(passed|failed)" || true
xcrun xcresulttool export attachments --path "$DD/shots.xcresult" --output-path "$OUT" >/dev/null
# Name files after the page (attachments come out with generated names plus a manifest).
python3 - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
for test in json.load(open(os.path.join(out, "manifest.json"))):
    for a in test.get("attachments", []):
        src = os.path.join(out, a["exportedFileName"])
        name = a.get("suggestedHumanReadableName", a["exportedFileName"]).split("_")[0]
        if os.path.exists(src):
            os.replace(src, os.path.join(out, f"{name}.png"))
PY
ls "$OUT"
