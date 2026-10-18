#!/bin/bash
# Exports the screenshot attachments of a LLMHubScreenshotUITests result bundle.
#
# Usage: extract_attachments.sh <xcresult> <outdir>
#
# Screenshots attached as "<lang>/<NN>_<feature>[_duo-<pose>].png" land in
# <outdir>/<lang>/<NN>_<feature>[_duo-<pose>].png. Generated result files attached as
# "outputs/<file>" land in <outdir>/outputs/<file>. Other attachments (UI snapshots,
# debug descriptions) are skipped. Needs Xcode 16+ (`xcresulttool export attachments`).
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "usage: $0 <xcresult> <outdir>" >&2
  exit 64
fi

XCRESULT="$1"
OUTDIR="$2"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/llmhub-attachments.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

xcrun xcresulttool export attachments --path "$XCRESULT" --output-path "$WORK" > /dev/null
mkdir -p "$OUTDIR"

/usr/bin/python3 -I - "$WORK" "$OUTDIR" <<'PY'
import json, os, re, shutil, sys

work, outdir = sys.argv[1], sys.argv[2]
languages = ["zh-TW", "ar", "da", "de", "en", "es", "fa", "fr", "he", "hi", "id", "it",
             "ja", "ko", "nl", "pl", "pt", "ru", "th", "tr", "uk", "vi"]
# xcresulttool drops the "/" from attachment names and appends "_<n>_<UUID>" before the
# extension, e.g. "zh-TW/07_translator.png" -> "zh-TW07_translator_0_<UUID>.png".
suffix = re.compile(r"_\d+_[0-9A-Fa-f-]{36}(?=(\.[^.]*)?$)")
screenshot = re.compile(r"^(?P<lang>" + "|".join(map(re.escape, languages)) + r")/?(?P<rest>\d\d_[^/]+\.png)$")

with open(os.path.join(work, "manifest.json")) as handle:
    manifest = json.load(handle)

count = 0
for test in manifest:
    for attachment in test.get("attachments", []):
        name = suffix.sub("", attachment["suggestedHumanReadableName"])
        source = os.path.join(work, attachment["exportedFileName"])
        match = screenshot.match(name)
        if match:
            target = os.path.join(outdir, match["lang"], match["rest"])
        elif name.startswith("outputs"):
            target = os.path.join(outdir, "outputs", name[len("outputs"):].lstrip("/"))
        else:
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(source, target)
        count += 1
print(f"Exported {count} files to {outdir}")
PY
