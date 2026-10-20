#!/bin/bash
# Captures the iPhone Duo simulator screenshots in one pose, in injection mode (media
# results come from real device outputs instead of generating in the simulator).
#
# Usage: run_duo_simulator.sh <xctestrun> <udid> <inner-landscape|outer-portrait> <device-outputs-dir> <outdir> [only-test]
#
#   <xctestrun>          from `xcodebuild build-for-testing -scheme LLMHubScreenshots
#                        -destination id=<udid>` (…/Build/Products/*.xctestrun)
#   <device-outputs-dir> folder with image.png, video.mp4, video_input.png, music.wav,
#                        upscale_input.png, upscale_output.png (the "outputs/" attachments
#                        of a device run, or Documents/screenshot_outputs from a device)
#   [only-test]          e.g. test07_translator; default runs every test
#
# Put the simulator in the pose first: a freshly booted iPhone Duo is folded (outer
# display); unfolding needs Device Hub. The test rotates the device itself.
set -euo pipefail

if [ $# -lt 5 ]; then
  sed -n '2,15p' "$0" >&2
  exit 64
fi
XCTESTRUN="$1"; UDID="$2"; POSE="$3"; OUTPUTS="$4"; OUTDIR="$5"; ONLY="${6:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"

xcrun simctl status_bar "$UDID" override --time 9:41 --dataNetwork wifi --wifiMode active \
  --wifiBars 3 --batteryState charged --batteryLevel 100

# The app must be installed for its data container to exist.
if ! CONTAINER="$(xcrun simctl get_app_container "$UDID" com.llmhub.llmhub data 2>/dev/null)"; then
  APP="$(dirname "$XCTESTRUN")/Debug-iphonesimulator/LLMHub.app"
  xcrun simctl install "$UDID" "$APP"
  CONTAINER="$(xcrun simctl get_app_container "$UDID" com.llmhub.llmhub data)"
fi
mkdir -p "$CONTAINER/Documents/screenshot_outputs"
cp "$OUTPUTS"/* "$CONTAINER/Documents/screenshot_outputs/"

RESULT="$OUTDIR/duo-$POSE.xcresult"
rm -rf "$RESULT"
mkdir -p "$OUTDIR"
TEST_FILTER="LLMHubScreenshotUITests/StoreScreenshotTests${ONLY:+/$ONLY}"

# XCUITest cannot capture either display correctly (the inner one is not listed, the outer
# one comes back sideways): the test asks for each shot through $CAPTURE/request and this
# loop takes it with simctl, answering in $CAPTURE/done. Display 1 is the outer display.
HOST_ENV=()
DISPLAY_ARG=()
[ "$POSE" = "outer-portrait" ] && DISPLAY_ARG=(--display=1)
if true; then
  CAPTURE="$OUTDIR/host-capture"
  rm -rf "$CAPTURE"; mkdir -p "$CAPTURE"
  HOST_ENV=(TEST_RUNNER_SCREENSHOT_HOST_CAPTURE_DIR="$CAPTURE")
  (
    last=""
    while true; do
      if [ -f "$CAPTURE/request" ]; then
        name="$(cat "$CAPTURE/request")"
        if [ -n "$name" ] && [ "$name" != "$last" ]; then
          mkdir -p "$OUTDIR/screenshots/$(dirname "$name")"
          xcrun simctl io "$UDID" screenshot ${DISPLAY_ARG[@]+"${DISPLAY_ARG[@]}"} --type=png "$OUTDIR/screenshots/$name" >/dev/null 2>&1
          printf '%s' "$name" > "$CAPTURE/done"
          last="$name"
        fi
      fi
      sleep 0.2
    done
  ) &
  CAPTURE_PID=$!
  trap 'kill $CAPTURE_PID 2>/dev/null' EXIT
fi

env ${HOST_ENV[@]+"${HOST_ENV[@]}"} TEST_RUNNER_SCREENSHOT_DUO_INJECT=1 TEST_RUNNER_SCREENSHOT_DUO_POSE="$POSE" \
  xcodebuild test-without-building -xctestrun "$XCTESTRUN" -destination "id=$UDID" \
  -only-testing:"$TEST_FILTER" -resultBundlePath "$RESULT" || echo "Some tests failed; extracting what was captured"

"$HERE/extract_attachments.sh" "$RESULT" "$OUTDIR/screenshots"
