#!/usr/bin/env bash
set -euo pipefail

# Builds llama.xcframework (iOS device, iOS Simulator, macOS; Metal) from the verified upstream source with
# llmhub-aion.patch applied: the `aion` architecture and its chat template, which upstream does not have yet
# (the same change as android/llama.cpp, see android/llama.cpp/LLMHUB_VERSION.md). The official binary
# xcframework cannot load Aion, so no slice comes from it any more.
VERSION=b12631
SOURCE_SHA256=bea5a657605d90dc2104ec7f3134cbec90e5fdbe1a2fdd50e66782cbedaff56b
PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/llmhub-llama-apple.XXXXXX")"
echo "Native dependency build files: $WORK_DIR"

curl --fail --location --retry 3 "https://github.com/ggml-org/llama.cpp/archive/refs/tags/$VERSION.tar.gz" -o "$WORK_DIR/source.tar.gz"
printf '%s  %s\n' "$SOURCE_SHA256" "$WORK_DIR/source.tar.gz" | shasum -a 256 -c -
tar -xzf "$WORK_DIR/source.tar.gz" -C "$WORK_DIR"

SOURCE_DIR="$WORK_DIR/llama.cpp-$VERSION"
patch -d "$SOURCE_DIR" -p1 --forward < "$PACKAGE_DIR/llmhub-aion.patch"
(cd "$SOURCE_DIR" && bash build-xcframework.sh ios-device ios-sim macos)

# Without the dSYMs that build-xcframework.sh bundles: they stay in the work directory.
xcrun xcodebuild -create-xcframework \
    -framework "$SOURCE_DIR/build-ios-device/framework/llama.framework" \
    -framework "$SOURCE_DIR/build-ios-sim/framework/llama.framework" \
    -framework "$SOURCE_DIR/build-macos/framework/llama.framework" \
    -output "$WORK_DIR/llama.xcframework"

rsync -a --delete "$WORK_DIR/llama.xcframework/" "$PACKAGE_DIR/llama.xcframework/"
echo "Updated $PACKAGE_DIR/llama.xcframework (device, simulator, macOS)."
