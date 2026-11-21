#!/usr/bin/env bash
set -euo pipefail

# Build only the missing simulator slice; keep upstream device/Metal binaries.
VERSION=b12613
SOURCE_SHA256=539f5b7a306c9392e0308094f9c39d1c793041bf18a2c5da5e93f79050be2eb7
ARTIFACT_SHA256=7075378ab937650128d1356d6952600ae2e65dbcf9528a07907b337290528a68
PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/llmhub-llama-apple.XXXXXX")"
echo "Native dependency build files: $WORK_DIR"

curl --fail --location --retry 3 "https://github.com/ggml-org/llama.cpp/archive/refs/tags/$VERSION.tar.gz" -o "$WORK_DIR/source.tar.gz"
curl --fail --location --retry 3 "https://github.com/ggml-org/llama.cpp/releases/download/$VERSION/llama-$VERSION-xcframework.zip" -o "$WORK_DIR/apple.zip"
printf '%s  %s\n' "$SOURCE_SHA256" "$WORK_DIR/source.tar.gz" "$ARTIFACT_SHA256" "$WORK_DIR/apple.zip" | shasum -a 256 -c -
tar -xzf "$WORK_DIR/source.tar.gz" -C "$WORK_DIR"
unzip -q "$WORK_DIR/apple.zip" -d "$WORK_DIR/upstream"

SOURCE_DIR="$WORK_DIR/llama.cpp-$VERSION"
UPSTREAM="$WORK_DIR/upstream/build-apple/llama.xcframework"
(cd "$SOURCE_DIR" && bash build-xcframework.sh ios-sim)

xcrun xcodebuild -create-xcframework \
    -framework "$UPSTREAM/ios-arm64/llama.framework" \
    -framework "$UPSTREAM/macos-arm64_x86_64/llama.framework" \
    -framework "$SOURCE_DIR/build-ios-sim/framework/llama.framework" \
    -output "$WORK_DIR/llama.xcframework"

rsync -a --delete "$WORK_DIR/llama.xcframework/" "$PACKAGE_DIR/llama.xcframework/"
echo "Updated $PACKAGE_DIR/llama.xcframework (device, simulator, macOS)."
