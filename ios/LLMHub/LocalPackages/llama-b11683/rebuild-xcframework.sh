#!/usr/bin/env bash
set -euo pipefail

# Build only the missing simulator slice; keep upstream device/Metal binaries.
VERSION=b11683
SOURCE_SHA256=f6dc7d2650fc75ca3c3bcee71a0e03243ec5339d7f727a854fd9d35da15ca7b7
ARTIFACT_SHA256=efdd0dabe6013653ed57a80b316d1600a7a349d86f8545beca3cd8a4804795bd
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
