#!/usr/bin/env bash
set -euo pipefail

# Builds LLMHubBmoe.xcframework (iOS device, iOS Simulator, macOS): a static library with BigMoeOnEdge's streaming
# engine (third_party/bigmoeonedge), llama.cpp's common helpers and the C bridge (include/llmhub_bmoe.h), for flash
# streaming of MoE GGUF models. It is compiled against the same llama.cpp source as ../llama.xcframework
# (b12631 + ../llmhub-aion.patch, checked by SHA-256) and links to that framework's llama, ggml and mtmd.
VERSION=b12631
SOURCE_SHA256=bea5a657605d90dc2104ec7f3134cbec90e5fdbe1a2fdd50e66782cbedaff56b
BRIDGE_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKAGE_DIR="$(dirname "$BRIDGE_DIR")"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/llmhub-bmoe-apple.XXXXXX")"
echo "Native dependency build files: $WORK_DIR"

curl --fail --location --retry 3 "https://github.com/ggml-org/llama.cpp/archive/refs/tags/$VERSION.tar.gz" -o "$WORK_DIR/source.tar.gz"
printf '%s  %s\n' "$SOURCE_SHA256" "$WORK_DIR/source.tar.gz" | shasum -a 256 -c -
tar -xzf "$WORK_DIR/source.tar.gz" -C "$WORK_DIR"
SOURCE_DIR="$WORK_DIR/llama.cpp-$VERSION"
patch -d "$SOURCE_DIR" -p1 --forward < "$PACKAGE_DIR/llmhub-aion.patch"
patch -d "$SOURCE_DIR" -p1 --forward < "$PACKAGE_DIR/llmhub-metal-nvfp4.patch"

build_slice() { # name sysroot archs deployment-flag
    local name=$1 sysroot=$2 archs=$3
    local build="$WORK_DIR/build-$name"
    local system=()
    if [ "$sysroot" != macosx ]; then system=(-DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_DEPLOYMENT_TARGET=17.5); else system=(-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0); fi
    cmake -S "$BRIDGE_DIR" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_OSX_SYSROOT="$sysroot" -DCMAKE_OSX_ARCHITECTURES="$archs" "${system[@]}" \
        -DLLAMA_SOURCE_DIR="$SOURCE_DIR" >/dev/null
    cmake --build "$build" --target llmhub_bmoe bmoe_core llama-common llama-common-base cpp-httplib
    mkdir -p "$WORK_DIR/lib-$name"
    # One archive with the bridge, the engine and llama.cpp's common helpers; not llama/ggml/mtmd (the framework's).
    libtool -static -o "$WORK_DIR/lib-$name/libLLMHubBmoe.a" \
        "$build/libllmhub_bmoe.a" \
        "$build/bigmoeonedge/libbmoe_core.a" \
        "$build/llama.cpp/common/libllama-common.a" \
        "$build/llama.cpp/common/libllama-common-base.a" \
        "$build/llama.cpp/vendor/cpp-httplib/libcpp-httplib.a"
}

build_slice ios-device iphoneos arm64
build_slice ios-sim iphonesimulator "arm64;x86_64"
build_slice macos macosx "arm64;x86_64"

HEADERS="$WORK_DIR/headers"
mkdir -p "$HEADERS"
cp "$BRIDGE_DIR/include/llmhub_bmoe.h" "$HEADERS/"
printf 'module LLMHubBmoe {\n    header "llmhub_bmoe.h"\n    export *\n}\n' > "$HEADERS/module.modulemap"

xcrun xcodebuild -create-xcframework \
    -library "$WORK_DIR/lib-ios-device/libLLMHubBmoe.a" -headers "$HEADERS" \
    -library "$WORK_DIR/lib-ios-sim/libLLMHubBmoe.a" -headers "$HEADERS" \
    -library "$WORK_DIR/lib-macos/libLLMHubBmoe.a" -headers "$HEADERS" \
    -output "$WORK_DIR/LLMHubBmoe.xcframework"

rsync -a --delete "$WORK_DIR/LLMHubBmoe.xcframework/" "$PACKAGE_DIR/LLMHubBmoe.xcframework/"
echo "Updated $PACKAGE_DIR/LLMHubBmoe.xcframework (device, simulator, macOS)."
