#!/usr/bin/env bash
# Builds libggml-hexagon.so (the Hexagon NPU backend's host library) from the vendored, patched llama.cpp
# tree and copies it, stripped, to app/src/main/jniLibs/arm64-v8a. The NPU-side skels (libggml-htp-v*.so in
# app/src/main/assets/llama_htp) stay the official release's: the patch changes the host side only.
#
# Needs Linux x86-64 (WSL works) with Qualcomm's Hexagon SDK and the Android NDK from the toolchain image llama.cpp's
# Snapdragon releases are built with (ghcr.io/snapdragon-toolchain/arm64-android:v0.8). Without Docker, unpack it with
# crane (github.com/google/go-containerregistry):
#   crane export --platform linux/amd64 ghcr.io/snapdragon-toolchain/arm64-android:v0.8 - | tar -x 'opt/hexagon/*' 'opt/android-ndk-*'
# Flags follow the release preset (docs/backend/snapdragon/CMakeUserPresets.json, arm64-android-snapdragon).
#
# usage: HEXAGON_SDK_ROOT=.../opt/hexagon/6.7.0.0 ANDROID_NDK_ROOT=.../opt/android-ndk-r29 bash android/tools/llama_hexagon/build.sh
set -euo pipefail
ANDROID_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LLAMA_DIR="$ANDROID_DIR/../llama.cpp"
OUT_DIR="$ANDROID_DIR/app/src/main/jniLibs/arm64-v8a"
: "${HEXAGON_SDK_ROOT:?set HEXAGON_SDK_ROOT}" "${ANDROID_NDK_ROOT:?set ANDROID_NDK_ROOT}"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/llmhub-llama-hexagon.XXXXXX")"
echo "Build files: $WORK_DIR"
FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE"
cmake -S "$LLAMA_DIR" -B "$WORK_DIR/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 \
    -DCMAKE_C_FLAGS="$FLAGS" -DCMAKE_CXX_FLAGS="$FLAGS" -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG" -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG" \
    -DHEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT" -DPREBUILT_LIB_DIR=android_aarch64 -DGGML_HEXAGON=ON -DGGML_OPENCL=OFF \
    -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF -DBUILD_SHARED_LIBS=ON -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_COMMON=OFF
cmake --build "$WORK_DIR/build" --target ggml-hexagon
"$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip" --strip-unneeded \
    -o "$OUT_DIR/libggml-hexagon.so" "$(find "$WORK_DIR/build" -name libggml-hexagon.so | head -1)"
echo "Updated $OUT_DIR/libggml-hexagon.so"
