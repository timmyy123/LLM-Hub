#!/usr/bin/env bash
# Builds libggml-opencl.so for the Snapdragon llama.cpp runtime from the vendored, patched
# android/llama.cpp tree (see android/llama.cpp/LLMHUB_VERSION.md), and copies the stripped
# library to app/src/main/jniLibs/arm64-v8a. All other Snapdragon libraries come unchanged from
# the official llama.cpp release; this one carries LLMHub patches.
#
# Flags follow the release preset (docs/backend/snapdragon/CMakeUserPresets.json,
# arm64-android-snapdragon) with -DGGML_HEXAGON=OFF -DBUILD_SHARED_LIBS=ON. The Khronos OpenCL
# headers and an NDK build of the ICD loader are built in a temporary directory; the NDK itself
# is not modified.
#
# usage: bash android/tools/llama_opencl/build.sh [path/to/ndk]
set -euo pipefail

OPENCL_SDK_TAG=v2026.05.29
ANDROID_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
NDK="${1:-${ANDROID_NDK_ROOT:-$HOME/Library/Android/sdk/ndk/29.0.13113456}}"
LLAMA_DIR="$ANDROID_DIR/llama.cpp"
OUT_DIR="$ANDROID_DIR/app/src/main/jniLibs/arm64-v8a"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/llmhub-llama-opencl.XXXXXX")"
TOOLCHAIN="$NDK/build/cmake/android.toolchain.cmake"
STRIP="$(ls "$NDK"/toolchains/llvm/prebuilt/*/bin/llvm-strip | head -1)"
echo "Build files: $WORK_DIR"

git clone --quiet --depth 1 --branch "$OPENCL_SDK_TAG" https://github.com/KhronosGroup/OpenCL-Headers "$WORK_DIR/OpenCL-Headers"
git clone --quiet --depth 1 --branch "$OPENCL_SDK_TAG" https://github.com/KhronosGroup/OpenCL-ICD-Loader "$WORK_DIR/OpenCL-ICD-Loader"
SDK="$WORK_DIR/opencl-sdk"
mkdir -p "$SDK/include" "$SDK/lib"
cp -R "$WORK_DIR/OpenCL-Headers/CL" "$SDK/include/"

cmake -S "$WORK_DIR/OpenCL-ICD-Loader" -B "$WORK_DIR/icd-build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 \
    -DANDROID_STL=c++_shared -DOPENCL_ICD_LOADER_HEADERS_DIR="$SDK/include" -DBUILD_TESTING=OFF
cmake --build "$WORK_DIR/icd-build"
cp "$WORK_DIR/icd-build/libOpenCL.so" "$SDK/lib/"

FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE"
cmake -S "$LLAMA_DIR" -B "$WORK_DIR/llama-build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 \
    -DCMAKE_C_FLAGS="$FLAGS" -DCMAKE_CXX_FLAGS="$FLAGS" \
    -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG" -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG" \
    -DOpenCL_INCLUDE_DIR="$SDK/include" -DOpenCL_LIBRARY="$SDK/lib/libOpenCL.so" -DCMAKE_PREFIX_PATH="$SDK" \
    -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF -DGGML_OPENCL=ON -DGGML_HEXAGON=OFF -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build "$WORK_DIR/llama-build" --target ggml-opencl

LIB="$(find "$WORK_DIR/llama-build" -name libggml-opencl.so | head -1)"
"$STRIP" --strip-unneeded -o "$OUT_DIR/libggml-opencl.so" "$LIB"
echo "Updated $OUT_DIR/libggml-opencl.so"
