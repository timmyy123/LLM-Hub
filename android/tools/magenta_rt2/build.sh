#!/bin/bash
# Builds the Magenta RealTime 2 engine (adreno-llms, Apache 2.0) for arm64 and installs it into the app:
#   app/src/main/jniLibs/arm64-v8a/libmagenta_rt2.so   the engine executable (run as a process, `--serve`)
#   app/src/main/jniLibs/arm64-v8a/libclblast.so       CLBlast, which the engine links
#   app/src/main/assets/magenta_rt2/kernels/*.cl       its OpenCL kernels (read from <cwd>/kernels at runtime)
#   app/src/main/assets/magenta_rt2/*LICENSE*          license notices
# The executable keeps the .so name so the APK installs it into nativeLibraryDir (useLegacyPackaging), the only
# app folder Android lets a process exec from. The weights are not bundled; the app downloads them (ModelData).
#
# Usage: tools/magenta_rt2/build.sh [WORK_DIR]   (needs the NDK, cmake, and an adb device for libOpenCL.so)
set -euo pipefail

COMMIT=2b1aee9fdd7d285b35292a9a73cc4f02c94bb65c   # a8nova/adreno-llms, 2026-10-31
APP="$(cd "$(dirname "$0")/../../app" && pwd)"
WORK="${1:-$(mktemp -d)}"
NDK="${ANDROID_NDK:-$(ls -d "$HOME/Library/Android/sdk/ndk/"* | sort -V | tail -1)}"
STRIP="$(ls "$NDK"/toolchains/llvm/prebuilt/*/bin/llvm-strip | head -1)"

if [ ! -d "$WORK/adreno-llms" ]; then
  git clone --quiet https://github.com/a8nova/adreno-llms.git "$WORK/adreno-llms"
fi
git -C "$WORK/adreno-llms" checkout --quiet "$COMMIT"
ENGINE="$WORK/adreno-llms/src/models/magenta-realtime-2"
(cd "$ENGINE" && ANDROID_NDK="$NDK" ADRENO_LLMS_CACHE="${ADRENO_LLMS_CACHE:-$WORK/cache}" NNOPT_DTYPE=fp16 ./scripts/build.sh --release)

"$STRIP" -o "$APP/src/main/jniLibs/arm64-v8a/libmagenta_rt2.so" "$ENGINE/build/fp16/magenta_realtime_2_inference_fp16"
"$STRIP" -o "$APP/src/main/jniLibs/arm64-v8a/libclblast.so" "$ENGINE/build/fp16/_deps/clblast-build/libclblast.so"
rm -rf "$APP/src/main/assets/magenta_rt2"
mkdir -p "$APP/src/main/assets/magenta_rt2/kernels"
cp "$ENGINE"/kernels/*.cl "$APP/src/main/assets/magenta_rt2/kernels/"
cp "$WORK/adreno-llms/LICENSE" "$APP/src/main/assets/magenta_rt2/adreno-llms-LICENSE"
cp "$ENGINE/build/fp16/_deps/clblast-src/LICENSE" "$APP/src/main/assets/magenta_rt2/CLBlast-LICENSE"
echo "$COMMIT" > "$APP/src/main/assets/magenta_rt2/VERSION"
echo "Installed adreno-llms $COMMIT into $APP"
