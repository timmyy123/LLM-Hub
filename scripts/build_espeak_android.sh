#!/usr/bin/env bash
# Cross-compile espeak-ng for Android NDK and stage everything Kokoro's
# EspeakG2P needs.
#
# Outputs into the Android module:
#   android/app/src/main/jniLibs/<abi>/libespeak-ng.so          (per ABI)
#   android/app/src/main/cpp/espeak-ng/include/speak_lib.h      (headers)
#   android/app/src/main/assets/espeak-ng-data/                 (voice data)
#
# When this finishes successfully, a clean Gradle build of the app will:
#   1. Compile src/main/cpp/espeak_jni.cpp into libespeak-jni.so
#   2. Package both .so files plus the espeak-ng-data assets into the APK
#   3. Make EspeakG2P.tryLoad() succeed at runtime
#
# Requirements on the build host:
#   - Android NDK r25 or newer ($ANDROID_NDK_HOME or $ANDROID_NDK_ROOT)
#   - cmake, make, git
#   - ~250 MB of disk + ~2 minutes per ABI on a recent laptop
#
# Pass ABIs to build via the ABIS env var, default arm64-v8a only (matches
# the Gradle abiFilter). Add `armeabi-v7a x86_64` for emulator/older devices.

set -euo pipefail

ABIS="${ABIS:-arm64-v8a}"
ANDROID_API="${ANDROID_API:-27}"
ESPEAK_REPO="${ESPEAK_REPO:-https://github.com/espeak-ng/espeak-ng.git}"
ESPEAK_REF="${ESPEAK_REF:-1.52.0}"

NDK="${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}"
if [[ -z "$NDK" ]]; then
    echo "error: ANDROID_NDK_HOME / ANDROID_NDK_ROOT not set" >&2
    exit 1
fi
TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake"
if [[ ! -f "$TOOLCHAIN_FILE" ]]; then
    echo "error: NDK toolchain file not found at $TOOLCHAIN_FILE" >&2
    exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$ROOT/build/espeak-android"
SRC="$WORK/espeak-ng"
mkdir -p "$WORK"

if [[ ! -d "$SRC/.git" ]]; then
    echo ">>> cloning espeak-ng @ $ESPEAK_REF"
    git clone --depth 1 --branch "$ESPEAK_REF" "$ESPEAK_REPO" "$SRC"
fi

# Stage 1 — build host espeak-ng so it can produce its data files.
HOST_BUILD="$WORK/host-build"
HOST_INSTALL="$WORK/host-install"
if [[ ! -x "$HOST_INSTALL/bin/espeak-ng" ]]; then
    echo ">>> building host espeak-ng (one-time, used to stage data files)"
    rm -rf "$HOST_BUILD"
    cmake -S "$SRC" -B "$HOST_BUILD" \
        -DCMAKE_INSTALL_PREFIX="$HOST_INSTALL" \
        -DBUILD_SHARED_LIBS=OFF -DUSE_ASYNC=OFF \
        -DUSE_LIBSONIC=OFF -DUSE_LIBPCAUDIO=OFF -DUSE_KLATT=ON \
        -DUSE_SPEECHPLAYER=ON -DUSE_MBROLA=OFF -DEXTRA_cmn=OFF \
        -DEXTRA_ru=OFF
    cmake --build "$HOST_BUILD" --target espeak-ng -j
    cmake --install "$HOST_BUILD"
fi

# Stage 2 — cross-compile per ABI.
for ABI in $ABIS; do
    echo ">>> building espeak-ng for $ABI"
    BUILD="$WORK/$ABI-build"
    INSTALL="$WORK/$ABI-install"
    rm -rf "$BUILD"
    cmake -S "$SRC" -B "$BUILD" \
        -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
        -DANDROID_ABI="$ABI" \
        -DANDROID_PLATFORM="android-$ANDROID_API" \
        -DCMAKE_INSTALL_PREFIX="$INSTALL" \
        -DBUILD_SHARED_LIBS=ON \
        -DUSE_ASYNC=OFF -DUSE_LIBSONIC=OFF -DUSE_LIBPCAUDIO=OFF \
        -DUSE_KLATT=ON -DUSE_SPEECHPLAYER=ON -DUSE_MBROLA=OFF
    cmake --build "$BUILD" --target espeak-ng -j
    cmake --install "$BUILD"

    JNI_DIR="$ROOT/android/app/src/main/jniLibs/$ABI"
    mkdir -p "$JNI_DIR"
    cp "$INSTALL/lib/libespeak-ng.so" "$JNI_DIR/libespeak-ng.so"
    echo "    → $JNI_DIR/libespeak-ng.so"
done

# Stage 3 — copy headers + data to the source tree.
HEADERS_DST="$ROOT/android/app/src/main/cpp/espeak-ng/include"
mkdir -p "$HEADERS_DST"
cp "$HOST_INSTALL/include/espeak-ng/speak_lib.h" "$HEADERS_DST/"
echo ">>> headers staged at $HEADERS_DST"

DATA_DST="$ROOT/android/app/src/main/assets/espeak-ng-data"
rm -rf "$DATA_DST"
mkdir -p "$DATA_DST"
cp -R "$HOST_INSTALL/share/espeak-ng-data/." "$DATA_DST/"
echo ">>> data staged at $DATA_DST"

echo
echo "✓ espeak-ng ready. Rebuild the Android app to package it."
echo "  EspeakG2P.tryLoad() should now return a non-null instance at runtime."
