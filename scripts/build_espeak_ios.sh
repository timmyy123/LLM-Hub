#!/usr/bin/env bash
# Build espeak-ng as a static library + XCFramework for iOS.
#
# Outputs:
#   ios/vendor/EspeakNG.xcframework/      (device + simulator slices)
#   ios/vendor/EspeakNG/include/          (public headers + module map)
#   ios/vendor/EspeakNG/share/espeak-ng-data/   (voice data, bundled in app)
#
# This is the FIRST half of espeak-ng on iOS. The SECOND half (which this
# script does NOT do) is wiring the XCFramework into the SwiftPM target or
# host app:
#   - Copy ios/vendor/EspeakNG.xcframework into the Xcode project's Frameworks
#   - Update Package.swift to add `.binaryTarget(name: "EspeakNG", path: "...")`
#     plus a clang module map exposing speak_lib.h
#   - Replace the body of `EspeakG2P.swift::tryLoad()` to actually call
#     espeak_Initialize / espeak_TextToPhonemes
#
# The reason that's manual: SwiftPM binary targets must resolve at build time,
# so we can't make the XCFramework optional from a single Package.swift the way
# the Android CMakeLists.txt early-returns. See docs/espeak-ng-setup.md.
#
# Requirements on the build host:
#   - macOS with Xcode 15+
#   - cmake, git
#   - ~5 minutes per slice

set -euo pipefail

ESPEAK_REPO="${ESPEAK_REPO:-https://github.com/espeak-ng/espeak-ng.git}"
ESPEAK_REF="${ESPEAK_REF:-1.52.0}"
IOS_DEPLOYMENT_TARGET="${IOS_DEPLOYMENT_TARGET:-17.0}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$ROOT/build/espeak-ios"
SRC="$WORK/espeak-ng"
OUT="$ROOT/ios/vendor"
mkdir -p "$WORK" "$OUT"

if [[ ! -d "$SRC/.git" ]]; then
    echo ">>> cloning espeak-ng @ $ESPEAK_REF"
    git clone --depth 1 --branch "$ESPEAK_REF" "$ESPEAK_REPO" "$SRC"
fi

# Stage 1 — host build to produce data files.
HOST_BUILD="$WORK/host-build"
HOST_INSTALL="$WORK/host-install"
if [[ ! -x "$HOST_INSTALL/bin/espeak-ng" ]]; then
    echo ">>> building host espeak-ng (one-time, used to stage data files)"
    rm -rf "$HOST_BUILD"
    cmake -S "$SRC" -B "$HOST_BUILD" \
        -DCMAKE_INSTALL_PREFIX="$HOST_INSTALL" \
        -DBUILD_SHARED_LIBS=OFF -DUSE_ASYNC=OFF \
        -DUSE_LIBSONIC=OFF -DUSE_LIBPCAUDIO=OFF -DUSE_KLATT=ON \
        -DUSE_SPEECHPLAYER=ON -DUSE_MBROLA=OFF
    cmake --build "$HOST_BUILD" --target espeak-ng -j
    cmake --install "$HOST_BUILD"
fi

build_slice() {
    local NAME="$1"; shift
    local SDK="$1"; shift
    local ARCH="$1"; shift
    local PLATFORM="$1"; shift

    local BUILD="$WORK/$NAME-build"
    local INSTALL="$WORK/$NAME-install"
    echo ">>> building espeak-ng for $NAME ($ARCH on $SDK)"
    rm -rf "$BUILD"
    cmake -S "$SRC" -B "$BUILD" \
        -DCMAKE_SYSTEM_NAME=iOS \
        -DCMAKE_OSX_SYSROOT="$SDK" \
        -DCMAKE_OSX_ARCHITECTURES="$ARCH" \
        -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_DEPLOYMENT_TARGET" \
        -DCMAKE_INSTALL_PREFIX="$INSTALL" \
        -DBUILD_SHARED_LIBS=OFF \
        -DUSE_ASYNC=OFF -DUSE_LIBSONIC=OFF -DUSE_LIBPCAUDIO=OFF \
        -DUSE_KLATT=ON -DUSE_SPEECHPLAYER=ON -DUSE_MBROLA=OFF
    cmake --build "$BUILD" --target espeak-ng -j
    cmake --install "$BUILD"
}

build_slice ios-device     "$(xcrun --sdk iphoneos        --show-sdk-path)" arm64 iOS
build_slice ios-simulator  "$(xcrun --sdk iphonesimulator --show-sdk-path)" "arm64;x86_64" iOS-simulator

# Stage 3 — assemble XCFramework.
XC_OUT="$OUT/EspeakNG.xcframework"
rm -rf "$XC_OUT"
xcodebuild -create-xcframework \
    -library "$WORK/ios-device-install/lib/libespeak-ng.a"   -headers "$WORK/ios-device-install/include" \
    -library "$WORK/ios-simulator-install/lib/libespeak-ng.a" -headers "$WORK/ios-simulator-install/include" \
    -output "$XC_OUT"
echo ">>> XCFramework written to $XC_OUT"

# Stage 4 — stage data files for bundling.
DATA_DST="$OUT/EspeakNG/share/espeak-ng-data"
rm -rf "$DATA_DST"
mkdir -p "$DATA_DST"
cp -R "$HOST_INSTALL/share/espeak-ng-data/." "$DATA_DST/"
echo ">>> data files at $DATA_DST"

cat <<EOF

✓ espeak-ng iOS slices ready.

Next steps (manual, see docs/espeak-ng-setup.md):
  1. Drag EspeakNG.xcframework into the host Xcode project's Frameworks group
     (or add as a SwiftPM binaryTarget in a separate package).
  2. Add EspeakNG/share/espeak-ng-data/ as a "Folder Reference" in Resources
     so it ships at Bundle.main.bundlePath/espeak-ng-data/.
  3. Replace EspeakG2P.swift::tryLoad() with the real implementation that
     calls espeak_Initialize / espeak_TextToPhonemes.
EOF
