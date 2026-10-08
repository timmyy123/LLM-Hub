#!/usr/bin/env bash

set -euo pipefail

GIT_CONFIG=$(git rev-parse --git-dir)
GIT_ROOT=$(git rev-parse --show-toplevel)

cd $GIT_ROOT

../bazel/bazel-dev build Apps/DrawThings:DrawThingsMac --config=release --linkopt=-L/System/iOSSupport/usr/lib/swift --swiftcopt=-Fsystem --swiftcopt=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/iOSSupport/System/Library/Frameworks/ --swiftcopt=-disable-autolinking-runtime-compatibility --linkopt=-weak_library --linkopt=/usr/lib/swift/libswiftPhotosUI.dylib --linkopt=-weak_library --linkopt=/usr/lib/swift/libswiftPhotos.dylib --catalyst_cpus=arm64,x86_64 --apple_platform_type=catalyst
