#!/bin/bash
# Builds the vendor NPU dispatch libraries that LiteRT-LM loads from the LiteRT source.
#
# Not used by the app since litertlm-android 0.20.0: it links the app's libLiteRt.so (litert 2.4.0), and
# the litert-npu-runtime 2.4.0 artifacts on Maven export LiteRtDispatchQueryInterface, so
# LiteRtLmEmbeddingService stages those. Kept for the case where a LiteRT-LM release again needs a
# dispatch ABI that Maven does not have yet (0.19.x did: Maven 2.3.x only exported LiteRtDispatchGetApi
# and every NPU model failed with "No usable Dispatch runtime found").
#
# To use it: set LITERT_REF to LITERT_REF in that release's WORKSPACE
# (https://github.com/google-ai-edge/LiteRT-LM/blob/v<version>/WORKSPACE), build, ship the three
# libraries in the APK and stage them in LiteRtLmEmbeddingService.prepareNpuLibraryDir, then run
# EmbeddingNpuLoadDeviceTest on a Snapdragon, a MediaTek and a Pixel device.
#
# Linux x86_64. Needs a QAIRT 2.50.0 SDK folder for the Qualcomm headers:
#   LITERT_QAIRT_SDK=/path/to/qairt/2.50.0.260828 ./build_dispatch.sh [work dir]
set -euo pipefail

LITERT_REF=bf49eb62406d15948261a840c9dcd8e4f13b478c  # litertlm-android 0.20.0 (last used: dcb657e8 for 0.19.1)
WORK=${1:-$HOME/litert_build}
: "${LITERT_QAIRT_SDK:?Set LITERT_QAIRT_SDK to a QAIRT 2.50.0 SDK folder}"
mkdir -p "$WORK" && cd "$WORK"

[ -x bazelisk ] || { curl -sL -o bazelisk https://github.com/bazelbuild/bazelisk/releases/download/v1.25.0/bazelisk-linux-amd64; chmod +x bazelisk; }
# NDK r25c is the newest LiteRT's configure accepts. Extract with unzip: the toolchain is full of symlinks.
[ -d android-ndk-r25c ] || { curl -sL -o ndk.zip https://dl.google.com/android/repository/android-ndk-r25c-linux.zip; unzip -q ndk.zip; rm ndk.zip; }
if [ ! -d LiteRT/.git ]; then
  mkdir -p LiteRT && cd LiteRT
  git init -q && git remote add origin https://github.com/google-ai-edge/LiteRT.git
  git fetch -q --depth 1 origin "$LITERT_REF" && git checkout -q FETCH_HEAD
  cd ..
fi

export ANDROID_NDK_HOME=$WORK/android-ndk-r25c ANDROID_NDK_API_LEVEL=26 ANDROID_NDK_VERSION=25
cd LiteRT
# The public BUILD file lists Google's internal profiling hooks (platforms/darwinn) as a dependency of
# the Tensor dispatch library, although the code that uses them is compiled out
# (LITERT_GOOGLE_TENSOR_HAS_HOOKS is not defined). Without this line the target does not load.
sed -i '/^        "\/\/litert\/vendors\/google_tensor\/hooks",$/d' litert/vendors/google_tensor/dispatch/BUILD
../bazelisk build -c opt --config=android_arm64 \
  --repo_env=ANDROID_NDK_HOME="$ANDROID_NDK_HOME" --repo_env=ANDROID_NDK_API_LEVEL=26 --repo_env=ANDROID_NDK_VERSION=25 \
  --repo_env=LITERT_QAIRT_SDK="$LITERT_QAIRT_SDK" --action_env=ANDROID_NDK_HOME="$ANDROID_NDK_HOME" \
  //litert/vendors/qualcomm/dispatch:dispatch_api_so //litert/vendors/mediatek/dispatch:dispatch_api_so \
  //litert/vendors/google_tensor/dispatch:dispatch_api_so

mkdir -p "$WORK/out"
for vendor in qualcomm:Qualcomm mediatek:MediaTek google_tensor:GoogleTensor; do
  "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip" --strip-unneeded \
    -o "$WORK/out/libLiteRtDispatch_${vendor#*:}.so" "bazel-bin/litert/vendors/${vendor%:*}/dispatch/libLiteRtDispatch_${vendor#*:}.so"
done
ls -la "$WORK/out"
