#!/bin/bash
set -euo pipefail
export LC_ALL=C
export COPYFILE_DISABLE=1

sdk=$1
arch=$2
minimum_os=$3
source_dir=$(cd "$4" && pwd)
support_dir=$(cd "$5" && pwd)
output="$PWD/$6"
[[ "$6" = /* ]] && output=$6
sanitizer=${7:-none}
license=$8
tools_dir=$(cd "$9" && pwd)
shift 9
codec_flags=()
codec_archives=()
codec_include_flags=""
pkgconfig_dirs=""
while (( $# )); do
  codec=$1
  prefix=$(cd "$2" && pwd)
  shift 2
  case "$codec" in
    svt-av1) feature=libsvtav1 ;;
    dav1d) feature=libdav1d ;;
    opus) feature=libopus ;;
    lame) feature=libmp3lame ;;
    libvmaf|libvpx) feature=$codec ;;
    x264|x265) feature=lib$codec ;;
    *) echo "Unsupported codec: $codec" >&2; exit 1 ;;
  esac
  codec_flags+=("--enable-$feature")
  codec_include_flags="$codec_include_flags -I$prefix/include"
  codec_archives+=("$prefix"/lib/*.a)
  pkgconfig_dirs="${pkgconfig_dirs:+$pkgconfig_dirs:}$prefix/lib/pkgconfig"
done
[[ "$license" == gpl ]] && codec_flags+=(--enable-gpl)
export PKG_CONFIG_LIBDIR="$pkgconfig_dirs"
export PKG_CONFIG_PATH=""
case "$sdk" in
  iphoneos) platform=ios; target="$arch-apple-ios$minimum_os" ;;
  iphonesimulator) platform=ios-simulator; target="$arch-apple-ios$minimum_os-simulator" ;;
  macosx) platform=macos; target="$arch-apple-macos$minimum_os" ;;
  *) echo "Unsupported FFmpeg SDK: $sdk" >&2; exit 1 ;;
esac

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/ffmpeg-build.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT
# Bazel sandbox inputs are symlinks; configure and patch must never write back
# through them into the shared external repository or a concurrent build.
cp -RL "$source_dir" "$work_dir/source"
cd "$work_dir/source"
patch -p1 < "$support_dir/ffmpeg-ios.patch" > "$work_dir/patch.log"
cp "$support_dir/FFmpegCommand.h" "$support_dir/ffmpeg_host.h" "$support_dir/ffmpeg_host.c" .
sdk_root=$(xcrun --sdk "$sdk" --show-sdk-path)
sdk_version=$(xcrun --sdk "$sdk" --show-sdk-version)
cc=$(xcrun --sdk "$sdk" --find clang)
# Bazel exports the target SDKROOT. Build-time generators must still be macOS
# executables when the library is cross-compiled for an iPhone or simulator.
host_sdk_root=$(xcrun --sdk macosx --show-sdk-path)
host_cc=$(xcrun --sdk macosx --find clang)
host_flags="-target $(uname -m)-apple-macos13.5 -isysroot $host_sdk_root"
cflags="-target $target -isysroot $sdk_root -O2 -fPIC -fvisibility=hidden$codec_include_flags"
ldflags="-target $target -isysroot $sdk_root"
for archive in ${codec_archives[@]+"${codec_archives[@]}"}; do ldflags="$ldflags -L$(dirname "$archive")"; done
if [[ "$sanitizer" == asan ]]; then
  # Image-wide registration uses a coalesced private-external guard. Hiding that
  # guard during archive encapsulation would register the final image twice.
  # Per-module registration retains global redzones and all ASan checks.
  cflags="$cflags -fsanitize=address -fno-omit-frame-pointer -mllvm -asan-globals-live-support=0"
  ldflags="$ldflags -fsanitize=address"
fi
"$cc" -c -target "$target" -isysroot "$sdk_root" "$support_dir/codec_configure.c" -o "$work_dir/codec_configure.o"
./configure --cc="$cc" --target-os=darwin --arch="${arch/arm64/aarch64}" \
  --host-cc="$host_cc" --host-cflags="$host_flags" --host-ldflags="$host_flags" \
  --enable-cross-compile --sysroot="$sdk_root" --disable-autodetect \
  --disable-shared --enable-static --enable-pic --disable-doc --disable-debug \
  --disable-ffplay --enable-ffprobe --disable-indevs --enable-indev=lavfi \
  --disable-outdevs --enable-pthreads --enable-videotoolbox --enable-audiotoolbox \
  --enable-securetransport --enable-zlib --disable-x86asm \
  --pkg-config="$tools_dir/bin/pkgconf" --pkg-config-flags=--static \
  ${codec_flags[@]+"${codec_flags[@]}"} --extra-libs=-lc++ \
  --extra-cflags="$cflags" --extra-ldflags="$ldflags $work_dir/codec_configure.o" \
  > "$work_dir/configure.log" 2>&1 || {
    cat "$work_dir/configure.log" ffbuild/config.log >&2; exit 1;
  }

# Apply redirection only to the private build, after configure's feature probes.
# Use upstream's object lists so codecs and generated tables stay in sync.
cat >> Makefile <<'MAKE'
CFLAGS += -DFFMPEG_HOST_REDIRECT -include $(SRC_PATH)/ffmpeg_host.h
embedded-ffmpeg: $(OBJS-ffmpeg) $(OBJS-ffprobe) $(FF_DEP_LIBS)
MAKE
if ! make -j8 embedded-ffmpeg > "$work_dir/make.log" 2>&1; then
  tail -100 "$work_dir/make.log" >&2
  exit 1
fi
# The host itself uses real libc; its hooks operate on resolved command paths.
"$cc" -c $cflags -I. ffmpeg_host.c -o ffmpeg_host.o
find fftools -name '*.o' > "$work_dir/objects.txt"
printf '%s\n' "$PWD/ffmpeg_host.o" >> "$work_dir/objects.txt"
find libav* libsw* -name '*.a' >> "$work_dir/objects.txt"
if (( ${#codec_archives[@]} )); then printf '%s\n' "${codec_archives[@]}" >> "$work_dir/objects.txt"; fi
printf '%s\n' _FFmpegCommandRun _FFprobeCommandRun > "$work_dir/exports.txt"
# Only the command ABI escapes: libav* and generic fftools symbols cannot bind
# to another embedded runtime (or another FFmpeg consumer) in the application.
ffmpeg_ld_classic=${FFMPEG_LD_CLASSIC:-$(xcrun --sdk "$sdk" --find ld-classic)}
"$ffmpeg_ld_classic" -r -d -arch "$arch" \
  -syslibroot "$sdk_root" -platform_version "$platform" "$minimum_os" "$sdk_version" \
  -exported_symbols_list "$work_dir/exports.txt" \
  -filelist "$work_dir/objects.txt" -o "$work_dir/FFmpeg.o"
xcrun nmedit -s "$work_dir/exports.txt" "$work_dir/FFmpeg.o"
xcrun libtool -static -D -o "$output" "$work_dir/FFmpeg.o"
xcrun strip -S -x "$output"
xcrun ranlib "$output"
xcrun nm -m "$output" > "$work_dir/symbols.txt"
awk '/ external / && !/ private external / && !/\(undefined\)/ { print $NF }' \
  "$work_dir/symbols.txt" | sort -u > "$work_dir/actual-exports.txt"
diff -u "$work_dir/exports.txt" "$work_dir/actual-exports.txt"
if grep -Eq '\(undefined\).* _(av_|avpriv_|avcodec_|avformat_|avio_|avfilter_|ff_|sws_|swr_)' "$work_dir/symbols.txt"; then
  echo "FFmpeg archive retains an unresolved internal library dependency" >&2
  exit 1
fi
