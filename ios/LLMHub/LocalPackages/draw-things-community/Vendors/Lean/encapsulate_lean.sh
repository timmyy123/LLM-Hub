#!/bin/bash
set -euo pipefail
export LC_ALL=C
export COPYFILE_DISABLE=1

if [[ $# -lt 7 ]]; then
  echo "usage: $0 <sdk> <arch> <minimum-os> <symbols> <roots> <output> <archives...>" >&2
  exit 1
fi
sdk=$1
arch=$2
minimum_os=$3
symbols=$4
roots=$5
output=$6
shift 6
case "$sdk" in
  iphoneos) platform=ios ;;
  iphonesimulator) platform=ios-simulator ;;
  macosx) platform=macos ;;
  *) echo "unsupported SDK: $sdk" >&2; exit 1 ;;
esac

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/lean-encapsulation.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT
sdk_root=$(xcrun --sdk "$sdk" --show-sdk-path)
sdk_version=$(xcrun --sdk "$sdk" --show-sdk-version)

# As with PythonIOS, allocate common globals and bind dependency references
# locally before stripping. Apply the export boundary to this object only, not
# the app: other embedded runtimes must retain their own public interfaces.
lean_ld_classic=${LEAN_LD_CLASSIC:-$(xcrun --sdk "$sdk" --find ld-classic)}
"$lean_ld_classic" -r -d -arch "$arch" \
  -syslibroot "$sdk_root" \
  -platform_version "$platform" "$minimum_os" "$sdk_version" \
  -bitcode_process_mode strip -exported_symbols_list "$symbols" \
  "@$roots" "$@" -o "$work_dir/Lean.o"
xcrun nmedit -s "$symbols" "$work_dir/Lean.o"
xcrun libtool -static -D -o "$work_dir/libLean.a" "$work_dir/Lean.o"
xcrun strip -S -x "$work_dir/libLean.a"
xcrun ranlib "$work_dir/libLean.a"
xcrun nm -m "$work_dir/libLean.a" > "$work_dir/symbols.txt"
if grep -Eq '\(common\)| private external ' "$work_dir/symbols.txt"; then
  echo "Lean archive retains common or private-external definitions" >&2
  exit 1
fi
awk '/ external / && !/ private external / && !/\(undefined\)/ { print $NF }' \
  "$work_dir/symbols.txt" | sort -u > "$work_dir/exports.txt"
diff -u "$symbols" "$work_dir/exports.txt"
# No allocator/libuv/GMP calls may escape to another runtime's implementation.
if grep -Eq '\(undefined\).* (__?mi_|_uv_|_mp[zqn]_|_mp_)' \
  "$work_dir/symbols.txt"; then
  echo "Lean archive contains an unbound private dependency" >&2
  exit 1
fi
cp "$work_dir/libLean.a" "$output"
