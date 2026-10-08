#!/bin/bash
set -euo pipefail
vendor_dir=$(cd "$(dirname "${1:?encapsulation script}")" && pwd)
fixture="$vendor_dir/Tests/Encapsulation"
work_dir=$(mktemp -d "${TEST_TMPDIR:?}/lean-encapsulation-test.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT

for source in api dependency main; do
  xcrun --sdk macosx clang -arch arm64 -mmacosx-version-min=13.5 -fcommon \
    -c "$fixture/$source.c" -o "$work_dir/$source.o"
done
xcrun libtool -static -o "$work_dir/input.a" "$work_dir/api.o" "$work_dir/dependency.o"
xcrun --sdk macosx clang -arch arm64 -mmacosx-version-min=13.5 \
  "$work_dir/main.o" "$work_dir/input.a" -o "$work_dir/unencapsulated"
status=0
"$work_dir/unencapsulated" || status=$?
[[ "$status" == 1 ]] || { echo "fixture did not demonstrate a collision" >&2; exit 1; }

bash "$vendor_dir/encapsulate_lean.sh" macosx arm64 13.5 \
  "$fixture/symbols.txt" "$fixture/roots.rsp" "$work_dir/encapsulated.a" "$work_dir/input.a"
xcrun --sdk macosx clang -arch arm64 -mmacosx-version-min=13.5 \
  "$work_dir/main.o" "$work_dir/encapsulated.a" -o "$work_dir/encapsulated"
"$work_dir/encapsulated"

# Even defining a switch as zero must fail: upstream tests macro presence.
for override in MI_MALLOC_OVERRIDE MI_OSX_INTERPOSE MI_OSX_ZONE; do
  if xcrun clang -fsyntax-only -include "$vendor_dir/include/LeanAllocatorPolicy.h" \
      -D"$override"=0 "$fixture/api.c" > "$work_dir/guard.txt" 2>&1; then
    echo "allocator policy accepted $override" >&2
    exit 1
  fi
  grep -q 'Embedded Lean must not override the host allocator' "$work_dir/guard.txt"
done
echo "Lean dependency function/common-symbol isolation and allocator policy passed"
