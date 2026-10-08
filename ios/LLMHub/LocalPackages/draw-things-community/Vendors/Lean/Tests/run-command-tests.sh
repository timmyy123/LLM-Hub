#!/bin/bash
set -euo pipefail
# Exercise the same ABI after local/debug-symbol stripping. Leave the Bazel
# artifact intact; the copy contains only system dylib dependencies.
binary="${TEST_TMPDIR:?}/LeanCommandValidation-stripped"
cp "$1" "$binary"
chmod u+w "$binary"
xcrun strip -S -x "$binary"
xcrun codesign --force --sign - "$binary"
shift
exec "$binary" "$@"
