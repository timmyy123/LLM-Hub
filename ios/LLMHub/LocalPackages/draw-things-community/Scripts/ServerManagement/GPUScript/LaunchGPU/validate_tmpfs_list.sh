#!/bin/bash

# Validate that every filename in a tmpfs list exists in model_list.
#
# Usage: ./validate_tmpfs_list.sh <tmpfs_list_file> [model_list_file]
#
# Defaults model_list_file to "model_list" in the script's directory.
# model_list is expected to be `ls -l` output; the filename is taken
# from the last whitespace-separated field.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TMPFS_LIST="${1:-}"
MODEL_LIST="${2:-${SCRIPT_DIR}/model_list}"

if [ -z "$TMPFS_LIST" ]; then
    echo "Usage: $0 <tmpfs_list_file> [model_list_file]"
    exit 1
fi

if [ ! -f "$TMPFS_LIST" ]; then
    echo "Error: tmpfs list '$TMPFS_LIST' not found"
    exit 1
fi

if [ ! -f "$MODEL_LIST" ]; then
    echo "Error: model list '$MODEL_LIST' not found"
    exit 1
fi

# Extract just the filenames (last column) from `ls -l` style model_list.
# Skip lines that don't look like file entries (no leading "-" perm bits).
NAMES_FILE=$(mktemp)
trap 'rm -f "$NAMES_FILE"' EXIT
awk '/^-/ {print $NF}' "$MODEL_LIST" | sort -u > "$NAMES_FILE"

missing=0
checked=0
while IFS= read -r line || [ -n "$line" ]; do
    # Skip blank lines and comments
    [[ -z "${line// }" ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue

    checked=$((checked + 1))
    if ! grep -Fxq "$line" "$NAMES_FILE"; then
        echo "MISSING: $line"
        missing=$((missing + 1))
    fi
done < "$TMPFS_LIST"

echo ""
echo "Checked: $checked"
echo "Missing: $missing"

if [ $missing -gt 0 ]; then
    exit 1
fi
echo "All entries valid"
