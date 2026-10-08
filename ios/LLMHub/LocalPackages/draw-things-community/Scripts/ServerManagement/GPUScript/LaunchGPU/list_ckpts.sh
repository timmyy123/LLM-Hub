#!/bin/bash

# A script to list files ending in .ckpt-tensordata, sorted by size.
# It has no external dependencies and uses standard shell utilities.

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Use the first command-line argument as the target directory.
# If no argument is provided, default to the current directory ('.').
SEARCH_DIR="${1:-.}"


# -----------------------------------------------------------------------------
# Main Logic
# -----------------------------------------------------------------------------

## Check 1: Ensure the target directory exists
if [ ! -d "$SEARCH_DIR" ]; then
    echo "Error: Directory '$SEARCH_DIR' not found. 🤷"
    exit 1
fi

## Check 2: Find matching files and handle the "no files found" case
# We use 'find' because it handles cases with no matching files gracefully,
# unlike 'ls *.ckpt-tensordata' which can error out.
# -maxdepth 1 ensures the search is not recursive.
# -print0 and xargs -0 handle filenames with spaces or special characters.
file_list=$(find "$SEARCH_DIR" -maxdepth 1 -type f -name "*.ckpt-tensordata" -print0)

if [ -z "$file_list" ]; then
    echo "No files ending in .ckpt-tensordata found in '$SEARCH_DIR'."
    exit 0
fi

## List and Sort Files
echo "✅ Files ending in .ckpt-tensordata in '$SEARCH_DIR', sorted by size:"
echo "-----------------------------------------------------------------"

# Use xargs to pass the null-terminated file list to 'ls'.
# ls flags explained:
# -l : Use a long listing format (required to show size).
# -h : Show sizes in human-readable format (e.g., 1K, 23M, 4G).
# -S : Sort by file size, largest first.
# -r : To sort smallest first, add this flag (e.g., ls -lhrS).
printf "%s" "$file_list" | xargs -0 ls -lhS
