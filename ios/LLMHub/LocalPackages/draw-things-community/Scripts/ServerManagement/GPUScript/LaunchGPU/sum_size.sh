#!/bin/bash

# A script to calculate the total size of files listed in an input file.

# --- Functions ---

##
# Displays usage information and exits the script.
#
usage() {
    echo "Usage: $0 <path_to_directory> <path_to_file_list>"
    echo ""
    echo "Arguments:"
    echo "  <path_to_directory>    The base directory where the files are located."
    echo "  <path_to_file_list>    A text file containing one filename per line."
    exit 1
}

# --- Argument Validation ---

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    usage
fi

BASE_DIR="$1"
FILE_LIST="$2"

# Check if the base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory '$BASE_DIR' not found. 🤷"
    exit 1
fi

# Check if the file list exists and is readable
if [ ! -r "$FILE_LIST" ]; then
    echo "Error: File list '$FILE_LIST' not found or cannot be read."
    exit 1
fi

# --- Main Logic ---

total_size_bytes=0
files_found=0
files_missing=0

echo "🔎 Calculating total size..."

# Loop through each line in the provided file list
while IFS= read -r filename || [[ -n "$filename" ]]; do
    # Skip any empty lines in the list
    if [[ -z "$filename" ]]; then
        continue
    fi

    # Construct the full path to the file
    full_path="${BASE_DIR}/${filename}"

    # Check if the file exists and is a regular file
    if [ -f "$full_path" ]; then
        # Get file size in bytes. This command is compatible with both Linux and macOS.
        file_size=$(stat -c%s "$full_path" 2>/dev/null || stat -f%z "$full_path" 2>/dev/null)
        
        # Add the file size to the running total
        total_size_bytes=$((total_size_bytes + file_size))
        files_found=$((files_found + 1))
    else
        # If the file doesn't exist, print a warning to the console
        echo "Warning: File not found, skipping: ${full_path}" >&2
        files_missing=$((files_missing + 1))
    fi
done < "$FILE_LIST"

# --- Display Results ---

# Convert total bytes to a human-readable format (KB, MB, GB, etc.) using awk
human_readable_size=$(echo "$total_size_bytes" | awk '{
    bytes = $1;
    suffixes[1] = "Bytes"; suffixes[2] = "KB"; suffixes[3] = "MB"; suffixes[4] = "GB"; suffixes[5] = "TB";
    i = 1;
    while (bytes >= 1024 && i < 5) {
        bytes /= 1024;
        i++;
    }
    printf "%.2f %s", bytes, suffixes[i];
}')

echo ""
echo "--- Calculation Complete ---"
echo "✅ Files Found:      ${files_found}"
echo "❌ Files Missing:    ${files_missing}"
echo "----------------------------"
echo "📊 Total Size:       ${human_readable_size} (${total_size_bytes} bytes)"
