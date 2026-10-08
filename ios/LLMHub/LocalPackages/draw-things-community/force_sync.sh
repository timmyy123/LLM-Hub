#!/bin/bash

# File containing the list of directories
DIR_LIST_FILE="SYNCLIST"

# Source repository path
SOURCE_REPO="../draw-things"

# Destination path where directories will be copied
DESTINATION_PATH="./"

# Check if the directory list file exists
if [[ ! -f "$DIR_LIST_FILE" ]]; then
    echo "Directory list file '$DIR_LIST_FILE' not found!"
    exit 1
fi

# Read directories from the file and process them
while IFS= read -r dir; do
    # Skip empty lines
    if [[ -z "$dir" ]]; then
        continue
    fi

    # Remove the directory if it exists
    rm -rf "$DESTINATION_PATH/$dir"
    echo "Removed directory: $DESTINATION_PATH/$dir"

    # Copy the directory from the source repo
    cp -r "$SOURCE_REPO/$dir" "$DESTINATION_PATH/$dir"
    echo "Copied directory: $dir from $SOURCE_REPO to $DESTINATION_PATH"

done < "$DIR_LIST_FILE"

echo "Process completed."
