#!/bin/bash

# Script to upload files to remote GPU servers
# Usage:
#   Single server:  ./UpdateScripts.sh root@hostname
#   All servers:    ./UpdateScripts.sh --all

# Exit immediately if a command exits with a non-zero status.
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
REMOTE_HOST=""
UPDATE_ALL=false

for arg in "$@"; do
    if [ "$arg" == "--all" ]; then
        UPDATE_ALL=true
    elif [ -z "$REMOTE_HOST" ]; then
        REMOTE_HOST="$arg"
    fi
done

# Function to upload files to a single server
upload_to_server() {
    local HOST="$1"

    echo "Target: $HOST"
    echo ""

    # Check if ssh is available
    if ! command -v ssh &> /dev/null; then
        echo "Error: ssh command not found. Please install openssh-client."
        return 1
    fi

    # Test SSH connection
    echo "Testing SSH connection to $HOST..."
    if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOST" "echo 'Connection successful'" 2>/dev/null < /dev/null; then
        echo "Error: Cannot connect to $HOST"
        echo "Please ensure:"
        echo "  1. The host is reachable"
        echo "  2. SSH keys are set up for passwordless authentication"
        echo "  3. You have root access"
        return 1
    fi
    echo "✅ SSH connection successful"
    echo ""

    # Remote directory
    REMOTE_DIR="/root/utils/"
    echo "Creating remote directory: $REMOTE_DIR"
    ssh "$HOST" "mkdir -p $REMOTE_DIR" < /dev/null

    # Read file list and copy files
    FILE_LIST="$SCRIPT_DIR/files_to_copy.txt"

    if [ ! -f "$FILE_LIST" ]; then
        echo "Error: File list not found at $FILE_LIST"
        echo "Please create files_to_copy.txt with the list of files to upload."
        return 1
    fi

    echo "Copying files from list: $FILE_LIST"
    while IFS= read -r file || [ -n "$file" ]; do
        # Skip empty lines and comments
        [[ -z "$file" || "$file" =~ ^[[:space:]]*# ]] && continue

        SOURCE_FILE="$SCRIPT_DIR/$file"
        if [ -f "$SOURCE_FILE" ]; then
            echo "  Removing old & copying: $file"
            ssh "$HOST" "rm -rf '$REMOTE_DIR/$file'" < /dev/null
            scp -q "$SOURCE_FILE" "$HOST:$REMOTE_DIR/"
            # Make .sh files executable
            if [[ "$file" == *.sh ]]; then
                ssh "$HOST" "chmod +x '$REMOTE_DIR/$file'" < /dev/null
            fi
        elif [ -d "$SOURCE_FILE" ]; then
            echo "  Removing old & copying directory: $file"
            ssh "$HOST" "rm -rf '$REMOTE_DIR/$file'" < /dev/null
            scp -rq "$SOURCE_FILE" "$HOST:$REMOTE_DIR/"
            # Make all .sh files in directory executable
            ssh "$HOST" "find '$REMOTE_DIR/$file' -name '*.sh' -exec chmod +x {} \;" < /dev/null
        else
            echo "  Warning: File not found: $file (skipping)"
        fi
    done < "$FILE_LIST"

    echo "✅ All files copied to $HOST:$REMOTE_DIR"
    return 0
}

# Handle --all flag: process all servers from gpu_servers.txt
if [ "$UPDATE_ALL" = true ]; then
    GPU_SERVERS_FILE="$SCRIPT_DIR/gpu_servers.txt"

    if [ ! -f "$GPU_SERVERS_FILE" ]; then
        echo "Error: gpu_servers.txt not found at $GPU_SERVERS_FILE"
        echo ""
        echo "Please create gpu_servers.txt with one server per line (format: user@hostname):"
        echo "  root@server1"
        echo "  root@server2"
        echo "  # comments are supported"
        exit 1
    fi

    echo "=================================================="
    echo "  Updating Scripts on All GPU Servers"
    echo "=================================================="
    echo "Reading servers from: $GPU_SERVERS_FILE"
    echo ""

    # Read server list and process each one
    SERVER_COUNT=0
    SUCCESS_COUNT=0
    FAILED_SERVERS=()

    while IFS= read -r line || [ -n "$line" ]; do
        # Skip empty lines and comments
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue

        # Hostname is the entire line (format: user@hostname)
        SERVER_HOST="$line"

        # Validate format
        if [[ ! "$SERVER_HOST" =~ @.+ ]]; then
            echo "⚠️  Skipping invalid line (expected user@hostname): $line"
            continue
        fi

        SERVER_COUNT=$((SERVER_COUNT + 1))

        echo "=================================================="
        echo "  Server $SERVER_COUNT: $SERVER_HOST"
        echo "=================================================="

        if upload_to_server "$SERVER_HOST"; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            echo "✅ Success: $SERVER_HOST"
        else
            FAILED_SERVERS+=("$SERVER_HOST")
            echo "❌ Failed: $SERVER_HOST"
        fi

        echo ""
    done < "$GPU_SERVERS_FILE"

    # Summary
    echo "=================================================="
    echo "  Summary"
    echo "=================================================="
    echo "Total servers: $SERVER_COUNT"
    echo "Successful: $SUCCESS_COUNT"
    echo "Failed: ${#FAILED_SERVERS[@]}"

    if [ ${#FAILED_SERVERS[@]} -gt 0 ]; then
        echo ""
        echo "Failed servers:"
        for server in "${FAILED_SERVERS[@]}"; do
            echo "  - $server"
        done
        exit 1
    fi

    echo ""
    echo "✅ All servers updated successfully!"
    exit 0
fi

# Single server mode
if [ -z "$REMOTE_HOST" ]; then
    echo "Usage:"
    echo "  Single server:  $0 root@hostname"
    echo "  All servers:    $0 --all"
    echo ""
    echo "The script will upload files listed in files_to_copy.txt"
    echo "to /root/utils/ on the remote server(s)."
    exit 1
fi

echo "=================================================="
echo "  Upload Files to Remote Server"
echo "=================================================="

if upload_to_server "$REMOTE_HOST"; then
    echo ""
    echo "=================================================="
    echo "✅ Upload complete!"
    echo ""
    echo "Files are located at: $REMOTE_HOST:/root/utils/"
    exit 0
else
    echo ""
    echo "=================================================="
    echo "❌ Upload failed!"
    exit 1
fi
