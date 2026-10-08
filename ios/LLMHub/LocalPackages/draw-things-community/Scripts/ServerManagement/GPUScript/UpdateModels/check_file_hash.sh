#!/bin/bash
# Check SHA256 hash of a file across all GPU servers
# Usage: ./check_file_hash.sh <relative_file_path>
# Example: ./check_file_hash.sh sd15/v1-5-pruned-emaonly.safetensors

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVERS_FILE="$SCRIPT_DIR/gpu_servers.csv"

if [ -z "$1" ]; then
    echo "Usage: $0 <relative_file_path>"
    echo "Example: $0 sd15/v1-5-pruned-emaonly.safetensors"
    exit 1
fi

FILE_PATH="$1"

echo "Checking: $FILE_PATH"
echo "========================================"

# Run all servers in parallel
# CSV format: remote_host, models_path [, nas_url]
while IFS=, read -r server models_path _; do
    # Skip comments and empty lines
    [[ "$server" =~ ^#.*$ || -z "$server" ]] && continue
    # Trim whitespace
    server=$(echo "$server" | xargs)
    models_path=$(echo "$models_path" | xargs)
    hostname="${server#*@}"
    full_path="$models_path/$FILE_PATH"
    (echo "$hostname ($models_path): $(ssh -o ConnectTimeout=10 "$server" "sha256sum \"$full_path\"" 2>&1)") &
done < "$SERVERS_FILE"
wait
