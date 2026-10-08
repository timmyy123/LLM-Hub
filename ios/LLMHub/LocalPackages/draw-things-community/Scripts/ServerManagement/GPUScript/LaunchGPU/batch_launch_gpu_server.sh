#!/bin/bash

# Batch launch GPU servers from a CSV configuration file
# Usage: ./batch_launch_gpu_server.sh <config.csv> [--dry-run]
#
# CSV format (one server per line):
#   remote_host, models_path, utils_path, lora_models_path, [--tmpfs-list <filename>]
#
# If the --tmpfs-list field is omitted, the tmpfs overlay is skipped and
# the models path is mounted directly.
#
# Example config.csv:
#   root@dfw-026-001, /mnt/models/official-models, /root/utils, /mnt/loraModels
#   root@dfw-026-002, /mnt/models/official-models, /root/utils, /mnt/loraModels, --tmpfs-list tmpfs_thpc.ls
#   root@dfw-026-003, /mnt/models/official-models, /root/utils, /mnt/loraModels, --tmpfs-list tmpfs_4090.ls

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_SCRIPT="${SCRIPT_DIR}/launch_gpu_server.sh"
LORA_CLEANUP_TIMER="${LORA_CLEANUP_TIMER:-drawthings-lora-cleanup.timer}"

# Initialize flags
DRY_RUN=false

# Parse arguments
CONFIG_FILE=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *)
            if [ -z "$CONFIG_FILE" ]; then
                CONFIG_FILE="$arg"
            fi
            ;;
    esac
done

# Check if config file is provided
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No config file provided"
    echo ""
    echo "Usage: $0 <config.csv> [--dry-run]"
    echo ""
    echo "Options:"
    echo "  --dry-run    Show what would be executed without actually running"
    echo ""
    echo "CSV format (one server per line):"
    echo "  remote_host, models_path, utils_path, lora_models_path, [--tmpfs-list <filename>]"
    echo ""
    echo "If --tmpfs-list is omitted, the tmpfs overlay is skipped and the models path"
    echo "is mounted directly."
    echo ""
    echo "Example config.csv:"
    echo "  root@dfw-026-001, /mnt/models/official-models, /root/utils, /mnt/loraModels"
    echo "  root@dfw-026-002, /mnt/models/official-models, /root/utils, /mnt/loraModels, --tmpfs-list tmpfs_thpc.ls"
    echo "  root@dfw-026-003, /mnt/models/official-models, /root/utils, /mnt/loraModels, --tmpfs-list tmpfs_4090.ls"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found"
    exit 1
fi

# Check if launch script exists
if [ ! -f "$LAUNCH_SCRIPT" ]; then
    echo "Error: launch_gpu_server.sh not found at $LAUNCH_SCRIPT"
    exit 1
fi

# Extract Docker image from launch script
DOCKER_IMAGE=$(grep -o 'docker pull [^[:space:]]*' "$LAUNCH_SCRIPT" | head -1 | awk '{print $3}')
if [ -z "$DOCKER_IMAGE" ]; then
    DOCKER_IMAGE="(unable to detect)"
fi

echo "=================================================="
echo "  Batch GPU Server Launch"
echo "=================================================="
echo "Config file: $CONFIG_FILE"
echo "Docker image: $DOCKER_IMAGE"
echo "Dry run: $DRY_RUN"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "🔍 DRY RUN MODE - No changes will be made"
    echo ""
fi

# Count total entries (excluding empty lines and comments)
TOTAL=$(grep -v '^#' "$CONFIG_FILE" | grep -v '^[[:space:]]*$' | wc -l | tr -d ' ')
echo "Total servers to launch: $TOTAL"
echo ""

# Process each line in the config file
LINE_NUM=0
SUCCESS_COUNT=0
FAIL_COUNT=0

# Use file descriptor 3 to avoid SSH consuming stdin
while IFS= read -r line <&3 || [ -n "$line" ]; do
    # Skip empty lines and comments
    if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
        continue
    fi

    LINE_NUM=$((LINE_NUM + 1))

    # Parse CSV line (trim whitespace from each field)
    IFS=',' read -ra FIELDS <<< "$line"

    REMOTE_HOST=$(echo "${FIELDS[0]}" | xargs)
    MODELS_PATH=$(echo "${FIELDS[1]}" | xargs)
    UTILS_PATH=$(echo "${FIELDS[2]}" | xargs)
    LORA_MODELS_PATH=$(echo "${FIELDS[3]}" | xargs)
    EXTRA_FLAGS=$(echo "${FIELDS[4]}" | xargs)

    echo "=================================================="
    echo "  [$LINE_NUM/$TOTAL] Launching: $REMOTE_HOST"
    echo "=================================================="
    echo "  Models path: $MODELS_PATH"
    echo "  Utils path: $UTILS_PATH"
    echo "  LoRA models path: $LORA_MODELS_PATH"
    echo "  Extra flags: ${EXTRA_FLAGS:-none}"
    echo ""

    if [ "$DRY_RUN" != true ]; then
        TIMER_STATE=$(ssh "$REMOTE_HOST" "systemctl is-enabled --quiet '$LORA_CLEANUP_TIMER' && systemctl is-active --quiet '$LORA_CLEANUP_TIMER' && echo ok || echo warning" 2>/dev/null || echo warning)
        if [ "$TIMER_STATE" = "ok" ]; then
            echo "LoRA cleanup timer: ok ($LORA_CLEANUP_TIMER)"
        else
            echo "Warning: LoRA cleanup timer is not enabled/active on $REMOTE_HOST ($LORA_CLEANUP_TIMER)"
        fi
        echo ""
    fi

    # Build command
    CMD="$LAUNCH_SCRIPT $REMOTE_HOST $MODELS_PATH $UTILS_PATH $LORA_MODELS_PATH"
    if [ -n "$EXTRA_FLAGS" ]; then
        CMD="$CMD $EXTRA_FLAGS"
    fi

    echo "Command: $CMD"
    echo ""

    # Execute launch script (redirect stdin from /dev/null to prevent SSH from consuming it)
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would execute the above command"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        if $CMD < /dev/null; then
            echo ""
            echo "✅ [$LINE_NUM/$TOTAL] Successfully launched: $REMOTE_HOST"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo ""
            echo "❌ [$LINE_NUM/$TOTAL] Failed to launch: $REMOTE_HOST"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    fi
    echo ""

done 3< "$CONFIG_FILE"

echo "=================================================="
if [ "$DRY_RUN" = true ]; then
    echo "  Batch Dry Run Complete"
else
    echo "  Batch Launch Complete"
fi
echo "=================================================="
echo "  Total: $TOTAL"
echo "  Success: $SUCCESS_COUNT"
echo "  Failed: $FAIL_COUNT"
echo "=================================================="

if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
fi
