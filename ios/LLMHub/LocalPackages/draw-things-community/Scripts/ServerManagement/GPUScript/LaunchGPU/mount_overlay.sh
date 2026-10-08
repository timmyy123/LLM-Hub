#!/bin/bash

# ==============================================================================
#  setup_overlay.sh (Corrected Version)
#
#  This version fixes the "wrong fs type" error by ensuring that
#  'upperdir' and 'workdir' are created on the same tmpfs filesystem.
# ==============================================================================

# --- Script Configuration ---
set -euo pipefail

# --- Global Variables (defaults, can be overridden by arguments) ---
SOURCE_DIR=""
LOWER_DIR=""
FILE_LIST=""
MERGED_DIR="/mnt/unified_view"
TMPFS_SIZE="512M"

# Internal paths are derived from the MERGED_DIR
TMPFS_PARENT="" # V2 CHANGE: This will be the single mountpoint for our tmpfs.
UPPER_DIR=""
WORK_DIR=""

# --- Functions ---

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] - INFO - $1"
}

error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] - ERROR - $1" >&2
}

usage() {
    echo "Usage: $0 -s <source_dir> -l <lower_dir> -f <file_list> -m <merged_dir> [-z <tmpfs_size>]"
    # ... (usage text remains the same)
    exit 1
}

cleanup() {
    log "Starting cleanup process..."
    set +e

    if mountpoint -q "$MERGED_DIR"; then
        log "Unmounting overlay at '$MERGED_DIR'..."
        umount "$MERGED_DIR"
    fi

    # V2 CHANGE: Unmount the parent tmpfs directory
    if mountpoint -q "$TMPFS_PARENT"; then
        log "Unmounting tmpfs at '$TMPFS_PARENT'..."
        umount "$TMPFS_PARENT"
    fi

    rmdir "$MERGED_DIR" 2>/dev/null
    # V2 CHANGE: Remove the parent tmpfs directory
    rmdir "$TMPFS_PARENT" 2>/dev/null

    log "Cleanup finished."
}

# --- Main Script ---

# (Argument parsing remains the same as before)
while getopts "s:l:f:m:z:h" opt; do
    case ${opt} in
        s) SOURCE_DIR=${OPTARG} ;;
        l) LOWER_DIR=${OPTARG} ;;
        f) FILE_LIST=${OPTARG} ;;
        m) MERGED_DIR=${OPTARG} ;;
        z) TMPFS_SIZE=${OPTARG} ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [[ -z "$SOURCE_DIR" || -z "$LOWER_DIR" || -z "$FILE_LIST" || -z "$MERGED_DIR" ]]; then
    error "Missing required arguments."
    usage
fi

# V2 CHANGE: Define a single parent staging directory for tmpfs.
# The upper and work directories will be created inside this.
PARENT_DIR=$(dirname "$MERGED_DIR")
MERGED_NAME=$(basename "$MERGED_DIR")
TMPFS_PARENT="${PARENT_DIR}/${MERGED_NAME}_staging"
UPPER_DIR="${TMPFS_PARENT}/upper"
WORK_DIR="${TMPFS_PARENT}/work"

# --- Pre-flight Checks (remain the same) ---
log "Starting OverlayFS setup script."
if [[ $EUID -ne 0 ]]; then
   error "This script must be run as root to perform mount operations."
   exit 1
fi
# ... (other checks remain the same)

# Resolve paths
SOURCE_DIR=$(readlink -f "$SOURCE_DIR")
LOWER_DIR=$(readlink -f "$LOWER_DIR")
FILE_LIST=$(readlink -f "$FILE_LIST")

# --- Execution ---

trap cleanup EXIT INT TERM

log "Step 1: Cleaning up any mounts from a previous run."
cleanup
trap cleanup EXIT INT TERM

log "Step 2: Creating required directories."
# V2 CHANGE: We only create the merged dir and the parent tmpfs dir initially.
mkdir -p "$MERGED_DIR" "$TMPFS_PARENT"
log "Created: '$MERGED_DIR', '$TMPFS_PARENT'"

log "Step 3: Mounting a single parent tmpfs."
mount -t tmpfs -o "size=${TMPFS_SIZE}" tmpfs "$TMPFS_PARENT"
log "Mounted tmpfs of size ${TMPFS_SIZE} at '$TMPFS_PARENT'."

# V2 CHANGE: Now create upper and work dirs *inside* the mounted tmpfs.
# This guarantees they are on the same filesystem.
mkdir -p "$UPPER_DIR" "$WORK_DIR"
log "Created upper and work directories inside tmpfs."

log "Step 4: Copying priority files from '$SOURCE_DIR' to upperdir."
pushd "$SOURCE_DIR" > /dev/null
copied_count=0
while IFS= read -r file_path || [[ -n "$file_path" ]]; do
    if [[ -z "${file_path// }" ]]; then continue; fi
    if [[ -e "$file_path" ]]; then
        cp -a --parents "$file_path" "$UPPER_DIR" # Copy to the new UPPER_DIR
        log "  -> Copied: $file_path"
        ((copied_count++))
    else
        log "  -> WARNING: File not found, skipping: '$file_path'"
    fi
done < "$FILE_LIST"
popd > /dev/null
log "Copied a total of $copied_count files."

log "Step 5: Mounting OverlayFS."
mount -t overlay overlay -o "lowerdir=${LOWER_DIR},upperdir=${UPPER_DIR},workdir=${WORK_DIR}" "$MERGED_DIR"
log "✅ Successfully mounted OverlayFS at '$MERGED_DIR'."
echo "--------------------------------------------------"
echo "  Lower Dir (base)   : ${LOWER_DIR}"
echo "  Upper Dir (tmpfs)  : ${UPPER_DIR}"
echo "  Unified View       : ${MERGED_DIR}"
echo "--------------------------------------------------"

trap - EXIT
exit 0
