#!/bin/bash

# Clean up remote-server LoRA cache files by configured usage ratio.
#
# Intended use:
#   a systemd timer on each GPU server runs this script once per day.
#
# Safety defaults:
#   - dry-run unless --execute is passed
#   - only direct child files with a 64-char SHA-256 basename are candidates
#   - files accessed within the last 2 days are ignored
#   - files currently open by any process are ignored

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_GPU_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LORA_PATH="/mnt/loraModels"
TARGET_SIZE="5T"
TRIGGER_PERCENT=90
TARGET_PERCENT=50
IGNORE_ACCESS_DAYS=2
EXECUTE=false
PROGRESS_INTERVAL="${LORA_CLEANUP_PROGRESS_INTERVAL:-1000}"
VERBOSE=false
DD_SITE="${DD_SITE:-us5.datadoghq.com}"
DD_API_KEY="${DD_API_KEY:-f395315025e2f0577c31e8b7fa5c7381}"
HOSTNAME_VALUE="${HOSTNAME:-$(hostname 2>/dev/null || printf 'unknown')}"

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --path <dir>              LoRA directory. Default: /mnt/loraModels"
    echo "  --target-size <size>      Quota denominator, e.g. 2T, 1500G, or auto. Default: 5T."
    echo "  --trigger-percent <n>     Cleanup trigger percentage. Default: 90."
    echo "  --target-percent <n>      Cleanup target percentage. Default: 50."
    echo "  --ignore-access-days <n>  Ignore files accessed within n days. Default: 2."
    echo "  --progress-interval <n>   Print scan/delete progress every n files. Default: 1000."
    echo "  --verbose                 Print each selected/deleted file."
    echo "  --execute                 Delete files. Without this, dry-run only."
    echo "  --dry-run                 Force dry-run mode."
    echo "  --datadog-api-key <key>   Datadog API key. Defaults to DD_API_KEY/DATADOG_API_KEY."
    echo "  --dd-site <site>          Datadog site. Default: us5.datadoghq.com."
    echo "  -h, --help                Show this help."
}

trim() {
    local value="$*"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

format_bytes() {
    local bytes="${1:-0}"
    awk -v bytes="$bytes" '
      BEGIN {
        split("B KiB MiB GiB TiB PiB", units, " ")
        value = bytes + 0
        unit = 1
        while (value >= 1024 && unit < 6) {
          value = value / 1024
          unit++
        }
        printf "%.2f %s", value, units[unit]
      }'
}

json_escape() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    value="${value//$'\n'/\\n}"
    value="${value//$'\r'/}"
    printf '%s' "$value"
}

parse_size_to_bytes() {
    local raw
    local raw_lower
    raw="$(trim "$1")"
    raw_lower="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')"
    if [ "$raw_lower" = "auto" ]; then
        printf 'auto'
        return 0
    fi
    awk -v raw="$raw" '
      BEGIN {
        value = raw
        gsub(/^[ \t]+|[ \t]+$/, "", value)
        upper = toupper(value)
        if (upper !~ /^[0-9]+([.][0-9]+)?[ \t]*([KMGTPE]?I?B?)?$/) {
          exit 1
        }
        number = upper
        sub(/[ \t]*([KMGTPE]?I?B?)$/, "", number)
        suffix = upper
        sub(/^[0-9]+([.][0-9]+)?[ \t]*/, "", suffix)
        multiplier = 1
        if (suffix ~ /^K/) multiplier = 1024
        else if (suffix ~ /^M/) multiplier = 1024 * 1024
        else if (suffix ~ /^G/) multiplier = 1024 * 1024 * 1024
        else if (suffix ~ /^T/) multiplier = 1024 * 1024 * 1024 * 1024
        else if (suffix ~ /^P/) multiplier = 1024 * 1024 * 1024 * 1024 * 1024
        else if (suffix ~ /^E/) multiplier = 1024 * 1024 * 1024 * 1024 * 1024 * 1024
        printf "%.0f", number * multiplier
      }'
}

du_bytes() {
    local path="$1"
    local output
    if output=$(du -sb "$path" 2>/dev/null); then
        printf '%s' "$output" | awk '{ print $1 }'
        return 0
    fi
    if output=$(du -sk "$path" 2>/dev/null); then
        printf '%s' "$output" | awk '{ print $1 * 1024 }'
        return 0
    fi
    return 1
}

df_capacity_bytes() {
    local path="$1"
    df -PB1 "$path" | awk 'NR == 2 { print $2 }'
}

percent_int() {
    local numerator="$1"
    local denominator="$2"
    awk -v n="$numerator" -v d="$denominator" 'BEGIN { if (d <= 0) print 0; else printf "%.0f", (n / d) * 100 }'
}

ratio_float() {
    local numerator="$1"
    local denominator="$2"
    awk -v n="$numerator" -v d="$denominator" 'BEGIN { if (d <= 0) print "0"; else printf "%.6f", n / d }'
}

bytes_for_percent() {
    local denominator="$1"
    local percent="$2"
    awk -v d="$denominator" -v p="$percent" 'BEGIN { printf "%.0f", d * p / 100 }'
}

is_positive_integer() {
    [[ "${1:-}" =~ ^[0-9]+$ ]] && [ "$1" -gt 0 ]
}

is_percent() {
    [[ "${1:-}" =~ ^[0-9]+$ ]] && [ "$1" -ge 0 ] && [ "$1" -le 100 ]
}

same_file_identity() {
    local file="$1"
    local expected_device="$2"
    local expected_inode="$3"
    local expected_size="$4"
    local stat_output
    stat_output=$(stat -Lc '%d	%i	%s' -- "$file" 2>/dev/null) || return 1
    IFS="$(printf '\t')" read -r device inode size <<< "$stat_output"
    [ "$device" = "$expected_device" ] && [ "$inode" = "$expected_inode" ] && [ "$size" = "$expected_size" ]
}

write_open_file_identity_set() {
    local output_file="$1"
    local fd
    local fd_target
    local tmp_file
    tmp_file="${output_file}.tmp"
    : > "$tmp_file"
    for fd in /proc/[0-9]*/fd/*; do
        [ -e "$fd" ] || continue
        fd_target=$(stat -Lc '%d:%i' -- "$fd" 2>/dev/null) || continue
        printf '%s\n' "$fd_target" >> "$tmp_file"
    done
    sort -u "$tmp_file" > "$output_file"
    rm -f "$tmp_file"
}

extract_datadog_key_from_start_script() {
    local start_script="${LAUNCH_GPU_DIR}/start_single_grpc.sh"
    [ -f "$start_script" ] || return 1
    awk '
      $1 == "-d" && $2 != "" {
        gsub(/\\$/, "", $2)
        gsub(/'\''|"/, "", $2)
        print $2
        exit
      }
    ' "$start_script"
}

datadog_post() {
    local status="$1"
    local initial_bytes="$2"
    local final_bytes="$3"
    local target_bytes="$4"
    local deleted_files="$5"
    local deleted_bytes="$6"
    local skipped_recent="$7"
    local skipped_open="$8"
    local initial_percent="$9"
    local final_percent="${10}"
    local metric_ratio="${11}"

    if [ -z "$DD_API_KEY" ]; then
        echo "Datadog: DD_API_KEY/DATADOG_API_KEY not set; skipping event and metric submission."
        return 0
    fi
    if ! command -v curl >/dev/null 2>&1; then
        echo "Datadog: curl not found; skipping event and metric submission."
        return 0
    fi

    local escaped_title
    local escaped_text
    local event_payload
    local metric_payload
    local now
    local text
    now=$(date +%s)
    text="Status: $status
Path: $LORA_PATH
Initial usage: ${initial_percent}% ($(format_bytes "$initial_bytes") / $(format_bytes "$target_bytes"))
Final usage: ${final_percent}% ($(format_bytes "$final_bytes") / $(format_bytes "$target_bytes"))
Deleted files: $deleted_files
Deleted bytes: $(format_bytes "$deleted_bytes")
Skipped recently used files: $skipped_recent
Skipped open files: $skipped_open"

    escaped_title=$(json_escape "Draw Things LoRA cleanup")
    escaped_text=$(json_escape "$text")
    event_payload="{\"title\":\"$escaped_title\",\"text\":\"$escaped_text\",\"alert_type\":\"info\",\"host\":\"$(json_escape "$HOSTNAME_VALUE")\",\"tags\":[\"usage_percent:${final_percent}\"]}"
    metric_payload="{\"series\":[{\"metric\":\"drawthings.lora.usage_ratio\",\"points\":[[$now,$metric_ratio]],\"type\":\"gauge\",\"host\":\"$(json_escape "$HOSTNAME_VALUE")\"}]}"

    if ! curl -fsS -X POST \
        -H "Content-Type: application/json" \
        -H "DD-API-KEY: $DD_API_KEY" \
        "https://api.${DD_SITE}/api/v1/events" \
        -d "$event_payload" >/dev/null; then
        echo "Datadog: failed to submit cleanup event."
    fi

    if ! curl -fsS -X POST \
        -H "Content-Type: application/json" \
        -H "DD-API-KEY: $DD_API_KEY" \
        "https://api.${DD_SITE}/api/v1/series" \
        -d "$metric_payload" >/dev/null; then
        echo "Datadog: failed to submit usage ratio metric."
    fi
}

while [ $# -gt 0 ]; do
    case "$1" in
        --path)
            [ $# -ge 2 ] || { echo "Error: --path requires a value"; exit 1; }
            LORA_PATH="$2"
            shift 2
            ;;
        --target-size)
            [ $# -ge 2 ] || { echo "Error: --target-size requires a value"; exit 1; }
            TARGET_SIZE="$2"
            shift 2
            ;;
        --trigger-percent)
            [ $# -ge 2 ] || { echo "Error: --trigger-percent requires a value"; exit 1; }
            TRIGGER_PERCENT="$2"
            shift 2
            ;;
        --target-percent)
            [ $# -ge 2 ] || { echo "Error: --target-percent requires a value"; exit 1; }
            TARGET_PERCENT="$2"
            shift 2
            ;;
        --ignore-access-days)
            [ $# -ge 2 ] || { echo "Error: --ignore-access-days requires a value"; exit 1; }
            IGNORE_ACCESS_DAYS="$2"
            shift 2
            ;;
        --progress-interval)
            [ $# -ge 2 ] || { echo "Error: --progress-interval requires a value"; exit 1; }
            PROGRESS_INTERVAL="$2"
            shift 2
            ;;
        --execute)
            EXECUTE=true
            shift
            ;;
        --dry-run)
            EXECUTE=false
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --datadog-api-key)
            [ $# -ge 2 ] || { echo "Error: --datadog-api-key requires a value"; exit 1; }
            DD_API_KEY="$2"
            shift 2
            ;;
        --dd-site)
            [ $# -ge 2 ] || { echo "Error: --dd-site requires a value"; exit 1; }
            DD_SITE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if [ ! -d "$LORA_PATH" ]; then
    echo "Error: LoRA directory does not exist: $LORA_PATH"
    exit 1
fi

if ! is_percent "$TRIGGER_PERCENT"; then
    echo "Error: --trigger-percent must be an integer between 0 and 100"
    exit 1
fi
if ! is_percent "$TARGET_PERCENT"; then
    echo "Error: --target-percent must be an integer between 0 and 100"
    exit 1
fi
if [ "$TARGET_PERCENT" -ge "$TRIGGER_PERCENT" ]; then
    echo "Error: --target-percent must be lower than --trigger-percent"
    exit 1
fi
if ! is_positive_integer "$IGNORE_ACCESS_DAYS"; then
    echo "Error: --ignore-access-days must be a positive integer"
    exit 1
fi
if ! is_positive_integer "$PROGRESS_INTERVAL"; then
    echo "Error: --progress-interval must be a positive integer"
    exit 1
fi

if [ -z "$DD_API_KEY" ]; then
    DD_API_KEY="$(extract_datadog_key_from_start_script || true)"
fi

TARGET_SIZE_BYTES=$(parse_size_to_bytes "$TARGET_SIZE") || {
    echo "Error: invalid --target-size value: $TARGET_SIZE"
    exit 1
}
if [ "$TARGET_SIZE_BYTES" = "auto" ]; then
    TARGET_SIZE_BYTES=$(df_capacity_bytes "$LORA_PATH")
fi
if ! is_positive_integer "$TARGET_SIZE_BYTES"; then
    echo "Error: failed to resolve target size for $LORA_PATH"
    exit 1
fi

INITIAL_BYTES=$(du_bytes "$LORA_PATH") || {
    echo "Error: failed to compute directory size: $LORA_PATH"
    exit 1
}
INITIAL_PERCENT=$(percent_int "$INITIAL_BYTES" "$TARGET_SIZE_BYTES")
TARGET_BYTES=$(bytes_for_percent "$TARGET_SIZE_BYTES" "$TARGET_PERCENT")
TRIGGER_BYTES=$(bytes_for_percent "$TARGET_SIZE_BYTES" "$TRIGGER_PERCENT")

echo "=================================================="
echo "  Draw Things LoRA Cleanup"
echo "=================================================="
echo "Host:             $HOSTNAME_VALUE"
echo "Path:             $LORA_PATH"
echo "Target size:      $(format_bytes "$TARGET_SIZE_BYTES")"
echo "Initial size:     $(format_bytes "$INITIAL_BYTES")"
echo "Initial usage:    ${INITIAL_PERCENT}%"
echo "Trigger:          ${TRIGGER_PERCENT}% ($(format_bytes "$TRIGGER_BYTES"))"
echo "Target:           ${TARGET_PERCENT}% ($(format_bytes "$TARGET_BYTES"))"
echo "Ignore accessed:  last ${IGNORE_ACCESS_DAYS} day(s)"
echo "Progress every:   ${PROGRESS_INTERVAL} file(s)"
echo "Mode:             $([ "$EXECUTE" = true ] && echo "EXECUTE" || echo "DRY RUN")"
echo "=================================================="

STATUS="noop"
DELETED_FILES=0
DELETED_BYTES=0
SKIPPED_RECENT=0
SKIPPED_OPEN=0

if [ "$INITIAL_BYTES" -le "$TRIGGER_BYTES" ]; then
    echo "Usage is at or below trigger; no cleanup needed."
    FINAL_BYTES=$(du_bytes "$LORA_PATH" || printf '%s' "$INITIAL_BYTES")
    FINAL_PERCENT=$(percent_int "$FINAL_BYTES" "$TARGET_SIZE_BYTES")
    FINAL_RATIO=$(ratio_float "$FINAL_BYTES" "$TARGET_SIZE_BYTES")
    datadog_post "$STATUS" "$INITIAL_BYTES" "$FINAL_BYTES" "$TARGET_SIZE_BYTES" \
        "$DELETED_FILES" "$DELETED_BYTES" "$SKIPPED_RECENT" "$SKIPPED_OPEN" \
        "$INITIAL_PERCENT" "$FINAL_PERCENT" "$FINAL_RATIO"
    exit 0
fi

NEEDED_BYTES=$((INITIAL_BYTES - TARGET_BYTES))
if [ "$NEEDED_BYTES" -le 0 ]; then
    echo "Computed cleanup need is <= 0; no cleanup needed."
    FINAL_BYTES=$(du_bytes "$LORA_PATH" || printf '%s' "$INITIAL_BYTES")
    FINAL_PERCENT=$(percent_int "$FINAL_BYTES" "$TARGET_SIZE_BYTES")
    FINAL_RATIO=$(ratio_float "$FINAL_BYTES" "$TARGET_SIZE_BYTES")
    datadog_post "$STATUS" "$INITIAL_BYTES" "$FINAL_BYTES" "$TARGET_SIZE_BYTES" \
        "$DELETED_FILES" "$DELETED_BYTES" "$SKIPPED_RECENT" "$SKIPPED_OPEN" \
        "$INITIAL_PERCENT" "$FINAL_PERCENT" "$FINAL_RATIO"
    exit 0
fi

echo "Cleanup required: need to free about $(format_bytes "$NEEDED_BYTES")."

CANDIDATES_FILE=$(mktemp "${TMPDIR:-/tmp}/draw-things-lora-candidates.XXXXXX")
SHUFFLED_FILE=$(mktemp "${TMPDIR:-/tmp}/draw-things-lora-shuffled.XXXXXX")
FILTERED_FILE=$(mktemp "${TMPDIR:-/tmp}/draw-things-lora-filtered.XXXXXX")
OPEN_IDENTITIES_FILE=$(mktemp "${TMPDIR:-/tmp}/draw-things-open-files.XXXXXX")
SKIPPED_OPEN_FILE=$(mktemp "${TMPDIR:-/tmp}/draw-things-skipped-open.XXXXXX")
trap 'rm -f "$CANDIDATES_FILE" "$SHUFFLED_FILE" "$FILTERED_FILE" "$OPEN_IDENTITIES_FILE" "$SKIPPED_OPEN_FILE"' EXIT

NOW=$(date +%s)
IGNORE_SECONDS=$((IGNORE_ACCESS_DAYS * 24 * 60 * 60))
SCAN_STARTED=$NOW
SCANNED_FILES=0
ELIGIBLE_FILES=0

echo "Scanning LoRA candidates. This can take a while on multi-TiB directories."

while IFS="$(printf '\t')" read -r -d '' device inode size atime file; do
    SCANNED_FILES=$((SCANNED_FILES + 1))
    [ "$size" -gt 0 ] || continue
    atime="${atime%%.*}"
    if [ $((NOW - atime)) -lt "$IGNORE_SECONDS" ]; then
        SKIPPED_RECENT=$((SKIPPED_RECENT + 1))
        if [ $((SCANNED_FILES % PROGRESS_INTERVAL)) -eq 0 ]; then
            elapsed=$(( $(date +%s) - SCAN_STARTED ))
            echo "scan progress: processed $SCANNED_FILES matching files, eligible $ELIGIBLE_FILES, skipped recent $SKIPPED_RECENT (${elapsed}s)"
        fi
        continue
    fi
    printf '%s\t%s\t%s\t%s\n' "$device" "$inode" "$size" "$file" >> "$CANDIDATES_FILE"
    ELIGIBLE_FILES=$((ELIGIBLE_FILES + 1))
    if [ $((SCANNED_FILES % PROGRESS_INTERVAL)) -eq 0 ]; then
        elapsed=$(( $(date +%s) - SCAN_STARTED ))
        echo "scan progress: processed $SCANNED_FILES matching files, eligible $ELIGIBLE_FILES, skipped recent $SKIPPED_RECENT (${elapsed}s)"
    fi
done < <(find "$LORA_PATH" -mindepth 1 -maxdepth 1 -type f ! -name '.*' -regextype posix-extended -regex '.*/[0-9A-Fa-f]{64}' -printf '%D\t%i\t%s\t%A@\t%p\0')

SCAN_ELAPSED=$(( $(date +%s) - SCAN_STARTED ))
echo "Scan complete: processed $SCANNED_FILES matching files in ${SCAN_ELAPSED}s."

CANDIDATE_COUNT=$(wc -l < "$CANDIDATES_FILE" | tr -d ' ')
echo "Eligible candidates: $CANDIDATE_COUNT"
echo "Skipped recently used candidates: $SKIPPED_RECENT"

if [ "$CANDIDATE_COUNT" -eq 0 ]; then
    echo "No eligible LoRA files to delete."
    STATUS="partial"
    FINAL_BYTES=$(du_bytes "$LORA_PATH" || printf '%s' "$INITIAL_BYTES")
    FINAL_PERCENT=$(percent_int "$FINAL_BYTES" "$TARGET_SIZE_BYTES")
    FINAL_RATIO=$(ratio_float "$FINAL_BYTES" "$TARGET_SIZE_BYTES")
    datadog_post "$STATUS" "$INITIAL_BYTES" "$FINAL_BYTES" "$TARGET_SIZE_BYTES" \
        "$DELETED_FILES" "$DELETED_BYTES" "$SKIPPED_RECENT" "$SKIPPED_OPEN" \
        "$INITIAL_PERCENT" "$FINAL_PERCENT" "$FINAL_RATIO"
    exit 0
fi

if command -v shuf >/dev/null 2>&1; then
    shuf "$CANDIDATES_FILE" > "$SHUFFLED_FILE"
else
    awk 'BEGIN { srand() } { print rand() "\t" $0 }' "$CANDIDATES_FILE" | sort -n | cut -f2- > "$SHUFFLED_FILE"
fi

DELETE_INPUT_FILE="$SHUFFLED_FILE"
if [ "$EXECUTE" = true ]; then
    echo "Snapshotting open file descriptors before deletion..."
    OPEN_SCAN_STARTED=$(date +%s)
    write_open_file_identity_set "$OPEN_IDENTITIES_FILE"
    OPEN_SCAN_ELAPSED=$(( $(date +%s) - OPEN_SCAN_STARTED ))
    OPEN_IDENTITY_COUNT=$(wc -l < "$OPEN_IDENTITIES_FILE" | tr -d ' ')
    echo "Open-file snapshot complete: $OPEN_IDENTITY_COUNT open file identity/identities (${OPEN_SCAN_ELAPSED}s)."
    echo "Filtering currently open LoRA candidates before deletion..."
    awk -F '\t' -v skipped_file="$SKIPPED_OPEN_FILE" '
      NR == FNR { open[$1] = 1; next }
      {
        key = $1 ":" $2
        if (key in open) {
          skipped++
        } else {
          print
        }
      }
      END { print skipped + 0 > skipped_file }
    ' "$OPEN_IDENTITIES_FILE" "$SHUFFLED_FILE" > "$FILTERED_FILE"
    SKIPPED_OPEN=$(cat "$SKIPPED_OPEN_FILE")
    echo "Skipped open candidates: $SKIPPED_OPEN"
    DELETE_INPUT_FILE="$FILTERED_FILE"
fi

DELETE_STARTED=$(date +%s)
PROCESSED_DELETE_CANDIDATES=0

while IFS="$(printf '\t')" read -r device inode size file; do
    [ -n "$file" ] || continue
    PROCESSED_DELETE_CANDIDATES=$((PROCESSED_DELETE_CANDIDATES + 1))
    if [ $((PROCESSED_DELETE_CANDIDATES % PROGRESS_INTERVAL)) -eq 0 ]; then
        elapsed=$(( $(date +%s) - DELETE_STARTED ))
        echo "delete progress: processed $PROCESSED_DELETE_CANDIDATES candidates, selected $DELETED_FILES files, freed/would-free $(format_bytes "$DELETED_BYTES") (${elapsed}s)"
    fi
    if [ "$DELETED_BYTES" -ge "$NEEDED_BYTES" ]; then
        break
    fi
    if [ "$EXECUTE" = true ]; then
        if ! same_file_identity "$file" "$device" "$inode" "$size"; then
            [ "$VERBOSE" = true ] && echo "skip changed: $file"
            continue
        fi
        if rm -f -- "$file"; then
            DELETED_FILES=$((DELETED_FILES + 1))
            DELETED_BYTES=$((DELETED_BYTES + size))
            [ "$VERBOSE" = true ] && echo "deleted $(format_bytes "$size"): $file"
        else
            echo "failed to delete: $file"
        fi
    else
        DELETED_FILES=$((DELETED_FILES + 1))
        DELETED_BYTES=$((DELETED_BYTES + size))
        [ "$VERBOSE" = true ] && echo "would delete $(format_bytes "$size"): $file"
    fi
done < "$DELETE_INPUT_FILE"

FINAL_BYTES=$(du_bytes "$LORA_PATH" || printf '%s' "$((INITIAL_BYTES - DELETED_BYTES))")
FINAL_PERCENT=$(percent_int "$FINAL_BYTES" "$TARGET_SIZE_BYTES")
FINAL_RATIO=$(ratio_float "$FINAL_BYTES" "$TARGET_SIZE_BYTES")

if [ "$FINAL_BYTES" -le "$TARGET_BYTES" ] || [ "$EXECUTE" != true ] && [ "$DELETED_BYTES" -ge "$NEEDED_BYTES" ]; then
    STATUS="deleted"
else
    STATUS="partial"
fi

echo "=================================================="
echo "  Cleanup Summary"
echo "=================================================="
echo "Status:           $STATUS"
if [ "$EXECUTE" = true ]; then
    echo "Deleted files:    $DELETED_FILES"
    echo "Deleted bytes:    $(format_bytes "$DELETED_BYTES")"
else
    echo "Would delete:     $DELETED_FILES file(s)"
    echo "Would free:       $(format_bytes "$DELETED_BYTES")"
fi
echo "Skipped open:     $SKIPPED_OPEN"
echo "Final size:       $(format_bytes "$FINAL_BYTES")"
echo "Final usage:      ${FINAL_PERCENT}%"
echo "=================================================="

datadog_post "$STATUS" "$INITIAL_BYTES" "$FINAL_BYTES" "$TARGET_SIZE_BYTES" \
    "$DELETED_FILES" "$DELETED_BYTES" "$SKIPPED_RECENT" "$SKIPPED_OPEN" \
    "$INITIAL_PERCENT" "$FINAL_PERCENT" "$FINAL_RATIO"

exit 0
