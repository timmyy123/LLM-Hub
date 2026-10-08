#!/bin/bash

# Clean up least-used LoRA model files on GPU servers.
#
# Safe default:
#   This script runs in dry-run mode unless --execute is passed.
#
# Usage:
#   ./cleanup_lora_models_before.sh
#   ./cleanup_lora_models_before.sh --limit 1000 --execute
#   ./cleanup_lora_models_before.sh --config config_lora.csv --before "Jan 1 2026" --dry-run
#
# CSV formats supported:
#   remote_host, lora_models_path
#   remote_host, models_path, utils_path, lora_models_path, [extra flags]

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config_lora.csv"
CUTOFF_DATE=""
LIMIT_COUNT=1000
CLEANUP_MODE="count"
EXECUTE=false
MAX_DEPTH=1
TIME_FIELD="access"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-10}"
LOG_DIR="${SCRIPT_DIR}/logs"

usage() {
    echo "Usage:"
    echo "  $0 [--limit 1000] [--config config_lora.csv] [--execute]"
    echo "  $0 --before <date> [--config config_lora.csv] [--execute]"
    echo "  $0 <config_lora.csv> [date] [--execute]"
    echo ""
    echo "Options:"
    echo "  --limit <n>       Delete the n least-used LoRA files per server. Default: 1000."
    echo "  --least-used <n>  Alias for --limit."
    echo "  --before <date>   Delete LoRA files whose selected timestamp is before this date."
    echo "                    Examples: 2026-01-01, \"Jan 1 2026\""
    echo "  --config <file>   CSV file to read. Defaults to config_lora.csv next to this script."
    echo "  --execute         Actually delete files. Without this, only prints matches."
    echo "  --dry-run         Force dry-run mode."
    echo "  --max-depth <n>   find(1) max depth under lora_models_path. Default: 1."
    echo "  --time-field <f>  Timestamp to compare: access, modified, changed. Default: access."
    echo "                    access is closest to least-used/opened; changed is inode change time."
    echo "  --log-dir <dir>   Directory for per-server logs. Defaults to ./logs next to this script."
    echo "  -h, --help        Show this help."
    echo ""
    echo "CSV formats supported:"
    echo "  remote_host, lora_models_path"
    echo "  remote_host, models_path, utils_path, lora_models_path, [extra flags]"
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

shell_quote() {
    local value="$1"
    printf "'%s'" "$(printf '%s' "$value" | sed "s/'/'\\\\''/g")"
}

safe_log_name() {
    printf '%s' "$1" | sed 's/[^A-Za-z0-9_.-]/_/g'
}

positional=()
while [ $# -gt 0 ]; do
    case "$1" in
        --before|--cutoff)
            if [ $# -lt 2 ]; then
                echo "Error: $1 requires a date value"
                exit 1
            fi
            CUTOFF_DATE="$2"
            CLEANUP_MODE="before"
            shift 2
            ;;
        --limit|--least-used|--count)
            if [ $# -lt 2 ]; then
                echo "Error: $1 requires a numeric value"
                exit 1
            fi
            LIMIT_COUNT="$2"
            CLEANUP_MODE="count"
            shift 2
            ;;
        --config)
            if [ $# -lt 2 ]; then
                echo "Error: --config requires a file path"
                exit 1
            fi
            CONFIG_FILE="$2"
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
        --max-depth)
            if [ $# -lt 2 ]; then
                echo "Error: --max-depth requires a numeric value"
                exit 1
            fi
            MAX_DEPTH="$2"
            shift 2
            ;;
        --time-field)
            if [ $# -lt 2 ]; then
                echo "Error: --time-field requires a value"
                exit 1
            fi
            TIME_FIELD="$2"
            shift 2
            ;;
        --log-dir)
            if [ $# -lt 2 ]; then
                echo "Error: --log-dir requires a directory path"
                exit 1
            fi
            LOG_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            positional+=("$1")
            shift
            ;;
    esac
done

if [ "${#positional[@]}" -gt 0 ]; then
    if [ -f "${positional[0]}" ]; then
        CONFIG_FILE="${positional[0]}"
        if [ "${#positional[@]}" -gt 1 ]; then
            CUTOFF_DATE="${positional[1]}"
            CLEANUP_MODE="before"
        fi
    elif [ -z "$CUTOFF_DATE" ]; then
        CUTOFF_DATE="${positional[0]}"
        CLEANUP_MODE="before"
    fi
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config file not found: $CONFIG_FILE"
    exit 1
fi

if ! [[ "$LIMIT_COUNT" =~ ^[0-9]+$ ]] || [ "$LIMIT_COUNT" -lt 1 ]; then
    echo "Error: --limit must be a positive integer"
    exit 1
fi

if ! [[ "$MAX_DEPTH" =~ ^[0-9]+$ ]] || [ "$MAX_DEPTH" -lt 1 ]; then
    echo "Error: --max-depth must be a positive integer"
    exit 1
fi

if [ "$CLEANUP_MODE" = "before" ] && [ -z "$CUTOFF_DATE" ]; then
    echo "Error: --before requires a cutoff date"
    exit 1
fi

case "$TIME_FIELD" in
    access|atime|used|open|opened)
        TIME_FIELD="access"
        ;;
    modified|modify|mtime|modification)
        TIME_FIELD="modified"
        ;;
    changed|change|ctime|metadata)
        TIME_FIELD="changed"
        ;;
    *)
        echo "Error: --time-field must be one of: access, modified, changed"
        exit 1
        ;;
esac

if ! command -v ssh >/dev/null 2>&1; then
    echo "Error: ssh command not found"
    exit 1
fi

if ! mkdir -p "$LOG_DIR"; then
    echo "Error: failed to create log directory: $LOG_DIR"
    exit 1
fi

RUN_ID=$(date '+%Y%m%d-%H%M%S')
TOTAL=$(grep -v '^[[:space:]]*#' "$CONFIG_FILE" | grep -v '^[[:space:]]*$' | wc -l | tr -d ' ')

echo "=================================================="
echo "  LoRA Model Cleanup"
echo "=================================================="
echo "Config file: $CONFIG_FILE"
echo "Cleanup mode: $CLEANUP_MODE"
if [ "$CLEANUP_MODE" = "before" ]; then
    echo "Cutoff date: $CUTOFF_DATE"
else
    echo "Limit:       $LIMIT_COUNT least-used file(s) per server"
fi
echo "Time field:  $TIME_FIELD"
echo "Mode:        $([ "$EXECUTE" = true ] && echo "EXECUTE" || echo "DRY RUN")"
echo "Max depth:   $MAX_DEPTH"
echo "SSH timeout: ${SSH_CONNECT_TIMEOUT}s"
echo "Log dir:     $LOG_DIR"
echo "Run id:      $RUN_ID"
echo "Servers:     $TOTAL"
echo "=================================================="
echo ""

if [ "$EXECUTE" != true ]; then
    echo "Dry-run mode: no files will be deleted. Re-run with --execute to remove matches."
    echo ""
fi

LINE_NUM=0
SUCCESS_COUNT=0
FAIL_COUNT=0
TOTAL_MATCHES=0
TOTAL_BYTES=0
TOTAL_DELETE_FAILURES=0
JOB_COUNT=0
PIDS=()
RESULT_FILES=()
LOG_FILES=()

cleanup_background_jobs() {
    echo ""
    echo "Stopping cleanup jobs..."
    local pid
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    exit 130
}

trap cleanup_background_jobs INT TERM

run_remote_cleanup() {
    local remote_host="$1"
    local lora_path="$2"
    local output_file="$3"
    local remote_cmd
    local pipe_status

    remote_cmd="bash -s -- $(shell_quote "$lora_path") $(shell_quote "$CLEANUP_MODE") $(shell_quote "$CUTOFF_DATE") $(shell_quote "$LIMIT_COUNT") $(shell_quote "$EXECUTE") $(shell_quote "$MAX_DEPTH") $(shell_quote "$TIME_FIELD")"

    echo "Connecting and scanning remote LoRA directory..."
    ssh -T \
        -o BatchMode=yes \
        -o ConnectTimeout="$SSH_CONNECT_TIMEOUT" \
        -o ServerAliveInterval=15 \
        -o ServerAliveCountMax=2 \
        "$remote_host" "$remote_cmd" <<'REMOTE_SCRIPT' \
      | tee "$output_file" \
      | awk -F '\t' '
          $1 == "MODE" { print "Remote cleanup mode: " $2; fflush() }
          $1 == "CUTOFF" { print "Remote cutoff: " $2; fflush() }
          $1 == "LIMIT" { print "Remote limit: " $2; fflush() }
          $1 == "TIME_FIELD" { print "Remote time field: " $2; fflush() }
          $1 == "MATCH" {
            printf "  match  %12s bytes  %s  %s\n", $2, $3, $4
            fflush()
          }
          $1 == "REMOVED" { print "  removed " $2; fflush() }
          $1 == "FAILED" { print "  failed  " $2; fflush() }
          $1 == "ERROR" { print "  error   " $2; fflush() }
        '
set -u

lora_path="$1"
cleanup_mode="$2"
cutoff_date="$3"
limit_count="$4"
execute="$5"
max_depth="$6"
time_field="$7"

if [ ! -d "$lora_path" ]; then
    echo "ERROR	LoRA models path does not exist: $lora_path"
    exit 10
fi

case "$time_field" in
    access)
        find_newer="-newerat"
        find_sort_format="%A@"
        stat_format="%x"
        ;;
    modified)
        find_newer="-newermt"
        find_sort_format="%T@"
        stat_format="%y"
        ;;
    changed)
        find_newer="-newerct"
        find_sort_format="%C@"
        stat_format="%z"
        ;;
    *)
        echo "ERROR	Unsupported time field: $time_field"
        exit 12
        ;;
esac

if ! [[ "$limit_count" =~ ^[0-9]+$ ]] || [ "$limit_count" -lt 1 ]; then
    echo "ERROR	Invalid limit count: $limit_count"
    exit 13
fi

echo "MODE	$cleanup_mode"
if [ "$cleanup_mode" = "before" ]; then
    if ! cutoff_display=$(date -d "$cutoff_date" '+%Y-%m-%d %H:%M:%S %Z' 2>/dev/null); then
        echo "ERROR	Could not parse cutoff date on remote host: $cutoff_date"
        exit 11
    fi
    echo "CUTOFF	$cutoff_display"
else
    echo "LIMIT	$limit_count"
fi
echo "TIME_FIELD	$time_field"

count=0
bytes=0
failed=0

if [ "$cleanup_mode" = "before" ]; then
    while IFS= read -r -d '' file; do
        size=$(stat -c '%s' -- "$file" 2>/dev/null || printf '0')
        timestamp=$(stat -c "$stat_format" -- "$file" 2>/dev/null | cut -d'.' -f1)
        echo "MATCH	$size	$timestamp	$file"
        count=$((count + 1))
        bytes=$((bytes + size))

        if [ "$execute" = true ]; then
            if rm -f -- "$file"; then
                echo "REMOVED	$file"
            else
                failed=$((failed + 1))
                echo "FAILED	$file"
            fi
        fi
    done < <(find "$lora_path" -mindepth 1 -maxdepth "$max_depth" -type f ! "$find_newer" "$cutoff_date" -print0)
elif [ "$cleanup_mode" = "count" ]; then
    while IFS= read -r -d '' entry; do
        count=$((count + 1))
        if [ "$count" -gt "$limit_count" ]; then
            break
        fi
        file="${entry#*$'\t'}"
        size=$(stat -c '%s' -- "$file" 2>/dev/null || printf '0')
        timestamp=$(stat -c "$stat_format" -- "$file" 2>/dev/null | cut -d'.' -f1)
        echo "MATCH	$size	$timestamp	$file"
        bytes=$((bytes + size))

        if [ "$execute" = true ]; then
            if rm -f -- "$file"; then
                echo "REMOVED	$file"
            else
                failed=$((failed + 1))
                echo "FAILED	$file"
            fi
        fi
    done < <(find "$lora_path" -mindepth 1 -maxdepth "$max_depth" -type f -printf "${find_sort_format}\t%p\0" | sort -z -n)
    if [ "$count" -gt "$limit_count" ]; then
        count="$limit_count"
    fi
else
    echo "ERROR	Unsupported cleanup mode: $cleanup_mode"
    exit 12
fi

echo "SUMMARY	$count	$bytes	$failed"

if [ "$failed" -gt 0 ]; then
    exit 13
fi
REMOTE_SCRIPT
    pipe_status=("${PIPESTATUS[@]}")
    return "${pipe_status[0]}"
}

run_server_job() {
    local line_num="$1"
    local total="$2"
    local remote_host="$3"
    local lora_path="$4"
    local log_file="$5"
    local result_file="$6"
    local output_file
    local status
    local summary
    local matches
    local bytes
    local delete_failures

    output_file=$(mktemp "${TMPDIR:-/tmp}/cleanup-lora-models.XXXXXX")

    {
        echo "=================================================="
        echo "  LoRA Cleanup Job"
        echo "=================================================="
        echo "Started:          $(date '+%Y-%m-%d %H:%M:%S %Z')"
        echo "Server:           $remote_host"
        echo "Server index:     $line_num/$total"
        echo "LoRA models path: $lora_path"
        echo "Cleanup mode:     $CLEANUP_MODE"
        if [ "$CLEANUP_MODE" = "before" ]; then
            echo "Cutoff date:      $CUTOFF_DATE"
        else
            echo "Limit:            $LIMIT_COUNT least-used file(s)"
        fi
        echo "Time field:       $TIME_FIELD"
        echo "Mode:             $([ "$EXECUTE" = true ] && echo "EXECUTE" || echo "DRY RUN")"
        echo "Max depth:        $MAX_DEPTH"
        echo "=================================================="
        echo ""

        if run_remote_cleanup "$remote_host" "$lora_path" "$output_file"; then
            status=0
        else
            status=$?
        fi

        summary=$(awk -F '\t' '$1 == "SUMMARY" { print $2 " " $3 " " $4 }' "$output_file" | tail -1)
        matches=$(printf '%s' "$summary" | awk '{ print $1 + 0 }')
        bytes=$(printf '%s' "$summary" | awk '{ print $2 + 0 }')
        delete_failures=$(printf '%s' "$summary" | awk '{ print $3 + 0 }')

        echo ""
        echo "=================================================="
        echo "  Server Summary"
        echo "=================================================="
        echo "Status:           $status"
        echo "Matched files:    $matches"
        echo "Matched size:     $(format_bytes "$bytes")"
        echo "Delete failures:  $delete_failures"
        echo "Finished:         $(date '+%Y-%m-%d %H:%M:%S %Z')"
        echo "=================================================="
    } > "$log_file" 2>&1

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$status" "$line_num" "$total" "$remote_host" "$lora_path" \
        "$matches" "$bytes" "$delete_failures" "$log_file" > "$result_file"

    rm -f "$output_file"
    return "$status"
}

while IFS= read -r line <&3 || [ -n "$line" ]; do
    if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
        continue
    fi

    LINE_NUM=$((LINE_NUM + 1))

    IFS=',' read -ra FIELDS <<< "$line"
    if [ "${#FIELDS[@]}" -lt 2 ]; then
        echo "[$LINE_NUM/$TOTAL] Skipping invalid row: $line"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    REMOTE_HOST=$(trim "${FIELDS[0]}")
    if [ "${#FIELDS[@]}" -ge 4 ]; then
        LORA_MODELS_PATH=$(trim "${FIELDS[3]}")
    else
        LORA_MODELS_PATH=$(trim "${FIELDS[1]}")
    fi

    if [ -z "$REMOTE_HOST" ] || [ -z "$LORA_MODELS_PATH" ]; then
        echo "[$LINE_NUM/$TOTAL] Skipping invalid row: $line"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    LOG_NAME=$(safe_log_name "$REMOTE_HOST")
    LOG_FILE="${LOG_DIR}/cleanup-lora-${RUN_ID}-${LINE_NUM}-${LOG_NAME}.log"
    RESULT_FILE=$(mktemp "${TMPDIR:-/tmp}/cleanup-lora-result.XXXXXX")
    rm -f "$RESULT_FILE"

    JOB_COUNT=$((JOB_COUNT + 1))
    echo "[$LINE_NUM/$TOTAL] Started $REMOTE_HOST"
    echo "  LoRA models path: $LORA_MODELS_PATH"
    echo "  Log: $LOG_FILE"

    run_server_job "$LINE_NUM" "$TOTAL" "$REMOTE_HOST" "$LORA_MODELS_PATH" "$LOG_FILE" "$RESULT_FILE" &
    PIDS[$JOB_COUNT]=$!
    RESULT_FILES[$JOB_COUNT]="$RESULT_FILE"
    LOG_FILES[$JOB_COUNT]="$LOG_FILE"
done 3< "$CONFIG_FILE"

echo ""
echo "Started $JOB_COUNT cleanup job(s) in parallel."
echo ""

if [ "$JOB_COUNT" -gt 0 ]; then
    for index in $(seq 1 "$JOB_COUNT"); do
        PID="${PIDS[$index]}"
        RESULT_FILE="${RESULT_FILES[$index]}"
        if wait "$PID"; then
            :
        else
            :
        fi

        if [ -f "$RESULT_FILE" ]; then
            IFS="$(printf '\t')" read -r STATUS SERVER_LINE SERVER_TOTAL REMOTE_HOST LORA_MODELS_PATH MATCHES BYTES DELETE_FAILURES LOG_FILE < "$RESULT_FILE"
            rm -f "$RESULT_FILE"
        else
            STATUS=1
            SERVER_LINE="$index"
            SERVER_TOTAL="$TOTAL"
            REMOTE_HOST="<unknown>"
            LORA_MODELS_PATH="<unknown>"
            MATCHES=0
            BYTES=0
            DELETE_FAILURES=0
            LOG_FILE="${LOG_FILES[$index]}"
        fi

        TOTAL_MATCHES=$((TOTAL_MATCHES + MATCHES))
        TOTAL_BYTES=$((TOTAL_BYTES + BYTES))
        TOTAL_DELETE_FAILURES=$((TOTAL_DELETE_FAILURES + DELETE_FAILURES))

        if [ "$STATUS" -eq 0 ]; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            echo "OK   [$SERVER_LINE/$SERVER_TOTAL] $REMOTE_HOST complete: $MATCHES file(s), $(format_bytes "$BYTES")"
        else
            FAIL_COUNT=$((FAIL_COUNT + 1))
            echo "FAIL [$SERVER_LINE/$SERVER_TOTAL] $REMOTE_HOST failed with exit code $STATUS"
        fi
        echo "   Log: $LOG_FILE"
    done
fi

trap - INT TERM

echo "=================================================="
echo "  Cleanup Summary"
echo "=================================================="
echo "Mode:             $([ "$EXECUTE" = true ] && echo "EXECUTE" || echo "DRY RUN")"
echo "Cleanup mode:     $CLEANUP_MODE"
if [ "$CLEANUP_MODE" = "count" ]; then
    echo "Limit:            $LIMIT_COUNT least-used file(s) per server"
else
    echo "Cutoff date:      $CUTOFF_DATE"
fi
echo "Time field:       $TIME_FIELD"
echo "Servers total:    $TOTAL"
echo "Servers success:  $SUCCESS_COUNT"
echo "Servers failed:   $FAIL_COUNT"
echo "Matched files:    $TOTAL_MATCHES"
echo "Matched size:     $(format_bytes "$TOTAL_BYTES")"
echo "Delete failures:  $TOTAL_DELETE_FAILURES"
echo "=================================================="

if [ "$FAIL_COUNT" -gt 0 ] || [ "$TOTAL_DELETE_FAILURES" -gt 0 ]; then
    exit 1
fi
