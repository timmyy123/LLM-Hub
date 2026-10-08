#!/bin/bash

# Batch install/update the LoRA cleanup systemd timer from the GPU server CSV.
# Usage:
#   ./LoRACleanup/batch_install_lora_cleanup_timer.sh <config.csv> [--dry-run] [--parallel|--concurrency <n>] [--run-now|--run-now-dry-run]
#
# Examples:
#   From LaunchGPU:
#     ./LoRACleanup/batch_install_lora_cleanup_timer.sh config.csv --parallel
#   From LaunchGPU/LoRACleanup:
#     ./batch_install_lora_cleanup_timer.sh ../config.csv --parallel

set -e

LORA_CLEANUP_SCHEDULE="${LORA_CLEANUP_SCHEDULE:-17 3 * * *}"
LORA_CLEANUP_TARGET_SIZE="${LORA_CLEANUP_TARGET_SIZE:-5T}"
LORA_CLEANUP_TRIGGER_PERCENT="${LORA_CLEANUP_TRIGGER_PERCENT:-90}"
LORA_CLEANUP_TARGET_PERCENT="${LORA_CLEANUP_TARGET_PERCENT:-50}"
LORA_CLEANUP_IGNORE_ACCESS_DAYS="${LORA_CLEANUP_IGNORE_ACCESS_DAYS:-2}"
LORA_CLEANUP_PROGRESS_INTERVAL="${LORA_CLEANUP_PROGRESS_INTERVAL:-1000}"
LORA_CLEANUP_CONCURRENCY="${LORA_CLEANUP_CONCURRENCY:-1}"
DD_SITE="${DD_SITE:-us5.datadoghq.com}"
DD_API_KEY="${DD_API_KEY:-${DATADOG_API_KEY:-}}"

DRY_RUN=false
RUN_NOW=false
RUN_NOW_DRY_RUN=false
VERIFY_ONLY=false
SYSTEMD_UNIT_NAME="${SYSTEMD_UNIT_NAME:-drawthings-lora-cleanup}"

usage() {
    echo "Usage: $0 <config.csv> [options]"
    echo ""
    echo "Examples:"
    echo "  From LaunchGPU:"
    echo "    ./LoRACleanup/batch_install_lora_cleanup_timer.sh config.csv --parallel"
    echo "  From LaunchGPU/LoRACleanup:"
    echo "    ./batch_install_lora_cleanup_timer.sh ../config.csv --parallel"
    echo ""
    echo "Options:"
    echo "  --dry-run             Show commands without running them."
    echo "  --verify-only         Only verify timer state; do not install/update."
    echo "  --parallel            Run all hosts concurrently."
    echo "  --concurrency <n>     Run up to n hosts concurrently."
    echo "  --run-now             Install timer, then immediately execute cleanup."
    echo "  --run-now-dry-run     Install timer, then immediately run cleanup dry-run."
}

shell_quote() {
    local value="$1"
    printf "'%s'" "$(printf '%s' "$value" | sed "s/'/'\\\\''/g")"
}

is_positive_integer() {
    [[ "${1:-}" =~ ^[0-9]+$ ]] && [ "$1" -gt 0 ]
}

append_remote_env_if_set() {
    local name="$1"
    local value="${!name:-}"
    if [ -n "$value" ]; then
        REMOTE_ENV_ARGS="${REMOTE_ENV_ARGS} ${name}=$(shell_quote "$value")"
    fi
}

build_install_command() {
    local utils_path="$1"
    local lora_models_path="$2"
    local REMOTE_ENV_ARGS=""
    local cmd

    append_remote_env_if_set "LORA_CLEANUP_SCHEDULE"
    append_remote_env_if_set "LORA_CLEANUP_TARGET_SIZE"
    append_remote_env_if_set "LORA_CLEANUP_TRIGGER_PERCENT"
    append_remote_env_if_set "LORA_CLEANUP_TARGET_PERCENT"
    append_remote_env_if_set "LORA_CLEANUP_IGNORE_ACCESS_DAYS"
    append_remote_env_if_set "LORA_CLEANUP_PROGRESS_INTERVAL"
    append_remote_env_if_set "DD_API_KEY"
    append_remote_env_if_set "DATADOG_API_KEY"
    append_remote_env_if_set "DD_SITE"

    cmd="cd $(shell_quote "$utils_path") && sudo env${REMOTE_ENV_ARGS} bash ./LoRACleanup/install_lora_cleanup_timer.sh --path $(shell_quote "$lora_models_path") --target-size $(shell_quote "$LORA_CLEANUP_TARGET_SIZE") --schedule $(shell_quote "$LORA_CLEANUP_SCHEDULE") --trigger-percent $(shell_quote "$LORA_CLEANUP_TRIGGER_PERCENT") --target-percent $(shell_quote "$LORA_CLEANUP_TARGET_PERCENT") --ignore-access-days $(shell_quote "$LORA_CLEANUP_IGNORE_ACCESS_DAYS") --progress-interval $(shell_quote "$LORA_CLEANUP_PROGRESS_INTERVAL")"
    if [ "$RUN_NOW" = true ]; then
        if [ "$RUN_NOW_DRY_RUN" = true ]; then
            cmd="$cmd --run-now-dry-run"
        else
            cmd="$cmd --run-now"
        fi
    fi
    printf '%s' "$cmd"
}

build_verify_command() {
    local lora_models_path="$1"
    local expected_path
    local unit_name

    expected_path=$(shell_quote "$lora_models_path")
    unit_name=$(shell_quote "$SYSTEMD_UNIT_NAME")
    printf '%s' "unit=${unit_name}; expected_path=${expected_path}; \
if ! command -v systemctl >/dev/null 2>&1; then echo 'state=missing-systemctl'; exit 2; fi; \
enabled=\$(systemctl is-enabled \"\${unit}.timer\" 2>/dev/null || true); \
active=\$(systemctl is-active \"\${unit}.timer\" 2>/dev/null || true); \
next=\$(systemctl list-timers \"\${unit}.timer\" --no-pager --all 2>/dev/null | awk 'NR==2 {print \$1\" \"\$2\" \"\$3\" \"\$4\" \"\$5\" \"\$6}'); \
exec_start=\$(systemctl show -p ExecStart --value \"\${unit}.service\" 2>/dev/null || true); \
echo \"enabled=\${enabled:-unknown}\"; \
echo \"active=\${active:-unknown}\"; \
echo \"next=\${next:-unknown}\"; \
if printf '%s\n' \"\$exec_start\" | grep -F -- \"--path \${expected_path}\" >/dev/null; then echo \"path=ok (\${expected_path})\"; path_ok=1; else echo \"path=warning expected \${expected_path}\"; echo \"exec_start=\$exec_start\"; path_ok=0; fi; \
if [ \"\$enabled\" = enabled ] && [ \"\$active\" = active ] && [ \"\$path_ok\" = 1 ]; then exit 0; fi; exit 2"
}

CONFIG_FILE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            ;;
        --verify-only)
            VERIFY_ONLY=true
            ;;
        --parallel)
            LORA_CLEANUP_CONCURRENCY=0
            ;;
        --concurrency)
            [ $# -ge 2 ] || { echo "Error: --concurrency requires a value"; exit 1; }
            LORA_CLEANUP_CONCURRENCY="$2"
            shift
            ;;
        --concurrency=*)
            LORA_CLEANUP_CONCURRENCY="${1#*=}"
            ;;
        --run-now)
            RUN_NOW=true
            ;;
        --run-now-dry-run)
            RUN_NOW=true
            RUN_NOW_DRY_RUN=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [ -z "$CONFIG_FILE" ]; then
                CONFIG_FILE="$1"
            else
                echo "Error: unknown argument: $1"
                usage
                exit 1
            fi
            ;;
    esac
    shift
done

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No config file provided"
    usage
    exit 1
fi
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

TOTAL=$(grep -v '^#' "$CONFIG_FILE" | grep -v '^[[:space:]]*$' | wc -l | tr -d ' ')
if [ "$LORA_CLEANUP_CONCURRENCY" = "0" ]; then
    LORA_CLEANUP_CONCURRENCY="$TOTAL"
fi
if ! is_positive_integer "$LORA_CLEANUP_CONCURRENCY"; then
    echo "Error: --concurrency must be a positive integer"
    exit 1
fi

echo "=================================================="
if [ "$VERIFY_ONLY" = true ]; then
    echo "  Batch LoRA Cleanup Timer Verify"
else
    echo "  Batch LoRA Cleanup Timer Install"
fi
echo "=================================================="
echo "Config file: $CONFIG_FILE"
echo "Dry run: $DRY_RUN"
echo "Verify only: $VERIFY_ONLY"
echo "Target size: $LORA_CLEANUP_TARGET_SIZE"
echo "Schedule: $LORA_CLEANUP_SCHEDULE"
echo "Run now: $RUN_NOW"
echo "Run-now mode: $([ "$RUN_NOW_DRY_RUN" = true ] && echo "dry-run" || echo "execute")"
echo "Concurrency: $LORA_CLEANUP_CONCURRENCY"
echo "Total servers: $TOTAL"
echo ""

LINE_NUM=0
SUCCESS_COUNT=0
FAIL_COUNT=0
STATUS_DIR=$(mktemp -d "${TMPDIR:-/tmp}/draw-things-lora-timer.XXXXXX")
trap 'rm -rf "$STATUS_DIR"' EXIT
ACTIVE_PIDS=""
ACTIVE_JOBS=0

while IFS= read -r line <&3 || [ -n "$line" ]; do
    if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
        continue
    fi

    LINE_NUM=$((LINE_NUM + 1))
    IFS=',' read -ra FIELDS <<< "$line"

    REMOTE_HOST=$(echo "${FIELDS[0]:-}" | xargs)
    UTILS_PATH=$(echo "${FIELDS[2]:-}" | xargs)
    LORA_MODELS_PATH=$(echo "${FIELDS[3]:-}" | xargs)

    if [ -z "$REMOTE_HOST" ] || [ -z "$UTILS_PATH" ] || [ -z "$LORA_MODELS_PATH" ]; then
        echo "Error: invalid config row $LINE_NUM; expected remote_host, models_path, utils_path, lora_models_path"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    if [ "$VERIFY_ONLY" = true ]; then
        CMD=$(build_verify_command "$LORA_MODELS_PATH")
        ACTION_LABEL="Timer verify"
    else
        CMD=$(build_install_command "$UTILS_PATH" "$LORA_MODELS_PATH")
        ACTION_LABEL="Timer install"
    fi
    echo "=================================================="
    echo "  [$LINE_NUM/$TOTAL] $ACTION_LABEL: $REMOTE_HOST"
    echo "=================================================="
    echo "  Utils path: $UTILS_PATH"
    echo "  LoRA models path: $LORA_MODELS_PATH"
    if [ "$VERIFY_ONLY" != true ]; then
        echo "  Command: $CMD"
    fi
    echo ""

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would execute timer $([ "$VERIFY_ONLY" = true ] && echo "verify" || echo "install") on $REMOTE_HOST"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        continue
    fi

    HOST_LABEL=$(printf '%s' "$REMOTE_HOST" | tr -c 'A-Za-z0-9._-' '_')
    STATUS_FILE="${STATUS_DIR}/${LINE_NUM}_${HOST_LABEL}.status"
    (
        {
            if ssh "$REMOTE_HOST" "$CMD" < /dev/null; then
                job_status=0
            else
                job_status=$?
            fi
            echo "$job_status" > "$STATUS_FILE"
            exit "$job_status"
        } 2>&1 | sed "s/^/[$REMOTE_HOST] /"
    ) &
    ACTIVE_PIDS="$ACTIVE_PIDS $!"
    ACTIVE_JOBS=$((ACTIVE_JOBS + 1))

    if [ "$ACTIVE_JOBS" -ge "$LORA_CLEANUP_CONCURRENCY" ]; then
        wait $ACTIVE_PIDS || true
        ACTIVE_PIDS=""
        ACTIVE_JOBS=0
    fi
done 3< "$CONFIG_FILE"

if [ "$ACTIVE_JOBS" -gt 0 ]; then
    wait $ACTIVE_PIDS || true
fi

if [ "$DRY_RUN" != true ]; then
    for status_file in "$STATUS_DIR"/*.status; do
        [ -e "$status_file" ] || continue
        status=$(cat "$status_file")
        if [ "$status" -eq 0 ]; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done
fi

echo "=================================================="
if [ "$VERIFY_ONLY" = true ]; then
    echo "  Batch LoRA Cleanup Timer Verify Complete"
else
    echo "  Batch LoRA Cleanup Timer Install Complete"
fi
echo "=================================================="
echo "  Total: $TOTAL"
echo "  Success: $SUCCESS_COUNT"
echo "  Failed: $FAIL_COUNT"
echo "=================================================="

if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
fi
