#!/bin/bash

# Install or update the managed daily systemd timer for LoRA cleanup on this GPU server.

set -e
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_GPU_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CLEANUP_SCRIPT="${SCRIPT_DIR}/cleanup_lora_models_by_usage.sh"
LOG_DIR="${LAUNCH_GPU_DIR}/logs"
LOG_FILE="${LOG_DIR}/lora-cleanup-cron.log"
SCHEDULE="17 3 * * *"
SYSTEMD_TIMER_TIME="03:17"
SYSTEMD_UNIT_NAME="drawthings-lora-cleanup"
LORA_PATH=""
TARGET_SIZE="5T"
TRIGGER_PERCENT=90
TARGET_PERCENT=50
IGNORE_ACCESS_DAYS=2
PROGRESS_INTERVAL="${LORA_CLEANUP_PROGRESS_INTERVAL:-1000}"
RUN_NOW=false
RUN_NOW_DRY_RUN=false
DD_SITE="${DD_SITE:-us5.datadoghq.com}"
DD_API_KEY="${DD_API_KEY:-f395315025e2f0577c31e8b7fa5c7381}"

usage() {
    echo "Usage: $0 --path <lora_models_path> [options]"
    echo ""
    echo "Options:"
    echo "  --path <dir>             LoRA models directory to clean. Required."
    echo "  --target-size <size>     Usage denominator. Default: 5T."
    echo "  --schedule <cron expr>   Legacy cron-style schedule. Default: 17 3 * * *"
    echo "  --timer-time <HH:MM>     systemd OnCalendar daily time. Default: 03:17."
    echo "  --trigger-percent <n>    Cleanup trigger percentage. Default: 90."
    echo "  --target-percent <n>     Cleanup target percentage. Default: 50."
    echo "  --ignore-access-days <n> Ignore files accessed within n days. Default: 2."
    echo "  --progress-interval <n>  Cleanup progress interval. Default: 1000."
    echo "  --run-now                Run cleanup immediately after installing timer."
    echo "  --run-now-dry-run        Run cleanup dry-run immediately after installing timer."
    echo "  --datadog-api-key <key>  Datadog API key. Defaults to DD_API_KEY/DATADOG_API_KEY."
    echo "  --dd-site <site>         Datadog site. Default: us5.datadoghq.com."
    echo "  -h, --help               Show this help."
}

shell_quote() {
    local value="$1"
    printf "'%s'" "$(printf '%s' "$value" | sed "s/'/'\\\\''/g")"
}

systemd_escape_value() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    printf '%s' "$value"
}

cron_schedule_to_timer_time() {
    local schedule="$1"
    local minute
    local hour
    read -r minute hour rest <<< "$schedule"
    if [[ "$minute" =~ ^[0-9]+$ ]] && [[ "$hour" =~ ^[0-9]+$ ]] && [ "$minute" -ge 0 ] && [ "$minute" -le 59 ] && [ "$hour" -ge 0 ] && [ "$hour" -le 23 ]; then
        printf '%02d:%02d' "$hour" "$minute"
        return 0
    fi
    return 1
}

is_positive_integer() {
    [[ "${1:-}" =~ ^[0-9]+$ ]] && [ "$1" -gt 0 ]
}

is_percent() {
    [[ "${1:-}" =~ ^[0-9]+$ ]] && [ "$1" -ge 0 ] && [ "$1" -le 100 ]
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
        --schedule)
            [ $# -ge 2 ] || { echo "Error: --schedule requires a value"; exit 1; }
            SCHEDULE="$2"
            SYSTEMD_TIMER_TIME="$(cron_schedule_to_timer_time "$SCHEDULE" || true)"
            if [ -z "$SYSTEMD_TIMER_TIME" ]; then
                echo "Error: --schedule must be a daily cron expression with numeric minute/hour, e.g. '17 3 * * *'"
                exit 1
            fi
            shift 2
            ;;
        --timer-time)
            [ $# -ge 2 ] || { echo "Error: --timer-time requires a value"; exit 1; }
            SYSTEMD_TIMER_TIME="$2"
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
        --run-now)
            RUN_NOW=true
            shift
            ;;
        --run-now-dry-run)
            RUN_NOW=true
            RUN_NOW_DRY_RUN=true
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

if [ -z "$LORA_PATH" ]; then
    echo "Error: --path is required"
    usage
    exit 1
fi
if [ ! -d "$LORA_PATH" ]; then
    echo "Error: LoRA models path not found: $LORA_PATH"
    exit 1
fi
if [ ! -f "$CLEANUP_SCRIPT" ]; then
    echo "Error: cleanup script not found: $CLEANUP_SCRIPT"
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
if ! [[ "$SYSTEMD_TIMER_TIME" =~ ^([01][0-9]|2[0-3]):[0-5][0-9]$ ]]; then
    echo "Error: --timer-time must use HH:MM in 24-hour time"
    exit 1
fi
if ! command -v systemctl >/dev/null 2>&1; then
    echo "Error: systemctl not found; systemd timer install is not available on this host"
    exit 1
fi

if [ -z "$DD_API_KEY" ]; then
    DD_API_KEY="$(extract_datadog_key_from_start_script || true)"
fi

mkdir -p "$LOG_DIR"
chmod +x "$CLEANUP_SCRIPT"

SERVICE_FILE="/etc/systemd/system/${SYSTEMD_UNIT_NAME}.service"
TIMER_FILE="/etc/systemd/system/${SYSTEMD_UNIT_NAME}.timer"
ENV_LINES="Environment=\"DD_SITE=$(systemd_escape_value "$DD_SITE")\""
if [ -n "$DD_API_KEY" ]; then
    ENV_LINES="${ENV_LINES}
Environment=\"DD_API_KEY=$(systemd_escape_value "$DD_API_KEY")\""
else
    echo "Warning: no Datadog API key found; cleanup still runs, Datadog emit will be skipped."
fi

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Draw Things LoRA cleanup
Documentation=file://${CLEANUP_SCRIPT}

[Service]
Type=oneshot
WorkingDirectory=${SCRIPT_DIR}
${ENV_LINES}
ExecStart=/bin/bash ${CLEANUP_SCRIPT} --path ${LORA_PATH} --target-size ${TARGET_SIZE} --trigger-percent ${TRIGGER_PERCENT} --target-percent ${TARGET_PERCENT} --ignore-access-days ${IGNORE_ACCESS_DAYS} --progress-interval ${PROGRESS_INTERVAL} --execute
StandardOutput=append:${LOG_FILE}
StandardError=append:${LOG_FILE}
EOF

cat > "$TIMER_FILE" <<EOF
[Unit]
Description=Run Draw Things LoRA cleanup daily

[Timer]
OnCalendar=*-*-* ${SYSTEMD_TIMER_TIME}:00
Persistent=true
RandomizedDelaySec=10m
Unit=${SYSTEMD_UNIT_NAME}.service

[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable --now "${SYSTEMD_UNIT_NAME}.timer"

echo "Installed managed LoRA cleanup systemd timer."
echo "LoRA path: $LORA_PATH"
echo "Target size: $TARGET_SIZE"
echo "Timer: daily at $SYSTEMD_TIMER_TIME"
echo "Service: $SERVICE_FILE"
echo "Timer file: $TIMER_FILE"
echo "Log file: $LOG_FILE"
echo "systemd verification:"
systemctl is-enabled "${SYSTEMD_UNIT_NAME}.timer" | sed 's/^/  enabled: /' || true
systemctl is-active "${SYSTEMD_UNIT_NAME}.timer" | sed 's/^/  active: /' || true
systemctl list-timers "${SYSTEMD_UNIT_NAME}.timer" --no-pager --all || true
echo "Inspect commands:"
echo "  systemctl status ${SYSTEMD_UNIT_NAME}.timer --no-pager"
echo "  systemctl status ${SYSTEMD_UNIT_NAME}.service --no-pager"
echo "  journalctl -u ${SYSTEMD_UNIT_NAME}.service -n 80 --no-pager"
echo "  tail -n 80 $LOG_FILE"

if [ "$RUN_NOW" = true ]; then
    RUN_ARGS=(
        /bin/bash "$CLEANUP_SCRIPT"
        --path "$LORA_PATH"
        --target-size "$TARGET_SIZE"
        --trigger-percent "$TRIGGER_PERCENT"
        --target-percent "$TARGET_PERCENT"
        --ignore-access-days "$IGNORE_ACCESS_DAYS"
        --progress-interval "$PROGRESS_INTERVAL"
    )
    if [ "$RUN_NOW_DRY_RUN" = true ]; then
        RUN_ARGS+=(--dry-run)
        echo "Running LoRA cleanup dry-run now..."
    else
        RUN_ARGS+=(--execute)
        echo "Running LoRA cleanup now..."
    fi
    echo "Immediate cleanup output is also appended to $LOG_FILE"
    "${RUN_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
    exit "${PIPESTATUS[0]}"
fi
