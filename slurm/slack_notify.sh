#!/usr/bin/env bash
# Slack Incoming Webhook helper for Slurm jobs (source this file).
#
# Configure one of:
#   export SLACK_WEBHOOK_URL='https://hooks.slack.com/services/...'
#   echo 'https://hooks.slack.com/services/...' > slurm/.slack_webhook_url
#
# If neither is set, min_slack_notify is a no-op.

min_slack_webhook_url() {
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        printf '%s' "$SLACK_WEBHOOK_URL"
        return 0
    fi
    local f="${SLACK_WEBHOOK_URL_FILE:-}"
    if [[ -z "$f" ]]; then
        local _here
        _here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        f="$_here/.slack_webhook_url"
    fi
    if [[ -f "$f" && -r "$f" ]]; then
        local url
        IFS= read -r url <"$f" || true
        url="${url//$'\r'/}"
        url="${url#"${url%%[![:space:]]*}"}"
        url="${url%"${url##*[![:space:]]}"}"
        if [[ -n "$url" ]]; then
            printf '%s' "$url"
        fi
    fi
}

min_slack_notify() {
    local status_msg="$1"
    local webhook
    webhook="$(min_slack_webhook_url)" || true
    [[ -n "$webhook" ]] || return 0

    local job="${SLURM_JOB_ID:-local}"
    local name="${SLURM_JOB_NAME:-unknown}"
    local host
    host="$(hostname -s 2>/dev/null || hostname)"
    local task=""
    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        task=" array=${SLURM_ARRAY_TASK_ID}"
        if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
            task+="/${SLURM_ARRAY_TASK_COUNT}"
        fi
    fi
    local bb="${BACKBONE:-}"
    local line="*${name}* job ${job}${task} @ ${host}"
    if [[ -n "$bb" ]]; then
        line+=" — *${bb}*"
    fi
    line+=" — ${status_msg}"

    local payload
    if command -v python3 >/dev/null 2>&1; then
        payload="$(python3 -c 'import json,sys; print(json.dumps({"text": sys.argv[1]}))' "$line")" || payload=""
    fi
    if [[ -z "$payload" ]]; then
        line="${line//\\/\\\\}"
        line="${line//\"/\\\"}"
        payload="{\"text\":\"$line\"}"
    fi

    curl -sS -m 15 -X POST -H 'Content-type: application/json' \
        -d "$payload" "$webhook" >/dev/null 2>&1 || true
}
