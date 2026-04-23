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

# Build a short human-readable summary of the failure (stderr / benchmark logs).
# Uses SLURM_SUBMIT_DIR, SLURM_JOB_ID, SLURM_ARRAY_*, LOG_DIR from the environment.
min_slack_failure_summary() {
    if ! command -v python3 >/dev/null 2>&1; then
        printf '%s' "(install python3 for detailed failure summaries)"
        return 0
    fi
    python3 -c '
import glob, os, re, sys

MAX_CHARS = 3500
TAIL_LINES = 120

def read_text(path: str) -> str:
    try:
        with open(path, "r", errors="replace") as f:
            return f.read()
    except OSError:
        return ""

def tail_lines(s: str, n: int) -> str:
    lines = s.splitlines()
    if len(lines) <= n:
        return "\n".join(lines)
    return "\n".join(lines[-n:])

def extract_from_log(text: str) -> str:
    if not text.strip():
        return ""
    # Prefer the last Python traceback
    idx = text.rfind("Traceback (most recent call last)")
    if idx != -1:
        chunk = text[idx:].strip()
        if len(chunk) > MAX_CHARS:
            chunk = chunk[: MAX_CHARS - 3] + "..."
        return chunk
    low = text.lower()
    for pat in (
        r"(cuda error[^\n]*(?:\n[^\n]*){0,40})",
        r"(runtimeerror[^\n]*(?:\n[^\n]*){0,40})",
        r"(out of memory[^\n]*(?:\n[^\n]*){0,30})",
        r"(killed[^\n]*(?:\n[^\n]*){0,15})",
    ):
        m = re.search(pat, low, re.IGNORECASE | re.DOTALL)
        if m:
            start = max(0, m.start() - 200)
            chunk = text[start : m.end() + 400].strip()
            if len(chunk) > MAX_CHARS:
                chunk = chunk[: MAX_CHARS - 3] + "..."
            return chunk
    chunk = tail_lines(text, TAIL_LINES).strip()
    if len(chunk) > MAX_CHARS:
        chunk = chunk[: MAX_CHARS - 3] + "..."
    return chunk

submit = os.environ.get("SLURM_SUBMIT_DIR", "")
jid = os.environ.get("SLURM_JOB_ID", "")
ajid = os.environ.get("SLURM_ARRAY_JOB_ID", "")
atid = os.environ.get("SLURM_ARRAY_TASK_ID", "")
log_dir = os.environ.get("LOG_DIR", "")

stderr_paths = []
if submit and jid:
    if ajid and atid not in (None, ""):
        for prefix in ("unlearn", "min-unlearn", "dino-bench", "min-dino-array"):
            stderr_paths.append(os.path.join(submit, f"slurm/logs/{prefix}_{ajid}_{atid}.err"))
            stderr_paths.append(os.path.join(submit, f"{prefix}-{ajid}_{atid}.err"))
    for prefix in ("unlearn", "min-unlearn", "dino-bench", "min-dino-array"):
        stderr_paths.append(os.path.join(submit, f"slurm/logs/{prefix}_{jid}.err"))
        stderr_paths.append(os.path.join(submit, f"{prefix}-{jid}.err"))

raw = ""
label = ""
for p in stderr_paths:
    if os.path.isfile(p) and os.path.getsize(p) > 0:
        raw = read_text(p)
        label = f"stderr: {os.path.basename(p)}"
        break

if not raw.strip() and log_dir and os.path.isdir(log_dir):
    logs = glob.glob(os.path.join(log_dir, "*.log"))
    logs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    best_raw = ""
    best_label = ""
    best_score = -1
    for p in logs[:15]:
        t = read_text(p)
        if not t.strip():
            continue
        low = t.lower()
        score = 0
        if "traceback" in low:
            score += 100
        if "error" in low or "exception" in low:
            score += 20
        if "cuda" in low or "out of memory" in low:
            score += 30
        if score > best_score or (score == best_score and len(t) > len(best_raw)):
            best_score = score
            best_raw = t
            best_label = f"log: {os.path.basename(p)}"
    if not best_raw.strip() and logs:
        p = logs[0]
        t = read_text(p)
        if t.strip():
            best_raw = t
            best_label = f"log: {os.path.basename(p)}"
    if best_raw.strip():
        raw = best_raw
        label = best_label

if not raw.strip():
    sys.stdout.write("No stderr or benchmark log content found for diagnostics.")
    sys.exit(0)

out = extract_from_log(raw)
if not out.strip():
    out = tail_lines(raw, 40)
summary = f"{label}\n{out}".strip()
if len(summary) > MAX_CHARS:
    summary = summary[: MAX_CHARS - 3] + "..."
sys.stdout.write(summary)
'
}

# Args: exit code, optional short label (prepended to the status line inside min_slack_notify).
min_slack_notify_failure() {
    local exit_code="${1:-?}"
    local label="${2:-}"
    local detail
    detail="$(min_slack_failure_summary)"
    local status="FAILED (exit ${exit_code})"
    [[ -n "$label" ]] && status="${label} — ${status}"
    min_slack_notify "$status" "$detail"
}

min_slack_notify() {
    local status_msg="$1"
    local detail="${2:-}"
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

    # Slack often renders top-level "text" as plain; Block Kit "mrkdwn" sections get formatting.
    # Top-level "text" remains the push-notification / accessibility fallback.
    local payload
    if command -v python3 >/dev/null 2>&1; then
        payload="$(python3 -c '
import json, sys
line = sys.argv[1]
detail = sys.argv[2] if len(sys.argv) > 2 else ""
body = {
    "text": line + (("\n" + detail[:500] + "...") if len(detail) > 500 else ("\n" + detail if detail else "")),
    "blocks": [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": line},
        },
    ],
}
if detail.strip():
    d = detail.strip()
    if len(d) > 2900:
        d = d[:2897] + "..."
    d = d.replace(chr(96), chr(39))
    body["blocks"].append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": "*Details:*\n```\n" + d + "\n```"},
    })
print(json.dumps(body))
' "$line" "$detail")" || payload=""
    fi
    if [[ -z "$payload" ]] && command -v jq >/dev/null 2>&1; then
        if [[ -n "${detail// }" ]]; then
            local d_safe
            d_safe="$(printf '%s' "$detail" | head -c 2900)"
            payload="$(jq -n --arg line "$line" --arg det "$d_safe" \
                '{text: ($line + "\n" + ($det | .[0:500])), blocks: [{type: "section", text: {type: "mrkdwn", text: $line}}, {type: "section", text: {type: "mrkdwn", text: ("*Details:*\n```\n" + $det + "\n```")}}]}')" \
                || payload=""
        else
            payload="$(jq -n --arg line "$line" \
                '{text: $line, blocks: [{type: "section", text: {type: "mrkdwn", text: $line}}]}')" \
                || payload=""
        fi
    fi
    if [[ -z "$payload" ]]; then
        line="${line//\\/\\\\}"
        line="${line//\"/\\\"}"
        payload="{\"text\":\"$line\"}"
    fi

    curl -sS -m 15 -X POST -H 'Content-type: application/json' \
        -d "$payload" "$webhook" >/dev/null 2>&1 || true
}
