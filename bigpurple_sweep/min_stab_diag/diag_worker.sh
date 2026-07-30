#!/bin/bash
# Atomic-mkdir claims worker for the instrumented probe queue.
# Queue row: run_id  base  model  seed  stages  flags
# Success test: "PROBE_FINAL:" in the log.
# Usage: diag_worker.sh <gpu>
set -uo pipefail

GPU="$1"
S=/gpfs/data/oermannlab/users/steelr04/min_stab_diag
QUEUE=$S/queue.tsv
MANIFEST=$S/manifest.tsv
CLAIMS=$S/claims
LOGS=$S/logs
PROBES=$S/probes
MAX_ATTEMPTS=3

host=$(hostname)
mkdir -p "$CLAIMS" "$LOGS" "$PROBES"

note() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$1" "$2" "$host" "$GPU" "$3" "$(date +%Y-%m-%dT%H:%M:%S)" "$4" >> "$MANIFEST"
}

EMPTY_SCANS=0
MAX_EMPTY=10

while true; do
  claimed=""
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    rid=$(printf '%s' "$line" | cut -f1)
    if mkdir "$CLAIMS/$rid" 2>/dev/null; then
      claimed="$line"
      break
    fi
  done < "$QUEUE"

  if [ -z "$claimed" ]; then
    EMPTY_SCANS=$((EMPTY_SCANS + 1))
    [ $EMPTY_SCANS -ge $MAX_EMPTY ] && { echo "[diag gpu$GPU@$host] queue quiet, exiting"; break; }
    sleep 60
    continue
  fi
  EMPTY_SCANS=0

  IFS=$'\t' read -r run_id base model seed stages flags <<< "$claimed"
  [ -z "$flags" ] && flags='-'
  log=$LOGS/${run_id}.log

  ok=0
  for attempt in $(seq 1 $MAX_ATTEMPTS); do
    note "$run_id" running "$attempt" "$log"
    bash $S/probe_run_one.sh "$base" "$model" "$GPU" "$seed" "$stages" "$flags" "$PROBES/$run_id" >> "$log" 2>&1
    rc=$?
    if grep -aq "PROBE_FINAL:" "$log"; then
      note "$run_id" done "$attempt" "$(grep -a 'PROBE_FINAL:' "$log" | tail -1 | head -c 200)"
      ok=1
      break
    else
      note "$run_id" retry "$attempt" "exit=$rc"
      sleep $((30 * attempt))
    fi
  done
  [ $ok -eq 0 ] && note "$run_id" FAILED "$MAX_ATTEMPTS" "see $log"
done
