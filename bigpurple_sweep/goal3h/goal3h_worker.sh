#!/bin/bash
# Goal-3-H worker: drains screen_queue.tsv (3-stage probes via min_stab_probe.py)
# then confirm_queue.tsv (full runs via min_run_one.sh) with atomic-mkdir claims.
# Queue rows:
#   screen:  run_id  base  model  seed  stages  flags     (probe_run_one.sh)
#   confirm: run_id  base  model  cell  backbone  seed  variant  (min_run_one.sh)
# Confirm rows are distinguished by a 7-column format (variant column present).
# Usage: goal3h_worker.sh <gpu>
set -uo pipefail

GPU="$1"
S=/gpfs/data/oermannlab/users/steelr04/goal3h
DIAG=/gpfs/data/oermannlab/users/steelr04/min_stab_diag
SWEEPBIN=/gpfs/data/oermannlab/users/steelr04/min_dinov3_paperhp/sweepbin
MANIFEST=$S/manifest.tsv
CLAIMS=$S/claims
LOGS=$S/logs
MAX_ATTEMPTS=3

host=$(hostname)
mkdir -p "$CLAIMS" "$LOGS" "$S/probes" "$S/results"

note() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$1" "$2" "$host" "$GPU" "$3" "$(date +%Y-%m-%dT%H:%M:%S)" "$4" >> "$MANIFEST"
}

EMPTY_SCANS=0
MAX_EMPTY=10

while true; do
  claimed=""
  for QUEUE in $S/screen_queue.tsv $S/confirm_queue.tsv; do
    [ -f "$QUEUE" ] || continue
    while IFS= read -r line; do
      [ -z "$line" ] && continue
      rid=$(printf '%s' "$line" | cut -f1)
      if mkdir "$CLAIMS/$rid" 2>/dev/null; then
        claimed="$line"
        break 2
      fi
    done < "$QUEUE"
  done

  if [ -z "$claimed" ]; then
    EMPTY_SCANS=$((EMPTY_SCANS + 1))
    [ $EMPTY_SCANS -ge $MAX_EMPTY ] && { echo "[g3h gpu$GPU@$host] queues quiet, exiting"; break; }
    sleep 60
    continue
  fi
  EMPTY_SCANS=0

  nf=$(printf '%s' "$claimed" | awk -F'\t' '{print NF}')
  rid=$(printf '%s' "$claimed" | cut -f1)
  log=$LOGS/${rid}.log
  ok=0
  for attempt in $(seq 1 $MAX_ATTEMPTS); do
    note "$rid" running "$attempt" "$log"
    if [ "$nf" -ge 7 ]; then
      IFS=$'\t' read -r run_id base model cell bb seed variant <<< "$claimed"
      bash $SWEEPBIN/min_run_one.sh "$base" "$model" "$GPU" "$seed" "$S/results/$variant" >> "$log" 2>&1
      rc=$?
      marker="FINAL:"
    else
      IFS=$'\t' read -r run_id base model seed stages flags <<< "$claimed"
      [ -z "$flags" ] && flags='-'
      bash $DIAG/probe_run_one.sh "$base" "$model" "$GPU" "$seed" "$stages" "$flags" "$S/probes/$run_id" >> "$log" 2>&1
      rc=$?
      marker="PROBE_FINAL:"
    fi
    if grep -aq "$marker" "$log"; then
      note "$rid" done "$attempt" "$(grep -a "$marker" "$log" | tail -1 | head -c 200)"
      ok=1
      break
    else
      note "$rid" retry "$attempt" "exit=$rc"
      sleep $((30 * attempt))
    fi
  done
  [ $ok -eq 0 ] && note "$rid" FAILED "$MAX_ATTEMPTS" "see $log"
done
