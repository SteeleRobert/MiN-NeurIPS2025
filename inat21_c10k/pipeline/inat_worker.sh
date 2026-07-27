#!/bin/bash
# Places365 worker: atomic-mkdir claims, tolerant of a long empty period while
# the dataset pipeline is being built (holds the node up to PATIENCE scans).
set -uo pipefail
GPU="$1"
S=/gpfs/data/oermannlab/users/steelr04/min_inat
QUEUE=$S/queue.tsv
MANIFEST=$S/manifest.tsv
CLAIMS=$S/claims
LOGS=$S/logs
MAX_ATTEMPTS=2
PATIENCE=720          # 720 x 60s = 12h of waiting for work before releasing
host=$(hostname)
mkdir -p "$CLAIMS" "$LOGS"
note() { printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$host" "$GPU" "$3" "$(date +%Y-%m-%dT%H:%M:%S)" "$4" >> "$MANIFEST"; }
empty=0
while true; do
  claimed=""
  if [ -f "$QUEUE" ]; then
    while IFS= read -r line; do
      [ -z "$line" ] && continue
      rid=$(printf '%s' "$line" | cut -f1)
      if mkdir "$CLAIMS/$rid" 2>/dev/null; then claimed="$line"; break; fi
    done < "$QUEUE"
  fi
  if [ -z "$claimed" ]; then
    empty=$((empty+1))
    [ $empty -ge $PATIENCE ] && { echo "[gpu$GPU@$host] no work for ${PATIENCE}m, releasing"; break; }
    sleep 60; continue
  fi
  empty=0
  IFS=$'\t' read -r run_id base model ds bb seed variant <<< "$claimed"
  log=$LOGS/${run_id}.log
  ok=0
  for attempt in $(seq 1 $MAX_ATTEMPTS); do
    note "$run_id" running "$attempt" "$log"
    bash $S/min_run_one.sh "$base" "$model" "$GPU" "$seed" "$S/results/$variant" >> "$log" 2>&1
    rc=$?
    if grep -aq "FINAL:" "$log"; then
      note "$run_id" done "$attempt" "$(grep -a 'FINAL:' "$log" | tail -1)"; ok=1; break
    else
      note "$run_id" retry "$attempt" "exit=$rc"; sleep 30
    fi
  done
  [ $ok -eq 0 ] && note "$run_id" FAILED "$MAX_ATTEMPTS" "see $log"
done
