#!/bin/bash
# Multi-node worker: claims runs with atomic mkdir instead of flock.
# (flock does NOT give cross-host mutual exclusion on GPFS; mkdir is atomic.)
# The queue file is never mutated, so runs can be appended at any time.
# Usage: stab_worker_mp.sh <gpu>
set -uo pipefail

GPU="$1"
S=/gpfs/data/oermannlab/users/steelr04/min_dinov3_paperhp
SWEEP=$S/sweepbin
QUEUE=$S/queue.tsv
MANIFEST=$S/manifest.tsv
CLAIMS=$S/claims
LOGS=$S/logs
MAX_ATTEMPTS=3

host=$(hostname)
mkdir -p "$CLAIMS" "$LOGS"

note() {  # run_id state attempt detail
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$1" "$2" "$host" "$GPU" "$3" "$(date +%Y-%m-%dT%H:%M:%S)" "$4" >> "$MANIFEST"
}

EMPTY_SCANS=0
MAX_EMPTY=10        # ~10 min idle before releasing the node

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
    if [ $EMPTY_SCANS -ge $MAX_EMPTY ]; then
      echo "[worker gpu$GPU@$host] queue quiet for $MAX_EMPTY scans, exiting"
      break
    fi
    sleep 60
    continue
  fi
  EMPTY_SCANS=0

  IFS=$'\t' read -r run_id base model ds bb seed variant <<< "$claimed"
  log=$LOGS/${run_id}.log
  results=$S/results/$variant

  ok=0
  for attempt in $(seq 1 $MAX_ATTEMPTS); do
    note "$run_id" running "$attempt" "$log"
    bash $SWEEP/min_run_one.sh "$base" "$model" "$GPU" "$seed" "$results" >> "$log" 2>&1
    rc=$?
    if grep -aq "FINAL:" "$log"; then
      note "$run_id" done "$attempt" "$(grep -a 'FINAL:' "$log" | tail -1)"
      ok=1
      break
    else
      # enroot's NVIDIA hook intermittently fails with an ldcache error on these
      # nodes; back off a little longer each time rather than burning attempts.
      note "$run_id" retry "$attempt" "exit=$rc"
      sleep $((30 * attempt))
    fi
  done
  [ $ok -eq 0 ] && note "$run_id" FAILED "$MAX_ATTEMPTS" "see $log"
done
