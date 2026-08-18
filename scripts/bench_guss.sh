#!/usr/bin/env bash
# Benchmark-session wrapper for a Jetson that is also running resident services.
#
# This box (an Orin Nano 8GB) normally serves an 8B LLM via llama-server and
# runs autonomous systemd timers that will RESTART the server if they fire and
# find it stopped — which would reload ~6GB mid-benchmark and silently corrupt
# every number after that point. So a bench session must:
#
#   1. pause the timers            4. lock clocks (MAXN assumed)
#   2. stop llama-server           5. run the sweep
#   3. drop page caches            6. restore everything (trap, runs on any exit)
#
# Every action is logged with a timestamp. Requires interactive sudo.
#
# Usage, from the repo root on the Jetson:
#   scripts/bench_guss.sh python -m bench.run --kernel all
set -euo pipefail

# Jetson wheels may need companion NVIDIA libs from pip on LD_LIBRARY_PATH.
[ -f scripts/jetson_env.sh ] && . scripts/jetson_env.sh

TIMERS=(nightly-report.timer guss-trader.timer)
SERVER=llama-server
LOG="bench/results/session-$(date +%Y%m%d-%H%M%S).log"
mkdir -p bench/results

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

restore() {
    log "RESTORE: starting ${SERVER} (drops caches first, as its unit does at boot)"
    sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || true
    sudo systemctl start "${SERVER}" || log "WARN: ${SERVER} failed to start — check manually"
    log "RESTORE: resuming timers: ${TIMERS[*]}"
    sudo systemctl start "${TIMERS[@]}" || log "WARN: timer restart failed — check manually"
    log "session log: ${LOG}"
}
trap restore EXIT

log "PAUSE: stopping timers: ${TIMERS[*]}"
sudo systemctl stop "${TIMERS[@]}"
log "PAUSE: stopping ${SERVER}"
sudo systemctl stop "${SERVER}"
log "dropping page caches"
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
log "locking clocks"
sudo jetson_clocks
nvpmodel -q 2>/dev/null | tee -a "$LOG" || true
free -h | tee -a "$LOG"

log "RUN: $*"
"$@" 2>&1 | tee -a "$LOG"
log "sweep finished"
