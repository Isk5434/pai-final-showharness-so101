#!/usr/bin/env bash
# Astra の動作トークンを Qwen3.5-4B に蒸留する学習を、GPU が空くのを待って無人で回す。
#   WAIT=outputs/eval_dm05_near_mix57_20k_te5/done bash scripts/train/vlm_astra_auto.sh
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
CONFIG=${CONFIG:-train/so101_qwen3_5_4b_astra.yaml}
LOG=outputs/train_vlm_astra_near_mix57; mkdir -p "$LOG"
log() { echo "[vlm_auto] $(date '+%F %T') $*" | tee -a "$LOG/auto.log"; }
[ -n "${WAIT:-}" ] && until [ -f "$WAIT" ]; do sleep 60; done
log "start config=$CONFIG"
CONFIG="$CONFIG" WANDB_PROJECT=${WANDB_PROJECT:-so101-showharness} bash scripts/train/train.sh > "$LOG/run.log" 2>&1
rc=$?; log "train rc=$rc"
[ "$rc" = 0 ] && echo done > "$LOG/done"
