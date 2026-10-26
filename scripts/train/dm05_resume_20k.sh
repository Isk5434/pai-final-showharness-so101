#!/usr/bin/env bash
# dm05_near_mix57 を 10,000 → 20,000 ステップへ再開し、その後 sim で評価する（無人）。
# 学習率は cosine（減衰 50,000 ステップ固定）なので、続きから回しても最初から 20,000 で回したのと同じ。
# GPU を取り合わないよう、10,000 ステップ版の評価（outputs/eval_dm05_near_mix57/done）を待ってから始める。
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
RUN=outputs/train_dm05_near_mix57; LOG=outputs/train_dm05_near_mix57_20k; mkdir -p "$LOG"
PY=/home/ubuntu/so101-showharness-deps/lerobot-dm05/.venv/bin
log() { echo "[resume20k] $(date '+%F %T') $*" | tee -a "$LOG/auto.log"; }
until [ -f outputs/eval_dm05_near_mix57/done ]; do sleep 60; done
log "10k eval done; resume to 20000"
"$PY/lerobot-train" --config_path="$RUN/run/checkpoints/last/pretrained_model/train_config.json" --resume=true \
  --steps=20000 > "$LOG/run.log" 2>&1
rc=$?; log "resume rc=$rc; last=$(readlink "$RUN/run/checkpoints/last")"
[ "$rc" = 0 ] || exit 1
echo done > "$LOG/done"
DATA=outputs/lerobot/so101_near_mix57 POSITIONS="near middle" TRAIN_DONE="$LOG/done" \
  CKPT="$RUN/run/checkpoints/020000/pretrained_model" bash scripts/eval/dm05_eval_auto.sh dm05_near_mix57_20k 10 7000
