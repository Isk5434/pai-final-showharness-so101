#!/usr/bin/env bash
# temporal ensemble（K ステップごとに予測して重なりを平均）で DM05 を評価する。前回（素の切り替え）と同じシード・
# 同じ判定・停止判定なしで、違いは動かし方だけにする。
#   bash scripts/eval/dm05_eval_te.sh [K=5] [N=30] [シード=7000] [出力名=eval_dm05_te5]
# 別の学習を評価するときは CKPT・DATA・POSITIONS・STALL_S（素の評価の既定は 8）を合わせ、
# GPU を取り合わないよう WAIT に先の評価の done を渡す。
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
K=${1:-5}; N=${2:-30}; SEED=${3:-7000}; TAG=${4:-eval_dm05_te$K}
PY=/home/ubuntu/so101-showharness-deps/lerobot-dm05/.venv/bin/python
CKPT=${CKPT:-outputs/train_dm05_success50/run/checkpoints/last/pretrained_model}
DATA=${DATA:-outputs/lerobot/so101_gpt_success50}; POSITIONS=${POSITIONS:-near middle far}; STALL_S=${STALL_S:-0}
OUT=outputs/$TAG; mkdir -p "$OUT"
[ -n "${WAIT:-}" ] && until [ -f "$WAIT" ]; do sleep 60; done
setsid nohup $PY scripts/eval/dm05_server.py --ckpt "$CKPT" --dataset "$DATA" \
  --port 8780 --ensemble "$K" > "$OUT/server.log" 2>&1 < /dev/null &
SP=$!
for i in $(seq 1 60); do grep -q "loaded" "$OUT/server.log" && break; sleep 10; done
grep "loaded" "$OUT/server.log" | tee -a "$OUT/auto.log" || { echo "server failed" | tee -a "$OUT/auto.log"; exit 1; }
echo "[te] $(date '+%F %T') start K=$K N=$N ckpt=$CKPT" | tee -a "$OUT/auto.log"
bash scripts/sim/run_in_teleop_docker.sh scripts/eval/dm05_eval_sim.py --episodes "$N" --seed0 "$SEED" \
  --positions $POSITIONS --stall_s "$STALL_S" --out "/workspace/out/$TAG/full" > "$OUT/full.log" 2>&1
echo "[te] $(date '+%F %T') done rc=$?" | tee -a "$OUT/auto.log"
kill $SP 2>/dev/null
echo done > "$OUT/done"
