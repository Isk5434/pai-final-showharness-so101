#!/usr/bin/env bash
# temporal ensemble（K ステップごとに予測して重なりを平均）で DM05 を評価する。前回（素の切り替え）と同じシード・
# 同じ判定・停止判定なしで、違いは動かし方だけにする。
#   bash scripts/eval/dm05_eval_te.sh [K=5] [N=30] [シード=7000] [出力名=eval_dm05_te5]
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
K=${1:-5}; N=${2:-30}; SEED=${3:-7000}; TAG=${4:-eval_dm05_te$K}
PY=/home/ubuntu/so101-showharness-deps/lerobot-dm05/.venv/bin/python
CKPT=outputs/train_dm05_success50/run/checkpoints/last/pretrained_model
OUT=outputs/$TAG; mkdir -p "$OUT"
setsid nohup $PY scripts/eval/dm05_server.py --ckpt "$CKPT" --dataset outputs/lerobot/so101_gpt_success50 \
  --port 8780 --ensemble "$K" > "$OUT/server.log" 2>&1 < /dev/null &
SP=$!
for i in $(seq 1 60); do grep -q "loaded" "$OUT/server.log" && break; sleep 10; done
grep "loaded" "$OUT/server.log" | tee -a "$OUT/auto.log" || { echo "server failed" | tee -a "$OUT/auto.log"; exit 1; }
echo "[te] $(date '+%F %T') start K=$K N=$N" | tee -a "$OUT/auto.log"
bash scripts/sim/run_in_teleop_docker.sh scripts/eval/dm05_eval_sim.py --episodes "$N" --seed0 "$SEED" \
  --positions near middle far --stall_s 0 --out "/workspace/out/$TAG/full" > "$OUT/full.log" 2>&1
echo "[te] $(date '+%F %T') done rc=$?" | tee -a "$OUT/auto.log"
kill $SP 2>/dev/null
echo done > "$OUT/done"
