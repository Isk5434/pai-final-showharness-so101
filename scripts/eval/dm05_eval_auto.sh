#!/usr/bin/env bash
# 学習の完了を待って、DM05 を sim で評価する（無人）:
#   1) 推論サーバを最後のチェックポイントで起動
#   2) 試運転（near 1 話・10 秒）で通信と動作を確認
#   3) 本番: near / middle / far 各 N 話（動画つき）
#   bash scripts/eval/dm05_eval_auto.sh [学習名=dm05_success50] [N=30] [シード=7000]
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
TAG=${1:-dm05_success50}; N=${2:-30}; SEED=${3:-7000}
PY=/home/ubuntu/so101-showharness-deps/lerobot-dm05/.venv/bin/python
OUT=outputs/eval_$TAG; mkdir -p "$OUT"
until [ -f "outputs/train_$TAG/done" ]; do sleep 60; done
CKPT=outputs/train_$TAG/run/checkpoints/last/pretrained_model
echo "[eval_auto] $(date '+%F %T') train done; ckpt=$CKPT" | tee -a "$OUT/auto.log"
setsid nohup $PY scripts/eval/dm05_server.py --ckpt "$CKPT" --dataset outputs/lerobot/so101_gpt_success50 --port 8780 \
  > "$OUT/server.log" 2>&1 < /dev/null &
SP=$!
for i in $(seq 1 60); do grep -q "loaded" "$OUT/server.log" && break; sleep 10; done
grep -q "loaded" "$OUT/server.log" || { echo "[eval_auto] server failed" | tee -a "$OUT/auto.log"; exit 1; }
bash scripts/sim/run_in_teleop_docker.sh scripts/eval/dm05_eval_sim.py --episodes 1 --seed0 6999 --positions near \
  --max_s 10 --out /workspace/out/eval_$TAG/smoke > "$OUT/smoke.log" 2>&1
grep "\[dm05_eval\]" "$OUT/smoke.log" | tee -a "$OUT/auto.log" || { echo "[eval_auto] smoke failed" | tee -a "$OUT/auto.log"; kill $SP; exit 1; }
echo "[eval_auto] $(date '+%F %T') full eval: $N per position" | tee -a "$OUT/auto.log"
bash scripts/sim/run_in_teleop_docker.sh scripts/eval/dm05_eval_sim.py --episodes "$N" --seed0 "$SEED" \
  --positions near middle far --out /workspace/out/eval_$TAG/full > "$OUT/full.log" 2>&1
echo "[eval_auto] $(date '+%F %T') full eval rc=$?" | tee -a "$OUT/auto.log"
kill $SP 2>/dev/null
echo done > "$OUT/done"
