#!/usr/bin/env bash
# dm05_eval_te.sh の続きから（PC 再起動などで止まったとき）。終わった話は episodes.jsonl から数えて飛ばし、
# 同じ出力先に追記する。位置ごとに「シード0 + 済んだ話数」から残りを回す。
#   bash scripts/eval/dm05_eval_te_resume.sh [K=5] [N=30] [シード=7000] [出力名=eval_dm05_te5]
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
K=${1:-5}; N=${2:-30}; SEED=${3:-7000}; TAG=${4:-eval_dm05_te$K}
PY=/home/ubuntu/so101-showharness-deps/lerobot-dm05/.venv/bin/python
CKPT=outputs/train_dm05_success50/run/checkpoints/last/pretrained_model
OUT=outputs/$TAG
setsid nohup $PY scripts/eval/dm05_server.py --ckpt "$CKPT" --dataset outputs/lerobot/so101_gpt_success50 \
  --port 8780 --ensemble "$K" > "$OUT/server_resume.log" 2>&1 < /dev/null &
SP=$!
for i in $(seq 1 60); do grep -q "loaded" "$OUT/server_resume.log" && break; sleep 10; done
grep -q "loaded" "$OUT/server_resume.log" || { echo "server failed" | tee -a "$OUT/auto.log"; exit 1; }
for pos in near middle far; do
  # grep -c は 0 件でも "0" を出して終了コード 1 を返す。|| echo 0 にすると "0\n0" になり、計算が壊れて
  # その位置が丸ごと飛ばされた（2026-10-13、far が 0 話のまま done になった）
  done_n=$(grep -c "\"position\": \"$pos\"" "$OUT/full/episodes.jsonl" 2>/dev/null; true)
  done_n=${done_n:-0}
  left=$((N - done_n)); [ "$left" -le 0 ] && continue
  echo "[te] $(date '+%F %T') resume $pos from seed $((SEED + done_n)) ($left left)" | tee -a "$OUT/auto.log"
  bash scripts/sim/run_in_teleop_docker.sh scripts/eval/dm05_eval_sim.py --episodes "$left" --seed0 $((SEED + done_n)) \
    --positions "$pos" --stall_s 0 --out "/workspace/out/$TAG/full" >> "$OUT/full.log" 2>&1
done
echo "[te] $(date '+%F %T') done" | tee -a "$OUT/auto.log"
kill $SP 2>/dev/null
echo done > "$OUT/done"
