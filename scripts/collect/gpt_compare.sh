#!/usr/bin/env bash
# 3 モデルを同じシードで少しずつ比べる（GPT サーバの起動・停止込み）。
#   [MODELS="..."] bash scripts/collect/gpt_compare.sh [エピソード数=2] [シード=3000] [出力名=gpt_trial]
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
N=${1:-2}; SEED=${2:-3000}; TAG=${3:-gpt_trial}
mkdir -p "outputs/$TAG"
python3 scripts/collect/gpt_operator_server.py --port 8770 --log outputs/$TAG/server.jsonl > outputs/${TAG}_server.log 2>&1 &
SPID=$!
sleep 2
for m in ${MODELS:-gpt-6-sol gpt-5.6-terra gpt-6-astra}; do
  bash scripts/sim/run_in_teleop_docker.sh scripts/collect/gpt_collect.py --model "$m" --effort low \
      --episodes "$N" --seed0 "$SEED" --out "/workspace/out/$TAG/$m" || true
done
kill "$SPID" 2>/dev/null || true
echo done > outputs/$TAG.done
