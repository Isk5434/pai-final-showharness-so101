#!/usr/bin/env bash
# GPT 操作の短い試験: 位置ごとに数話（同じシード）。GPT サーバの起動・停止込み。
#   [EFFORT=low] [EXTRA="--scene_shift 0 0"] bash scripts/collect/gpt_trial.sh <出力名> [話数=3] [シード=5000] [位置="middle near"] [モデル=gpt-6-astra]
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
TAG=$1; N=${2:-3}; SEED=${3:-5000}; POS=${4:-"middle near"}; MODEL=${5:-gpt-6-astra}
mkdir -p "outputs/$TAG"
setsid nohup python3 scripts/collect/gpt_operator_server.py --port 8770 --log outputs/$TAG/server.jsonl \
  > outputs/${TAG}_server.log 2>&1 < /dev/null &
SP=$!; sleep 2
for t in $POS; do
  bash scripts/sim/run_in_teleop_docker.sh scripts/collect/${COLLECTOR:-gpt_collect.py} --model "$MODEL" --effort "${EFFORT:-low}" \
    --target "$t" --episodes "$N" --seed0 "$SEED" --out "/workspace/out/$TAG/$t" ${EXTRA:-}
done
kill "$SP" 2>/dev/null || true
echo done > outputs/$TAG.done
