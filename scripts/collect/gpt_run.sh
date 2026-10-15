#!/usr/bin/env bash
# 1 モデルで N 話を無人で集める。取るバイアルの位置はシード % 3 で near/middle/far を回すので、
# N=150 なら 3 位置が 50 話ずつになる。途中で sim や GPT サーバが落ちても、episodes.jsonl に
# 記録済みの話を飛ばして続きのシードから再開する。成功/失敗の最終判断は動画を見て人がする。
#   bash scripts/collect/gpt_run.sh [話数=150] [シード=4000] [出力名=gpt_astra150] [モデル=gpt-6-astra]
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
N=${1:-150}; SEED=${2:-4000}; TAG=${3:-gpt_astra150}; MODEL=${4:-gpt-6-astra}
OUT=outputs/$TAG/$MODEL
mkdir -p "outputs/$TAG"
start_server() {
  setsid nohup python3 scripts/collect/gpt_operator_server.py --port 8770 --log outputs/$TAG/server.jsonl \
    >> outputs/${TAG}_server.log 2>&1 < /dev/null &
  SPID=$!; sleep 2
}
start_server
tries=0
while :; do
  done_n=$( [ -f "$OUT/episodes.jsonl" ] && wc -l < "$OUT/episodes.jsonl" || echo 0 )
  [ "$done_n" -ge "$N" ] && break
  [ "$tries" -ge 10 ] && { echo "[gpt_run] 再開が 10 回続いたので中止" ; break; }
  kill -0 "$SPID" 2>/dev/null || start_server
  echo "[gpt_run] $(date '+%F %T') $done_n/$N 済み → シード $((SEED + done_n)) から"
  bash scripts/sim/run_in_teleop_docker.sh scripts/collect/gpt_collect.py --model "$MODEL" --effort low --target cycle \
      --episodes $((N - done_n)) --seed0 $((SEED + done_n)) --out "/workspace/out/$TAG/$MODEL"
  tries=$((tries + 1))
done
kill "$SPID" 2>/dev/null || true
echo done > outputs/$TAG.done
