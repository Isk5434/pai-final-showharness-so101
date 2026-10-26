#!/usr/bin/env bash
# Astra を蒸留した Qwen3.5-4B を sim で評価する（無人）。学習の完了を待ち、指定エポックのチェックポイントごとに
# policy_server.py を立て、Astra の収集と同じ gpt_collect_v150.py（--vlm_server）で near / middle を回す。
# 判定は収集の記録と同じ厳しい in-hole（target_in_hole）。シードは DM0.5 の評価と同じ 7000〜。
#   bash scripts/eval/vlm_astra_eval_auto.sh [エポック="5 10"] [N=10] [シード=7000]
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
EPOCHS=${1:-5 10}; N=${2:-10}; SEED=${3:-7000}
RUN=outputs/train/qwen3_5_4b_astra_near_mix57; BASE=${BASE:-Qwen/Qwen3.5-4B}
LF_PY=/home/ubuntu/so101-showharness-deps/Show-Harness/third_party/LlamaFactory/.venv/bin/python
OUT=outputs/eval_vlm_astra_near_mix57; mkdir -p "$OUT"
log() { echo "[vlm_eval] $(date '+%F %T') $*" | tee -a "$OUT/auto.log"; }
until [ -f outputs/train_vlm_astra_near_mix57/done ]; do sleep 60; done
STEPS=$(ls -d "$RUN"/checkpoint-* | sed 's/.*checkpoint-//' | sort -n)   # エポックごとの保存（昇順）
for ep in $EPOCHS; do
  CK="$RUN/checkpoint-$(echo "$STEPS" | sed -n "${ep}p")"
  [ -d "$CK" ] || { log "epoch $ep: no checkpoint"; continue; }
  TAG=ep$ep; log "$TAG ckpt=$CK"
  FLA_TILELANG=0 setsid nohup $LF_PY scripts/eval/policy_server.py --adapter "$CK" --base "$BASE" --port 8765 \
    > "$OUT/server_$TAG.log" 2>&1 < /dev/null &
  SP=$!
  for i in $(seq 1 60); do curl -s -o /dev/null -X POST localhost:8765/act 2>/dev/null && break; sleep 10; done
  bash scripts/sim/run_in_teleop_docker.sh scripts/collect/gpt_collect_v150.py --model smoke --vlm_server http://localhost:8765 \
    --target near --episodes 1 --seed0 6999 --max_tokens 3 --out "/workspace/out/eval_vlm_astra_near_mix57/smoke_$TAG" \
    > "$OUT/smoke_$TAG.log" 2>&1
  grep "\[GPT収集\].*トークン=3" "$OUT/smoke_$TAG.log" | tee -a "$OUT/auto.log" \
    || { log "$TAG smoke failed"; kill $SP; exit 1; }
  for pos in near middle; do
    bash scripts/sim/run_in_teleop_docker.sh scripts/collect/gpt_collect_v150.py --model "qwen3.5-4b-astra-$TAG" \
      --vlm_server http://localhost:8765 --target "$pos" --episodes "$N" --seed0 "$SEED" \
      --out "/workspace/out/eval_vlm_astra_near_mix57/$TAG/$pos" > "$OUT/${TAG}_$pos.log" 2>&1
    log "$TAG $pos rc=$?"
  done
  kill $SP 2>/dev/null; sleep 10
done
echo done > "$OUT/done"
