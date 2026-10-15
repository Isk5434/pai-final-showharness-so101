#!/usr/bin/env bash
# GR00T N1.7 を DM0.5 と同じ条件で学習・評価する（無人）:
#   1) GPU が空くのを待つ（DM0.5 の temporal ensemble 評価の done）
#   2) 学習: 同じ 50 話（LeRobot v2.1 に変換）、1 万ステップ、バッチ 8、腕・グリッパとも絶対関節角
#   3) 推論サーバ（temporal ensemble、K=5）→ 試運転 1 話 → 本番: 各位置 30 話（シード 7000〜、60 秒、停止判定なし）
# 専用 env: /home/ubuntu/so101-showharness-deps/Isaac-GR00T-n17（Isaac-GR00T d2b7e75、uv.lock どおり）
#   bash scripts/train/gr00t_auto.sh [待つ印=outputs/eval_dm05_te5/done]
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
WAIT=${1:-outputs/eval_dm05_te5/done}
G=/home/ubuntu/so101-showharness-deps/Isaac-GR00T-n17
DATA=$REPO/outputs/lerobot/gr00t/local/so101_gpt_success50
RUN=$REPO/outputs/train_gr00t_success50
EVAL=outputs/eval_gr00t_te5
mkdir -p "$RUN" "$EVAL"
log() { echo "[gr00t_auto] $(date '+%F %T') $*" | tee -a "$RUN/auto.log"; }

until [ -f "$WAIT" ]; do sleep 60; done
if [ -d "$RUN/run/checkpoint-10000" ]; then
  log "checkpoint-10000 exists; skipping training"
else
log "GPU free ($WAIT); training"
# finetune.sh は中で `python` を呼ぶので、専用 env の python が先に見えるようにする（公式手順の uv run と同じ。
# 素の bash で呼ぶとシステムの python になり tyro が無くて落ちた、2026-10-13）
# deepspeed は import 時に CUDA_HOME/bin/nvcc -V を見る。このマシンには toolkit が無いので、torch(cu128) と
# 同じ 12.8 の nvcc を持つ conda env を CUDA_HOME に使う（公式 README の「export CUDA_HOME」と同じ、2026-10-14）
CUDA_HOME=${CUDA_HOME:-/home/ubuntu/miniconda3/envs/lerobot-arena}
( cd "$G" && CUDA_HOME="$CUDA_HOME" PATH="$G/.venv/bin:$PATH" VIRTUAL_ENV="$G/.venv" CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 GLOBAL_BATCH_SIZE=8 MAX_STEPS=10000 SAVE_STEPS=2000 USE_WANDB=0 \
    bash examples/finetune.sh --base-model-path nvidia/GR00T-N1.7-3B --dataset-path "$DATA" \
    --modality-config-path "$REPO/configs/gr00t/so101_config.py" --embodiment-tag NEW_EMBODIMENT \
    --output-dir "$RUN/run" ) > "$RUN/train.log" 2>&1
rc=$?; log "training rc=$rc"
[ "$rc" = 0 ] || exit 1
fi
# パスの so101-showharness にも「-」があるので sort -t- は列がずれ、文字列順で 8000 を選んでいた（2026-10-14）
CKPT=$(ls -d "$RUN"/run/checkpoint-* | sort -V | tail -1)
log "checkpoint $CKPT"

setsid nohup "$G/.venv/bin/python" scripts/eval/gr00t_server.py --ckpt "$CKPT" --ensemble 5 --port 8780 \
  > "$EVAL/server.log" 2>&1 < /dev/null &
SP=$!
for i in $(seq 1 90); do grep -q "loaded" "$EVAL/server.log" && break; sleep 10; done
grep -q "loaded" "$EVAL/server.log" || { log "server failed"; exit 1; }
bash scripts/sim/run_in_teleop_docker.sh scripts/eval/dm05_eval_sim.py --episodes 1 --seed0 6999 --positions near \
  --max_s 10 --stall_s 0 --label "GR00T N1.7" --out "/workspace/out/eval_gr00t_te5/smoke" > "$EVAL/smoke.log" 2>&1
grep -q "\[dm05_eval\]" "$EVAL/smoke.log" || { log "smoke failed (see $EVAL/smoke.log, $EVAL/server.log)"; kill $SP; exit 1; }
log "smoke ok; full eval"
bash scripts/sim/run_in_teleop_docker.sh scripts/eval/dm05_eval_sim.py --episodes 30 --seed0 7000 \
  --positions near middle far --stall_s 0 --label "GR00T N1.7" --out "/workspace/out/eval_gr00t_te5/full" \
  > "$EVAL/full.log" 2>&1
log "full eval rc=$?"
kill $SP 2>/dev/null
echo done > "$EVAL/done"
