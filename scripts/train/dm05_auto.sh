#!/usr/bin/env bash
# DM05（lerobot/dm05_base）を SO-101 の LeRobot データで FT する。無人で:
#   1) データ変換の完了を待つ
#   2) 試運転（20 step）で OOM しない最大バッチを探す（8 → 4 → 2、全部だめなら LoRA）
#   3) 本番（STEPS step, SAVE ごとに checkpoint）
# 専用 env: /home/ubuntu/so101-showharness-deps/lerobot-dm05（lerobot main 2577da0e, uv.lock どおり）
#   bash scripts/train/dm05_auto.sh [データ=outputs/lerobot/so101_gpt_success50] [出力名=dm05_success50] [STEPS=10000]
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
DATA=${1:-outputs/lerobot/so101_gpt_success50}; TAG=${2:-dm05_success50}; STEPS=${3:-10000}; SAVE=${SAVE:-2000}
PY=/home/ubuntu/so101-showharness-deps/lerobot-dm05/.venv/bin
LOG=outputs/train_$TAG; mkdir -p "$LOG"
echo "[dm05_auto] $(date '+%F %T') start" | tee -a "$LOG/auto.log"

CONV_LOG=${CONV_LOG:-outputs/to_lerobot50.log}
until [ -f "$DATA/meta/info.json" ] && grep -q "kept .*skipped" "$CONV_LOG" 2>/dev/null; do sleep 30; done
echo "[dm05_auto] $(date '+%F %T') dataset ready: $(python3 -c "import json;d=json.load(open('$DATA/meta/info.json'));print(d['total_episodes'],'eps',d['total_frames'],'frames')")" | tee -a "$LOG/auto.log"

common=(--dataset.repo_id=local/so101_gpt_success --dataset.root="$DATA" --dataset.video_backend=pyav
        --policy.path=lerobot/dm05_base --policy.chunk_size=30 --policy.n_action_steps=15
        --policy.push_to_hub=false --policy.device=cuda --wandb.enable=false --num_workers=4)

peak_mem() {  # GPU メモリのピークを記録しながらコマンドを走らせる（出力は $out.log）。"rc peakMiB" を返す
  local out=$1; shift
  ( while sleep 2; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; done ) > "$out.mem" &
  local mp=$!
  "$@" > "$out.log" 2>&1; local rc=$?
  kill $mp 2>/dev/null
  echo "$rc $(sort -n "$out.mem" | tail -1)"
}

BS=""; PEFT=""
# 2026-10-09: 全パラメータの FT は L40S 46GB で bs=2 でも OOM（PyTorch だけで 42GB）。PEFT_ONLY=1 で LoRA から
for bs in $([ "${PEFT_ONLY:-0}" = 1 ] || echo 8 4 2); do
  rm -rf "$LOG/smoke_bs$bs"
  res=$(peak_mem "$LOG/smoke_bs$bs" "$PY/lerobot-train" "${common[@]}" --batch_size=$bs --steps=20 --log_freq=5 \
        --save_freq=100000 --output_dir="$LOG/smoke_bs$bs")
  rc=${res%% *}; mem=${res##* }
  echo "[dm05_auto] smoke bs=$bs rc=$rc peak_mem=${mem}MiB" | tee -a "$LOG/auto.log"
  if [ "$rc" = 0 ]; then BS=$bs; break; fi
done
if [ -z "$BS" ]; then
  # --policy.use_peft は「学習済み LoRA を読む」ための設定。新しく LoRA を張るのは --peft.*。
  # DM05 は既定の対象層を持たないので all-linear。行動の入出力層は新しいロボット用に全部学習する。
  BS=8; PEFT="--peft.target_modules=all-linear --peft.r=32 --peft.full_training_modules=[action_in_proj,action_out_proj]"
  res=$(peak_mem "$LOG/smoke_peft" "$PY/lerobot-train" "${common[@]}" $PEFT --batch_size=$BS --steps=20 --log_freq=5 \
        --save_freq=100000 --output_dir="$LOG/smoke_peft")
  echo "[dm05_auto] smoke peft bs=$BS -> ${res}" | tee -a "$LOG/auto.log"
  [ "${res%% *}" = 0 ] || { echo "[dm05_auto] all smoke runs failed; stop" | tee -a "$LOG/auto.log"; exit 1; }
fi

echo "[dm05_auto] $(date '+%F %T') full run: bs=$BS steps=$STEPS ${PEFT:-full-ft}" | tee -a "$LOG/auto.log"
"$PY/lerobot-train" "${common[@]}" $PEFT --batch_size=$BS --steps=$STEPS --log_freq=50 --save_freq=$SAVE \
  --output_dir="$LOG/run" > "$LOG/run.log" 2>&1
echo "[dm05_auto] $(date '+%F %T') full run rc=$?" | tee -a "$LOG/auto.log"
echo done > "$LOG/done"
