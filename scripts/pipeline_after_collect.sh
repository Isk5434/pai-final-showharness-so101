#!/usr/bin/env bash
# 収集が終わったら、学習データ化 → LeRobot 変換 → 本番学習 → チェックポイントごとの sim 評価まで
# を自動で続ける（セッションが切れても進むように、切り離して起動する）。
#   setsid nohup bash scripts/pipeline_after_collect.sh > outputs/pipeline_v1.log 2>&1 &
# 各段の完了は outputs/pipeline_v1.<段>.done に残す。途中から再実行すると完了済みの段は飛ばす。
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
SH=${SHOW_HARNESS_ROOT:-/home/ubuntu/so101-showharness-deps/Show-Harness}
LFPY=$SH/third_party/LlamaFactory/.venv/bin/python
COLLECT=${COLLECT:-outputs/collect_v1}
EXTRA=${EXTRA:-outputs/collect_test8}
NAME=${NAME:-so101_vials_oracle_v1}
CONFIG=${CONFIG:-train/so101_qwen3_5_2b.yaml}
TRAIN_OUT=${TRAIN_OUT:-outputs/train/oracle_v1}
EVAL_EPISODES=${EVAL_EPISODES:-20}
EVAL_SEED0=${EVAL_SEED0:-9000}
P=outputs/pipeline_v1

stamp() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
done_() { [ -f "$P.$1.done" ]; }
mark() { date > "$P.$1.done"; }

stamp "収集の完了を待つ: $COLLECT"
until [ -f "$COLLECT.done" ]; do sleep 60; done
stamp "収集完了: 採用 $(grep -c '"valid": true' "$COLLECT/episodes.jsonl") / 全 $(wc -l < "$COLLECT/episodes.jsonl")"

if ! done_ export; then
  stamp "1) 学習レイアウトへ書き出し"
  python3 scripts/train/export_rollouts.py --src "$COLLECT" $EXTRA --out "data/train_view/$NAME"
  mark export
fi
if ! done_ prepare; then
  stamp "2) Show-Harness 学習データ化（v5・回転込み）"
  SRC="data/train_view/$NAME" NAME="$NAME" bash scripts/train/prepare_dataset.sh
  mark prepare
fi
if ! done_ lerobot; then
  stamp "3) LeRobot 変換（VLA 比較用）"
  bash scripts/sim/run_in_teleop_docker.sh scripts/convert/traj_to_lerobot.py \
      --src "/workspace/out/${COLLECT#outputs/}" --out "/workspace/out/lerobot/$NAME"
  mark lerobot
fi
if ! done_ train; then
  stamp "4) 本番学習: $CONFIG"
  CONFIG="$CONFIG" WANDB_PROJECT=so101-showharness bash scripts/train/train.sh
  mark train
fi

stamp "5) チェックポイントごとの sim 評価（シード $EVAL_SEED0〜、$EVAL_EPISODES エピソード）"
for ck in $(ls -d "$TRAIN_OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n) "$TRAIN_OUT"; do
  tag=$(basename "$ck"); [ "$ck" = "$TRAIN_OUT" ] && tag=final
  done_ "eval_$tag" && continue
  [ -f "$ck/adapter_model.safetensors" ] || continue
  stamp "  評価: $tag"
  FLA_TILELANG=0 "$LFPY" scripts/eval/policy_server.py --adapter "$ck" --port 8765 > "outputs/policy_server_$tag.log" 2>&1 &
  SPID=$!
  for _ in $(seq 1 60); do grep -q "待ち受け" "outputs/policy_server_$tag.log" 2>/dev/null && break; sleep 5; done
  bash scripts/sim/run_in_teleop_docker.sh scripts/eval/sim_eval_vlm.py --server http://localhost:8765 \
      --episodes "$EVAL_EPISODES" --seed0 "$EVAL_SEED0" --out "/workspace/out/eval_v1/$tag" || true
  kill "$SPID" 2>/dev/null || true
  wait "$SPID" 2>/dev/null || true
  stamp "  $tag: $(cat outputs/eval_v1/$tag/summary.json 2>/dev/null)"
  mark "eval_$tag"
done
stamp "パイプライン完了"
