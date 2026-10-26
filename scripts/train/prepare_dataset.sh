#!/usr/bin/env bash
# Show-Harness train/scripts/prepare_dataset.sh と同じ手順を、回転トークン込みの変換器で行う。
#   SRC=<書き出し済みビュー>  NAME=<データセット名>  [VERSION=v5]  [OUT=...]
#   SRC=data/train_view/oracle_v1 NAME=so101_vials_oracle_v1 bash scripts/train/prepare_dataset.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SH=${SHOW_HARNESS_ROOT:-/home/ubuntu/so101-showharness-deps/Show-Harness}
SRC="${SRC:?SRC が必要}"; NAME="${NAME:?NAME が必要}"
VERSION="${VERSION:-v5}"    # RT_* を含むのは v5（v3 は平行移動とグリッパーのみ）
OUT="${OUT:-$REPO/data/train_data/$NAME}"
source "$SH/train/scripts/llamafactory_env.sh"   # LF_ROOT / DATA_PREP_PYTHON（上流と同じ）
PY="${DATA_PREP_PYTHON:-$SH/.venv/bin/python}"
mkdir -p "$OUT"
dirs=(); maps=()
for d in "$SRC"/*/; do
  T="$(basename "$d")"
  [ -d "$d/rollout_000" ] || continue
  if [ -f "$d/task_text.txt" ]; then TASK_TEXT="$(cat "$d/task_text.txt")"   # export_rollouts.py --list
  else TASK_TEXT="$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['task_text'])" "$d/rollout_000/metadata.json")"; fi
  echo "[$T] $(find -L "$d" -maxdepth 1 -type d -name 'rollout_*' | wc -l) エピソード  指示: $TASK_TEXT"
  dirs+=("$d"); maps+=("${T}=${TASK_TEXT}")
done
[ "${#dirs[@]}" -gt 0 ] || { echo "rollout_000 を含むタスクが無い: $SRC" >&2; exit 1; }
SHOW_HARNESS_ROOT="$SH" "$PY" "$REPO/scripts/train/rollouts_to_alpaca_rt.py" "${dirs[@]}" \
    --version "$VERSION" --task-map "${maps[@]}" --output "$OUT/rollouts.json"
(cd "$SH" && "$PY" train/data_preparation/register_dataset.py "$NAME" --samples "$OUT/rollouts.json")
