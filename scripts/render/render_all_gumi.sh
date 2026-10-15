#!/usr/bin/env bash
# 収集フォルダの採用エピソードをすべて GUMI パネル付きで描き直す（<収集>/video_gumi/）。
#   bash scripts/render/render_all_gumi.sh outputs/collect_v1 [速度]
set -euo pipefail
SRC=${1:?収集フォルダ}; SPEED=${2:-1.0}
for d in "$SRC"/traj/ep_*; do
  ep=$(basename "$d")
  out="$SRC/video_gumi/$ep.mp4"
  [ -f "$out" ] && continue
  python3 "$(dirname "$0")/render_gumi_video.py" "$d" --out "$out" --speed "$SPEED"
done
