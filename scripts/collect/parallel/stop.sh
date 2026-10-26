#!/usr/bin/env bash
# 並列収集を止める。各ワーカーは今の話を書き終えてからではなく、STOP を見た時点の区切り（チャンクの終わり）で止まる。
#   bash scripts/collect/parallel/stop.sh        # チャンクの終わりで止める
#   bash scripts/collect/parallel/stop.sh --now  # すぐ止める（書きかけの話は捨てる）
REPO=$(cd "$(dirname "$0")/../../.." && pwd); RUN=${RUN:-near200}
touch "$REPO/outputs/$RUN/STOP"
if [ "${1:-}" = "--now" ]; then
  pkill -f "scripts/collect/parallel/worker.sh" || true
  docker ps --format '{{.Names}}' | grep '^so101sh_w' | xargs -r docker rm -f
fi
