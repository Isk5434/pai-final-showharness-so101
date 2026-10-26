#!/usr/bin/env bash
# 並列収集を始める。ワーカーはそれぞれ独立したプロセスとして裏で動き、このシェルを閉じても続く。
#   RUN=near200 bash scripts/collect/parallel/launch.sh <N> [worker.sh への引数...]
# 止める：bash scripts/collect/parallel/stop.sh   状況：bash scripts/collect/parallel/status.sh
set -euo pipefail
N=$1; shift
REPO=$(cd "$(dirname "$0")/../../.." && pwd); cd "$REPO"
RUN=${RUN:-near200}
mkdir -p "outputs/$RUN"; rm -f "outputs/$RUN/STOP"
for K in $(seq 0 $((N - 1))); do
  RUN=$RUN setsid nohup bash scripts/collect/parallel/worker.sh "$K" "$@" > "outputs/$RUN/w$K.nohup" 2>&1 < /dev/null &
  echo "w$K pid=$!"
  sleep 20  # Isaac の起動が重なってメモリが跳ねないよう、少しずらす
done
