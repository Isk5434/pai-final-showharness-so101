#!/usr/bin/env bash
# Show-Harness train/scripts/train.sh を、この機（L40S・CUDA toolkit なし）向けの環境変数つきで呼ぶ。
#   CONFIG=train/so101_qwen3_5_2b_smoke.yaml bash scripts/train/train.sh
# FLA_TILELANG=0: fla の gated-delta-rule 逆伝播が tilelang を選ぶと、tilelang は CUDA toolkit
#   （nvcc）を要求して "No CUDA or HIP or MPS available" で落ちる。tilelang は Hopper(sm90)
#   向けの回避策で、L40S(Ada, sm89) は標準の Triton 経路でよい（fla の docstring より）。
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SH=${SHOW_HARNESS_ROOT:-/home/ubuntu/so101-showharness-deps/Show-Harness}
CONFIG="${CONFIG:?CONFIG が必要}"
[ -f "$CONFIG" ] && CONFIG=$(realpath "$CONFIG") || CONFIG="$REPO/$CONFIG"   # 下で Show-Harness に cd するので絶対パスに
export FLA_TILELANG="${FLA_TILELANG:-0}"
cd "$SH" && CONFIG="$CONFIG" bash train/scripts/train.sh
