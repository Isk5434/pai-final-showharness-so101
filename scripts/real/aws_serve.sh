#!/usr/bin/env bash
# 実機用の推論サーバを AWS で起動する。127.0.0.1 だけで待ち受けるので、ローカル PC からは SSH トンネルで届く
# （インターネットには開かない）。
#   bash scripts/real/aws_serve.sh dm05 <学習名> [学習データ]   # 例: dm05 dm05_near_mix57 outputs/lerobot/so101_near_mix57
#   bash scripts/real/aws_serve.sh showharness                  # GPT-6 Astra（Codex のログインを使う）
# ポートは PORT で変えられる（既定 dm05=8780、showharness=8770）。
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
MODE=${1:?dm05 か showharness}
mkdir -p outputs/real_server
case $MODE in
  dm05)
    TAG=${2:?学習名（outputs/train_<学習名>）}
    DATA=${3:-$(grep -o 'outputs/lerobot/[^ ]*' "outputs/train_$TAG/auto.log" 2>/dev/null | head -1)}
    [ -n "$DATA" ] || { echo "学習データの場所を 3 つ目の引数で渡す" >&2; exit 2; }
    CKPT=outputs/train_$TAG/run/checkpoints/last/pretrained_model
    [ -d "$CKPT" ] || { echo "チェックポイントが無い: $CKPT" >&2; exit 2; }
    exec /home/ubuntu/so101-showharness-deps/lerobot-dm05/.venv/bin/python scripts/eval/dm05_server.py \
      --ckpt "$CKPT" --dataset "$DATA" --host 127.0.0.1 --port "${PORT:-8780}" ;;
  showharness)
    exec python3 scripts/collect/gpt_operator_server.py --host 127.0.0.1 --port "${PORT:-8770}" \
      --log "outputs/real_server/gpt_$(date +%Y%m%d_%H%M%S).jsonl" ;;
  *) echo "dm05 か showharness" >&2; exit 2 ;;
esac
