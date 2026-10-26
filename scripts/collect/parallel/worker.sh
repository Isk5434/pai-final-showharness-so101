#!/usr/bin/env bash
# 並列収集のワーカー 1 つ。ほかのワーカーと何も共有しない（GPT サーバ・ポート・コンテナ・ネットワーク・
# Isaac のキャッシュ・出力・シード範囲がすべて別）。共有するのは GPU 1 枚と Codex のログイン（サブスクの利用枠）だけ。
#
#   bash scripts/collect/parallel/worker.sh <K> [--chunk 5] [--target_succ 200] [--position near]
#
# - GPT サーバ：ホストの 127.0.0.1 ではなく docker の bridge から届く 0.0.0.0:$((8770+K))、ログは自分のフォルダに
# - コンテナ：名前 so101sh_w<K>、自分専用の bridge ネットワーク（ポートや Isaac の通信がほかと衝突しない）
# - キャッシュ：/home/ubuntu/sim2real/isaac-sim-cache を最初に複製した自分専用のコピー
# - シード：100000 + K*10000 から。前回の続きから再開する（episodes.jsonl の最大シード + 1）
# - 終了条件：全ワーカー合計で、狙いの vial が穴に刺さった話（target_in_hole）が --target_succ に達したら
set -uo pipefail
K=$1; shift
CHUNK=5; TARGET_SUCC=200; POSITION=near; MODEL=gpt-6-astra; EFFORT=low
while [ $# -gt 0 ]; do
  case $1 in
    --chunk) CHUNK=$2; shift 2;; --target_succ) TARGET_SUCC=$2; shift 2;;
    --position) POSITION=$2; shift 2;; --model) MODEL=$2; shift 2;; --effort) EFFORT=$2; shift 2;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done

REPO=$(cd "$(dirname "$0")/../../.." && pwd); cd "$REPO"
RUN=${RUN:-near200}
ROOT=$REPO/outputs/$RUN
W=$ROOT/w$K
mkdir -p "$W"
PORT=$((8770 + K))
NET=so101sh_net_w$K
CACHE=/home/ubuntu/sim2real/isaac-sim-cache-w$K
SEED_BASE=$((100000 + K * 10000))
log() { echo "[w$K] $(date '+%F %T') $*" | tee -a "$W/worker.log"; }

# 自分専用の Isaac キャッシュ（共有キャッシュへの同時書き込みを避ける）
[ -d "$CACHE" ] || cp -a /home/ubuntu/sim2real/isaac-sim-cache "$CACHE"
# 自分専用の bridge ネットワーク
docker network inspect "$NET" >/dev/null 2>&1 || docker network create "$NET" >/dev/null

# 自分専用の GPT 操作サーバ（自分のネットワークのゲートウェイだけで待ち受ける＝ほかのワーカーやインターネットからは届かない）
GW=$(docker network inspect -f '{{(index .IPAM.Config 0).Gateway}}' "$NET")
python3 scripts/collect/gpt_operator_server.py --host "$GW" --port "$PORT" --log "$W/server.jsonl" >> "$W/server.out" 2>&1 &
SP=$!
trap 'kill $SP 2>/dev/null; docker rm -f so101sh_w$K >/dev/null 2>&1' EXIT
sleep 2
kill -0 $SP 2>/dev/null || { log "GPT server failed to start (port $PORT)"; exit 1; }
log "start: server=$GW:$PORT net=$NET cache=$CACHE seeds from $SEED_BASE, position=$POSITION, target_succ=$TARGET_SUCC (all workers)"

total_succ() {  # 全ワーカー合計の厳しい成功数（読むだけ。ほかのワーカーの動作には触れない）
  cat "$ROOT"/w*/*/episodes.jsonl 2>/dev/null | python3 -c '
import json,sys
print(sum(1 for l in sys.stdin if l.strip() and json.loads(l).get("target_in_hole")))'
}
next_seed() {
  cat "$W"/*/episodes.jsonl 2>/dev/null | python3 -c "
import json,sys
s=[json.loads(l)['seed'] for l in sys.stdin if l.strip()]
print(max(s)+1 if s else $SEED_BASE)"
}

FLOOR=0  # 同じシードで落ち続けるときに飛ばすための下限
while true; do
  [ -f "$ROOT/STOP" ] && { log "STOP file found"; break; }
  n=$(total_succ)
  [ "$n" -ge "$TARGET_SUCC" ] && { log "target reached: $n >= $TARGET_SUCC"; break; }
  s0=$(next_seed); [ "$s0" -lt "$FLOOR" ] && s0=$FLOOR
  log "chunk: seeds $s0..$((s0 + CHUNK - 1)) (total strict successes so far: $n)"
  NAME=so101sh_w$K NETWORK=$NET ISAAC_CACHE=$CACHE OUT_ROOT=$ROOT \
  DOCKER_EXTRA="--memory=${MEM:-7g}" \
    bash scripts/sim/run_in_teleop_docker.sh scripts/collect/gpt_collect_v150.py \
      --server "http://$GW:$PORT" --model "$MODEL" --effort "$EFFORT" \
      --target "$POSITION" --episodes "$CHUNK" --seed0 "$s0" --out "/workspace/out/w$K/$MODEL" \
      >> "$W/collect.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    if [ "$(next_seed)" -le "$s0" ]; then FLOOR=$((s0 + 1)); log "chunk rc=$rc with no episode recorded; skip seed $s0"; fi
    log "chunk exited rc=$rc (see collect.log); retry after 30 s"; sleep 30
  fi
done
log "done"
