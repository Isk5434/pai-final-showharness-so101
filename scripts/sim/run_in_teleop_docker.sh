#!/usr/bin/env bash
# Run a script from this repo inside teleop-docker against the Workshop v1.0 env, headless.
#   bash scripts/sim/run_in_teleop_docker.sh scripts/sim/smoke_workshop_env.py [args...]
# Show-Harness is mounted read-only and put on PYTHONPATH (nothing is pip-installed into the image).
# Outputs land in $OUT_ROOT (default: <repo>/outputs, git-ignored).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
WORKSHOP=${SO101_WORKSHOP_ROOT:-/home/ubuntu/sim2real/Sim-to-Real-SO-101-Workshop-v1.0}
SHOW_HARNESS=${SHOW_HARNESS_ROOT:-/home/ubuntu/so101-showharness-deps/Show-Harness}
CACHE=${ISAAC_CACHE:-/home/ubuntu/sim2real/isaac-sim-cache}
OUT_ROOT=${OUT_ROOT:-$REPO/outputs}
IMAGE=${IMAGE:-teleop-docker:latest}
NAME=${NAME:-so101sh_$(date +%H%M%S)}

[ $# -ge 1 ] || { echo "usage: $0 <script.py> [args...]" >&2; exit 2; }
SCRIPT=$1; shift
# Quote every argument for the inner `bash -lc`: an unquoted "a|b" there is a pipe (it broke
# a run once: --grid_world "x|y|z" piped python's stdout into nonexistent commands).
ARGS=""
[ $# -gt 0 ] && ARGS=$(printf '%q ' "$@")  # no args: printf would emit one empty '' argument
mkdir -p "$OUT_ROOT"

docker run --rm --name "$NAME" --gpus all --network=host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e PYTHONPATH=/workspace/so101-showharness/src:/workspace/Show-Harness \
  -v "$WORKSHOP/docker/env:/root/env" \
  -v "$WORKSHOP:/workspace/Sim-to-Real-SO-101-Workshop" \
  -v "$REPO:/workspace/so101-showharness" \
  -v "$SHOW_HARNESS:/workspace/Show-Harness:ro" \
  -v "$OUT_ROOT:/workspace/out" \
  -v "$CACHE/kit:/isaac-sim/kit/cache:rw" \
  -v "$CACHE/ov:/root/.cache/ov:rw" \
  -v "$CACHE/pip:/root/.cache/pip:rw" \
  -v "$CACHE/glcache:/root/.cache/nvidia/GLCache:rw" \
  -v "$CACHE/computecache:/root/.nv/ComputeCache:rw" \
  -v "$CACHE/logs:/root/.nvidia-omniverse/logs:rw" \
  -v "$CACHE/data:/root/.local/share/ov/data:rw" \
  -v "$CACHE/documents:/root/Documents:rw" \
  "$IMAGE" \
  bash -lc "
    cd /workspace/Sim-to-Real-SO-101-Workshop
    python3 /workspace/so101-showharness/$SCRIPT --headless $ARGS
    rc=\$?
    chown -R $(id -u):$(id -g) /workspace/out || true
    exit \$rc
  "
