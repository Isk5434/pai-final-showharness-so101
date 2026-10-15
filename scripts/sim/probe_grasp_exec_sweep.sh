#!/usr/bin/env bash
# probe_grasp_exec.py をずらし量ごとに回す（1 ずらし量 = 1 プロセス）。
#   bash scripts/sim/probe_grasp_exec_sweep.sh [出力名=probe_exec] [シード0=5000] [シード1=5009]
cd "$(dirname "$0")/../.."
TAG=${1:-probe_exec}; S0=${2:-5000}; S1=${3:-5009}
for sx in 0 -0.03 -0.05; do
  NAME=so101sh_probe_$$ bash scripts/sim/run_in_teleop_docker.sh scripts/sim/probe_grasp_exec.py \
    --shift "$sx" 0 --seeds "$S0" "$S1" --out "/workspace/out/$TAG/sx$sx" > "outputs/${TAG}_sx$sx.log" 2>&1
done
echo done > "outputs/$TAG.done"
