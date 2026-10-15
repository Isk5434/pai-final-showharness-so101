#!/usr/bin/env bash
# バイアルとラックを一緒に横（+y）へずらして、実際に掴めるかを測る。
cd "$(dirname "$0")/../.."
TAG=${1:-probe_exec_y}
for dy in 0.03 0.05; do
  NAME=so101sh_probey_$$ bash scripts/sim/run_in_teleop_docker.sh scripts/sim/probe_grasp_exec.py \
    --shift 0 "$dy" --rack_shift 0 "$dy" --seeds 5000 5009 --cases near:60,middle:50,far:50,far:40 \
    --out "/workspace/out/$TAG/dy$dy" > "outputs/${TAG}_dy$dy.log" 2>&1
done
echo done > "outputs/$TAG.done"
