#!/usr/bin/env bash
# ローカル PC へ持っていく最小の束（実機クライアント＋必要な部品だけ）を作る。git も Show-Harness 全体（約 500MB）も要らない。
#   bash scripts/real/make_local_bundle.sh   → outputs/real_bundle.tgz
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
SH=${SHOW_HARNESS_ROOT:-/home/ubuntu/so101-showharness-deps/Show-Harness}
T=$(mktemp -d); B=$T/so101_real; R=$B/so101-showharness
mkdir -p "$R/src/so101_showharness/oracle" "$R/src/so101_showharness/real" "$R/configs" "$R/scripts/real" \
         "$B/Show-Harness/scripts/trajectory/real2sim"
cp src/so101_showharness/{__init__.py,controller.py,exec.py,kinematics.py,units.py,operator_prompt.py} "$R/src/so101_showharness/"
cp src/so101_showharness/oracle/vials_to_rack.py "$R/src/so101_showharness/oracle/"
cp src/so101_showharness/real/{__init__.py,so101_robot.py,robot_io.py,camera_fix.py} "$R/src/so101_showharness/real/"
cp configs/{so101_chain.json,so101_gripper_hull.json,motion_criteria.yaml,real_setup.json} "$R/configs/"
cp scripts/real/{run_dm05.py,run_showharness.py,camera_check.py,preflight.py,calibrate.py,camera_calib.py,requirements-local.txt} "$R/scripts/real/"
[ -d outputs/real_reference ] || python3 scripts/real/make_reference.py
cp -r outputs/real_reference "$R/real_reference"
python3 scripts/real/make_targets.py >/dev/null && cp -r outputs/real_targets "$R/print" && rm -f "$R/print/check_fix.png"
cp "$SH"/scripts/trajectory/real2sim/{__init__.py,atomic_tokenizer.py} "$B/Show-Harness/scripts/trajectory/real2sim/"
cp "$SH/LICENSE" "$B/Show-Harness/LICENSE"
echo "Show-Harness $(git -C "$SH" rev-parse --short HEAD)（showlab/Show-Harness、Apache-2.0）から atomic_tokenizer.py だけ" > "$B/Show-Harness/FROM.txt"
echo "so101-showharness $(git rev-parse --short HEAD)、$(date '+%F %T') に作成" > "$B/VERSION.txt"
tar -C "$T" -czf outputs/real_bundle.tgz so101_real
rm -rf "$T"
ls -la outputs/real_bundle.tgz
