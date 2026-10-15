"""位置ごとの把持の傾きを sim の画像で見せる: 掴む位置・掴んで 10 cm 持ち上げた所を、外部カメラと手首カメラで撮る。

比較用に「一番奥を、これまでの 60° で」も試す（届かなければ、止まった所を撮る）。動きは記録しない
goto_pose（オラクルと同じ把持点: _choose_side）。GPT は不使用。

    bash scripts/sim/run_in_teleop_docker.sh scripts/sim/render_grasp_poses.py --seed 5002 --out /workspace/out/grasp_poses
"""
import argparse
import json
import os

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--seed", type=int, default=5002)
ap.add_argument("--cases", default="near:60,middle:50,far:40,far:60")
ap.add_argument("--out", default="/workspace/out/grasp_poses")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from so101_showharness.exec import So101AtomicExec  # noqa: E402
from so101_showharness.oracle.vials_to_rack import EpisodeInvalid, VialsToRackOracle  # noqa: E402
from so101_showharness.sim.scene import VialsRackScene  # noqa: E402
from so101_showharness.sim.workshop_backend import So101WorkshopBackend, make_env  # noqa: E402

POSITIONS = ("near", "middle", "far")


def snap(be, path):
    ext, wr = be.raw_frames()
    Image.fromarray(np.concatenate([ext, wr], axis=1)).save(path)


def main():
    os.makedirs(args.out, exist_ok=True)
    env = make_env(seed=args.seed, fixed_look=True, preplaced_vial=False)
    be = So101WorkshopBackend(env, jaw_open_rad=0.45)
    ex = So101AtomicExec(be)
    scene, report = None, []
    for case in args.cases.split(","):
        pos, deg = case.split(":")
        be.reset(args.seed)
        ex.grip_cmd, ex.hold_target = 1.0, None
        scene = scene or VialsRackScene(be)
        be.hand_floor_task_z = scene.table_top + 0.003
        be.workspace = {"min": [0.05, -0.30, scene.table_top + 0.002], "max": [0.45, 0.25, 0.32]}
        rack_c = scene.rack_box()[0] / 2 + scene.rack_box()[1] / 2
        vial = sorted(scene.vials(), key=lambda v: float(np.linalg.norm((v.pos - rack_c)[:2])))[POSITIONS.index(pos)]
        pitch = np.radians(float(deg))
        orc = VialsToRackOracle(be, ex, scene, writer=None, rng=np.random.default_rng(args.seed))
        row = {"case": case, "vial": vial.name}
        try:
            roll, tcp, gap = orc._choose_side(vial, vial.pos, pitch)
            row["plan"] = f"ok (gap {gap * 1000:.0f} mm)"
        except EpisodeInvalid as e:
            # 計画が通らない: 届く傾き（40°）で求めた把持点へ、この傾きのまま向かってみる
            roll, tcp, _ = orc._choose_side(vial, vial.pos, np.radians(40.0))
            row["plan"] = f"refused: {e}"
        row["approach_ok"] = bool(ex.goto_pose(tcp + [0, 0, 0.06], pitch, roll=roll, budget=600))
        row["grasp_pose_ok"] = bool(ex.goto_pose(tcp, pitch, roll=roll, budget=600))
        ex.quiesce()
        row["tcp_err_mm"] = round(float(np.linalg.norm(be.tcp_pos() - tcp)) * 1000, 1)
        snap(be, os.path.join(args.out, f"{pos}_{deg}_grasp.png"))
        if row["grasp_pose_ok"]:
            ex.grasp()
            row["holding_width_mm"] = round(be.gripper_width() * 1000, 1)
            row["lift_ok"] = bool(ex.goto_pose(tcp + [0, 0, 0.10], pitch, roll=roll, budget=600))
            ex.quiesce()
            snap(be, os.path.join(args.out, f"{pos}_{deg}_lift.png"))
        report.append(row)
        print("[render_grasp_poses]", json.dumps(row), flush=True)
    with open(os.path.join(args.out, "report.jsonl"), "w") as f:
        for r in report:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    try:
        main()
    finally:
        os._exit(0)
