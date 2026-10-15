"""把持を「実際に動かして」確かめる: 位置 × 傾き × バイアルのずらし量で、掴んで 10 cm 持ち上げられるか。

probe_reach.py は運動学の判定（IK・関節限界・隙間）だけで、一番奥を 40° で 17/20 と出したが、
実際に動かすと指がバイアルを押して止まった（render_grasp_poses, 2026-10-08）。こちらは
goto（記録しない）で実際に降ろし、GRASP して持ち上げ、指の隙間で保持を判定する。GPT 不使用。
ずらし量は env 作成時の設定なので、1 プロセス 1 ずらし量。

    bash scripts/sim/run_in_teleop_docker.sh scripts/sim/probe_grasp_exec.py --shift -0.03 0 \\
        --cases far:40,far:50,middle:50,near:60 --seeds 5000 5009 --out /workspace/out/probe_exec/sx-0.03
"""
import argparse
import json
import os

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--shift", type=float, nargs=2, default=(0.0, 0.0), help="バイアル 3 本のずらし量 [m] (world x, y)")
ap.add_argument("--rack_shift", type=float, nargs=2, default=(0.0, 0.0), help="ラックのずらし量 [m] (world x, y)")
ap.add_argument("--cases", default="near:60,middle:50,far:40,far:50")
ap.add_argument("--seeds", type=int, nargs=2, default=(5000, 5009))
ap.add_argument("--out", default="/workspace/out/probe_exec")
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
HOLD_MIN_WIDTH_M = 0.015


def main():
    os.makedirs(args.out, exist_ok=True)
    env = make_env(seed=args.seeds[0], fixed_look=True, preplaced_vial=False, vial_shift=tuple(args.shift),
                   rack_shift=tuple(args.rack_shift))
    be = So101WorkshopBackend(env, jaw_open_rad=0.45)
    ex = So101AtomicExec(be)
    scene, rows = None, []
    for seed in range(args.seeds[0], args.seeds[1] + 1):
        for case in args.cases.split(","):
            pos, deg = case.split(":")
            be.reset(seed)
            ex.grip_cmd, ex.hold_target = 1.0, None
            scene = scene or VialsRackScene(be)
            be.hand_floor_task_z = scene.table_top + 0.003
            be.workspace = {"min": [0.05, -0.30, scene.table_top + 0.002], "max": [0.45, 0.25, 0.32]}
            rack_c = scene.rack_box()[0] / 2 + scene.rack_box()[1] / 2
            vial = sorted(scene.vials(), key=lambda v: float(np.linalg.norm((v.pos - rack_c)[:2])))[POSITIONS.index(pos)]
            pitch = np.radians(float(deg))
            row = {"shift": list(args.shift), "rack_shift": list(args.rack_shift), "seed": seed, "case": case, "vial": vial.name,
                   "reach_xy_m": round(float(np.linalg.norm(vial.pos[:2])), 3)}
            orc = VialsToRackOracle(be, ex, scene, writer=None, rng=np.random.default_rng(seed))
            try:
                roll, tcp, gap = orc._choose_side(vial, vial.pos, pitch)
            except EpisodeInvalid as e:
                row.update(stage="plan", ok=False, why=str(e))
                rows.append(row)
                print("[probe_exec]", json.dumps(row), flush=True)
                continue
            row["plan_gap_mm"] = round(gap * 1000, 1)
            ok = ex.goto_pose(tcp + [0, 0, 0.06], pitch, roll=roll, budget=600)
            ok = ok and ex.goto_pose(tcp, pitch, roll=roll, budget=600)
            ex.quiesce()
            row["grasp_pose_err_mm"] = round(float(np.linalg.norm(be.tcp_pos() - tcp)) * 1000, 1)
            if not ok:
                row.update(stage="reach", ok=False)
            else:
                ex.grasp()
                w = be.gripper_width()
                lifted = ex.goto_pose(tcp + [0, 0, 0.10], pitch, roll=roll, budget=600)
                ex.quiesce()
                w2 = be.gripper_width()
                row.update(width_mm=round(w * 1000, 1), width_after_lift_mm=round(w2 * 1000, 1), lift_ok=bool(lifted))
                row["ok"] = bool(w >= HOLD_MIN_WIDTH_M and lifted and w2 >= HOLD_MIN_WIDTH_M)
                row["stage"] = "done" if row["ok"] else ("grasp" if w < HOLD_MIN_WIDTH_M else "lift")
            if seed == args.seeds[0]:
                ext, wr = be.raw_frames()
                Image.fromarray(np.concatenate([ext, wr], axis=1)).save(os.path.join(args.out, f"{pos}_{deg}_seed{seed}.png"))
            rows.append(row)
            print("[probe_exec]", json.dumps(row), flush=True)
    with open(os.path.join(args.out, "exec.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    try:
        main()
    finally:
        os._exit(0)
