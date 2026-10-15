"""Where can the SO-101 hold a given hand orientation? (pure numpy, no sim)

Grid over the task region (task frame = world orientation at the robot base, +X toward the
vials). For each point and each candidate tool pitch, solve IK and record whether it is
reached with margin. A point counts as GOOD only if all six 2 cm neighbours are also
reachable with the same orientation: that is what executing a token there requires.

Two orientation modes:
  yaw   pitch + closing-axis yaw held (across the vials, i.e. along world Y), via Wrist_Roll
  roll  pitch held, Wrist_Roll fixed (yaw then follows the base rotation)

    python scripts/analysis/reachability.py --out outputs/reachability.json
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from so101_showharness.kinematics import So101Kinematics  # noqa: E402

R_WB = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # base yawed +90 deg in world
BASE_W = np.array([-0.05, 0.0, 0.0])
SEEDS = [np.array(s) for s in (
    [-0.2736, -0.6109, -0.0745, 1.5148, -1.6034, 0.0],
    [0.0, 0.3, -0.4, 1.4, -1.57, 0.0],
    [0.0, -0.2, 0.6, 0.9, -1.57, 0.0],
    [0.0, 0.6, -1.0, 1.2, -1.57, 0.0],
)]
XS_W = np.arange(0.10, 0.34, 0.03)
YS_W = np.arange(-0.22, 0.15, 0.03)
ZS = np.arange(0.04, 0.27, 0.03)
STEPS = np.vstack([np.eye(3), -np.eye(3)]) * 0.02


def task_to_base(p):
    return R_WB.T @ p


def reach(kin, seeds, p_task, pitch, yaw, tol_m=5e-4, tol_rad=np.radians(0.5), margin=0.03,
          iters=60):
    for s in seeds:
        if s is None:
            continue
        q, err = kin.solve(s, task_to_base(p_task), pitch, yaw, iters=iters)
        if (np.linalg.norm(err[:3]) < tol_m and np.all(np.abs(err[3:]) < tol_rad)
                and kin.joint_limit_margin(q)[:5].min() > margin):
            return q
    return None


def one(job):
    pdeg, mode = job
    kin = So101Kinematics()
    pitch = np.radians(pdeg)
    yaw = 0.0 if mode == "yaw" else None  # closing axis along world Y == base X
    grid = []
    for xw in XS_W:
        for yw in YS_W:
            warm = None  # walk up the z column, seeding each solve with the one below
            for z in ZS:
                p_task = np.array([xw, yw, z]) - BASE_W
                q = reach(kin, [warm] + SEEDS, p_task, pitch, yaw)
                warm = q if q is not None else warm
                good = q is not None and all(
                    reach(kin, [q], p_task + d, pitch, yaw, iters=30) is not None for d in STEPS)
                grid.append({"world": [round(xw, 3), round(yw, 3), round(z, 3)],
                             "reach": q is not None, "good": bool(good)})

    def frac(sel):
        pts = [g for g in grid if sel(g["world"])]
        return round(sum(g["good"] for g in pts) / max(len(pts), 1), 3), len(pts)

    res = {
        "grid": grid,
        "good_vials_zone(z 0.06-0.16)": frac(lambda w: 0.18 <= w[0] <= 0.28 and -0.18 <= w[1] <= 0.02 and 0.06 <= w[2] <= 0.16),
        "good_rack_zone(z 0.14-0.24)": frac(lambda w: 0.14 <= w[0] <= 0.24 and 0.04 <= w[1] <= 0.12 and 0.14 <= w[2] <= 0.24),
        "good_all": frac(lambda w: True),
    }
    print(f"{pdeg:.0f}/{mode}", {k: v for k, v in res.items() if k != "grid"}, flush=True)
    return f"{pdeg:.0f}/{mode}", res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/reachability.json")
    ap.add_argument("--jobs", default="75/yaw,60/yaw,47/yaw,90/roll,75/roll,60/roll,47/roll")
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()
    jobs = [(float(j.split("/")[0]), j.split("/")[1]) for j in args.jobs.split(",")]
    with Pool(args.procs) as pool:
        out = dict(pool.map(one, jobs))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
