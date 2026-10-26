"""operator_prompt.direction_table（real_setup.json のカメラ）が、v150 の sim カメラでの計算と同じ表を出すかを確かめる。

    bash scripts/sim/run_in_teleop_docker.sh tests/sim/check_direction_table.py
"""
import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402

from so101_showharness.controller import ROLL_CLOSING_HORIZONTAL  # noqa: E402
from so101_showharness.exec import So101AtomicExec  # noqa: E402
from so101_showharness.kinematics import quat_wxyz_to_mat  # noqa: E402
from so101_showharness.operator_prompt import Camera, cameras_from_setup, direction_table  # noqa: E402
from so101_showharness.oracle.vials_to_rack import OracleParams  # noqa: E402
from so101_showharness.sim.workshop_backend import So101WorkshopBackend, make_env  # noqa: E402


def sim_cam(be, key):
    c = be.u.scene[key]
    return Camera(c.data.intrinsic_matrices[0].cpu().numpy(),
                  c.data.pos_w[0].cpu().numpy() - be.robot.data.root_pos_w[0].cpu().numpy(),
                  quat_wxyz_to_mat(c.data.quat_w_ros[0].cpu().numpy()))


p = OracleParams()
env = make_env("Lerobot-So101-Teleop-Vials-To-Rack-DR", seed=100000, fixed_look=True, preplaced_vial=False, look="white_arm")
be = So101WorkshopBackend(env, jaw_open_rad=p.jaw_open_rad)
ex = So101AtomicExec(be, step_m=0.02)
setup = json.load(open("/workspace/so101-showharness/configs/real_setup.json"))
bad = 0
rel, rel_b, kin_err, per_body = [], [], [], {}
for seed in (100000, 100001, 100002):
    be.reset(seed)
    begin = np.asarray(p.begin_world) - be.robot.data.root_pos_w[0].cpu().numpy()
    ex.goto_pose(begin, np.radians(p.grasp_pitch_deg), roll=ROLL_CLOSING_HORIZONTAL)
    for tok in [None, "MV_FWD", "MV_DOWN", "MV_LEFT", "RT_PITCH_FWD", "MV_UP"]:
        if tok:
            ex.run(tok)
        w = sim_cam(be, "camera_ego")
        g = be.kin.link_poses(be.q_obs())["gripper"]
        rg, pg = be._r_world_base() @ g[:3, :3], be._r_world_base() @ g[:3, 3]
        rel.append(np.concatenate([rg.T @ (w.c - pg), (rg.T @ w.r).ravel()]))
        bi = be.robot.body_names.index("gripper")
        bp = be.robot.data.body_pos_w[0, bi].cpu().numpy() - be.robot.data.root_pos_w[0].cpu().numpy()
        br = quat_wxyz_to_mat(be.robot.data.body_quat_w[0, bi].cpu().numpy())
        rel_b.append(np.concatenate([br.T @ (w.c - bp), (br.T @ w.r).ravel()]))
        for bn in be.robot.body_names:
            j = be.robot.body_names.index(bn)
            p_ = be.robot.data.body_pos_w[0, j].cpu().numpy() - be.robot.data.root_pos_w[0].cpu().numpy()
            r_ = quat_wxyz_to_mat(be.robot.data.body_quat_w[0, j].cpu().numpy())
            per_body.setdefault(bn, []).append(np.concatenate([r_.T @ (w.c - p_), (r_.T @ w.r).ravel()]))
        kin_err.append([float(np.linalg.norm(bp - pg)), float(np.abs(br - rg).max())])
        a = direction_table(be, sim_cam(be, "camera_external_D455"), sim_cam(be, "camera_ego"))
        b = direction_table(be, *cameras_from_setup(setup, be))
        same = a == b
        bad += not same
        print(f"[dir] seed={seed} after={tok} same={same}", flush=True)
        if not same:
            print(a, "\n---\n", b, flush=True)
print(f"[dir] mismatches={bad}", flush=True)
rel = np.asarray(rel)
print("[rel] mean", json.dumps(rel.mean(0).round(6).tolist()), flush=True)
print("[rel] std max", float(rel.std(0).max()), flush=True)
rel_b = np.asarray(rel_b)
print("[relb] mean", json.dumps(rel_b.mean(0).round(6).tolist()), "std max", float(rel_b.std(0).max()), flush=True)
for bn, v in per_body.items():
    v = np.asarray(v)
    print("[body]", bn, "std max", round(float(v.std(0).max()), 6), "mean", json.dumps(v.mean(0).round(6).tolist()), flush=True)
print("[kin] pos err max", float(np.max(np.asarray(kin_err)[:, 0])), "rot err max", float(np.max(np.asarray(kin_err)[:, 1])), flush=True)
env.close()
app.close()
