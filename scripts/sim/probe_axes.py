"""Measure what each atomic token physically does on the SO-101 Workshop sim.

For every MV_* token: run it ``--reps`` times through Show-Harness's closed-loop
``AtomicExec`` from a mid-workspace pose, then undo it. Records achieved TCP displacement
(task frame), off-axis error, control steps used, and the external-camera frames before /
after, so the token -> image-direction mapping can be checked by eye. Also measures the
empty grasp / release widths and the drift while holding.

Phase 1 GO: every token moves along its axis with the right sign, and the per-token travel
varies by less than +-20 %.
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Lerobot-So101-Teleop-Vials-To-Rack-DR")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--step_m", type=float, default=0.02)
parser.add_argument("--reps", type=int, default=3)
parser.add_argument("--max_ctrl_steps", type=int, default=24)
parser.add_argument("--ticks_per_ctrl", type=int, default=2)
parser.add_argument("--no_gravity_bias", action="store_true")
parser.add_argument("--prepose", default="MV_DOWN,MV_DOWN,MV_FWD",
                    help="tokens run first to leave the reset pose (it sits near a wrist limit)")
parser.add_argument("--out", default="/workspace/out/probe_axes")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from scripts.trajectory.real2sim.atomic_tokenizer import MOVE_DIRS, OPPOSITE, AtomicExec  # noqa: E402
from so101_showharness.sim.workshop_backend import So101WorkshopBackend, make_env  # noqa: E402


def project(be, p_task):
    """Task-frame point -> external_D455 pixel (u right, v down)."""
    cam = be.u.scene["camera_external_D455"]
    k = cam.data.intrinsic_matrices[0].cpu().numpy()
    c_pos = cam.data.pos_w[0].cpu().numpy()
    from so101_showharness.kinematics import quat_wxyz_to_mat
    r_wc = quat_wxyz_to_mat(cam.data.quat_w_ros[0].cpu().numpy())  # ROS optical: z fwd, x right, y down
    base_w = be.robot.data.root_pos_w[0].cpu().numpy()
    p_w = base_w + np.asarray(p_task)  # task frame = world orientation at the base origin
    pc = r_wc.T @ (p_w - c_pos)
    uv = k @ pc
    return (uv[:2] / uv[2]).tolist()


def main():
    os.makedirs(args.out, exist_ok=True)
    env = make_env(args.task, seed=args.seed)
    be = So101WorkshopBackend(env, ticks_per_ctrl=args.ticks_per_ctrl,
                              use_gravity_bias=not args.no_gravity_bias)
    be.reset(args.seed)
    ex = AtomicExec(be, step_m=args.step_m, max_cmd_m=args.step_m,
                    max_ctrl_steps=args.max_ctrl_steps)
    result = {"args": vars(args), "reset_tcp": be.tcp_pos().tolist(),
              "reset_pitch_deg": float(np.degrees(be.pitch_ref)),
              "bias_after_reset": be.q_bias.tolist()}

    for tok in [t for t in args.prepose.split(",") if t]:
        ex.move(tok)
    ex.quiesce()
    result["probe_origin_tcp"] = be.tcp_pos().tolist()

    # hold drift
    p0 = be.tcp_pos()
    ex.hold(30)
    result["hold_30ctrl_drift_mm"] = float(np.linalg.norm(be.tcp_pos() - p0) * 1000)

    per_token = {}
    for tok in MOVE_DIRS:
        ext0, _ = be.raw_frames()
        moves = []
        start_first = be.tcp_pos()
        for _ in range(args.reps):
            start = be.tcp_pos()
            c0 = be.ctrl_step_count
            ex.move(tok)
            d = be.tcp_pos() - start
            axis = MOVE_DIRS[tok]
            along = float(d @ axis)
            moves.append({
                "delta_mm": (d * 1000).tolist(),
                "along_mm": along * 1000,
                "off_axis_mm": float(np.linalg.norm(d - along * axis) * 1000),
                "ctrl_steps": be.ctrl_step_count - c0,
            })
        ex.quiesce()
        ext1, _ = be.raw_frames()
        px0 = project(be, start_first)
        px1 = project(be, be.tcp_pos())
        img_dir = np.subtract(px1, px0)
        Image.fromarray(np.concatenate([ext0, ext1], axis=1)).save(
            os.path.join(args.out, f"{tok}_before_after.png"))
        for _ in range(args.reps):
            ex.move(OPPOSITE[tok])
        ex.quiesce()
        along = np.array([m["along_mm"] for m in moves])
        per_token[tok] = {
            "moves": moves,
            "along_mean_mm": float(along.mean()),
            "along_rel_spread": float((along.max() - along.min()) / max(along.mean(), 1e-9)),
            "off_axis_max_mm": float(max(m["off_axis_mm"] for m in moves)),
            "ctrl_steps_mean": float(np.mean([m["ctrl_steps"] for m in moves])),
            "external_px_start": px0,
            "external_px_delta": img_dir.tolist(),
        }
    result["tokens"] = per_token

    result["grasp_empty_width_m"] = ex.grasp()
    result["release_width_m"] = ex.release()
    result["bias_final"] = be.q_bias.tolist()

    all_along = np.array([m["along_mm"] for t in per_token.values() for m in t["moves"]])
    result["summary"] = {
        "along_mm_min": float(all_along.min()),
        "along_mm_max": float(all_along.max()),
        "along_mm_mean": float(all_along.mean()),
        "all_signs_correct": bool((all_along > 0).all()),
        "within_20pct": bool(((all_along / (args.step_m * 1000)) > 0.8).all()
                             and ((all_along / (args.step_m * 1000)) < 1.2).all()),
        "off_axis_max_mm": float(max(t["off_axis_max_mm"] for t in per_token.values())),
    }
    with open(os.path.join(args.out, "calibration.json"), "w") as f:
        json.dump(result, f, indent=2)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
