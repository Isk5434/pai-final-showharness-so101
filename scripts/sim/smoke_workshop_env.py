"""Phase 0 smoke test: drive the Sim-to-Real SO-101 Workshop (v1.0) env from an external script.

Runs inside teleop-docker (see scripts/sim/run_in_teleop_docker.sh). Checks the Phase 0 GO
criteria: the env resets and steps, joints follow commanded absolute positions, and the
ego / external_D455 cameras plus joint state can be read. Writes summary.json and
first/last camera frames to --out.
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Lerobot-So101-Teleop-Vials-To-Rack-DR")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--steps", type=int, default=120, help="env steps per phase (60 Hz)")
parser.add_argument("--probe_joint", default="Rotation")
parser.add_argument("--probe_rad", type=float, default=0.3)
parser.add_argument("--out", default="/workspace/out/smoke")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from PIL import Image  # noqa: E402

import sim_to_real_so101.tasks  # noqa: E402,F401


def describe(x):
    if isinstance(x, dict):
        return {k: describe(v) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    return str(type(x))


def save_frames(obs, tag):
    for key, val in obs.get("visual", {}).items():
        if not key.startswith("rgb"):
            continue
        img = val[0].detach().cpu().numpy()
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        Image.fromarray(img[..., :3]).save(os.path.join(args.out, f"{tag}_{key}.png"))


def main():
    os.makedirs(args.out, exist_ok=True)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.seed = args.seed
    env = gym.make(args.task, cfg=cfg)
    u = env.unwrapped
    robot = u.scene["robot"]
    joint_names = list(robot.data.joint_names)

    obs, _ = env.reset()
    save_frames(obs, "reset")
    q0 = obs["policy"]["joint_pos_obs"][0].clone()

    # hold: command the reset pose, joints should stay put
    action = q0.unsqueeze(0).clone()
    for _ in range(args.steps):
        obs, *_ = env.step(action)
    q_hold = obs["policy"]["joint_pos_obs"][0].clone()

    # probe: offset one joint, it should move toward the target
    j = joint_names.index(args.probe_joint)
    action[0, j] += args.probe_rad
    for _ in range(args.steps):
        obs, *_ = env.step(action)
    q_probe = obs["policy"]["joint_pos_obs"][0].clone()
    save_frames(obs, "probe")

    summary = {
        "task": args.task,
        "step_dt_s": float(u.step_dt),
        "physics_dt_s": float(u.physics_dt),
        "decimation": int(u.cfg.decimation),
        "episode_length_s": float(u.cfg.episode_length_s),
        "joint_names": joint_names,
        "action_space": str(env.action_space),
        "obs": describe(obs),
        "cameras": {
            k[len("camera_"):]: {"height": getattr(u.scene.cfg, k).height, "width": getattr(u.scene.cfg, k).width}
            for k in vars(u.scene.cfg) if k.startswith("camera_")
        },
        "q_reset": q0.tolist(),
        "hold_max_abs_drift_rad": float((q_hold - q0).abs().max()),
        "probe": {
            "joint": args.probe_joint,
            "target_rad": float(action[0, j]),
            "reached_rad": float(q_probe[j]),
            "error_rad": float((action[0, j] - q_probe[j]).abs()),
            "other_joints_max_abs_change_rad": float(
                torch.cat([(q_probe - q_hold)[:j], (q_probe - q_hold)[j + 1 :]]).abs().max()
            ),
        },
    }
    if "ee_frame_state" in obs["policy"]:
        summary["ee_frame_state_after_probe"] = obs["policy"]["ee_frame_state"][0].tolist()
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[SMOKE] " + json.dumps(summary["probe"]) + f" hold_drift={summary['hold_max_abs_drift_rad']:.4f}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
