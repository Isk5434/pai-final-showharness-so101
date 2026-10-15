"""特権オラクルで「バイアル → ラック」のデモを収集する（Workshop v1.0 sim、teleop-docker 内）。

1エピソードごとに3種類を同時に書く:
  <task>/ep_XXXXXX/   Show-Harness 形式（実行前フレーム + トークン）→ VLM の FT 用
  traj/ep_XXXXXX/     30 Hz 軌道（観測関節・指令関節・判断 ID・生カメラ動画）→ VLA 用
  video/ep_XXXXXX.mp4 閉ループ確認動画
成功かつ途中中断なしのエピソードだけを採用。それ以外は _rejected/ に移す（学習に混ぜない）。

    bash scripts/sim/run_in_teleop_docker.sh scripts/collect/oracle_collect.py \
        --episodes 5 --seed0 1000 --out /workspace/out/collect_test
"""

import argparse
import json
import os
import shutil
import subprocess
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Lerobot-So101-Teleop-Vials-To-Rack-DR")
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--seed0", type=int, default=1000)
parser.add_argument("--out", default="/workspace/out/collect")
parser.add_argument("--no_video", action="store_true", help="閉ループ確認動画を書かない")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from scripts.trajectory.real2sim.atomic_tokenizer import RolloutWriter  # noqa: E402
from so101_showharness.exec import So101AtomicExec  # noqa: E402
from so101_showharness.oracle.vials_to_rack import (  # noqa: E402
    INSTRUCTION, EpisodeInvalid, OracleParams, VialsToRackOracle)
from so101_showharness.recorder.trajectory import TrajectoryRecorder  # noqa: E402
from so101_showharness.recorder.video import EpisodeVideo  # noqa: E402
from so101_showharness.sim.scene import VialsRackScene  # noqa: E402
from so101_showharness.sim.workshop_backend import So101WorkshopBackend, make_env  # noqa: E402

REPO = "/workspace/so101-showharness"
TASK_DIR = INSTRUCTION.lower().replace(" ", "_")


def git_commit() -> str:
    """どのコードで収集したかを残す（コンテナに git が無くても .git から読む）。"""
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        pass
    try:
        head = open(f"{REPO}/.git/HEAD").read().strip()
        if head.startswith("ref: "):
            ref = head[5:]
            path = f"{REPO}/.git/{ref}"
            if os.path.exists(path):
                return open(path).read().strip()
            for line in open(f"{REPO}/.git/packed-refs"):
                if line.strip().endswith(ref):
                    return line.split()[0]
        return head
    except Exception:
        return "unknown"


def main():
    os.makedirs(args.out, exist_ok=True)
    crit = yaml.safe_load(open(f"{REPO}/configs/motion_criteria.yaml"))
    env = make_env(args.task, seed=args.seed0)
    params = OracleParams()
    be = So101WorkshopBackend(env, jaw_open_rad=params.jaw_open_rad)
    ex = So101AtomicExec(be, step_m=0.02)
    commit = git_commit()
    scene = None
    log_path = os.path.join(args.out, "episodes.jsonl")
    n_ok = 0
    for i in range(args.episodes):
        seed = args.seed0 + i
        name = f"ep_{seed:06d}"
        t0 = time.time()
        be.reset(seed)
        ex.lattice_origin, ex.hold_target, ex.grip_cmd = None, None, 1.0
        if scene is None:
            scene = VialsRackScene(be)
        table = scene.table_top
        be.hand_floor_task_z = table + crit["hand_floor_above_table_m"]
        be.workspace = {"min": [0.05, -0.30, table + 0.002], "max": [0.45, 0.25, 0.32]}

        rollout_dir = os.path.join(args.out, TASK_DIR, name)
        writer = RolloutWriter(rollout_dir)
        traj = TrajectoryRecorder(os.path.join(args.out, "traj", name))
        be.step_hooks = [traj.on_step]
        vid = None
        if not args.no_video:
            vid = EpisodeVideo(os.path.join(args.out, "video", f"{name}.mp4"))
            be.step_hooks.append(vid.on_step)
        rng = np.random.default_rng(seed)
        oracle = VialsToRackOracle(be, ex, scene, writer, rng, params)
        row = {"episode": name, "seed": seed, "instruction": INSTRUCTION, "commit": commit,
               "table_top_task_z": table,
               "vial_bottom_minus_table_mm": round((min(v.pos[2] for v in scene.vials())
                                                    - scene.vial_diameter / 2 - table) * 1000, 2)}
        try:
            res = oracle.run()
            row.update(res)
            row["valid"] = bool(res["success"])
            row["reason"] = None if res["success"] else "not_placed"
        except EpisodeInvalid as e:
            row.update({"success": False, "valid": False, "reason": str(e),
                        "tokens": len(oracle.log.tokens), "events": oracle.log.events})
        meta = {"instruction": INSTRUCTION, "task": INSTRUCTION, "task_text": INSTRUCTION,  # task_text: Show-Harness prepare_dataset.sh
                "seed": seed, "embodiment": "so101",
                "sim": "Sim-to-Real-SO-101-Workshop v1.0", "env": args.task, "commit": commit,
                "success": row["success"], "valid": row["valid"], "reason": row["reason"],
                "step_m": ex.step_m, "params": vars(params)}
        writer.close(meta)
        traj.close({"episode": name, **meta})
        if vid is not None:
            vid.close()
        be.step_hooks = []
        row["seconds"] = round(time.time() - t0, 1)
        if row["valid"]:
            n_ok += 1
        else:  # 学習に混ぜない
            for sub, src in ((TASK_DIR, rollout_dir), ("traj", os.path.join(args.out, "traj", name))):
                dst = os.path.join(args.out, "_rejected", sub, name)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)
        with open(log_path, "a") as f:
            f.write(json.dumps(row, default=float, ensure_ascii=False) + "\n")
        print(f"[収集] {name} 成功={row['success']} 採用={row['valid']} 理由={row['reason']} "
              f"トークン={row.get('tokens')} {row['seconds']}s", flush=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"episodes": args.episodes, "valid": n_ok, "commit": commit}, f)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
