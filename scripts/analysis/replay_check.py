"""記録した話を sim で再生し、最後に狙いの vial が穴に刺さっているかを、評価と同じ厳しい判定で確かめる（GPT は使わない）。

v150 の収集（gpt_collect_v150.py）と同じ環境・同じシードでリセットし、traj/<ep>/frames.jsonl の関節指令 q_cmd を
1 フレーム（制御 1 ステップ = 2 ティック）ずつ流す。再現できたかは、再生中の関節角と記録の関節角のずれで見る。

    bash scripts/sim/run_in_teleop_docker.sh scripts/analysis/replay_check.py \\
        --list /workspace/out/replay/list.json --out /workspace/out/replay/result.jsonl
list.json: [{"dir": "<outputs からの相対>", "episode": "ep_004000", "seed": 4000, "target_vial": "vial_2"}, ...]
"""
import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Lerobot-So101-Teleop-Vials-To-Rack-DR")
parser.add_argument("--list", required=True)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402

from so101_showharness.oracle.vials_to_rack import OracleParams  # noqa: E402
from so101_showharness.sim.scene import VialsRackScene  # noqa: E402
from so101_showharness.sim.workshop_backend import So101WorkshopBackend, make_env  # noqa: E402

OUT_ROOT = "/workspace/out"


def strict_in_hole(scene, v) -> bool:
    """dm05_eval_sim.py の in_hole と同じ。"""
    if v.up_z <= 0.85:
        return False
    bottom = v.pos[2] - scene.vial_length / 2
    return any(np.linalg.norm((v.pos - s)[:2]) < 0.015 and bottom < s[2] - 0.010 for s in scene.slots())


def main():
    items = json.load(open(args.list))
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["episode_key"] for l in open(args.out) if l.strip()}
    p = OracleParams()
    env = make_env(args.task, seed=items[0]["seed"], fixed_look=True, preplaced_vial=False, look="white_arm")
    be = So101WorkshopBackend(env, jaw_open_rad=p.jaw_open_rad)
    scene = None
    for it in items:
        key = f'{it["dir"]}/{it["episode"]}'
        if key in done:
            continue
        frames = [json.loads(l) for l in open(os.path.join(OUT_ROOT, it["dir"], "traj", it["episode"], "frames.jsonl"))]
        be.reset(it["seed"])
        scene = scene or VialsRackScene(be)
        # 最初の並び（収集と同じ決め方：ラックの中心に近い順に near / middle / far）
        rack_c = scene.rack_box()[0] / 2 + scene.rack_box()[1] / 2
        by_dist = sorted(scene.vials(), key=lambda v: float(np.linalg.norm((v.pos - rack_c)[:2])))
        start_pos = {v.name: p_ for v, p_ in zip(by_dist, ("near", "middle", "far"))}
        dev = []
        for f in frames:
            q = np.asarray(f["q_cmd"], dtype=np.float64)
            for _ in range(be.ticks_per_ctrl):
                be._tick(q)
            dev.append(float(np.abs(be.q_obs()[:5] - np.asarray(f["q_obs"][:5])).max()))
        vials = scene.vials()
        in_hole = [v.name for v in vials if strict_in_hole(scene, v)]
        final = {}
        for v in vials:  # 傾いていても穴に入っているかを後で判定するための、vial ごとの最後の状態
            d_xy = [float(np.linalg.norm((v.pos - s_)[:2])) for s_ in scene.slots()]
            k = int(np.argmin(d_xy))
            final[v.name] = {"start": start_pos[v.name], "up_z": round(float(v.up_z), 3),
                             "slot_xy_m": round(d_xy[k], 4),
                             "below_rim_m": round(float(scene.slots()[k][2] - (v.pos[2] - scene.vial_length / 2)), 4),
                             "center_below_rim_m": round(float(scene.slots()[k][2] - v.pos[2]), 4)}
        tgt = next(v for v in vials if v.name == it["target_vial"])
        bottom = float(tgt.pos[2] - scene.vial_length / 2)
        depth = max(float(s[2] - bottom) for s in scene.slots()
                    if np.linalg.norm((tgt.pos - s)[:2]) < 0.015) if any(
            np.linalg.norm((tgt.pos - s)[:2]) < 0.015 for s in scene.slots()) else None
        row = {"episode_key": key, **it, "frames": len(frames), "target_in_hole": it["target_vial"] in in_hole,
               "vials_in_hole": in_hole, "final": final, "target_up_z": round(float(tgt.up_z), 3),
               "target_depth_m": round(depth, 4) if depth is not None else None,
               "max_joint_dev_rad": round(max(dev), 4), "final_joint_dev_rad": round(dev[-1], 4)}
        with open(args.out, "a") as fo:
            fo.write(json.dumps(row) + "\n")
        print(f"[replay] {key} in_hole={row['target_in_hole']} up_z={row['target_up_z']} depth={row['target_depth_m']} "
              f"dev max={row['max_joint_dev_rad']} final={row['final_joint_dev_rad']}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
