"""DM05 を Workshop sim で閉ループ評価する（teleop-docker 内）。推論はホストの dm05_server.py。

位置（near / middle / far）ごとに N 話。1 制御ステップ（30 Hz = 2 tick）ごとに、生の外部・手首画像と関節角を
サーバへ送り、返った絶対関節指令をそのまま 2 tick 送る（学習データの q_cmd と同じ意味・同じ周期）。

成功は 2 通り記録する:
  strict   … 狙った位置のバイアルが立って（軸の鉛直成分 > 0.85）、穴の入口中心から 15 mm 以内、下端が入口より
              10 mm 以上下（= 穴に刺さっている）。これが本当の成功。
  sim      … Workshop の vial_placed（離した瞬間に 45° 以内・ラックの外形内なら成功。甘い）
strict を 1 秒保ったら終了。上限 --max_s 秒。観測関節角が --stall_s 秒のあいだ全関節 --stall_rad 以内しか
動かなければ「停止」として打ち切る（失敗した方策は姿勢が固まったまま上限まで回るため）。stop に終わり方を残す。
見た目と配置は学習データの大半（150 話収集）に合わせる。

    bash scripts/sim/run_in_teleop_docker.sh scripts/eval/dm05_eval_sim.py --episodes 30 --seed0 7000 \\
        --positions near middle far --out /workspace/out/eval_dm05
"""
import argparse
import base64
import io
import json
import os
import time
import urllib.request

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--server", default="http://localhost:8780")
ap.add_argument("--episodes", type=int, default=30)
ap.add_argument("--seed0", type=int, default=7000)
ap.add_argument("--positions", nargs="+", default=["near", "middle", "far"])
ap.add_argument("--max_s", type=float, default=60.0)
ap.add_argument("--stall_s", type=float, default=8.0, help="0 で停止判定なし")
ap.add_argument("--stall_rad", type=float, default=0.02)
ap.add_argument("--look", default="white_arm")
ap.add_argument("--out", default="/workspace/out/eval_dm05")
ap.add_argument("--label", default="DM05", help="動画に焼き込むモデル名")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from so101_showharness.recorder.video import EpisodeVideo  # noqa: E402
from so101_showharness.sim.scene import VialsRackScene  # noqa: E402
from so101_showharness.sim.workshop_backend import So101WorkshopBackend, make_env  # noqa: E402

INSTR = {"near": "Pick up the vial closest to the rack and place it in the rack",
         "middle": "Pick up the middle vial and place it in the rack",
         "far": "Pick up the vial farthest from the rack and place it in the rack"}
POSITIONS = ("near", "middle", "far")


def png_b64(a: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(a[..., :3]).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def act(payload: dict) -> dict:
    r = urllib.request.Request(f"{args.server}/act", data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=120).read())


def in_hole(scene, v) -> bool:
    if v.up_z <= 0.85:
        return False
    bottom = v.pos[2] - scene.vial_length / 2
    return any(np.linalg.norm((v.pos - s)[:2]) < 0.015 and bottom < s[2] - 0.010 for s in scene.slots())


def main():
    os.makedirs(os.path.join(args.out, "video"), exist_ok=True)
    env = make_env(seed=args.seed0, fixed_look=True, preplaced_vial=False, look=args.look)
    be = So101WorkshopBackend(env, jaw_open_rad=0.45)
    scene, log = None, open(os.path.join(args.out, "episodes.jsonl"), "a")
    for pos in args.positions:
        for i in range(args.episodes):
            seed = args.seed0 + i
            name = f"{pos}_{seed}"
            be.reset(seed)
            scene = scene or VialsRackScene(be)
            rack_c = scene.rack_box()[0] / 2 + scene.rack_box()[1] / 2
            target = sorted(scene.vials(), key=lambda v: float(np.linalg.norm((v.pos - rack_c)[:2])))[POSITIONS.index(pos)].name
            vid = EpisodeVideo(os.path.join(args.out, "video", f"{name}.mp4"), task=f"[{args.label}] {INSTR[pos]}")
            be.step_hooks = [vid.on_step]
            t0, infer, held_strict, strict, first = time.time(), [], 0, False, True
            n_ctrl = int(args.max_s * 30)
            stall_n, q_hist, stop = int(args.stall_s * 2), [], "timeout"  # q_hist は 0.5 秒ごと
            for k in range(n_ctrl):
                ext, wr = be.raw_frames()
                res = act({"reset": first, "task": INSTR[pos], "state": be.q_obs().tolist(),
                           "front_png_b64": png_b64(ext), "wrist_png_b64": png_b64(wr)})
                first = False
                infer.append(res["infer_s"])
                q = np.asarray(res["action"], dtype=np.float64)
                be.annotation = {"token": "DM05", "phase": f"t={k / 30:.1f}s"}
                for _ in range(be.ticks_per_ctrl):
                    be._tick(q)
                if k % 15 == 0:
                    strict = in_hole(scene, scene.vial(target))
                    held_strict = held_strict + 15 if strict else 0
                    if held_strict >= 30:
                        stop = "success"
                        break
                    q_hist = (q_hist + [be.q_obs()])[-(stall_n + 1):]
                    if stall_n and len(q_hist) > stall_n and not strict and \
                            float(np.ptp(np.asarray(q_hist), axis=0).max()) < args.stall_rad:
                        stop = "stall"
                        break
            vid.close()
            be.step_hooks = []
            any_in = [v.name for v in scene.vials() if in_hole(scene, v)]
            row = {"position": pos, "seed": seed, "target_vial": target, "strict_success": bool(strict),
                   "vials_in_hole": any_in, "sim_success": be.success(), "stop": stop, "ctrl_steps": k + 1,
                   "mean_infer_s": round(float(np.mean(infer)), 3), "wall_s": round(time.time() - t0, 1)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print("[dm05_eval]", json.dumps(row), flush=True)
    log.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        os._exit(0)
