"""FT 済み VLM の閉ループ sim 評価（teleop-docker 内）。推論は policy_server.py に HTTP で問い合わせる。

各エピソード: 評価用シード（学習に使っていない）で reset → オラクルと同じ開始姿勢（ピッチ 60°、
ロール −90°）へ移動（記録しない）→ 毎ステップ「prepared_pair 済みの2画像 + 直近の手」を送り、
返ってきた1トークンを So101AtomicExec で実行 → DONE か上限ステップで終了 → Workshop の
成功判定（vial_placed）。閉ループ動画と、ステップごとの記録（トークン・拒否・遅延）を残す。

    bash scripts/sim/run_in_teleop_docker.sh scripts/eval/sim_eval_vlm.py \\
        --server http://localhost:8765 --episodes 10 --seed0 9000 --out /workspace/out/eval_smoke
"""

import argparse
import base64
import io
import json
import os
import time
import urllib.request

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Lerobot-So101-Teleop-Vials-To-Rack-DR")
parser.add_argument("--server", default="http://localhost:8765")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--seed0", type=int, default=9000, help="評価用シード（学習は 1000〜/2000〜）")
parser.add_argument("--max_steps", type=int, default=80)
parser.add_argument("--out", default="/workspace/out/eval")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from core.action_units import MOVE_ATOMS, RT_ATOMS  # noqa: E402
from so101_showharness.controller import ROLL_CLOSING_HORIZONTAL  # noqa: E402
from so101_showharness.exec import So101AtomicExec  # noqa: E402
from so101_showharness.oracle.vials_to_rack import INSTRUCTION, OracleParams  # noqa: E402
from so101_showharness.recorder.video import EpisodeVideo  # noqa: E402
from so101_showharness.sim.scene import VialsRackScene  # noqa: E402
from so101_showharness.sim.workshop_backend import So101WorkshopBackend, make_env  # noqa: E402

REPO = "/workspace/so101-showharness"
HISTORY = set(MOVE_ATOMS) | set(RT_ATOMS)  # core/runners/mvtoken.py の HISTORY_ATOMS と同じ
HISTORY_MAX = 5


def png_b64(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ask(server: str, req: dict) -> dict:
    r = urllib.request.Request(f"{server}/act", data=json.dumps(req).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=120).read())


def main():
    os.makedirs(args.out, exist_ok=True)
    crit = yaml.safe_load(open(f"{REPO}/configs/motion_criteria.yaml"))
    p = OracleParams()
    env = make_env(args.task, seed=args.seed0)
    be = So101WorkshopBackend(env, jaw_open_rad=p.jaw_open_rad)
    ex = So101AtomicExec(be, step_m=0.02)
    scene = None
    n_ok = 0
    log_path = os.path.join(args.out, "episodes.jsonl")
    for i in range(args.episodes):
        seed = args.seed0 + i
        name = f"ep_{seed:06d}"
        be.reset(seed)
        ex.lattice_origin, ex.hold_target, ex.grip_cmd = None, None, 1.0
        scene = scene or VialsRackScene(be)
        be.hand_floor_task_z = scene.table_top + crit["hand_floor_above_table_m"]
        be.workspace = {"min": [0.05, -0.30, scene.table_top + 0.002], "max": [0.45, 0.25, 0.32]}
        vid = EpisodeVideo(os.path.join(args.out, "video", f"{name}.mp4"))
        be.step_hooks = [vid.on_step]
        begin = np.asarray(p.begin_world) - be.robot.data.root_pos_w[0].cpu().numpy()
        ex.goto_pose(begin, np.radians(p.grasp_pitch_deg), roll=ROLL_CLOSING_HORIZONTAL)
        recent, steps, success, t0 = [], [], False, time.time()
        for k in range(args.max_steps):
            av, wr = be.grab_frames()
            res = ask(args.server, {"task": INSTRUCTION, "recent_moves": recent,
                                    "gripper_closed": ex.gripper_closed,
                                    "agentview_png_b64": png_b64(av), "wrist_png_b64": png_b64(wr)})
            tok = res["token"]
            rec = {"step": k, "token": tok, "in_vocab": res["in_vocab"], "latency_s": res["latency_s"]}
            if tok == "DONE":
                steps.append(rec)
                break
            if not res["in_vocab"]:
                rec["blocked"] = "out_of_vocab"
            else:
                be.decision_id = k
                ex.run(tok)
                rec["blocked"] = ex.last_blocked
                if tok in HISTORY and not ex.last_blocked:
                    recent.insert(0, tok)
                    del recent[HISTORY_MAX:]
            steps.append(rec)
            if be.success():
                success = True
        ex.hold(30)
        success = success or be.success()
        vid.close()
        be.step_hooks = []
        n_ok += success
        row = {"episode": name, "seed": seed, "success": success, "steps": len(steps),
               "ended_with_done": bool(steps and steps[-1]["token"] == "DONE"),
               "blocked": sum(1 for s in steps if s.get("blocked")),
               "out_of_vocab": sum(1 for s in steps if not s["in_vocab"]),
               "mean_latency_s": round(float(np.mean([s["latency_s"] for s in steps])), 3) if steps else None,
               "seconds": round(time.time() - t0, 1), "tokens": [s["token"] for s in steps]}
        with open(log_path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[評価] {name} 成功={success} ステップ={len(steps)} 拒否={row['blocked']} "
              f"語彙外={row['out_of_vocab']} DONE終了={row['ended_with_done']}", flush=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"episodes": args.episodes, "success": n_ok, "rate": n_ok / max(args.episodes, 1)}, f)
    print(f"[評価] 成功率 {n_ok}/{args.episodes}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
