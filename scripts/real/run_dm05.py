"""DM0.5 を実機 SO-101 で動かす（ローカル PC 側）。推論は AWS の dm05_server.py に SSH トンネル越しで頼む。

    # AWS 側：bash scripts/real/aws_serve.sh dm05 <学習名>
    # ローカル：ssh -N -L 8780:127.0.0.1:8780 <AWS>   （別の端末で張りっぱなし）
    python scripts/real/run_dm05.py --port /dev/ttyACM0 --robot_id follower_arm_1 --front 0 --wrist 2 \\
        --position near --dry_run            # まずは腕を動かさずに通しで確認
    python scripts/real/run_dm05.py ... --position near   # 本番

1 回の往復で n_action_steps 手（既定 15 手 = 0.5 秒分）を受け取り、30 Hz で順に送る（sim 評価の同期実行と同じ考え方）。
指令は SafeArm が「可動範囲」と「1 回あたり --max_step_rad」で必ず制限する。Ctrl-C でその場に止め、開始姿勢へゆっくり戻る。
各話の記録（関節・指令・推論時間の jsonl と front/wrist の動画）を --out に残す。
--mock で実機なし（sim の収集動画をカメラ代わり）に、サーバとの往復まで通しで試せる。
"""
import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from so101_showharness.real.robot_io import MockArm, SafeArm, apply_camera_fix, connect_so101, open_cameras  # noqa: E402

INSTR = {"near": "Pick up the vial closest to the rack and place it in the rack",
         "middle": "Pick up the middle vial and place it in the rack",
         "far": "Pick up the vial farthest from the rack and place it in the rack"}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--server", default="http://127.0.0.1:8780")
ap.add_argument("--port", help="SO-101 のシリアルポート（Linux: /dev/ttyACM0、Windows: COM3 など）")
ap.add_argument("--robot_id", default="follower_arm_1", help="lerobot-calibrate で付けた id")
ap.add_argument("--front", default="0", help="外部カメラの番号またはパス")
ap.add_argument("--wrist", default="2", help="手首カメラの番号またはパス")
ap.add_argument("--position", default="near", choices=list(INSTR))
ap.add_argument("--episodes", type=int, default=1)
ap.add_argument("--max_s", type=float, default=60.0, help="1 話の上限（秒）")
ap.add_argument("--max_step_rad", type=float, default=0.08, help="1 回の指令で各関節が動いてよい量")
ap.add_argument("--jpeg", type=int, default=95, help="送る画像の JPEG 品質（0 で PNG。学習は無圧縮）")
ap.add_argument("--setup", default=str(REPO / "configs/real_setup.json"))
ap.add_argument("--camera_fix", default=None, help="camera_calib.py align が作る camera_fix.json（front を sim のカメラの形に直す）")
ap.add_argument("--out", default=str(REPO / "outputs/real_dm05"))
ap.add_argument("--dry_run", action="store_true", help="腕には何も送らない（推論と記録だけ）")
ap.add_argument("--mock", action="store_true", help="実機なし。--front/--wrist に動画ファイルを渡す")
args = ap.parse_args()


def enc(img: np.ndarray) -> str:
    buf = io.BytesIO()
    if args.jpeg > 0:
        Image.fromarray(img).save(buf, format="JPEG", quality=args.jpeg)
    else:
        Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ask(payload: dict) -> dict:
    r = urllib.request.Request(f"{args.server}/chunk", data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=60).read())


def main():
    setup = json.load(open(args.setup))
    q_start = np.asarray(setup["start_q_rad"])
    robot = MockArm(q_start) if args.mock else connect_so101(args.port, args.robot_id)
    arm = SafeArm(robot, max_step_rad=args.max_step_rad, dry_run=args.dry_run)
    cams = apply_camera_fix(open_cameras(args.front, args.wrist, mock=args.mock), args.camera_fix, setup)
    task = INSTR[args.position]
    os.makedirs(args.out, exist_ok=True)
    print(f"[real_dm05] server={args.server} task='{task}' dry_run={args.dry_run} mock={args.mock}", flush=True)
    try:
        for ep in range(args.episodes):
            name = time.strftime("%Y%m%d_%H%M%S") + f"_{args.position}"
            if not args.dry_run:
                print("[real_dm05] 開始姿勢へ移動", flush=True)
                arm.move_to(q_start, duration_s=3.0)
            input(f"[real_dm05] 話 {ep + 1}/{args.episodes}：vial を並べたら Enter（Ctrl-C で中止）") if not args.mock else None
            log = open(os.path.join(args.out, f"{name}.jsonl"), "w")
            vids = {k: [] for k in ("front", "wrist")}
            t0, first, k, rtts = time.time(), True, 0, []
            while time.time() - t0 < args.max_s:
                q = arm.q()
                front, wrist = cams["front"].read(), cams["wrist"].read()
                ta = time.time()
                res = ask({"reset": first, "task": task, "state": q.tolist(),
                           "front_png_b64": enc(front), "wrist_png_b64": enc(wrist)})
                rtt = time.time() - ta
                rtts.append(rtt)
                first = False
                vids["front"].append(front)
                vids["wrist"].append(wrist)
                tick = time.monotonic()
                for a in res["actions"]:
                    sent = arm.send_q(a)
                    log.write(json.dumps({"k": k, "t": round(time.time() - t0, 3), "q_obs": q.tolist(),
                                          "q_policy": a, "q_sent": np.asarray(sent).round(5).tolist(),
                                          "infer_s": res["infer_s"], "rtt_s": round(rtt, 3)}) + "\n")
                    k += 1
                    tick += 1 / 30
                    time.sleep(max(0.0, tick - time.monotonic()))
                    q = arm.q()
            log.close()
            arm.hold()
            save_videos(name, vids)
            print(f"[real_dm05] {name}: {k} 手、往復 平均 {np.mean(rtts):.3f}s 最大 {np.max(rtts):.3f}s、"
                  f"制限に当たった指令 {arm.clipped} 回", flush=True)
    except KeyboardInterrupt:
        print("\n[real_dm05] 中止：その場で止める", flush=True)
        arm.hold()
    finally:
        if not args.dry_run and not args.mock:
            print("[real_dm05] 開始姿勢へ戻す", flush=True)
            arm.move_to(q_start, duration_s=4.0)
        for c in cams.values():
            c.close()
        arm.disconnect()


def save_videos(name: str, vids: dict) -> None:
    """推論に使ったコマ（約 2 Hz）を動画に残す（OpenCV だけで書く。ローカル PC に余計なものを入れないため）。"""
    import cv2

    for key, frames in vids.items():
        if not frames:
            continue
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(os.path.join(args.out, f"{name}_{key}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 2, (w, h))
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()


if __name__ == "__main__":
    main()
