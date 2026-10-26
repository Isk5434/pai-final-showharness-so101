"""実機の前にローカル PC で流す点検（何も動かさない）。Windows / Linux どちらでも動く。

    python scripts/real/preflight.py [--robot_id follower_arm_1]

1. Python と必要なパッケージ（lerobot 0.4.1 など）
2. シリアルポート（SO-101 の候補）
3. カメラ（番号 0〜5 を 640x480 で開けるか。開けたものはサムネイルを preflight_cam<N>.jpg に保存）
4. キャリブレーションのファイル
5. SSH トンネル（127.0.0.1:8780 = DM0.5、:8770 = Show-Harness）
"""
import argparse
import importlib
import os
import socket
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--robot_id", default="follower_arm_1")
ap.add_argument("--max_cam", type=int, default=6)
ap.add_argument("--server_test", action="store_true",
                help="トンネル越しに本当に推論させる（参照画像を送る。Show-Harness は GPT を 1 回呼ぶ）")
args = ap.parse_args()
OK, NG = "  [OK] ", "  [NG] "

print("== 1. Python とパッケージ")
print(f"{OK if sys.version_info[:2] >= (3, 10) else NG}Python {sys.version.split()[0]}（3.10 以上）")
for mod, want in (("lerobot", "0.4.1"), ("cv2", None), ("numpy", None), ("PIL", None), ("yaml", None), ("serial", None)):
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "")
        if mod == "lerobot":
            import importlib.metadata as md
            v = md.version("lerobot")
        print(f"{OK if (want is None or v == want) else NG}{mod} {v}" + (f"（{want} が必要）" if want and v != want else ""))
    except Exception as e:  # noqa: BLE001
        print(f"{NG}{mod} が読めない: {e}")

print("== 2. シリアルポート")
try:
    from serial.tools import list_ports
    ports = list(list_ports.comports())
    for p in ports:
        print(f"{OK}{p.device}  {p.description}")
    if not ports:
        print(f"{NG}見つからない（USB と電源を確認）")
except Exception as e:  # noqa: BLE001
    print(f"{NG}{e}")

print("== 3. カメラ（640x480 で開けるもの）")
try:
    import cv2
    for i in range(args.max_cam):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(i)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ok, f = cap.read()
        cap.release()
        if ok:
            cv2.imwrite(f"preflight_cam{i}.jpg", f)
            good = f.shape[:2] == (480, 640)
            print(f"{OK if good else NG}番号 {i}: {f.shape[1]}x{f.shape[0]} → preflight_cam{i}.jpg" + ("" if good else "（640x480 にならない）"))
    print("     → 画像を見て、外部（front）と手首（wrist）の番号を控える")
except Exception as e:  # noqa: BLE001
    print(f"{NG}{e}")

print("== 4. キャリブレーション")
calib = Path(os.environ.get("HF_LEROBOT_CALIBRATION", Path.home() / ".cache/huggingface/lerobot/calibration")) \
    / "robots" / "so101_follower" / f"{args.robot_id}.json"
print(f"{OK if calib.exists() else NG}{calib}" + ("" if calib.exists() else "（無い → lerobot-calibrate を実行）"))

print("== 5. SSH トンネル")
for port, name in ((8780, "DM0.5"), (8770, "Show-Harness")):
    s = socket.socket()
    s.settimeout(2)
    try:
        s.connect(("127.0.0.1", port))
        print(f"{OK}{name} 127.0.0.1:{port} につながる")
    except OSError:
        print(f"{NG}{name} 127.0.0.1:{port} につながらない（トンネルと AWS のサーバを確認）")
    finally:
        s.close()

if args.server_test:
    import base64
    import io
    import json
    import time
    import urllib.request

    from PIL import Image

    root = Path(__file__).resolve().parents[2]
    setup = json.load(open(root / "configs/real_setup.json"))
    imgs = {k: Image.open(root / "real_reference" / f"{k}.png").convert("RGB") for k in ("front", "wrist")}

    def b64(im, fmt):
        buf = io.BytesIO()
        im.save(buf, format=fmt, **({"quality": 95} if fmt == "JPEG" else {}))
        return base64.b64encode(buf.getvalue()).decode()

    def post(url, payload, timeout):
        t0 = time.time()
        r = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(r, timeout=timeout).read()), time.time() - t0

    print("== 6. 推論の通し試験（ロボットは動かさない）")
    try:
        for i in range(3):
            res, dt = post("http://127.0.0.1:8780/chunk", {
                "reset": i == 0, "task": "Pick up the vial closest to the rack and place it in the rack",
                "state": setup["start_q_rad"], "front_png_b64": b64(imgs["front"], "JPEG"),
                "wrist_png_b64": b64(imgs["wrist"], "JPEG")}, 60)
            print(f"{OK}DM0.5 {len(res['actions'])} 手を受信  往復 {dt:.2f}s（うち推論 {res['infer_s']:.2f}s）")
    except Exception as e:  # noqa: BLE001
        print(f"{NG}DM0.5: {e}")
    try:
        res, dt = post("http://127.0.0.1:8770/decide", {
            "model": "gpt-6-astra", "effort": "low", "episode": "preflight", "step": 0,
            "prompt": "Connectivity test. Answer with exactly one token on the last line: MV_UP",
            "external_png_b64": b64(imgs["front"], "PNG"), "wrist_png_b64": b64(imgs["wrist"], "PNG")}, 300)
        good = res["token"] == "MV_UP" and not res["error"]
        print(f"{OK if good else NG}Show-Harness 応答 '{res['token']}'  往復 {dt:.1f}s" + (f"  エラー: {res['error'][-200:]}" if res["error"] else ""))
    except Exception as e:  # noqa: BLE001
        print(f"{NG}Show-Harness: {e}")
