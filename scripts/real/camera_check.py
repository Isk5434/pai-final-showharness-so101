"""実機カメラを sim と同じ画角に合わせる（ローカル PC 側）。ライブ映像に、sim の同じ姿勢の画像を半透明で重ねて表示する。

DM0.5 は sim の画像だけで学習しているので、front（外部）カメラの位置・向きが sim とずれるほど本番で崩れる。
アームを開始姿勢に置いた状態で、ライブ映像（実線）と sim（半透明）の机・ラック・アームが重なるようにカメラを動かす。

    python scripts/real/camera_check.py --front 0 --wrist 2 --ref_dir real_reference
    # キー：q 終了 / s 今の画面を保存 / 1 front・2 wrist 切り替え / + - 重ねる濃さ

--ref_dir には AWS で作った参照画像（front.png / wrist.png）を置く：
    AWS: python3 scripts/real/make_reference.py → outputs/real_reference/ を scp でローカルへ
--move_start を付けると、先にアームを開始姿勢へゆっくり動かす（--port / --robot_id が必要）。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from so101_showharness.real.robot_io import SafeArm, connect_so101, open_cameras  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--front", default="0")
ap.add_argument("--wrist", default="2")
ap.add_argument("--ref_dir", default="real_reference")
ap.add_argument("--move_start", action="store_true")
ap.add_argument("--port")
ap.add_argument("--robot_id", default="follower_arm_1")
args = ap.parse_args()


def main():
    import cv2

    if args.move_start:
        setup = json.load(open(REPO / "configs/real_setup.json"))
        arm = SafeArm(connect_so101(args.port, args.robot_id))
        arm.move_to(np.asarray(setup["start_q_rad"]), duration_s=3.0)
    cams = open_cameras(args.front, args.wrist)
    refs = {k: cv2.cvtColor(cv2.imread(str(Path(args.ref_dir) / f"{k}.png")), cv2.COLOR_BGR2RGB)
            for k in ("front", "wrist")}
    view, alpha = "front", 0.5
    print("q 終了 / s 保存 / 1 front・2 wrist / + - 濃さ", flush=True)
    while True:
        live = cams[view].read()
        mix = (live * (1 - alpha) + refs[view] * alpha).astype(np.uint8)
        diff = np.abs(live.astype(int) - refs[view].astype(int)).mean()
        show = cv2.cvtColor(np.hstack([mix, live]), cv2.COLOR_RGB2BGR)
        cv2.putText(show, f"{view}  sim overlay {alpha:.1f}  mean|diff| {diff:.0f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("camera_check (left: live+sim, right: live)", show)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        if k == ord("1"):
            view = "front"
        if k == ord("2"):
            view = "wrist"
        if k in (ord("+"), ord("=")):
            alpha = min(0.9, alpha + 0.1)
        if k == ord("-"):
            alpha = max(0.1, alpha - 0.1)
        if k == ord("s"):
            p = f"camera_check_{view}_{time.strftime('%H%M%S')}.png"
            cv2.imwrite(p, show)
            print("saved", p, flush=True)
    for c in cams.values():
        c.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
