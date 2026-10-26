"""外部（front）カメラを sim に合わせる測定ツール（ローカル PC 側）。印刷物は AWS の make_targets.py で作る。

1) レンズを測る（1 回だけ。カメラを替えたらやり直し）：checkerboard.pdf を板に貼り、いろいろな角度・距離で写す
     python scripts/real/camera_calib.py lens --front 0
     キー：c = 今の 1 枚を使う（角が見つかったときだけ）/ k = 計算して保存（12 枚以上）/ q = 終了
     → camera_front_lens.json（焦点距離・歪み）

2) 位置と向きを合わせる：aruco_sheet.pdf を「基部から前へ 30.0 cm・右へ 8.0 cm」に置き、FAR の辺を奥へ
     python scripts/real/camera_calib.py align --front 0
     画面に「あと何 cm・何度」と出るので、カメラを動かして位置を合わせる（位置が 0.5 cm 以内で緑）。
     向きの残りはソフトで消せるので、位置を優先する。s で保存 → camera_fix.json
     run_dm05.py / run_showharness.py に --camera_fix camera_fix.json を付けると、画像を sim のカメラで撮った形に直して送る。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from so101_showharness.real.camera_fix import (  # noqa: E402
    FrontRectifier, estimate_pose, pose_diff, rotation_given_center)
from so101_showharness.real.robot_io import _Grabber  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("mode", choices=["lens", "align"])
ap.add_argument("--front", default="0")
ap.add_argument("--lens", default="camera_front_lens.json")
ap.add_argument("--out", default="camera_fix.json")
ap.add_argument("--setup", default=str(REPO / "configs/real_setup.json"))
ap.add_argument("--pos_tol_cm", type=float, default=0.5)
ap.add_argument("--rot_tol_deg", type=float, default=1.0)
args = ap.parse_args()


def put(img, lines, color=(0, 255, 255)):
    import cv2

    for i, t in enumerate(lines):
        cv2.putText(img, t, (10, 25 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(img, t, (10, 25 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def lens(cam):
    import cv2

    pattern, sq = (9, 6), 0.025
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * sq
    objs, imgs = [], []
    while True:
        rgb = cam.read()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)
        show = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if found:
            cv2.drawChessboardCorners(show, pattern, corners, found)
        put(show, [f"lens: {len(imgs)} shots (12+), c=capture k=compute q=quit", "found" if found else "board not found"])
        cv2.imshow("camera_calib lens", show)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            return
        if k == ord("c") and found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
            objs.append(objp)
            imgs.append(corners)
            print(f"captured {len(imgs)}", flush=True)
        if k == ord("k") and len(imgs) >= 12:
            rms, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, gray.shape[::-1], None, None)
            json.dump({"K": K.tolist(), "dist": dist.ravel().tolist(), "rms_px": rms, "shots": len(imgs),
                       "time": time.strftime("%F %T")}, open(args.lens, "w"), indent=1)
            print(f"saved {args.lens}  rms={rms:.3f}px  fx={K[0, 0]:.1f} fy={K[1, 1]:.1f}（sim は 514.7）", flush=True)
            return


def align(cam, setup):
    import cv2

    e = setup["external"]
    k_sim, c_sim, r_sim = np.asarray(e["K"]), np.asarray(e["pos_task_m"]), np.asarray(e["R_task_cam_ros"])
    z = float(setup["table_top_task_z_m"])
    if Path(args.lens).exists():
        lj = json.load(open(args.lens))
        K, dist = np.asarray(lj["K"]), np.asarray(lj["dist"])
        lens_note = f"lens {args.lens}"
    else:
        K, dist = k_sim, np.zeros(5)
        lens_note = "lens 未測定（sim の値で代用。先に lens モードを推奨）"
    hist = []
    while True:
        rgb = cam.read()
        show = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        p = estimate_pose(rgb, K, dist, z)
        if p is None:
            put(show, ["markers not found (need 2+ of id 0-3)", lens_note])
        else:
            hist = (hist + [(p["c"], p["R"])])[-15:]
            c = np.mean([h[0] for h in hist], axis=0)
            d = pose_diff(c, p["R"], c_sim, r_sim)
            dx, dy, dz = d["dpos_cm"]
            good_pos = d["pos_err_cm"] <= args.pos_tol_cm
            good = good_pos and d["rot_err_deg"] <= args.rot_tol_deg
            lines = [f"move camera: {'back' if dx > 0 else 'forward'} {abs(dx):.1f}cm  "
                     f"{'right' if dy > 0 else 'left'} {abs(dy):.1f}cm  {'down' if dz > 0 else 'up'} {abs(dz):.1f}cm",
                     f"pos err {d['pos_err_cm']:.2f}cm  rot err {d['rot_err_deg']:.2f}deg "
                     f"(tilt {d['tilt_deg']:+.1f} pan {d['pan_deg']:+.1f} roll {d['roll_deg']:+.1f})",
                     f"markers {sorted(p['ids'])}  reproj {p['reproj_px']:.2f}px  {lens_note}",
                     "POSITION OK - s=save (rotation fixed in software)" if good_pos else "s=save anyway  q=quit"]
            put(show, lines, (0, 255, 0) if good else (0, 255, 255))
        cv2.imshow("camera_calib align", show)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            return
        if k == ord("s") and hist:
            c = np.mean([h[0] for h in hist], axis=0)
            d = pose_diff(c, hist[-1][1], c_sim, r_sim)
            # 位置が合っていれば、位置を sim に固定して向きだけ解く（そのほうが正確）
            center = c_sim if d["pos_err_cm"] <= 1.0 else c
            R = rotation_given_center(rgb, K, dist, z, center)
            if R is None:
                print("markers not found at save; try again", flush=True)
                continue
            fix = {"K": K.tolist(), "dist": np.ravel(dist).tolist(), "R_task_cam_ros": R.tolist(),
                   "measured_pos_task_m": c.tolist(), "pos_err_cm": d["pos_err_cm"], "rot_err_deg": d["rot_err_deg"],
                   "correct_rotation": True, "time": time.strftime("%F %T")}
            json.dump(fix, open(args.out, "w"), indent=1)
            fixed = FrontRectifier(fix, setup)(rgb)
            cv2.imwrite(args.out.replace(".json", "_preview.jpg"),
                        cv2.cvtColor(np.hstack([rgb, fixed]), cv2.COLOR_RGB2BGR))
            print(f"saved {args.out}（位置の差 {d['pos_err_cm']:.2f} cm は残る。向きとレンズは補正される）", flush=True)


def main():
    import cv2

    setup = json.load(open(args.setup))
    cam = _Grabber(args.front, "front")
    try:
        lens(cam) if args.mode == "lens" else align(cam, setup)
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
