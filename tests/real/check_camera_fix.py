"""camera_fix の合成試験：印刷するマーカー紙を仮想カメラで「撮影」し、
(1) 姿勢推定が真の姿勢を復元するか、(2) 向きとレンズが違うカメラの画像を sim カメラへ描き直せるか、を確かめる。

    python3 tests/real/check_camera_fix.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts/real")]
from so101_showharness.real.camera_fix import SHEET_CENTER, FrontRectifier, estimate_pose, pose_diff  # noqa: E402
import make_targets  # noqa: E402

setup = json.load(open(REPO / "configs/real_setup.json"))
e = setup["external"]
K_SIM, C_SIM, R_SIM = np.array(e["K"]), np.array(e["pos_task_m"]), np.array(e["R_task_cam_ros"])
Z = setup["table_top_task_z_m"]
MM = make_targets.MM
sheet = np.asarray(make_targets.aruco_sheet())
H_PAGE, W_PAGE = sheet.shape[:2]


def page_to_task(u, v):
    return np.array([SHEET_CENTER[0] + (H_PAGE / 2 - v) / MM / 1000, SHEET_CENTER[1] - (u - W_PAGE / 2) / MM / 1000, Z])


def render(K, R, c, dist=None):
    """紙（机の上の平面）を、K・向き R・位置 c のカメラで撮った 640x480 画像（背景は灰色）。歪みは dist で足す。"""
    corners_page = np.array([[0, 0], [W_PAGE, 0], [W_PAGE, H_PAGE], [0, H_PAGE]], dtype=np.float64)
    pts = np.array([page_to_task(u, v) for u, v in corners_page])
    rvec, _ = cv2.Rodrigues(R.T)
    tvec = -R.T @ c
    img_pts, _ = cv2.projectPoints(pts, rvec, tvec, K, np.zeros(5))
    Hm = cv2.getPerspectiveTransform(corners_page.astype(np.float32), img_pts.reshape(4, 2).astype(np.float32))
    out = cv2.warpPerspective(sheet, Hm, (640, 480), borderValue=(128, 128, 128))
    if dist is not None:  # 歪みのあるレンズで撮った画像にする（理想像から歪み画像へ写す）
        m1, m2 = cv2.initUndistortRectifyMap(K, np.zeros(5), None, K, (640, 480), cv2.CV_32FC1)
        ys, xs = np.mgrid[0:480, 0:640].astype(np.float32)
        und = cv2.undistortPoints(np.stack([xs.ravel(), ys.ravel()], 1).reshape(-1, 1, 2), K, dist, P=K).reshape(480, 640, 2)
        out = cv2.remap(out, und[..., 0], und[..., 1], cv2.INTER_LINEAR)
    return out


def rot(axis, deg):
    v = np.zeros(3)
    v["xyz".index(axis)] = np.radians(deg)
    return cv2.Rodrigues(v)[0]


ok = True
# (1) sim と同じカメラ → 差 0 のはず
p = estimate_pose(render(K_SIM, R_SIM, C_SIM), K_SIM, np.zeros(5), Z)
d = pose_diff(p["c"], p["R"], C_SIM, R_SIM)
print(f"[1] sim と同じ姿勢: 位置誤差 {d['pos_err_cm']:.2f} cm、向き誤差 {d['rot_err_deg']:.2f}°、再投影 {p['reproj_px']:.2f}px、ids {p['ids']}")
ok &= d["pos_err_cm"] < 0.5 and d["rot_err_deg"] < 0.5

# (2) 位置 +2 cm 前・−1 cm 左、下へ 3° 首振り、光軸回り 2° → 推定がその差を当てるか
c2 = C_SIM + np.array([0.02, -0.01, 0.0])
R2 = R_SIM @ rot("x", 3.0) @ rot("z", 2.0)
p = estimate_pose(render(K_SIM, R2, c2), K_SIM, np.zeros(5), Z)
d = pose_diff(p["c"], p["R"], C_SIM, R_SIM)
print(f"[2] ずらしたカメラ: 推定 dpos {d['dpos_cm']} cm（真値 [2.0, -1.0, 0.0]）、tilt {d['tilt_deg']:.2f}° roll {d['roll_deg']:.2f}°（真値 3, 2）")
ok &= np.allclose(d["dpos_cm"], [2.0, -1.0, 0.0], atol=0.5) and abs(d["tilt_deg"] - 3) < 0.5 and abs(d["roll_deg"] - 2) < 0.5

# (3) レンズ（焦点距離 −15%・歪みあり）と向き（pan 4°・tilt −3°）が違うが位置は同じカメラ → 描き直すと sim の画像と重なるか
K3 = K_SIM.copy(); K3[0, 0] *= 0.85; K3[1, 1] *= 0.85
dist3 = np.array([-0.12, 0.03, 0, 0, 0])
R3 = R_SIM @ rot("y", 4.0) @ rot("x", -3.0)
real = render(K3, R3, C_SIM, dist=dist3)
p = estimate_pose(real, K3, dist3, Z)
from so101_showharness.real.camera_fix import rotation_given_center  # noqa: E402
R3_est = rotation_given_center(real, K3, dist3, Z, C_SIM)  # 位置は合わせ済みとして向きだけ解く
fixed = FrontRectifier({"K": K3.tolist(), "dist": dist3.tolist(), "R_task_cam_ros": R3_est.tolist()}, setup)(real)
ideal = render(K_SIM, R_SIM, C_SIM)
before = estimate_pose(real, K_SIM, np.zeros(5), Z)  # 補正なしで sim の K とみなした場合（使わないが参考）
det = lambda im: {int(np.ravel(i)[0]): c.reshape(4, 2) for c, i in zip(*cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)).detectMarkers(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY))[:2])}  # noqa: E731
a, b, r = det(fixed), det(ideal), det(real)
px_after = np.mean([np.linalg.norm(a[i] - b[i], axis=1).mean() for i in b if i in a])
px_before = np.mean([np.linalg.norm(r[i] - b[i], axis=1).mean() for i in b if i in r])
fixed_true = FrontRectifier({"K": K3.tolist(), "dist": dist3.tolist(), "R_task_cam_ros": R3.tolist()}, setup)(real)
t = det(fixed_true)
px_true = np.mean([np.linalg.norm(t[i] - b[i], axis=1).mean() for i in b if i in t])
r_err = np.degrees(np.linalg.norm(cv2.Rodrigues(R3_est.T @ R3)[0]))
print(f"[3] レンズと向きが違うカメラ: マーカーの角のずれ 補正前 {px_before:.1f}px → 補正後 {px_after:.2f}px"
      f"（真の向きで補正すると {px_true:.2f}px。向きだけ解いた誤差 {r_err:.3f}°）")
ok &= px_after < 1.5
cv2.imwrite(str(REPO / "outputs/real_targets/check_fix.png"), cv2.cvtColor(np.hstack([real, fixed, ideal]), cv2.COLOR_RGB2BGR))
print("ALL OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
