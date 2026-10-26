"""実機の外部カメラを sim のカメラに合わせるための計算（印刷する目印の定義・姿勢の推定・画像の補正）。

- 目印：A4 横 1 枚に ArUco（DICT_4X4_50、id 0〜3、一辺 50 mm）を 4 つ。紙の中心をタスク座標（基部が原点、+X 前方、+Y 左）の
  SHEET_CENTER に置き、紙の上辺（"FAR" と書いた側）をロボットから遠い向きにする。マーカーの中心はそこから ±65 mm（前後）、±100 mm（左右）。
- 姿勢：検出したマーカーの角（タスク座標が既知）から solvePnP で実カメラの位置と向きを求め、sim のカメラとの差を出す。
- 補正：向きの差とレンズの違い（内部パラメータ・歪み）は、画像を「sim のカメラで撮った画像」に描き直して消せる
  （回転だけなら画像の変換で正確に表せる）。位置の差は消せないので、カメラを動かして合わせる。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SHEET_CENTER = (0.30, -0.08)   # タスク座標 (x, y) [m]
MARKER_M = 0.05
MARKER_OFFSETS = {0: (+0.065, +0.10), 1: (+0.065, -0.10), 2: (-0.065, +0.10), 3: (-0.065, -0.10)}  # (dx, dy) [m]
ARUCO_DICT = "DICT_4X4_50"


def marker_corners_task(table_z: float) -> dict[int, np.ndarray]:
    """id -> 4x3（ArUco の角の順：左上・右上・右下・左下。紙の上 = +X、紙の右 = −Y）。"""
    s = MARKER_M / 2
    out = {}
    for i, (dx, dy) in MARKER_OFFSETS.items():
        c = np.array([SHEET_CENTER[0] + dx, SHEET_CENTER[1] + dy, table_z])
        out[i] = np.array([c + [s, s, 0], c + [s, -s, 0], c + [-s, -s, 0], c + [-s, s, 0]])
    return out


def _detector():
    import cv2

    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT))
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(d, p)


def estimate_pose(img_rgb: np.ndarray, K: np.ndarray, dist: np.ndarray, table_z: float):
    """画像から実カメラの姿勢（タスク座標の位置 c と向き R：ROS 規約）。見つからなければ None。"""
    import cv2

    corners, ids, _ = _detector().detectMarkers(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY))
    if ids is None:
        return None
    known = marker_corners_task(table_z)
    obj, imgp, seen = [], [], []
    for cs, i in zip(corners, ids.ravel()):
        if int(i) in known:
            obj.append(known[int(i)])
            imgp.append(cs.reshape(4, 2))
            seen.append(int(i))
    if len(seen) < 2:
        return None
    obj, imgp = np.concatenate(obj).astype(np.float64), np.concatenate(imgp).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, imgp, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    r_cw, _ = cv2.Rodrigues(rvec)          # task -> camera
    R = r_cw.T                             # camera -> task（列がカメラの軸）
    c = (-R @ tvec).ravel()
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = float(np.linalg.norm(proj.reshape(-1, 2) - imgp, axis=1).mean())
    return {"c": c, "R": R, "ids": seen, "reproj_px": err}


def rotation_given_center(img_rgb: np.ndarray, K: np.ndarray, dist: np.ndarray, table_z: float, center) -> np.ndarray | None:
    """カメラの位置を center に固定して、向きだけを求める（各マーカー角への視線と、実際の方向を SVD で合わせる）。

    平面のマーカーから位置と向きを同時に解くと 0.4° ほど混ざる。位置を物理的に合わせた後は、こちらで向きを出すほうが正確。
    """
    import cv2

    corners, ids, _ = _detector().detectMarkers(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY))
    if ids is None:
        return None
    known = marker_corners_task(table_z)
    rays, dirs = [], []
    for cs, i in zip(corners, ids.ravel()):
        if int(i) not in known:
            continue
        n = cv2.undistortPoints(cs.reshape(-1, 1, 2).astype(np.float64), K, dist).reshape(-1, 2)
        for (x, y), P in zip(n, known[int(i)]):
            r = np.array([x, y, 1.0])
            rays.append(r / np.linalg.norm(r))
            d = P - np.asarray(center)
            dirs.append(d / np.linalg.norm(d))
    if len(rays) < 8:
        return None
    # R（カメラ → タスク）で R @ ray ≈ dir となる回転（Wahba 問題）
    u, _, vt = np.linalg.svd(np.asarray(dirs).T @ np.asarray(rays))
    return u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt


def pose_diff(c_real, R_real, c_sim, R_sim) -> dict:
    """実カメラ − sim カメラ。位置はタスク座標（前 +X / 左 +Y / 上 +Z）、向きは sim カメラから見た回転（度）。"""
    import cv2

    d_rot = R_sim.T @ R_real               # sim カメラ座標での実カメラの向き
    rv, _ = cv2.Rodrigues(d_rot)
    deg = np.degrees(rv.ravel())           # x: 上下の首振り（tilt）、y: 左右（pan）、z: 光軸回り（roll）
    dp = np.asarray(c_real) - np.asarray(c_sim)
    return {"dpos_cm": (100 * dp).round(2).tolist(), "tilt_deg": float(deg[0]), "pan_deg": float(deg[1]),
            "roll_deg": float(deg[2]), "pos_err_cm": float(100 * np.linalg.norm(dp)),
            "rot_err_deg": float(np.degrees(np.linalg.norm(rv)))}


class FrontRectifier:
    """実カメラの画像を「sim の外部カメラ（K_sim、sim の向き）で撮った画像」に描き直す。位置の差は残る。"""

    def __init__(self, fix: dict, setup: dict) -> None:
        import cv2

        e = setup["external"]
        k_sim, r_sim = np.asarray(e["K"]), np.asarray(e["R_task_cam_ros"])
        k_real, dist = np.asarray(fix["K"]), np.asarray(fix["dist"])
        r_real = np.asarray(fix["R_task_cam_ros"]) if fix.get("correct_rotation", True) else r_sim
        w, h = setup["image_size"]
        # initUndistortRectifyMap の R は「元のカメラ座標 → 描き直す先のカメラ座標」
        self.map1, self.map2 = cv2.initUndistortRectifyMap(k_real, dist, r_sim.T @ r_real, k_sim, (w, h), cv2.CV_32FC1)
        self._cv2 = cv2

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self._cv2.remap(img, self.map1, self.map2, self._cv2.INTER_LINEAR, borderMode=self._cv2.BORDER_CONSTANT)


def load_fix(path) -> dict | None:
    p = Path(path) if path else None
    return json.load(open(p)) if p and p.exists() else None
