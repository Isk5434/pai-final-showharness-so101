"""実機 SO-101 の入出力（ローカル PC 側）。DM0.5 と Show-Harness の両方の実機クライアントが使う。

- SafeArm: LeRobot の SO101Follower を包み、指令を「関節の可動範囲」と「1 回あたりの変化量」で必ず制限する。
  lerobot 0.4.1（Workshop の real-robot イメージと同じ版）の Robot API（get_observation / send_action）だけを使う。
- Cameras: front（外部）と wrist（手首）を OpenCV で別スレッドから読み続け、いつ読んでも最新の 1 枚を返す
  （学習・sim と同じ 640x480・RGB）。
- --mock 用の MockArm / MockCameras：実機なしでパイプライン全体（AWS の推論サーバとの往復まで）を試すため。

単位はすべて units.py（Workshop と同じ式）で rad ⇔ LeRobot の値を変換する。
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional

import numpy as np

from so101_showharness.units import LEROBOT_JOINTS, MAXS_DEG, MINS_DEG, rad_to_raw, raw_to_rad

Q_MIN = np.radians(MINS_DEG)
Q_MAX = np.radians(MAXS_DEG)
IMAGE_W, IMAGE_H = 640, 480


# ---------------------------------------------------------------------- arm
class SafeArm:
    """関節指令を必ず制限して送る薄い包み。So101RealBackend にもこのまま robot として渡せる。

    max_step_rad: 1 回の send_action で各関節が「今の位置」から動いてよい量（rad）。超える分は切る。
    """

    def __init__(self, robot, max_step_rad: float = 0.08, dry_run: bool = False) -> None:
        self.robot = robot
        self.max_step_rad = float(max_step_rad)
        self.dry_run = bool(dry_run)
        self.clipped = 0  # 制限に当たった回数（ログ用）
        self._last_cmd: Optional[np.ndarray] = None

    # LeRobot の Robot API と同じ形（So101RealBackend がそのまま呼ぶ）
    def get_observation(self) -> dict:
        return self.robot.get_observation()

    def send_action(self, action: dict) -> dict:
        q_cmd = raw_to_rad([action[f"{j}.pos"] for j in LEROBOT_JOINTS])
        q_safe = self.limit(q_cmd)
        self._last_cmd = q_safe
        raw = rad_to_raw(q_safe)
        out = {f"{j}.pos": float(v) for j, v in zip(LEROBOT_JOINTS, raw)}
        if not self.dry_run:
            self.robot.send_action(out)
        return out

    # rad で使う側
    def q(self) -> np.ndarray:
        obs = self.robot.get_observation()
        return raw_to_rad([obs[f"{j}.pos"] for j in LEROBOT_JOINTS])

    def send_q(self, q_cmd) -> np.ndarray:
        raw = rad_to_raw(np.asarray(q_cmd, dtype=np.float64))
        self.send_action({f"{j}.pos": float(v) for j, v in zip(LEROBOT_JOINTS, raw)})
        return self._last_cmd

    def limit(self, q_cmd) -> np.ndarray:
        q_cmd = np.clip(np.asarray(q_cmd, dtype=np.float64), Q_MIN, Q_MAX)
        # dry run では腕が動かないので、前の指令を基準にして「動いたつもり」で制限する
        base = self._last_cmd if (self.dry_run and self._last_cmd is not None) else self.q()
        d = np.clip(q_cmd - base, -self.max_step_rad, self.max_step_rad)
        if np.any(np.abs(q_cmd - base) > self.max_step_rad + 1e-9):
            self.clipped += 1
        return base + d

    def move_to(self, q_target, duration_s: float = 3.0, hz: float = 30.0) -> None:
        """今の姿勢から q_target へ、余弦補間でゆっくり動かす（Workshop の move_to_pose と同じ形）。"""
        start = self.q()
        n = max(1, int(duration_s * hz))
        for i in range(1, n + 1):
            a = (1 - np.cos(np.pi * i / n)) / 2
            self.send_q(start + a * (np.asarray(q_target) - start))
            time.sleep(1.0 / hz)

    def hold(self) -> None:
        """今の位置をそのまま指令して止める。"""
        if not self.dry_run:
            self.send_q(self.q())

    def disconnect(self) -> None:
        self.robot.disconnect()


def lerobot_lite() -> None:
    """torch / accelerate / datasets を入れていない軽い環境でも LeRobot の SO-101 を使えるようにする。

    lerobot 0.4.1 では、腕の制御（motors_bus → lerobot.utils.utils）が読み込み時に torch・accelerate・datasets を
    import するが、腕の制御で実際に使うのは enter_pressed / move_cursor_up / init_logging だけで、これらは torch を使わない。
    本物が入っていればそれを使い、無いものだけ「名前だけの代用品」を差し込む（呼ばれたら例外になるので、使われれば気づく）。
    """
    import importlib.util
    import sys
    import types

    class _Absent:
        def __init__(self, name):
            self._name = name

        def __getattr__(self, attr):
            return _Absent(f"{self._name}.{attr}")

        def __call__(self, *a, **k):
            raise RuntimeError(f"{self._name} は軽量モードでは使えない（torch 等を入れていない）")

        def __or__(self, other):  # 型注釈の X | Y
            return self

        __ror__ = __or__

    def stub(name: str, **attrs):
        top = name.split(".")[0]
        if name in sys.modules or (top not in sys.modules and importlib.util.find_spec(top) is not None):
            return  # 本物がある（または差し込み済み）
        m = types.ModuleType(name)
        m.__getattr__ = lambda attr, _n=name: _Absent(f"{_n}.{attr}")  # type: ignore[attr-defined]
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m

    stub("torch")
    stub("accelerate", Accelerator=_Absent("accelerate.Accelerator"))
    if importlib.util.find_spec("datasets") is None:
        for n in ("datasets", "datasets.utils"):
            stub(n)
        stub("datasets.utils.logging", disable_progress_bar=lambda: None, enable_progress_bar=lambda: None)


def connect_so101(port: str, robot_id: str, max_relative_target: Optional[float] = 10.0):
    """LeRobot の SO101Follower に接続する（キャリブレーションは lerobot-calibrate で済ませておく）。

    max_relative_target は LeRobot 側の安全制限（LeRobot の単位。腕は −100〜100）で、SafeArm の制限に重ねて効く。
    """
    lerobot_lite()
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id, cameras={},
                                              max_relative_target=max_relative_target))
    robot.connect()
    return robot


class MockArm:
    """実機の代わり（--mock）。指令に一次遅れで追従するだけ。"""

    def __init__(self, q0) -> None:
        self.q = np.asarray(q0, dtype=np.float64).copy()
        self.target = self.q.copy()

    def get_observation(self) -> dict:
        self.q += 0.5 * (self.target - self.q)
        return {f"{j}.pos": float(v) for j, v in zip(LEROBOT_JOINTS, rad_to_raw(self.q))}

    def send_action(self, action: dict) -> dict:
        self.target = raw_to_rad([action[f"{j}.pos"] for j in LEROBOT_JOINTS])
        return action

    def disconnect(self) -> None:
        pass


# ---------------------------------------------------------------------- cameras
class _Grabber:
    def __init__(self, source, name: str) -> None:
        import cv2

        src = int(source) if str(source).isdigit() else str(source)
        # Windows の既定（MSMF）は開くのが遅く解像度指定も効きにくいので DirectShow を使う
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if (os.name == "nt" and isinstance(src, int)) else cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"camera '{name}' ({source}) を開けない")
        if frame.shape[:2] != (IMAGE_H, IMAGE_W):
            raise RuntimeError(f"camera '{name}' が {frame.shape[1]}x{frame.shape[0]}。学習と同じ 640x480 にする")
        self.name, self._cv2 = name, cv2
        self._frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._lock = threading.Lock()
        self._stop = False
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while not self._stop:
            ok, frame = self.cap.read()
            if ok:
                rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
                with self._lock:
                    self._frame = rgb

    def read(self) -> np.ndarray:
        with self._lock:
            return self._frame.copy()

    def close(self) -> None:
        self._stop = True
        time.sleep(0.1)
        self.cap.release()


class _VideoLoop:
    """--mock 用：動画ファイル（sim の収集動画など）を先頭から読み、終わったら最後のコマを返し続ける。"""

    def __init__(self, path: str) -> None:
        import cv2

        cap, self.frames = cv2.VideoCapture(str(path)), []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            self.frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        if not self.frames:
            raise RuntimeError(f"動画を読めない: {path}")
        self.i = 0

    def read(self) -> np.ndarray:
        f = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return f

    def close(self) -> None:
        pass


def open_cameras(front, wrist, mock: bool = False) -> dict:
    """{"front": ..., "wrist": ...}。各要素は read() -> RGB uint8 (480, 640, 3)。"""
    if mock:
        return {"front": _VideoLoop(front), "wrist": _VideoLoop(wrist)}
    return {"front": _Grabber(front, "front"), "wrist": _Grabber(wrist, "wrist")}


class _Rectified:
    """front カメラの画像を FrontRectifier で sim のカメラの形に直して返す包み。"""

    def __init__(self, cam, rect) -> None:
        self.cam, self.rect = cam, rect

    def read(self) -> np.ndarray:
        return self.rect(self.cam.read())

    def close(self) -> None:
        self.cam.close()


def apply_camera_fix(cams: dict, fix_path, setup: dict) -> dict:
    """--camera_fix があれば front を補正付きに差し替える（camera_calib.py align が作る camera_fix.json）。"""
    if not fix_path:
        return cams
    import json

    from so101_showharness.real.camera_fix import FrontRectifier

    fix = json.load(open(fix_path))
    print(f"[camera_fix] {fix_path}: 位置の差 {fix.get('pos_err_cm', float('nan')):.2f} cm（残る）、"
          f"向き {fix.get('rot_err_deg', float('nan')):.2f}° とレンズを補正", flush=True)
    return {**cams, "front": _Rectified(cams["front"], FrontRectifier(fix, setup))}
