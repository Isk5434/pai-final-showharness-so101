"""実機 SO-101 用の入出力（制御ロジックは sim と同じ So101ArmController）。

ローカル PC（SO-101 を USB 接続）で動かす。必要なのは LeRobot の SO-101 follower が提供する
2つだけ: ``get_observation() -> {"<joint>.pos": raw, ...}`` と
``send_action({"<joint>.pos": raw, ...})``（lerobot 0.4 以降の Robot API）。単位変換は
units.py（Workshop と同じ式）。カメラは ``cameras`` に {名前: read() を持つ物} を渡す。

- タスク座標: sim と同じ（ワールドの向き、基部が原点、+X がロボット前方の作業台）。基部リンクは
  タスク座標に対して +90° ヨー（sim の Workshop 配置と同じ）。
- 1 tick = ``1 / control_hz``（既定 30 Hz）。sim の 60 Hz・2 tick/制御と同じ 30 Hz 制御になる
  よう ticks_per_ctrl=1。
- 力センサが無いので、接触は「関節差 + 進まない」で推定する（contact_is_measured=False）。
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from so101_showharness.controller import So101ArmController
from so101_showharness.units import LEROBOT_JOINTS, rad_to_raw, raw_to_rad

R_WORLD_BASE = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # base +90 deg yaw


class So101RealBackend(So101ArmController):
    def __init__(self, robot, cameras: Optional[dict] = None, control_hz: float = 30.0, **kw) -> None:
        kw.setdefault("ticks_per_ctrl", 1)
        kw.setdefault("tick_dt", 1.0 / control_hz)
        super().__init__(**kw)
        self.robot = robot
        self.cameras = cameras or {}
        self._next_tick = None
        self.contact_is_measured = False

    # ------------------------------------------------------------ I/O
    def _read_q(self) -> np.ndarray:
        obs = self.robot.get_observation()
        return raw_to_rad([obs[f"{j}.pos"] for j in LEROBOT_JOINTS])

    def _send(self, q_cmd: np.ndarray) -> None:
        raw = rad_to_raw(q_cmd)
        self.robot.send_action({f"{j}.pos": float(v) for j, v in zip(LEROBOT_JOINTS, raw)})
        # hold the tick rate (sleep until the next tick boundary)
        now = time.monotonic()
        if self._next_tick is None:
            self._next_tick = now
        self._next_tick += self.tick_dt
        delay = self._next_tick - now
        if delay > 0:
            time.sleep(delay)
        else:
            self._next_tick = now  # fell behind: do not try to catch up

    def _r_world_base(self) -> np.ndarray:
        return R_WORLD_BASE

    # ------------------------------------------------------------ task / cameras
    def success(self) -> bool:
        return False  # 実機では人が判定する

    def raw_frames(self):
        return self.cameras["external"].read(), self.cameras["wrist"].read()

    def grab_frames(self):
        from scripts.trajectory.real2sim.atomic_tokenizer import prepared_pair

        return prepared_pair(self, *self.raw_frames())

    # Show-Harness camera contract: same as the sim backend (256x256 letterbox, no rotation)
    agentview_rotation_degrees = 0
    agentview_flip = None
    agentview_square_size = 256
    wrist_rotation_degrees = 0
    wrist_flip = None
    wrist_square_size = 256

    def reset(self, seed: int = 0, settle_ctrl_steps: int = 30) -> None:
        self._next_tick = None
        self._begin_episode(settle_ctrl_steps)
