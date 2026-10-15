"""sim（rad、Workshop の関節順）⇔ 実機 LeRobot 単位（腕 −100〜100、グリッパー 0〜100）。

Workshop の utils/lerobot_interface.py（get_mapped_actions_vectorized /
get_raw_actions_from_radians）と同じ式。sim と実機、学習データ（LeRobot 変換）で1か所を共有する。
"""
from __future__ import annotations

import numpy as np

LEROBOT_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
# Workshop SO101_USD_MAPPING（度）
MINS_DEG = np.array([-110, -100, -100, -95, -160, -10.0])
MAXS_DEG = np.array([110, 100, 90, 95, 160, 100.0])


def rad_to_raw(q) -> np.ndarray:
    n = (np.degrees(np.asarray(q, dtype=np.float64)) - MINS_DEG) / (MAXS_DEG - MINS_DEG)
    raw = np.empty(6)
    raw[:5] = n[:5] * 200.0 - 100.0
    raw[5] = n[5] * 100.0
    return raw


def raw_to_rad(raw) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    n = np.empty(6)
    n[:5] = (raw[:5] + 100.0) / 200.0
    n[5] = raw[5] / 100.0
    return np.radians(MINS_DEG + n * (MAXS_DEG - MINS_DEG))
