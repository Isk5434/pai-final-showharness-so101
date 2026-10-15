"""実機バックエンドを、LeRobot の SO-101 follower を真似た偽ロボットで確認（ハード不要）。"""
import numpy as np
import pytest

pytest.importorskip("scripts.trajectory.real2sim.atomic_tokenizer")

from kinematic_arm import Q_RESET  # noqa: E402
from so101_showharness.exec import So101AtomicExec  # noqa: E402
from so101_showharness.real.so101_robot import So101RealBackend  # noqa: E402
from so101_showharness.units import LEROBOT_JOINTS, rad_to_raw, raw_to_rad  # noqa: E402


class FakeFollower:
    """get_observation / send_action だけを持つ。関節は指令に一次遅れで追従（rad 空間）。"""

    def __init__(self):
        self.q = Q_RESET.copy()

    def get_observation(self):
        return {f"{j}.pos": float(v) for j, v in zip(LEROBOT_JOINTS, rad_to_raw(self.q))}

    def send_action(self, action):
        target = raw_to_rad([action[f"{j}.pos"] for j in LEROBOT_JOINTS])
        self.q += 0.6 * (target - self.q)


def test_units_round_trip():
    q = np.array([0.3, -0.5, 0.7, 1.2, -1.57, 0.4])
    assert np.allclose(raw_to_rad(rad_to_raw(q)), q, atol=1e-12)


def test_tokens_on_the_fake_follower():
    be = So101RealBackend(FakeFollower(), control_hz=1000.0)  # fast ticks for the test
    be.reset()
    ex = So101AtomicExec(be, step_m=0.02)
    assert ex.goto_pose(np.array([0.26, -0.04, 0.12]), np.radians(60), roll=-np.pi / 2, budget=600)
    for t in ("MV_FWD", "MV_DOWN", "MV_BACK", "MV_UP"):
        p0 = be.tcp_pos()
        ex.move(t)
        assert ex.last_blocked is None, (t, ex.last_blocked)
        assert abs(np.linalg.norm(be.tcp_pos() - p0) - 0.02) < 1e-3, t
