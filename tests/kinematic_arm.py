"""A physics-free SO-101 for testing the controller logic.

Each tick the joints move a fraction ``alpha`` of the way to (command - sag): a first-order
lag plus a constant gravity sag on the arm joints, the two effects the real controller has
to cope with. World orientation of the base matches the Workshop (+90 deg yaw).
"""
import numpy as np

from so101_showharness.controller import So101ArmController

Q_RESET = np.array([-0.2736, -0.6109, -0.0745, 1.5148, -1.6034, -0.1465])
R_WB = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


class KinematicArm(So101ArmController):
    def __init__(self, q0=Q_RESET, alpha: float = 0.5, sag=(0.0, 0.008, 0.012, 0.006, 0.0, 0.0),
                 wall_y_max=None, **kw):
        super().__init__(**kw)
        # Optional obstacle: the TCP cannot pass task-frame y = wall_y_max (the joints simply
        # stop, like a hand pressed against the rack).
        self.wall_y_max = wall_y_max
        self.q = np.asarray(q0, dtype=np.float64).copy()
        self.alpha = float(alpha)
        self.sag = np.asarray(sag, dtype=np.float64)

    def _read_q(self):
        return self.q.copy()

    def _send(self, q_cmd):
        q_next = self.q + self.alpha * (np.asarray(q_cmd) - self.sag - self.q)
        if self.wall_y_max is not None:
            y = (R_WB @ self.kin.tcp(q_next)[0])[1]
            if y > self.wall_y_max:
                return  # blocked: the arm does not move this tick
        self.q = q_next

    def _r_world_base(self):
        return R_WB

    def success(self):
        return False

    def grab_frames(self):
        raise NotImplementedError("no cameras on the kinematic arm")

    def reset(self, seed: int = 0):
        self._begin_episode()
