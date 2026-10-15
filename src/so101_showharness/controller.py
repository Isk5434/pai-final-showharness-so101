"""SO-101 token-level arm controller, independent of where the arm lives.

The same class drives the Workshop sim (sim/workshop_backend.py), the real arm (real/, Phase
5) and a kinematic stand-in for unit tests (tests/kinematic_arm.py). Subclasses provide only
I/O: read the joints, send one tick of joint targets, and the world<-base rotation. Every
decision about MOTION lives here, so what is verified in sim is what runs on hardware.

Frames
------
TASK frame: world orientation, origin at the robot base. +X from the robot toward the
vials/rack, +Z up. Show-Harness MV_* tokens are unit steps along its axes. FK/IK work in the
base-link frame (yawed +90 deg in the Workshop world); conversion happens here.

Control
-------
One CONTROL step = ``ticks_per_ctrl`` ticks (a tick is one env.step, 60 Hz). Each control
step: TCP target = measured TCP + capped delta; IK from the measured joints holding the
references (tool pitch in the arm plane, plus EITHER the Wrist_Roll joint angle ``roll_ref``
(default) OR the closing-axis yaw ``yaw_ref``).

Which second reference: Pitch, Elbow and Wrist_Pitch are parallel, so at the constant roll
ROLL_CLOSING_HORIZONTAL (= -pi/2, measured) the jaw closes along the arm-plane normal, i.e.
HORIZONTALLY, in every pose. Holding the yaw instead keeps the closing direction's compass
heading but tilts it out of horizontal as the base turns (13-22 deg measured over the vial
area at pitch 47): the fixed finger dips and grasps fail. So roll is the default. The SO-101's PD gains are low, so gravity leaves a steady-state joint
offset; it is learned whenever the arm is at rest and added to the command (``q_bias``).

Exactness
---------
``feasible`` answers "can the hand be put THERE with THIS orientation, within tolerance and
off the joint limits" before a token runs; So101AtomicExec refuses tokens that fail it, so a
recorded token always means the motion it names.

Every tick is offered to ``step_hooks`` (video, trajectory recorder) with observed AND
commanded joints.
"""
from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from scripts.trajectory.real2sim.atomic_tokenizer import AtomicSimEnv
from so101_showharness.kinematics import So101Kinematics, mat_to_quat_wxyz


DEFAULT_HULL = Path(__file__).resolve().parents[2] / "configs" / "so101_gripper_hull.json"

# Wrist_Roll at which the jaw's closing axis is the arm-plane normal (horizontal in any pose).
# Measured on the Workshop chain (scripts: tests/test_kinematics.py checks it).
ROLL_CLOSING_HORIZONTAL = -np.pi / 2


class So101ArmController(AtomicSimEnv):
    delta_bound_m = 0.02

    def __init__(
        self,
        kin: Optional[So101Kinematics] = None,
        ticks_per_ctrl: int = 2,
        tick_dt: float = 1.0 / 60.0,
        jaw_open_rad: float = 0.9,
        jaw_close_rad: float = -0.17,
        bias_rest_tol_rad: float = 2e-3,
        bias_alpha: float = 0.5,
        use_gravity_bias: bool = True,
        bias_max_rad: float = 0.08,
        bias_rate_rad: float = 0.01,
        contact_tol_rad: float = 0.08,
        fixed_finger_face_m: float = 0.007,
        hand_floor_task_z: Optional[float] = None,
        hull_path: Optional[Path] = DEFAULT_HULL,
        workspace: Optional[dict] = None,
        tol_pos_m: float = 1e-3,
        tol_ori_rad: float = float(np.radians(1.0)),
        min_joint_margin_rad: float = 0.02,
    ) -> None:
        self.kin = kin or So101Kinematics()
        self.ticks_per_ctrl = int(ticks_per_ctrl)
        self.tick_dt = float(tick_dt)
        self.jaw_open_rad = float(jaw_open_rad)
        self.jaw_close_rad = float(jaw_close_rad)
        self.bias_rest_tol_rad = float(bias_rest_tol_rad)
        self.bias_alpha = float(bias_alpha)
        self.use_gravity_bias = bool(use_gravity_bias)
        # Gravity sag measured on the Workshop SO-101 is <= 0.018 rad (the real arm may sag more
        # at full reach). The bias is learned at rest, rate-limited and bounded. It can no
        # longer cause a runaway (commands are relative to q_des, not the measured joints);
        # contact is reported by ``in_contact`` and the executor undoes what a push taught.
        self.bias_max_rad = float(bias_max_rad)
        self.bias_rate_rad = float(bias_rate_rad)
        self.contact_tol_rad = float(contact_tol_rad)
        # The SO-101 has ONE moving jaw. The TCP (lerobot URDF gripper_frame) is on the FIXED
        # finger; an object is held on the moving-jaw side of it, along the closing axis.
        # Measured from FK: closed moving-jaw tip at +0.007 m from the TCP along gripper +x.
        self.fixed_finger_face_m = float(fixed_finger_face_m)
        # True when in_contact() reads a physical sensor (sim); False = inferred from the joint
        # gap (real arm without force sensing), which then also needs "no progress".
        self.contact_is_measured = False
        # Floor for the WHOLE hand, not just the TCP: the lowest point of the gripper + jaw
        # collision hulls must stay above it (the fingertip region reaches 5.5 mm below the
        # TCP at pitch 47; a box model said 23 mm and was wrong). None = not checked.
        self.hand_floor_task_z = hand_floor_task_z
        self._hull = None
        if hull_path is not None and Path(hull_path).exists():
            h = json.loads(Path(hull_path).read_text())
            self._hull = {k: np.asarray(h[k], dtype=np.float64) for k in ("gripper", "jaw")}
        self.step_hooks: list[Callable[[dict], None]] = []
        # Labels for the current activity (token, phase, blocked reason): video + log.
        self.annotation: dict = {}
        # TASK-frame box; a token whose end point leaves it is refused.
        self.workspace = workspace or {"min": [0.05, -0.30, 0.0], "max": [0.45, 0.25, 0.30]}
        self.tol_pos_m = float(tol_pos_m)
        self.tol_ori_rad = float(tol_ori_rad)
        self.min_joint_margin_rad = float(min_joint_margin_rad)
        self.q_des = np.zeros(6)   # IK solution: where the arm should be
        self.q_bias = np.zeros(6)  # learned gravity sag, added to the command
        self.pitch_ref = 0.0
        # Small integral trim on the IK pitch target, set by the executor while settling: the
        # gravity bias leaves ~0.2 deg of pitch at full reach (measured). Never a reference.
        self.pitch_trim = 0.0
        self.yaw_ref: Optional[float] = None  # set: hold the closing-axis yaw (via Wrist_Roll)
        self.roll_ref: float = ROLL_CLOSING_HORIZONTAL  # used while yaw_ref is None
        self.tick_count = 0
        self.ctrl_step_count = 0
        self.decision_id: Optional[int] = None
        self.last_tcp_target_task: Optional[np.ndarray] = None

    # ------------------------------------------------------------ I/O (subclass)
    @abc.abstractmethod
    def _read_q(self) -> np.ndarray:
        """Measured joints, rad, Workshop order (Rotation..Jaw)."""

    @abc.abstractmethod
    def _send(self, q_cmd: np.ndarray) -> None:
        """Apply ``q_cmd`` for ONE tick and return when the tick is over."""

    @abc.abstractmethod
    def _r_world_base(self) -> np.ndarray:
        """3x3 rotation of the robot base link in the task/world frame."""

    def grasped(self) -> bool:
        return False

    def raw_frames(self):
        raise NotImplementedError

    # ------------------------------------------------------------ frames
    def _base_to_task(self, v: np.ndarray) -> np.ndarray:
        return self._r_world_base() @ v

    def _task_to_base(self, v: np.ndarray) -> np.ndarray:
        return self._r_world_base().T @ v

    # ------------------------------------------------------------ readback
    def q_obs(self) -> np.ndarray:
        return np.asarray(self._read_q(), dtype=np.float64)

    def tcp_pos(self) -> np.ndarray:
        return self._base_to_task(self.kin.tcp(self.q_obs())[0])

    def tcp_rotmat(self) -> np.ndarray:
        return self._r_world_base() @ self.kin.tcp(self.q_obs())[1]

    def tcp_pose7(self) -> list[float]:
        return [*self.tcp_pos().tolist(), *mat_to_quat_wxyz(self.tcp_rotmat()).tolist()]

    def gripper_width(self) -> float:
        """Approximate fingertip opening (m): chord of the ~0.08 m moving jaw swung from its
        closed limit. Good enough for empty-grasp detection; not a contact measurement."""
        jaw = self.q_obs()[5]
        return float(2 * 0.08 * np.sin(max(jaw - self.kin.lower[5], 0.0) / 2))

    def orientation_error(self) -> tuple[float, float]:
        """(pitch, yaw) error: references minus the measured hand, rad."""
        q = self.q_obs()
        dp = float((self.pitch_ref - self.kin.tool_pitch(q) + np.pi) % (2 * np.pi) - np.pi)
        dy = 0.0
        if self.yaw_ref is not None:
            dy = float((self.yaw_ref - self.kin.tool_yaw(q)[0] + np.pi / 2) % np.pi - np.pi / 2)
        else:
            dy = float(self.roll_ref - q[4])  # roll mode: the second reference is the joint
        return dp, dy

    # ------------------------------------------------------------ actuation
    def _tick(self, q_cmd: np.ndarray) -> None:
        self._send(q_cmd)
        self.tick_count += 1
        if self.step_hooks:
            rec = {
                "env_step": self.tick_count,
                "t": self.tick_count * self.tick_dt,
                "ctrl_step": self.ctrl_step_count,
                "decision_id": self.decision_id,
                "q_obs": self.q_obs(),
                "q_cmd": np.asarray(q_cmd, dtype=np.float64).copy(),  # what was sent
                "q_des": self.q_des.copy(),
                "tcp_obs": self.tcp_pose7(),
                "tcp_target": None if self.last_tcp_target_task is None
                else self.last_tcp_target_task.tolist(),
                "pitch_ref": self.pitch_ref,
                "yaw_ref": self.yaw_ref,
                "roll_ref": self.roll_ref,
                "annotation": dict(self.annotation),
                "contacts": self.contact_report() if hasattr(self, "contact_report") else [],
                "backend": self,
            }
            for hook in self.step_hooks:
                hook(rec)

    def _command(self) -> np.ndarray:
        q = self.q_des + self.q_bias
        q[:5] = np.clip(q[:5], self.kin.lower[:5], self.kin.upper[:5])
        return q

    def _update_bias(self, q_before: np.ndarray, q_cmd: np.ndarray) -> None:
        """At rest the measured joints sit ``sag`` below the command: learn it (EMA)."""
        if not self.use_gravity_bias:
            return
        q = self.q_obs()
        if np.max(np.abs(q[:5] - q_before[:5])) < self.bias_rest_tol_rad:
            sag = q_cmd - q
            sag[5] = 0.0  # the jaw stalls on objects by design; never "compensate" a grasp
            # learn gradually (<= bias_rate_rad per update) and within +-bias_max_rad. A push
            # against an obstacle would also look like sag; the executor restores the bias
            # from before the token whenever a token is aborted for contact or stall.
            step = np.clip(self.bias_alpha * (sag - self.q_bias), -self.bias_rate_rad, self.bias_rate_rad)
            self.q_bias = np.clip(self.q_bias + step, -self.bias_max_rad, self.bias_max_rad)

    def _ctrl_step(self, q_before: np.ndarray) -> None:
        q_cmd = self._command()
        self.ctrl_step_count += 1
        for _ in range(self.ticks_per_ctrl):
            self._tick(q_cmd)
        self._update_bias(q_before, q_cmd)

    def tcp_for_grasp(self, obj_center_task, width_m: float, gap_m: float = 0.002) -> np.ndarray:
        """TCP position that puts an object of ``width_m`` (centred at ``obj_center_task``)
        between the fingers, ``gap_m`` clear of the fixed finger, with the CURRENT orientation.

        The closing axis (gripper +x, from the fixed finger toward the moving jaw) is taken
        from the measured hand; with the yaw held it is horizontal.
        """
        c = self.tcp_rotmat()[:, 0]
        return np.asarray(obj_center_task, dtype=np.float64) - (self.fixed_finger_face_m + gap_m + width_m / 2) * c

    def hand_points(self, q) -> np.ndarray:
        """Gripper + jaw collision-hull vertices in the task frame at joints ``q`` (N, 3)."""
        poses = self.kin.link_poses(q)
        r = self._r_world_base()
        pts = [(r @ (poses[link][:3, :3] @ p.T + poses[link][:3, 3:4])).T for link, p in self._hull.items()]
        return np.vstack(pts)

    def hand_lowest_z(self, q) -> float:
        """Lowest task-frame z of the gripper + jaw collision hulls at joints ``q``."""
        poses = self.kin.link_poses(q)
        r = self._r_world_base()
        low = np.inf
        for link, pts in self._hull.items():
            t = poses[link]
            z = (r @ (t[:3, :3] @ pts.T + t[:3, 3:4]))[2]
            low = min(low, float(z.min()))
        return low

    def in_contact(self) -> bool:
        """Arm held away from its IK target by more than gravity can account for."""
        return bool(np.max(np.abs(self.q_des[:5] - self.q_obs()[:5])) > self.contact_tol_rad)

    def _solve(self, q_seed, target_task, pitch, yaw, roll=None):
        q_seed = np.asarray(q_seed, dtype=np.float64).copy()
        if yaw is None:
            q_seed[4] = self.roll_ref if roll is None else roll  # roll mode: the joint IS the reference
        q, err = self.kin.solve(q_seed, self._task_to_base(np.asarray(target_task)), pitch, yaw)
        return q, err

    def feasible(self, target_task, pitch: Optional[float] = None, yaw: Optional[float] = None,
                 use_yaw_ref: bool = True, roll: Optional[float] = None) -> tuple[bool, str]:
        """Can the hand be put at ``target_task`` with this orientation, exactly?

        Reasons for "no": outside the workspace box, IK residual above tolerance (unreachable
        / orientation cannot be held), or a joint within ``min_joint_margin_rad`` of a limit.
        """
        target_task = np.asarray(target_task, dtype=np.float64)
        pitch = self.pitch_ref if pitch is None else pitch
        if yaw is None and use_yaw_ref:
            yaw = self.yaw_ref
        lo, hi = np.asarray(self.workspace["min"]), np.asarray(self.workspace["max"])
        if np.any(target_task < lo) or np.any(target_task > hi):
            axis = "xyz"[int(np.argmax(np.maximum(lo - target_task, target_task - hi)))]
            return False, f"workspace({axis})"
        q, err = self._solve(self.q_obs(), target_task, pitch, yaw, roll)
        if yaw is None and not (self.kin.lower[4] + self.min_joint_margin_rad
                                <= q[4] <= self.kin.upper[4] - self.min_joint_margin_rad):
            return False, "joint_limit(Wrist_Roll)"
        if np.linalg.norm(err[:3]) > self.tol_pos_m:
            return False, f"ik_pos({np.linalg.norm(err[:3]) * 1000:.1f}mm)"
        if err.size > 3 and np.max(np.abs(err[3:])) > self.tol_ori_rad:
            return False, f"ik_ori({np.degrees(np.max(np.abs(err[3:]))):.1f}deg)"
        if self.hand_floor_task_z is not None and self._hull is not None:
            q_chk = q.copy()
            q_chk[5] = self.q_des[5]  # with the jaw as currently commanded
            low = self.hand_lowest_z(q_chk)
            if low < self.hand_floor_task_z:
                return False, f"hand_floor({(self.hand_floor_task_z - low) * 1000:.1f}mm)"
        margin = self.kin.joint_limit_margin(q)[:5].min()
        if margin < self.min_joint_margin_rad:
            return False, f"joint_limit({margin:.3f}rad)"
        return True, "ok"

    def commanded_tcp(self) -> np.ndarray:
        """TCP of the COMMANDED joints q_des (model, no sag): the setpoint the arm is chasing."""
        return self._base_to_task(self.kin.tcp(self.q_des)[0])

    def apply_delta(self, delta_m, grip_cmd: float, max_cmd_m: float) -> None:
        self.apply_delta_pose(delta_m, 0.0, 0.0, grip_cmd, max_cmd_m, 0.0)

    def apply_delta_pose(self, delta_m, d_pitch: float, d_yaw: float, grip_cmd: float,
                         max_cmd_m: float, max_cmd_rad: float) -> None:
        """ONE control step: the COMMANDED TCP moves by (capped) ``delta_m``; the pitch / yaw
        references move by (capped) ``d_pitch`` / ``d_yaw``. Orientation is in SO-101 terms
        (in-plane pitch, and the closing-axis yaw OR, in roll mode, the Wrist_Roll angle:
        +d_yaw = counter-clockwise seen from above for a downward hand).

        Everything here is relative to the COMMANDED state (q_des), never the measured one.
        Solving from the measured, sagging joints re-adds whatever sag the bias does not cover
        on every step, and that integrates: measured on the kinematic arm with 0.05 rad of
        unmodelled sag, the hand walked 278 mm off in 90 steps (and the rack-contact runaway
        was the same loop). From q_des, residual sag is a constant offset that the executor's
        bounded integral term closes.
        """
        delta = np.asarray(delta_m, dtype=np.float64)
        n = np.linalg.norm(delta)
        if n > max_cmd_m:
            delta = delta * (max_cmd_m / n)
        if d_pitch:
            self.pitch_ref += float(np.clip(d_pitch, -max_cmd_rad, max_cmd_rad))
        if d_yaw:
            step = float(np.clip(d_yaw, -max_cmd_rad, max_cmd_rad))
            if self.yaw_ref is not None:
                self.yaw_ref += step
            else:
                self.roll_ref += step
        q_before = self.q_obs()
        target_task = self.commanded_tcp() + delta
        self.last_tcp_target_task = target_task
        q_des, _ = self._solve(self.q_des, target_task, self.pitch_ref + self.pitch_trim, self.yaw_ref)
        q_des[5] = self.jaw_open_rad if grip_cmd > 0 else self.jaw_close_rad
        self.q_des = q_des
        self._ctrl_step(q_before)

    # ------------------------------------------------------------ lifecycle helpers
    def _begin_episode(self, settle_ctrl_steps: int = 30) -> None:
        """Adopt the current pose as the command and learn the gravity sag at rest."""
        self.q_des = self.q_obs()
        self.q_des[5] = self.jaw_open_rad
        self.q_bias = np.zeros(6)
        self.pitch_ref = self.kin.tool_pitch(self.q_des)
        self.yaw_ref = None
        self.roll_ref = float(self.q_des[4])  # keep the reset roll until told otherwise
        self.pitch_trim = 0.0
        self.annotation = {"phase": "reset"}
        self.tick_count = 0
        self.ctrl_step_count = 0
        self.decision_id = None
        self.last_tcp_target_task = None
        for _ in range(settle_ctrl_steps):
            self._ctrl_step(self.q_obs())
