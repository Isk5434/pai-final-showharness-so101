"""Show-Harness ``AtomicExec`` for SO-101: tokens are checked before they run, executed at a
speed the arm can follow, and finished only when the arm is at REST on the target.

Translation (MV_*): one token == ``step_m`` of TCP travel in the task frame, orientation held.
The END POINT (and midpoint) is checked first with ``backend.feasible``; an infeasible token
is refused whole (the arm does not move, ``last_blocked`` says why). Executing it partway
would bend the motion off its axis or drop the orientation: a label that lies.

Rotation (RT_*): the SO-101 can only tilt the hand about the arm-plane normal, and turn it
about the vertical through Wrist_Roll. So:
  RT_PITCH_FWD / RT_PITCH_BACK  in-plane tool pitch -/+ ``rot_step_rad`` (FWD tips the
                                fingertips outward, away from the base, as in Show-Harness)
  RT_YAW_CCW / RT_YAW_CW        closing-axis yaw +/- ``rot_step_rad`` (seen from above)
  RT_ROLL_*                     not realisable -> always refused
The TCP position is held during a rotation.

Why not Show-Harness's own loop as is (measured on the Workshop sim, 2026-10-03):
  * it commands up to ``step_m`` (2 cm) per control step: the low-gain SO-101 joints swing
    past the target, 5.5 mm overshoot on MV_UP/DOWN, and the wrist roll lags the base on
    MV_LEFT/RIGHT (5 deg transient yaw error);
  * its ``quiesce`` holds by commanding "zero delta from the MEASURED pose", so the setpoint
    follows the hand and an overshoot is never pulled back;
  * a token returns as soon as it is within tolerance, still moving, so the next frame is
    taken mid-motion.
Here every motion is one ``_servo`` toward a fixed TARGET with per-step caps
(``max_cmd_m``, ``rot_rate_rad``), and it only returns once the arm has settled there.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from scripts.trajectory.real2sim.atomic_tokenizer import MOVE_DIRS, AtomicExec

ROT_TOKENS = {
    "RT_PITCH_FWD": (-1.0, 0.0),
    "RT_PITCH_BACK": (1.0, 0.0),
    "RT_YAW_CCW": (0.0, 1.0),
    "RT_YAW_CW": (0.0, -1.0),
}
UNSUPPORTED = ("RT_ROLL_LEFT", "RT_ROLL_RIGHT")


def _wrap_pi(a: float) -> float:
    return float((a + np.pi / 2) % np.pi - np.pi / 2)


class So101AtomicExec(AtomicExec):
    def __init__(
        self,
        backend,
        max_cmd_m: float = 0.005,             # 5 mm per control step (30 Hz) = 15 cm/s
        rot_rate_rad: float = float(np.radians(0.5)),  # 0.5 deg per control step (1 deg: 3.04 mm TCP drift)
        settle_tol_m: float = 0.0003,         # "at rest": TCP moved < 0.3 mm in one control step
        settle_tol_rad: float = float(np.radians(0.05)),
        servo_max_ctrl_steps: int = 90,       # 3 s: give up and report
        i_max_m: float = 0.004,               # integral trim limits (position / pitch)
        i_max_rad: float = float(np.radians(1.0)),
        stall_steps_max: int = 20,            # at rest short of the target this long = stall
        **kw,
    ) -> None:
        kw.setdefault("rot_tol_rad", float(np.radians(0.2)))
        kw.setdefault("tol_m", 0.001)
        super().__init__(backend, max_cmd_m=max_cmd_m, **kw)
        self.rot_rate_rad = float(rot_rate_rad)
        self.settle_tol_m = float(settle_tol_m)
        self.settle_tol_rad = float(settle_tol_rad)
        self.servo_max_ctrl_steps = int(servo_max_ctrl_steps)
        self.i_max_m = float(i_max_m)
        self.i_max_rad = float(i_max_rad)
        self.stall_steps_max = int(stall_steps_max)
        self.last_blocked: Optional[str] = None
        self.last_converged: bool = True
        self.hold_target: Optional[np.ndarray] = None  # TCP position held between tokens
        self.last_contact: bool = False
        self.last_stall: bool = False
        self.last_abort: Optional[dict] = None

    def _label(self, token: str, **extra) -> None:
        self.backend.annotation = {"token": token, **extra}

    def _refuse(self, token: str, why: str) -> float:
        self.last_blocked = why
        self._label(token, blocked=why)
        return 0.0

    # -------------------------------------------------------------- the one primitive
    def _servo(self, target, pitch_t: Optional[float] = None, yaw_t: Optional[float] = None,
               budget: Optional[int] = None, roll_t: Optional[float] = None) -> bool:
        """Drive the TCP to ``target`` (and the references to pitch_t / yaw_t or roll_t),
        capped per control step, and return once the arm is AT REST there. Returns
        convergence. ``yaw_t`` switches the backend to yaw mode; ``roll_t`` (roll mode) moves
        the Wrist_Roll reference."""
        be = self.backend
        target = np.asarray(target, dtype=np.float64)
        pitch_t = be.pitch_ref if pitch_t is None else pitch_t
        if yaw_t is not None and be.yaw_ref is None:
            be.yaw_ref = be.kin.tool_yaw(be.q_obs())[0]
        self.hold_target = target
        prev_p, prev_o = be.tcp_pos(), self._orient()
        contact_steps = stall_steps = 0
        # Integral action for the last millimetre: gravity sag that the joint bias does not
        # remove leaves the arm AT REST 1.5 mm / 0.2 deg short (measured in free space). While
        # at rest and not at the target, aim past it by the accumulated residual.
        i_pos = np.zeros(3)  # per target; the pitch trim persists like the gravity bias
        bias_before, trim_before = be.q_bias.copy(), be.pitch_trim
        for _ in range(budget or self.servo_max_ctrl_steps):
            d_pitch = pitch_t - be.pitch_ref
            if yaw_t is not None:
                d_yaw = _wrap_pi(yaw_t - be.yaw_ref)
            elif roll_t is not None and be.yaw_ref is None:
                d_yaw = roll_t - be.roll_ref
            else:
                d_yaw = 0.0
            # The COMMANDED TCP walks to (target + integral trim) at <= 5 mm per control step,
            # decelerating over the last centimetre (a constant 5 mm step swung the low-gain
            # joints 2.3 mm past). The measured hand follows; it only feeds the integral.
            err = (target + i_pos) - be.commanded_tcp()
            cap = min(self.max_cmd_m, max(0.001, 0.5 * float(np.linalg.norm(err))))
            be.apply_delta_pose(err, d_pitch, d_yaw, self.grip_cmd, cap, self.rot_rate_rad)
            p, o = be.tcp_pos(), self._orient()
            ep, ey = be.orientation_error()
            at_target = (np.linalg.norm(target - p) < self.tol_m
                         and abs(pitch_t - be.pitch_ref) < 1e-12
                         and (yaw_t is None or abs(_wrap_pi(yaw_t - be.yaw_ref)) < 1e-12)
                         and (roll_t is None or be.yaw_ref is not None or abs(roll_t - be.roll_ref) < 1e-12)
                         and abs(ep) < self.rot_tol_rad and abs(ey) < self.rot_tol_rad)
            at_rest = (np.linalg.norm(p - prev_p) < self.settle_tol_m
                       and abs(o[0] - prev_o[0]) < self.settle_tol_rad
                       and abs(_wrap_pi(o[1] - prev_o[1])) < self.settle_tol_rad)
            # Two signatures of "something is in the way": a joint held far from its IK target
            # while the hand makes no progress (hard push), or the hand at rest short of the
            # target (soft stop; the joint gap of a 1 cm deficit is only ~0.04 rad).
            # The gap alone is NOT contact: at full reach gravity sag + motion lag exceed any
            # fixed threshold while the hand moves exactly right (measured: a correct 20.2 mm
            # MV_FWD at x = 0.26 m was flagged, and so was the vial approach).
            progress = float(np.linalg.norm(target - prev_p) - np.linalg.norm(target - p))
            # measured contact (sim sensor) is trusted as is; the kinematic fallback (no sensor,
            # e.g. the real arm) additionally needs "no progress" to avoid full-reach false fires
            pushing = be.in_contact() and (be.contact_is_measured
                                           or (progress < 0.0005 and not at_target))
            contact_steps = contact_steps + 1 if pushing else 0
            short = at_rest and not at_target
            if short:  # integrate the residual (bounded: a few mm / a degree, never a reference)
                i_pos = np.clip(i_pos + 0.5 * (target - p), -self.i_max_m, self.i_max_m)
                be.pitch_trim = float(np.clip(be.pitch_trim + 0.5 * ep, -self.i_max_rad, self.i_max_rad))
            # a stall is "at rest and short of the target for stall_steps_max control steps"
            # (the integral had that long to close the gap) - not "short for a moment"
            stall_steps = stall_steps + 1 if short else 0
            prev_p, prev_o = p, o
            if contact_steps >= 3 or stall_steps >= self.stall_steps_max:
                # Something is holding the arm off its target. Stop pushing: adopt where the
                # hand IS as the target, and report. Never keep driving into it (the gravity
                # bias used to learn the push and the arm ran away).
                q = be.q_obs()
                self.last_abort = {
                    "kind": "contact" if contact_steps >= 3 else "stall",
                    "pos_err_mm": round(float(np.linalg.norm(target - p) * 1000), 2),
                    "pitch_err_deg": round(float(np.degrees(ep)), 3),
                    "second_err_deg": round(float(np.degrees(ey)), 3),
                    "joint_gap_rad": round(float(np.max(np.abs(be.q_des[:5] - q[:5]))), 4),
                    "bias_rad": [round(float(b), 4) for b in be.q_bias[:5]],
                    "contacts": be.contact_report() if hasattr(be, "contact_report") else [],
                }
                if be.hand_floor_task_z is not None and be._hull is not None:
                    self.last_abort["hand_low_above_floor_mm"] = round(
                        (be.hand_lowest_z(q) - be.hand_floor_task_z) * 1000, 2)
                self.hold_target = be.tcp_pos()
                be.pitch_ref, yaw_now = self._orient()
                if be.yaw_ref is not None:
                    be.yaw_ref = yaw_now
                else:
                    be.roll_ref = float(be.q_obs()[4])
                # undo what the push may have taught the bias / trim, then command = where the
                # arm IS (minus the sag): nothing left to push with
                be.q_bias, be.pitch_trim = bias_before, trim_before
                be.q_des[:5] = q[:5] - be.q_bias[:5]
                be.apply_delta_pose(np.zeros(3), 0.0, 0.0, self.grip_cmd, self.max_cmd_m, 0.0)
                self.last_contact = self.last_abort["kind"] == "contact"
                self.last_stall = self.last_abort["kind"] == "stall"
                self.last_converged = False
                return False
            if at_target and at_rest:
                self.last_converged = True
                return True
        self.last_converged = False
        self.last_contact = False
        return False

    def _orient(self) -> tuple[float, float]:
        q = self.backend.q_obs()
        return self.backend.kin.tool_pitch(q), self.backend.kin.tool_yaw(q)[0]

    # -------------------------------------------------------------- dry check
    def check(self, token: str) -> tuple[bool, str]:
        """Would ``token`` be executed (not refused)? Nothing moves.

        Generators call this BEFORE recording a frame, so a refused token never becomes a
        training label. Gripper tokens are always executable.
        """
        be = self.backend
        if token in MOVE_DIRS:
            node = self.lattice_node(be.tcp_pos())
            target = node + MOVE_DIRS[token] * self.step_m
            for probe in (target, (node + target) / 2):
                ok, why = be.feasible(probe)
                if not ok:
                    return False, why
            return True, "ok"
        if token.startswith("RT_"):
            if token in UNSUPPORTED or token not in ROT_TOKENS:
                return False, "unsupported_on_5dof"
            sp, sy = ROT_TOKENS[token]
            pitch_t = be.pitch_ref + sp * self.rot_step_rad
            yaw_t = None if be.yaw_ref is None else be.yaw_ref + sy * self.rot_step_rad
            roll_t = be.roll_ref + sy * self.rot_step_rad if be.yaw_ref is None else None
            return be.feasible(self.lattice_node(be.tcp_pos()), pitch_t, yaw_t,
                               use_yaw_ref=False, roll=roll_t)
        if token in ("GRASP", "RELEASE"):
            return True, "ok"
        return False, f"unknown_token({token})"

    # -------------------------------------------------------------- translation
    def move(self, token: str) -> float:
        be = self.backend
        start = be.tcp_pos()
        node = self.lattice_node(start)
        target = node + MOVE_DIRS[token] * self.step_m
        for probe in (target, (node + target) / 2):
            ok, why = be.feasible(probe)
            if not ok:
                return self._refuse(token, why)
        self.last_blocked = None
        self.last_contact = self.last_stall = False
        self.last_abort = None
        self._label(token)
        if not self._servo(target):
            # moved PART of the way: recording it as this token would be a lie
            self.last_blocked = ("contact" if self.last_contact else
                                 "stall" if self.last_stall else "not_converged")
            self._label(token, blocked=self.last_blocked)
        return float(np.linalg.norm(be.tcp_pos() - start))

    # -------------------------------------------------------------- rotation
    def rotate(self, token: str) -> float:
        be = self.backend
        if token in UNSUPPORTED or token not in ROT_TOKENS:
            return self._refuse(token, "unsupported_on_5dof")
        sp, sy = ROT_TOKENS[token]
        pitch_t = be.pitch_ref + sp * self.rot_step_rad
        # yaw mode: turn the closing-axis heading; roll mode: turn the hand about its own axis
        # (Wrist_Roll; +roll is counter-clockwise from above, measured)
        yaw_t = None if be.yaw_ref is None else be.yaw_ref + sy * self.rot_step_rad
        roll_t = be.roll_ref + sy * self.rot_step_rad if be.yaw_ref is None else None
        pos = self.lattice_node(be.tcp_pos())
        ok, why = be.feasible(pos, pitch_t, yaw_t, use_yaw_ref=False, roll=roll_t)
        if not ok:
            return self._refuse(token, why)
        self.last_blocked = None
        self.last_contact = False
        self._label(token)
        o0, r0 = self._orient(), be.q_obs()[4]
        if not self._servo(pos, pitch_t, yaw_t, roll_t=roll_t):
            self.last_blocked = ("contact" if self.last_contact else
                                 "stall" if self.last_stall else "not_converged")
            self._label(token, blocked=self.last_blocked)
        o1, r1 = self._orient(), be.q_obs()[4]
        turn = _wrap_pi(o1[1] - o0[1]) if be.yaw_ref is not None else r1 - r0
        return float(np.hypot(o1[0] - o0[0], turn))

    # -------------------------------------------------------------- holding / gripper
    def _hold_point(self) -> np.ndarray:
        return self.backend.tcp_pos() if self.hold_target is None else self.hold_target

    def quiesce(self, tol_m=None, max_steps=None) -> int:
        tok = self.backend.annotation.get("token", "-")
        self._label(tok, phase="settle")
        c0 = self.backend.ctrl_step_count
        self._servo(self._hold_point(), budget=max_steps)
        return self.backend.ctrl_step_count - c0

    def hold(self, n: int = 1) -> None:
        target = self._hold_point()
        for _ in range(n):
            self.backend.apply_delta_pose(target - self.backend.commanded_tcp(), 0.0, 0.0,
                                          self.grip_cmd, self.max_cmd_m, 0.0)

    def _gripper(self, token: str, cmd: float) -> float:
        # At rest first: closing while moving drags the object, opening while moving throws
        # it (Show-Harness's own measured failure modes). Then hold the TCP still while the
        # jaw moves, and let it settle again.
        self._label(token, phase="settle")
        self._servo(self._hold_point())
        self._label(token)
        self.grip_cmd = cmd
        self.hold(self.gripper_steps)
        self._servo(self._hold_point())
        return self.backend.gripper_width()

    def grasp(self) -> float:
        return self._gripper("GRASP", -1.0)

    def release(self) -> float:
        return self._gripper("RELEASE", 1.0)

    def run(self, token: str) -> float:
        """Execute any vocabulary token; returns metres (MV), radians (RT) or width (gripper)."""
        if token in MOVE_DIRS:
            return self.move(token)
        if token.startswith("RT_"):
            return self.rotate(token)
        if token == "GRASP":
            return self.grasp()
        if token == "RELEASE":
            return self.release()
        raise ValueError(f"unknown token {token!r}")

    def goto_pose(self, pos_task, pitch: float, yaw: Optional[float] = None,
                  roll: Optional[float] = None, tol_m: float = 1e-3, budget: int = 400) -> bool:
        """Servo (NOT a token, never recorded as one) to a start pose with an orientation:
        pitch + yaw (yaw mode) or pitch + Wrist_Roll ``roll`` (roll mode, the default)."""
        self._label("-", phase="goto_pose")
        return self._servo(pos_task, pitch, yaw, budget=budget, roll_t=roll)
