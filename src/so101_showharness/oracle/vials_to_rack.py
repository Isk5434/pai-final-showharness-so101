"""特権オラクル: 横倒しのバイアルを1本つかみ、立ててラックの空いた穴へ落とす。

人のテレオペデモ（sreetz-nv/so101_teleop_vials_rack_left, 3 エピソードを解析）と同じ流れ:
  - 手首ロールは全期間 −90° 前後（= 指が水平に閉じる）
  - 手先ピッチ 56〜62° で、バイアルの真横に指を下ろして閉じる
  - 持ち上げてから手先を前に約 70° 倒し（ピッチ −3〜−18°）、ラックの上 16〜20 cm で離す

記録の約束（Show-Harness と同じ）:
  - 各トークンの「実行前」のフレームとトークンを1組記録する
  - 記録する前に ``exec.check`` で実行できることを確かめる。拒否されるトークンは記録しない
  - 途中で接触・未収束になったら、ラベルが嘘になるのでエピソード全体を不採用にする
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from scripts.trajectory.real2sim.atomic_tokenizer import MOVE_DIRS, token_for_axis
from so101_showharness.controller import ROLL_CLOSING_HORIZONTAL

INSTRUCTION = "Pick up the vial and place it in the rack"  # Workshop の既定指示文


class EpisodeInvalid(RuntimeError):
    """記録済みのラベルが実際の動きと一致しなくなった（接触・未収束・計画不能）。"""


@dataclass
class OracleParams:
    grasp_pitch_deg: float = 60.0         # 人のデモ: 56〜62°
    release_pitch_min_deg: float = -25.0  # これより前には倒さない
    release_pitch_target_deg: float = -20.0  # 人のデモの離す時のピッチ（−3〜−18°）
    min_rotate_z_m: float = 0.14          # 回転中はバイアルが下にぶら下がるので、この高さ以上で回す
    upright_cos: float = float(np.cos(np.radians(15.0)))  # バイアル軸が鉛直から 15° 以内で「立った」
    begin_world: tuple = (0.20, -0.02, 0.17)  # 開始姿勢（記録しない移動で向かう）
    approach_height_m: float = 0.06       # 把持点の上この高さから降りる
    grasp_below_axis_m: float = 0.004     # 指先をバイアル軸より 4 mm 下へ（8 mm はテーブル床まで 0.1 mm で余裕なし）
    lift_to_z_m: float = 0.16             # 持ち上げ後の TCP 高さ（タスク座標）
    release_clearance_m: float = 0.02     # バイアル下端を穴の入口のこの高さ上で離す（最低値）
    hand_rack_clearance_m: float = 0.010  # 離す点で手（衝突形状）とラックの最小隙間
    grasp_along_axis_jitter_m: float = 0.01  # 把持位置を軸方向にばらつかせる（多様性）
    tol_m: float = 0.011                  # 2 cm 格子での到達判定（Show-Harness 既定）
    max_tokens: int = 120
    jaw_open_rad: float = 0.45            # 把持前の開き。人のデモ 0.36〜0.46（0.9 だと上から押し出す）
    # 把持位置は実験で決定（scripts/sim/probe_grasp.py、2026-10-03、18 条件中 14 成功）:
    # ピッチ 60°・中心 35 mm・軸の 4 mm 下を選択。近傍（35/0, 35/8, 31/4）もすべて成功。
    grasp_center_from_tcp_m: float = 0.035


@dataclass
class EpisodeLog:
    tokens: list = field(default_factory=list)
    events: list = field(default_factory=list)


class VialsToRackOracle:
    def __init__(self, backend, executor, scene, writer, rng: np.random.Generator,
                 params: Optional[OracleParams] = None) -> None:
        self.be = backend
        self.ex = executor
        self.scene = scene
        self.writer = writer
        self.rng = rng
        self.p = params or OracleParams()
        self.log = EpisodeLog()

    # ------------------------------------------------------------ recorded execution
    def emit(self, token: str) -> None:
        if self.writer.step >= self.p.max_tokens:
            raise EpisodeInvalid("token_budget")
        ok, why = self.ex.check(token)
        if not ok:
            raise EpisodeInvalid(f"refused_before_record:{token}:{why}")
        av, wr = self.be.grab_frames()
        kind = "move" if token in MOVE_DIRS else ("rotate" if token.startswith("RT_") else "gripper")
        self.be.decision_id = self.writer.step
        self.writer.add_step(token=token, kind=kind, agentview=av, wrist=wr,
                             gripper_closed=self.ex.gripper_closed,
                             ee_pose=self.be.tcp_pose7(), width=self.be.gripper_width())
        self.log.tokens.append(token)
        self.ex.run(token)
        if self.ex.last_blocked:  # 実行中に中断: このラベルはもう正しくない
            self.log.events.append({"abort": token, "detail": self.ex.last_abort})
            raise EpisodeInvalid(f"aborted:{token}:{self.ex.last_blocked}")

    def emit_done(self) -> None:
        av, wr = self.be.grab_frames()
        self.be.decision_id = self.writer.step
        self.writer.add_step(token="DONE", kind="done", agentview=av, wrist=wr,
                             gripper_closed=self.ex.gripper_closed,
                             ee_pose=self.be.tcp_pose7(), width=self.be.gripper_width())
        self.log.tokens.append("DONE")

    # ------------------------------------------------------------ planners
    def _axis_runs(self, target: np.ndarray, axes) -> list[str]:
        node = self.ex.lattice_node(self.be.tcp_pos())
        toks = []
        for ax in axes:
            n = int(np.round((target[ax] - node[ax]) / self.ex.step_m))
            toks += [token_for_axis(ax, np.sign(n))] * abs(n)
        return toks

    def go_to(self, target: np.ndarray, axes_order=None, tries: int = 4) -> None:
        """格子上をマンハッタン移動。拒否される軸は順番を入れ替えて試す（記録はしない）。"""
        target = np.asarray(target, dtype=np.float64)
        for _ in range(tries * 8):
            err = target - self.be.tcp_pos()
            if np.all(np.abs(err) <= self.p.tol_m):
                return
            order = list(axes_order) if axes_order else list(self.rng.permutation(3))
            moved = False
            for ax in order:
                toks = self._axis_runs(target, [ax])
                if not toks:
                    continue
                if self.ex.check(toks[0])[0]:
                    for t in toks:
                        if not self.ex.check(t)[0]:
                            break
                        self.emit(t)
                    moved = True
                    break
            if not moved:
                raise EpisodeInvalid(f"stuck_at:{np.round(self.be.tcp_pos(), 3).tolist()}->{np.round(target, 3).tolist()}")
        raise EpisodeInvalid("go_to_no_convergence")

    def _rotation_station(self, n_rot: int, radius_steps: int = 4) -> Optional[np.ndarray]:
        """Nearest lattice point (current lattice) from which the next ``n_rot`` RT_PITCH_FWD
        tokens are all feasible, staying above ``min_rotate_z_m`` (the vial hangs below)."""
        be, ex, p = self.be, self.ex, self.p
        here = ex.lattice_node(be.tcp_pos())
        pitches = [be.pitch_ref - np.radians(10.0) * k for k in range(n_rot + 1)]
        rng = range(-radius_steps, radius_steps + 1)
        cands = sorted(((abs(dx) + abs(dy) + abs(dz), dx, dy, dz)
                        for dx in rng for dy in rng for dz in range(0, radius_steps + 1)))
        for _, dx, dy, dz in cands:  # nearest first: the first fully feasible one wins
            pt = here + ex.step_m * np.array([dx, dy, dz])
            if pt[2] < p.min_rotate_z_m:
                continue
            if all(be.feasible(pt, pc, None, use_yaw_ref=False, roll=be.roll_ref)[0] for pc in pitches):
                return pt
        return None

    def _hand_rack_gap(self, tcp_task) -> float:
        """Gap between the hand (gripper + jaw collision hulls, current orientation and jaw
        command) and the rack, if the TCP were at ``tcp_task``."""
        be = self.be
        q = be.q_obs().copy()
        q[4] = be.roll_ref
        q, _ = be.kin.solve(q, be._task_to_base(np.asarray(tcp_task)), be.pitch_ref, None, iters=80)
        q[5] = be.q_des[5]
        return self.scene.rack_gap(be.hand_points(q))

    def _make_room_for(self, token: str, tries: int = 3) -> None:
        """A token is refused here (joint limit / reach): step the hand (recorded tokens) to a
        neighbouring lattice point from which it is executable. Moves are tried in an order
        that tends to relieve the wrist (toward the base, then up / down / out)."""
        ex = self.ex
        for _ in range(tries):
            for mv in ("MV_BACK", "MV_UP", "MV_DOWN", "MV_FWD"):
                if not ex.check(mv)[0]:
                    continue
                self.emit(mv)
                if ex.check(token)[0]:
                    self.log.events.append({"made_room": mv, "for": token})
                    return
                break
        raise EpisodeInvalid(f"refused:{token}:{ex.check(token)[1]}")

    # ------------------------------------------------------------ grasp side
    def _obstacle_gap(self, pts: np.ndarray, exclude: str) -> tuple[float, str]:
        """Min distance from hand points to the rack (oriented box) and the other vials
        (capsules), and which obstacle it is."""
        sc = self.scene
        gaps = [(sc.rack_gap(pts), "rack")]
        for v in sc.vials():
            if v.name == exclude:
                continue
            a = v.pos - v.axis * sc.vial_length / 2
            ab = v.axis * sc.vial_length
            t = np.clip(((pts - a) @ ab) / (ab @ ab), 0.0, 1.0)
            gaps.append((float((np.linalg.norm(pts - (a + t[:, None] * ab), axis=1) - sc.vial_diameter / 2).min()), v.name))
        return min(gaps)

    def _choose_side(self, vial, grasp_center, pitch):
        be, p = self.be, self.p
        best = None
        sides = [(r, b) for r in (ROLL_CLOSING_HORIZONTAL, ROLL_CLOSING_HORIZONTAL + np.pi)
                 for b in (p.grasp_below_axis_m, 0.0)]  # 0 mm below the axis also holds (probe_grasp)
        for roll, below in sides:
            if not (be.kin.lower[4] < roll < be.kin.upper[4]):
                continue
            # closing axis for this roll at the grasp: solve the pose, read the hand frame
            q = be.q_obs().copy()
            q[4] = roll
            q, _ = be.kin.solve(q, be._task_to_base(grasp_center + [0, 0, 0.06]), pitch, None, iters=200)
            c = be._r_world_base() @ be.kin.tcp(q)[1][:, 0]
            tcp = grasp_center - c * p.grasp_center_from_tcp_m - [0, 0, below]
            worst, why, nearest = np.inf, None, None
            for dz in np.linspace(p.approach_height_m, 0.0, 7):  # the whole descent
                # the side must be EXECUTABLE (floor / reach / joint limits), not only clear:
                # roll +90 hangs the open jaw lower and was refused at the last 2 cm (measured)
                ok, reason = be.feasible(tcp + [0, 0, dz], pitch, None, use_yaw_ref=False, roll=roll)
                if not ok:
                    worst, why = -np.inf, reason
                    break
                qq, _ = be.kin.solve(q, be._task_to_base(tcp + [0, 0, dz]), pitch, None, iters=100)
                qq[5] = be.jaw_open_rad
                g, obj = self._obstacle_gap(be.hand_points(qq), vial.name)
                if g < worst:
                    worst, nearest = g, obj
            self.log.events.append({"side_roll_deg": round(float(np.degrees(roll))), "below_mm": round(below * 1000),
                                    "worst_gap_mm":
                                    None if not np.isfinite(worst) else round(worst * 1000, 1),
                                    "nearest": nearest, "refused": why})
            if best is None or worst > best[2]:
                best = (roll, tcp, worst)
        if best is None or not np.isfinite(best[2]) or best[2] < 0.005:
            raise EpisodeInvalid(f"no_clear_grasp_side(gap={0 if best is None else best[2] * 1000:.1f}mm)")
        return best

    # ------------------------------------------------------------ the task
    def run(self) -> dict:
        p, be, ex, sc = self.p, self.be, self.ex, self.scene
        lying = sc.lying_vials()
        free = [s for s in sc.slots() if not sc.slot_occupied(s)]
        if not lying or not free:
            raise EpisodeInvalid("no_lying_vial_or_free_slot")
        # 指はアーム平面に垂直に閉じる（ロール −90°）ので、バイアル軸がアームの向きと
        # ずれている角 φ だけ、立てたときに傾きが残る（最大で up_z = cos φ）。ずれの小さい
        # バイアルを選ぶ（RT_YAW で合わせるのは今後の課題）。
        def mismatch(v):
            heading = np.arctan2(v.pos[1], v.pos[0])
            ax = np.arctan2(v.axis[1], v.axis[0])
            d = (ax - heading + np.pi / 2) % np.pi - np.pi / 2
            return abs(d)
        vial = min(lying, key=mismatch)
        if np.cos(mismatch(vial)) < p.upright_cos:
            raise EpisodeInvalid(f"vial_axis_vs_arm_heading({np.degrees(mismatch(vial)):.0f}deg)")
        slot = min(free, key=lambda s: np.linalg.norm((s - vial.pos)[:2]))
        self.log.events.append({"vial": vial.name, "vial_pos": vial.pos.tolist(), "slot": slot.tolist()})

        pitch = np.radians(p.grasp_pitch_deg)
        along = float(self.rng.uniform(-p.grasp_along_axis_jitter_m, p.grasp_along_axis_jitter_m))
        grasp_center = vial.pos + vial.axis * along
        # 1) どちら側から挟むか: 手首ロール −90° と +90° はどちらも指が水平に閉じるが、固定指の
        #    側が逆になる。降下経路全体で、手の衝突形状と障害物（ラック・他のバイアル）の
        #    隙間が大きい方を選ぶ（隣のラックに手の甲が当たる失敗を実測 → 対策）。
        roll, grasp_tcp, clearance = self._choose_side(vial, grasp_center, pitch)
        gap = p.grasp_center_from_tcp_m - be.fixed_finger_face_m - sc.vial_diameter / 2
        self.log.events.append({"grasp_roll_deg": float(np.degrees(roll)), "clearance_mm": round(clearance * 1000, 1)})
        # 格子を把持点に合わせ、開始姿勢を格子点に置く → トークンで把持点ちょうどに着く
        ex.lattice_origin = grasp_tcp.copy()
        begin = np.asarray(p.begin_world) - (be.robot.data.root_pos_w[0].cpu().numpy())
        begin = ex.lattice_origin + ex.step_m * np.round((begin - ex.lattice_origin) / ex.step_m)
        if not ex.goto_pose(begin, pitch, roll=roll):
            raise EpisodeInvalid("begin_pose_unreachable")
        be.decision_id = None
        self.log.events.append({"begin_task": begin.tolist(), "grasp_tcp": grasp_tcp.tolist(), "gap_m": gap})

        # 2) 把持点の上へ → 降下 → 把持
        above = grasp_tcp + [0, 0, p.approach_height_m]
        self.go_to(above)
        self.go_to(grasp_tcp, axes_order=[2])
        self.emit("GRASP")

        # 3) 持ち上げ: 届く高さまで（遠いバイアルではピッチ 60° のまま 16 cm に届かない）
        while be.tcp_pos()[2] < p.lift_to_z_m - p.tol_m and ex.check("MV_UP")[0]:
            self.emit("MV_UP")
        held = sc.vial(vial.name)
        if held.pos[2] < vial.pos[2] + 0.03:
            raise EpisodeInvalid("vial_not_lifted")

        # 4) 前に倒してバイアルを立てる。今の位置で全部回せないなら、近くで「回転列が全部
        #    実行可能な格子点」（回転ステーション）を探してそこへ移ってから回す
        n_rot = int(np.ceil((np.degrees(be.pitch_ref) - p.release_pitch_target_deg) / 10.0))
        station = self._rotation_station(n_rot)
        if station is not None and np.linalg.norm(station - be.tcp_pos()) > p.tol_m:
            self.log.events.append({"rotation_station": station.tolist()})
            self.go_to(station, axes_order=[2, 0, 1] if station[2] > be.tcp_pos()[2] else [0, 1, 2])
        while True:
            v = sc.vial(vial.name)
            if v.up_z >= p.upright_cos:
                break
            if np.degrees(be.pitch_ref) - 10 < p.release_pitch_min_deg:
                raise EpisodeInvalid(f"cannot_upright(up_z={v.up_z:.2f})")
            if not ex.check("RT_PITCH_FWD")[0]:
                self._make_room_for("RT_PITCH_FWD")
            self.emit("RT_PITCH_FWD")

        # 5) 穴の上へ: バイアル中心を穴の真上、下端を入口の少し上に
        v = sc.vial(vial.name)
        bottom = v.pos - v.axis * np.sign(v.axis[2]) * sc.vial_length / 2
        tcp = be.tcp_pos()
        target = np.r_[slot[:2] - (v.pos - tcp)[:2], slot[2] + p.release_clearance_m + (tcp - bottom)[2]]
        ex.lattice_origin = None  # ここからは相対の 2 cm（把持格子はもう関係ない）
        # 手は上向き（ピッチ −20°）なのでグリッパー本体は TCP より下にある。そのまま水平に
        # ラック上へ入ると本体がラック上面をこする（接触センサで 2.9〜3.9 N を実測）。
        # 離す点で手とラックの隙間が足りなければ 2 cm ずつ上げる（人も 16〜20 cm 上から落とす）。
        raised = 0
        while self._hand_rack_gap(target) < p.hand_rack_clearance_m and raised < 5:
            target = target + [0, 0, ex.step_m]
            raised += 1
        # 経路: 先に真上へ（離す高さまで）、次に水平。水平経路も隙間を確認する
        if target[2] > be.tcp_pos()[2]:
            self.go_to(np.r_[be.tcp_pos()[:2], target[2]], axes_order=[2])
        path_gap = min(self._hand_rack_gap(be.tcp_pos() + (target - be.tcp_pos()) * t)
                       for t in np.linspace(0.0, 1.0, 6))
        self.log.events.append({"release_raised_steps": raised, "path_gap_mm": round(path_gap * 1000, 1)})
        if path_gap < p.hand_rack_clearance_m / 2:
            raise EpisodeInvalid(f"release_path_too_close_to_rack({path_gap * 1000:.1f}mm)")
        self.go_to(target, axes_order=[0, 1, 2])
        self.log.events.append({"release_tcp": be.tcp_pos().tolist(), "release_target": target.tolist()})

        # 6) 離す → 判定
        self.emit("RELEASE")
        ok = False
        for _ in range(60):
            ex.hold(1)
            if be.success():
                ok = True
                break
        self.emit_done()
        return {"success": ok, "tokens": len(self.log.tokens), "events": self.log.events}
