"""Workshop「バイアル → ラック」シーンの特権情報（オラクル専用。方策には渡さない）。

すべてタスク座標（ワールドの向き・ロボット基部が原点）で返す。
ラックの穴（`Rack_Left/Body1/Mesh/top_*`）は、初期化時に USD からラック基準の局所姿勢を
読み、以後はラックの現在姿勢に掛けて求める（Fabric 有効時は USD の子 prim の姿勢が
更新されないため、直接読むと古い値になる）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.trajectory.real2sim.atomic_tokenizer import to_np
from so101_showharness.kinematics import quat_wxyz_to_mat

VIALS = ("vial_1", "vial_2", "vial_3")
RACK = "rack_left"
RACK_PRIM = "/World/envs/env_0/Rack_Left"


def _tf(pos, rot) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = pos
    return t


@dataclass
class Vial:
    name: str
    pos: np.ndarray      # 幾何中心（軸上）。※ルート原点ではない（原点は円錐側の端から約 17 mm）
    rot: np.ndarray      # 3x3
    axis: np.ndarray     # 長手方向の単位ベクトル（局所 z、+ がキャップ側）
    root: np.ndarray     # ルート原点（円錐の先端寄り）

    @property
    def up_z(self) -> float:
        """軸の鉛直成分の絶対値（1 = 立っている、0 = 横倒し）。"""
        return float(abs(self.axis[2]))


class VialsRackScene:
    def __init__(self, backend) -> None:
        from pxr import Usd, UsdGeom

        self.be = backend
        self.u = backend.u
        stage = self.u.sim.stage
        xc = UsdGeom.XformCache()
        rack_prim = stage.GetPrimAtPath(RACK_PRIM)
        rack_w = np.array(xc.GetLocalToWorldTransform(rack_prim)).T
        self.slot_local = []
        for p in Usd.PrimRange(rack_prim):
            if p.GetName().startswith("top_"):
                slot_w = np.array(xc.GetLocalToWorldTransform(p)).T
                self.slot_local.append(np.linalg.inv(rack_w) @ slot_w)
        if not self.slot_local:
            raise RuntimeError("ラックの穴（top_*）が見つからない")
        # テーブル面 = マット上面。マットは静的物体で、DR はヨー回転だけなので高さは変わらず、
        # USD のワールド外形が正しい（バイアルから推定すると数 mm ずれる）。
        mb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False) \
            .ComputeWorldBound(stage.GetPrimAtPath("/World/envs/env_0/Mat")).ComputeAlignedRange()
        self.table_top = float(mb.GetMax()[2]) - float(self._base_w()[2])
        rb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False) \
            .ComputeUntransformedBound(rack_prim).ComputeAlignedRange()
        self.rack_local_box = np.array([list(rb.GetMin()), list(rb.GetMax())])
        # 寸法と幾何中心は「局所」外形から取る。ワールド外形は Fabric 有効時に USD が物理に
        # 追従せず、乱択前の初期位置のまま古くなる（測って確認: 3本とも同じ x 範囲だった）。
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False)
        r = cache.ComputeUntransformedBound(stage.GetPrimAtPath("/World/envs/env_0/Vial_1")).ComputeAlignedRange()
        lo, hi = np.array(r.GetMin()), np.array(r.GetMax())
        ext = hi - lo
        self.vial_center_local = (lo + hi) / 2           # ルート原点 → 幾何中心（局所座標）
        self.vial_diameter = float(np.sort(ext)[0])
        self.vial_length = float(np.sort(ext)[-1])

    def _base_w(self) -> np.ndarray:
        return to_np(self.be.robot.data.root_pos_w[0])

    def vials(self) -> list[Vial]:
        out = []
        for name in VIALS:
            obj = self.u.scene[name]
            rot = quat_wxyz_to_mat(to_np(obj.data.root_quat_w[0]))
            root = to_np(obj.data.root_pos_w[0]) - self._base_w()
            out.append(Vial(name, root + rot @ self.vial_center_local, rot, rot[:, 2], root))
        return out

    def vial(self, name: str) -> Vial:
        return next(v for v in self.vials() if v.name == name)

    def rack_box(self, margin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """ラックの現在の外形（タスク座標の AABB）。USD のワールド外形は古いので使わない。"""
        rack = self.u.scene[RACK]
        rot = quat_wxyz_to_mat(to_np(rack.data.root_quat_w[0]))
        pos = to_np(rack.data.root_pos_w[0]) - self._base_w()
        corners = np.array([[x, y, z] for x in self.rack_local_box[:, 0] for y in self.rack_local_box[:, 1]
                            for z in self.rack_local_box[:, 2]])
        w = (rot @ corners.T).T + pos
        return w.min(0) - margin, w.max(0) + margin

    def rack_gap(self, pts: np.ndarray) -> float:
        """Exact distance from points (task frame) to the rack as an ORIENTED box.

        The world AABB of a yawed rack (+-0.5 rad randomisation) is much larger than the rack
        and flagged clear grasps as collisions; testing in the rack's own frame is exact.
        """
        rack = self.u.scene[RACK]
        rot = quat_wxyz_to_mat(to_np(rack.data.root_quat_w[0]))
        pos = to_np(rack.data.root_pos_w[0]) - self._base_w()
        local = (np.asarray(pts) - pos) @ rot  # = rot.T @ (p - pos) per row
        lo, hi = self.rack_local_box
        return float(np.linalg.norm(np.maximum(np.maximum(lo - local, local - hi), 0.0), axis=1).min())

    def slots(self) -> list[np.ndarray]:
        """穴の入口の中心（タスク座標）。"""
        rack = self.u.scene[RACK]
        rack_t = _tf(to_np(rack.data.root_pos_w[0]), quat_wxyz_to_mat(to_np(rack.data.root_quat_w[0])))
        return [(rack_t @ loc)[:3, 3] - self._base_w() for loc in self.slot_local]

    def slot_occupied(self, slot: np.ndarray, vials: list[Vial] | None = None) -> bool:
        for v in vials or self.vials():
            if v.up_z > 0.7 and np.linalg.norm((v.pos - slot)[:2]) < 0.02:
                return True
        return False

    def lying_vials(self) -> list[Vial]:
        vs = self.vials()
        slots = self.slots()
        return [v for v in vs if v.up_z < 0.5
                and all(np.linalg.norm((v.pos - s)[:2]) > 0.04 for s in slots)]
