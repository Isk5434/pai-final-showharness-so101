"""Workshop (v1.0) sim I/O for the SO-101 controller.

Runs under the Isaac kit python (teleop-docker), after ``AppLauncher`` has started the app.
All motion logic is in controller.So101ArmController; this file only reads the joints and
cameras from the env and steps it. One tick = one env.step (60 Hz).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from scripts.trajectory.real2sim.atomic_tokenizer import prepared_pair, to_np
from so101_showharness.controller import So101ArmController
from so101_showharness.kinematics import quat_wxyz_to_mat

DEFAULT_TASK = "Lerobot-So101-Teleop-Vials-To-Rack-DR"


# Links whose contacts mean "the hand/forearm touched something". The base sits on the table
# and the shoulder/upper arm never come near the scene in this task.
CONTACT_LINKS = ("lower_arm", "wrist", "gripper", "jaw")
# Filters must name the RIGID BODY prim: for the rack that is Rack_Left/Body1 (Rack_Left is
# only its Xform). With "Rack_Left" every rack contact landed in "other" (measured 2-3 N).
CONTACT_FILTERS = ("Vial_1", "Vial_2", "Vial_3", "Rack_Left/Body1")

ROBOT_PRIM = "/World/envs/env_0/Robot"
# 爪（挟む部分）のメッシュ。固定爪はグリッパ本体と一体のメッシュなので本体ごと黒になる。
# バインドはメッシュの親 Xform に strongerThanDescendants で付いているので、そこを差し替える
# （メッシュ側に付けても親が勝つ。測定済み）。
JAW_MESHES = ("gripper/visuals/wrist_roll_follower_so101_v1", "jaw/visuals/moving_jaw_so101_v1")
# 実機（2026-10-08 の写真）に合わせる: アームのリンクは黒、グリッパ本体と爪は白。
# 値は Workshop の ROBOT_COLORS["black"] / ["white"]。
# look 名 -> (アームの色, グリッパと爪の色)。"real" が実機（既定）。"white_arm" は 2026-10-05〜06 の
# 150 話収集のときの見た目（near 10/39 成功の実績）。黒いマットに黒いアームが重なると GPT が手の位置を
# 見失うかを比べるために残す。
LOOKS = {"real": ((0.08, 0.08, 0.08), (0.95, 0.95, 0.95)),
         "white_arm": ((0.95, 0.95, 0.95), (0.08, 0.08, 0.08))}
ARM_COLOR, JAW_COLOR = LOOKS["real"]


def apply_fixed_look(env, env_ids, look: str) -> None:
    """アームを黒、グリッパと爪を白に固定する（実機に合わせる。DR の毎リセットの色替えは make_env で外す）。

    sim 開始前（EventTerm の mode="prestartup"）に呼ぶ。開始後に USD を書き換えても、インスタンス化
    された部品の材質とバインドの変更がレンダラに届かなかった（測定: アームは USD 既定の黄色のまま）。

    3D プリント部品は全部 1 つの材質（material_a_3d_printed）を共有しているので、爪用にその複製を
    作って白にし、爪のメッシュだけに割り当てる。メッシュはインスタンス化されていて書き換えられない
    ため、その link の visuals だけインスタンスを解く（見た目だけ。衝突形状と物理は触らない）。
    """
    from pxr import Gf, Sdf, UsdShade

    stage = env.sim.stage
    src_mat = stage.GetPrimAtPath(f"{ROBOT_PRIM}/Looks/material_a_3d_printed")
    src = src_mat.GetChild("Shader")
    src.GetAttribute("inputs:diffuse_color_constant").Set(Gf.Vec3f(*LOOKS[look][0]))
    # 複製: 材質の出力と MDL の指定・入力値を写す（参照先レイヤの spec は直接コピーできない）
    mat = UsdShade.Material.Define(stage, f"{ROBOT_PRIM}/Looks/material_jaw")
    sh = UsdShade.Shader.Define(stage, mat.GetPath().AppendChild("Shader"))
    for a in src.GetAttributes():
        if a.HasAuthoredValue():
            sh.GetPrim().CreateAttribute(a.GetName(), a.GetTypeName(), custom=False).Set(a.Get())
    sh.GetPrim().GetAttribute("inputs:diffuse_color_constant").Set(Gf.Vec3f(*LOOKS[look][1]))
    for out in ("surface", "displacement", "volume"):  # MDL 材質の出力（outputs:mdl:*）
        mat.CreateOutput(f"mdl:{out}", Sdf.ValueTypeNames.Token).ConnectToSource(sh.ConnectableAPI(), "out")
    with Sdf.ChangeBlock():
        for m in JAW_MESHES:
            stage.GetPrimAtPath(f"{ROBOT_PRIM}/{m.split('/')[0]}/visuals").SetInstanceable(False)
    for m in JAW_MESHES:
        prim = stage.GetPrimAtPath(f"{ROBOT_PRIM}/{m}")
        if not prim.IsValid():
            raise RuntimeError(f"爪の部品が見つからない: {m}")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants)


def make_env(task: str = DEFAULT_TASK, seed: int = 0, device: str = "cuda:0",
             episode_length_s: float = 3600.0, hand_contact: bool = True,
             fixed_look: bool = False, preplaced_vial: bool = True,
             vial_shift: tuple[float, float] = (0.0, 0.0), rack_shift: tuple[float, float] = (0.0, 0.0),
             look: str = "real"):
    """Create the Workshop env for token collection: no time-out, no auto-reset.

    The non-Eval task variants already have ``terminations = None``; the long episode length
    only keeps ``episode_length_buf`` from ever mattering. Success is read from the
    ``subtask_terms`` observations instead of a termination.

    fixed_look: アーム黒・グリッパと爪は白に固定（実機に合わせる。DR の色替えを外す。照明・マット回転などほかの DR はそのまま）。
    preplaced_vial: False で、リセット時に 1/3 の確率で 1 本をラックに挿しておく初期配置をやめる
    （いつも 3 本が台の上に並ぶ）。
    """
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg

    import sim_to_real_so101.tasks  # noqa: F401  (registers the gym ids)

    cfg = parse_env_cfg(task, device=device, num_envs=1)
    cfg.seed = seed
    if hand_contact:
        # Measured contact on the arm's distal links, with per-object breakdown, so contact is
        # OBSERVED instead of inferred from the joint gap (which false-fires at full reach).
        from isaaclab.sensors import ContactSensorCfg

        cfg.scene.contact_hand = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/(" + "|".join(CONTACT_LINKS) + ")",
            update_period=0.0, history_length=1, debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/" + n for n in CONTACT_FILTERS],
        )
    cfg.episode_length_s = episode_length_s
    if getattr(cfg, "terminations", None) is not None:
        raise ValueError(f"{task} has terminations (Eval variant?); use a non-Eval task id")
    if fixed_look:
        from isaaclab.managers import EventTermCfg

        if hasattr(cfg.events, "reset_set_robot_visual_material"):
            cfg.events.reset_set_robot_visual_material = None
        cfg.events.fixed_look = EventTermCfg(func=apply_fixed_look, mode="prestartup", params={"look": look})
        cfg.scene.replicate_physics = False  # prestartup の条件（1 env なので影響なし）
    if any(vial_shift):
        # 3 本の初期位置（乱択の中心）をまとめてずらす [m, ワールド x/y]。-x がロボット側。
        # 一番奥は手首ピッチの可動域の端で掴めなかった（2026-10-08, render_grasp_poses）。
        for name in ("vial_1", "vial_2", "vial_3"):
            v = getattr(cfg.scene, name)
            x, y, z = v.init_state.pos
            v.init_state.pos = (x + vial_shift[0], y + vial_shift[1], z)
    if any(rack_shift):
        # ラックも一緒にずらす（バイアルだけ +y にずらすと手前のバイアルがラックに重なる）
        x, y, z = cfg.scene.rack_left.init_state.pos
        cfg.scene.rack_left.init_state.pos = (x + rack_shift[0], y + rack_shift[1], z)
    if not preplaced_vial:
        cfg.events.reset_vials_setup.params["rack_placement_prob"] = 0.0
    return gym.make(task, cfg=cfg)


class So101WorkshopBackend(So101ArmController):
    # Show-Harness camera contract (same as upstream configs/robot_so101_workshop.yaml, PR #6,
    # measured by its check_views.py): agentview 90 deg CCW -> image bottom = MV_FWD, left =
    # MV_LEFT; wrist 180 deg -> fingertips on top, left = MV_LEFT. Both letterboxed to 256.
    # Before 2026-10-08 both were stored unrotated (data collected earlier is NOT in this
    # orientation). Only the RECORDED frames change; raw_frames() (what the GPT operator sees)
    # stays raw.
    agentview_rotation_degrees = 90
    agentview_flip = None
    agentview_square_size = 256
    wrist_rotation_degrees = 180
    wrist_flip = None
    wrist_square_size = 256

    def __init__(self, env, **kw) -> None:
        self.env = env
        self.u = env.unwrapped
        self.robot = self.u.scene["robot"]
        kw.setdefault("tick_dt", float(self.u.step_dt))
        super().__init__(**kw)
        self._obs: Any = None
        self.contact_is_measured = "contact_hand" in self.u.scene.keys()

    # ------------------------------------------------------------ I/O
    def _read_q(self) -> np.ndarray:
        return to_np(self._obs["policy"]["joint_pos_obs"][0])

    def _send(self, q_cmd: np.ndarray) -> None:
        import torch

        action = torch.tensor(q_cmd, dtype=torch.float32, device=self.u.device).unsqueeze(0)
        self._obs, *_ = self.env.step(action)

    def _r_world_base(self) -> np.ndarray:
        return quat_wxyz_to_mat(to_np(self.robot.data.root_quat_w[0]))

    # ------------------------------------------------------------ measured contact
    def contact_report(self, min_force_n: float = 0.5) -> list[dict]:
        """Links touching something, with the net force and what it is pressing on."""
        if "contact_hand" not in self.u.scene.keys():
            return []
        sensor = self.u.scene["contact_hand"]
        net = to_np(sensor.data.net_forces_w[0])            # (links, 3)
        mat = sensor.data.force_matrix_w
        mat = None if mat is None else to_np(mat[0])        # (links, filters, 3)
        out = []
        for i, link in enumerate(sensor.body_names):
            f = float(np.linalg.norm(net[i]))
            if f < min_force_n:
                continue
            with_ = {}
            explained = np.zeros(3)
            if mat is not None:
                for j, name in enumerate(CONTACT_FILTERS):
                    fj = float(np.linalg.norm(mat[i, j]))
                    explained += mat[i, j]
                    if fj >= min_force_n:
                        with_[name] = round(fj, 2)
            # unexplained = the VECTOR residual (|net| - sum|f_j| is not a force: it read 2-3 N
            # of phantom "other" contact whenever the grip forces were not collinear)
            other = float(np.linalg.norm(net[i] - explained))
            if other >= min_force_n:
                with_["other"] = round(other, 2)
            out.append({"link": link, "force_n": round(f, 2), "with": with_})
        return out

    def in_contact(self, threshold_n: float = 2.0) -> bool:
        """Measured: any distal link pressed with more than ``threshold_n`` against anything
        except a vial held between closed fingers (that force is the grasp itself)."""
        if "contact_hand" not in self.u.scene.keys():
            return super().in_contact()
        closed = self.q_des[5] < (self.jaw_open_rad + self.jaw_close_rad) / 2
        for c in self.contact_report(threshold_n):
            w = dict(c["with"])
            if closed and c["link"] in ("gripper", "jaw"):
                for name in ("Vial_1", "Vial_2", "Vial_3"):
                    w.pop(name, None)
            if any(f >= threshold_n for f in w.values()):
                self.last_contact = c
                return True
        return False

    # ------------------------------------------------------------ task / cameras
    def success(self) -> bool:
        return bool(to_np(self._obs["subtask_terms"]["vial_placed"]).reshape(-1)[0] > 0.5)

    def grasped(self) -> bool:
        return bool(to_np(self._obs["subtask_terms"]["vial_grasped"]).reshape(-1)[0] > 0.5)

    def raw_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """(external_D455, ego) uint8 HWC 480x640, untransformed."""
        vis = self._obs["visual"]
        return to_np(vis["rgb_external_D455"][0])[..., :3], to_np(vis["rgb_ego"][0])[..., :3]

    def grab_frames(self) -> tuple[np.ndarray, np.ndarray]:
        return prepared_pair(self, *self.raw_frames())

    # ------------------------------------------------------------ lifecycle
    def reset(self, seed: int, settle_ctrl_steps: int = 30) -> None:
        self._obs, _ = self.env.reset(seed=seed)
        self._begin_episode(settle_ctrl_steps)

    def close(self) -> None:
        self.env.close()
