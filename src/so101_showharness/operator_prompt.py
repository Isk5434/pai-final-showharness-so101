"""Show-Harness の GPT 操作（収集 v150）と同じ指示文を組み立てる部品。sim の収集と実機の両方で使う。

PROMPT・状態の言い方は scripts/collect/gpt_collect_v150.py（凍結版）からそのまま写したもの。違いは「各トークンで
画像のどちらへ動くか」の表（direction_table）を、sim のカメラではなく Camera（K と姿勢）から計算する点だけ。
実機では configs/real_setup.json のカメラ（sim から書き出した値、または実測値）を使う。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.trajectory.real2sim.atomic_tokenizer import MOVE_DIRS

# 位置 → (指示文, プロンプト中の言い方)。位置はラックからの距離の順（near が一番近い）。
TARGETS = {
    "near": ("Pick up the vial closest to the rack and place it in the rack", "the vial CLOSEST TO THE RACK"),
    "middle": ("Pick up the middle vial and place it in the rack", "the MIDDLE vial"),
    "far": ("Pick up the vial farthest from the rack and place it in the rack", "the vial FARTHEST FROM THE RACK"),
}
TARGET_CYCLE = ("near", "middle", "far")
HISTORY = set(MOVE_DIRS) | {"RT_PITCH_FWD", "RT_PITCH_BACK", "RT_YAW_CW", "RT_YAW_CCW"}

PROMPT = """You are operating a real-world-like robot arm (SO-101, 5 joints) in simulation, one small action at a time.

TASK: {task}
There are three white vials (blue caps) lying on a black mat and a yellow rack with four holes.
Pick up {target} of the three, tilt it cap-up and drop it into an EMPTY hole of the yellow rack.

IMAGES
- Image 1: fixed external camera. The robot base is on the right side of the image.
- Image 2: wrist camera mounted on the gripper (moves with the hand). The two gripper fingers are at the bottom.
  The gripper has ONE fixed finger and ONE moving jaw; an object is held between them.

ACTIONS (each MV_* moves the fingertips 2 cm; each RT_PITCH_* tilts the hand 10 degrees)
{table}
- RT_PITCH_FWD: tilt the hand forward (fingertips swing forward and up). RT_PITCH_BACK: the opposite.
- GRASP: close the gripper. RELEASE: open it. DONE: only after RELEASE, when the vial has dropped into a rack hole.

HOW THIS TASK IS DONE (a successful demonstration does exactly this)
1. Move above {target} so that it is between the two fingers in the wrist view, then MV_DOWN until the
   STATE "Fingertip height above the mat" is about 1.5 cm (vial between the fingers). Grasping higher than
   ~2.5 cm closes on nothing.
2. GRASP. Right after it, STATE "Gripper" must say HOLDING; if it says CLOSED ON NOTHING the grasp missed.
3. MV_UP about 5 times to lift the vial clear of the mat.
4. Still lifted and AWAY from the rack, RT_PITCH_FWD until the STATE line "Hand tilt" says the release tilt is
   reached (about 8 times from the grasp tilt). Do not judge this from the images and do NOT rotate further:
   the vial does not have to look perfectly vertical. Never rotate while the hand is over or next to the rack.
5. With the tilt done, MV_UP until "Fingertip height above the mat" is about 18-20 cm, then move toward the rack
   and stop as soon as the vial is right above an EMPTY hole (do not overshoot past the rack).
   Do NOT MV_DOWN over the rack: release from that height, the vial drops into the hole by itself.
6. RELEASE, then DONE.
If "Gripper" says CLOSED ON NOTHING, the grasp missed because the vial is NOT between the fingers:
RELEASE, MV_UP once, then move sideways so the vial is centred between the fingers in the wrist view, MV_DOWN
back to ~1.5 cm and GRASP again. GRASP right after RELEASE without moving is refused (it would miss the same way).
If an action is REFUSED, do not repeat it; choose another action (usually MV_UP or moving away from the rack).
Never push the hand into the rack, the mat or the other vials.

STATE
- Step: {step} of {max_tokens}
- Gripper: {gripper}
- Hand tilt: {tilt}
- Fingertip height above the mat: {height}
- Recent actions (newest first): {recent}
- Previous action: {last_result}

Look at both images, say in one or two short sentences where the gripper is relative to what it needs to reach,
then give exactly ONE action token from this list ALONE on the last line:
MV_FWD MV_BACK MV_LEFT MV_RIGHT MV_UP MV_DOWN RT_PITCH_FWD RT_PITCH_BACK GRASP RELEASE DONE"""


HOLD_MIN_WIDTH_M = 0.015  # 閉じた指の隙間がこれ以上 = 何かを挟んでいる（バイアル径 34 mm、空振りはほぼ 0）


def gripper_state(ex, be) -> str:
    """固有感覚だけ（指の隙間）から把持状態を言う。実機でもグリッパ位置から同じことが分かる。"""
    w = be.gripper_width()
    if not ex.gripper_closed:
        return "open"
    if w >= HOLD_MIN_WIDTH_M:
        return f"CLOSED and HOLDING something (finger gap {1000 * w:.0f} mm)"
    return f"CLOSED ON NOTHING (finger gap {1000 * w:.0f} mm) - the grasp missed"


def tilt_state(be, p) -> str:
    """指令ピッチ（関節角から決まる固有感覚）と、離す姿勢までの残り回数。"""
    now = np.degrees(be.pitch_ref)
    left = int(np.ceil((now - p.release_pitch_target_deg) / 10.0 - 1e-6))
    if left > 0:
        return f"{now:+.0f} deg (grasp tilt {p.grasp_pitch_deg:+.0f}, release tilt {p.release_pitch_target_deg:+.0f}): {left} more RT_PITCH_FWD to reach the release tilt"
    return f"{now:+.0f} deg: release tilt REACHED - do not rotate any more"


def _dir_words(v) -> str:
    """image-plane vector (u right, v down) -> 'up', 'down-left', ..."""
    u, w = float(v[0]), float(v[1])
    if np.hypot(u, w) < 1.5:
        return "barely moves"
    ang = np.degrees(np.arctan2(-w, u)) % 360  # 0 = right, 90 = up
    names = ["right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right"]
    return names[int(((ang + 22.5) % 360) // 45)]


@dataclass
class Camera:
    """ピンホール。K（3x3）と、タスク座標でのカメラ位置 c・向き r（ROS 規約：+z 前、+y 下）。"""
    K: np.ndarray
    c: np.ndarray
    r: np.ndarray

    def px(self, p) -> np.ndarray:
        pc = self.r.T @ (np.asarray(p) - self.c)
        uv = self.K @ pc
        return uv[:2] / uv[2]


def cameras_from_setup(setup: dict, be) -> tuple[Camera, Camera]:
    """real_setup.json から（外部、手首）。

    手首カメラ：mode "v150_fixed"（既定）はタスク座標の固定値をそのまま使う。sim の camera_ego は姿勢が更新されず
    基部に対して固定だったので、v150 の収集で GPT が見た表と同じものを出すにはこれを使う。
    mode "attached" は gripper リンクに固定とみなし、今の関節角から置き直す（実測した取り付け位置がある場合）。
    """
    e = setup["external"]
    ext = Camera(np.asarray(e["K"]), np.asarray(e["pos_task_m"]), np.asarray(e["R_task_cam_ros"]))
    w = setup["wrist"]
    if w.get("mode", "v150_fixed") == "v150_fixed":
        return ext, Camera(np.asarray(w["K"]), np.asarray(w["pos_task_m"]), np.asarray(w["R_task_cam_ros"]))
    g = be.kin.link_poses(be.q_obs())["gripper"]
    r_wb = be._r_world_base()
    rg, pg = r_wb @ g[:3, :3], r_wb @ g[:3, 3]
    wrist = Camera(np.asarray(w["K"]), rg @ np.asarray(w["pos_in_gripper_m"]) + pg,
                   rg @ np.asarray(w["R_gripper_cam_ros"]))
    return ext, wrist


def direction_table(be, ext: Camera, wrist: Camera) -> str:
    """今の姿勢で、各 MV_* が画像のどちらへ手を動かすか（v150 の direction_table と同じ計算）。"""
    tcp = be.tcp_pos()
    ahead = tcp + be.tcp_rotmat() @ (be.kin.tcp_offset / np.linalg.norm(be.kin.tcp_offset)) * 0.05
    lines = []
    for tok, d in MOVE_DIRS.items():
        d = d * 0.02
        e = ext.px(tcp + d) - ext.px(tcp)
        moved = Camera(wrist.K, wrist.c + d, wrist.r)
        shift = moved.px(ahead) - wrist.px(ahead)
        lines.append(f"- {tok}: in image 1 the gripper moves {_dir_words(e)}; "
                     f"in image 2 the gripper heads toward the {_dir_words(-shift)} of the image")
    return "\n".join(lines)
