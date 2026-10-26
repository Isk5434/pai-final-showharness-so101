"""【比較用の凍結版】150 話収集（2026-10-05, commit dc6ba6e）のときの収集コードそのまま。変更は
見た目を当時と同じ白いアームにする look="white_arm" だけ。今の gpt_collect.py との比較にだけ使う。

サブスクの GPT（Codex）に操作させて「バイアル → ラック」のデモを集める（teleop-docker 内）。

取るバイアルは並んだ 3 本の位置で決める（ラックに近い順に near / middle / far）。--target cycle なら
シード % 3 で回すので、連続した 3k 個のシードで 3 位置が k 話ずつ同数になる。

オラクルとの違いは「誰が次の1手を決めるか」だけ。実行・記録・選別は同じ:
  - GPT が見るのは生のカメラ画像 2 枚（外部 640x480・手首 640x480）と、指の開閉・ステップ数・
    直近の手・前の手の結果（拒否理由）。バイアルの位置などの特権情報は渡さない
  - 各トークンが「いま」画像の上でどちらへ動くかを、カメラの姿勢から毎ステップ計算してプロンプトに
    書く（Show-Harness 論文の zero-shot と同じ考え方。SO-101 のカメラ配置用に実測で作る）
  - 実行は検証済みの So101AtomicExec。拒否された手は記録しない（理由を次のプロンプトに伝える）
  - 記録はオラクルと同じ 3 種（Show-Harness 形式・30 Hz 軌道・GUMI 動画）＋ GPT の生の応答
  - sim の成功判定（vial_placed）は緩い（離した瞬間に傾き 45° 以内・ラックの外形内なら成功）ので、
    採否は人が動画を見て決める。全話をその場に残し、自動判定は meta / episodes.jsonl に記録だけする
  - 見た目はアーム白・爪黒に固定し、ラックに最初から挿してあるバイアルはなし（いつも 3 本が並ぶ）

ホストで gpt_operator_server.py を先に起動しておくこと。

--vlm_server を付けると、GPT の代わりに FT 済み VLM（scripts/eval/policy_server.py）が1手を決める（Astra を
蒸留した VLM を、デモを集めたときと同じ開始姿勢・実行部・拒否・判定で評価する）。入力は学習データと同じ
Show-Harness v5 の文面と prepared_pair 済みの2画像で、拒否理由は伝わらない（学習データに無いため）。

    bash scripts/sim/run_in_teleop_docker.sh scripts/collect/gpt_collect.py \\
        --model gpt-6-astra --episodes 3 --seed0 4000 --target cycle --out /workspace/out/gpt_trial/gpt-6-astra
"""

import argparse
import base64
import io
import json
import os
import shutil
import time
import urllib.request

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Lerobot-So101-Teleop-Vials-To-Rack-DR")
parser.add_argument("--server", default="http://localhost:8770")
parser.add_argument("--model", required=True)
parser.add_argument("--vlm_server", default=None,
                    help="GPT の代わりに FT 済み VLM（policy_server.py の /act）に聞く。収集と同じ開始姿勢・実行部・判定で評価する")
parser.add_argument("--target", default="cycle", choices=["cycle", *("near", "middle", "far")],
                    help="取るバイアルの位置。cycle = シード % 3 で near/middle/far を回す")
parser.add_argument("--effort", default="low")
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--seed0", type=int, default=3000)
parser.add_argument("--max_tokens", type=int, default=70, help="実行するトークンの上限")
parser.add_argument("--max_calls", type=int, default=110, help="GPT への問い合わせの上限（拒否込み）")
parser.add_argument("--out", default="/workspace/out/gpt_collect")
parser.add_argument("--max_wait_s", type=float, default=7200, help="1手あたり GPT エラーで待つ合計の上限")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from scripts.trajectory.real2sim.atomic_tokenizer import MOVE_DIRS, RolloutWriter  # noqa: E402
from so101_showharness.controller import ROLL_CLOSING_HORIZONTAL  # noqa: E402
from so101_showharness.exec import So101AtomicExec  # noqa: E402
from so101_showharness.kinematics import quat_wxyz_to_mat  # noqa: E402
from so101_showharness.oracle.vials_to_rack import OracleParams  # noqa: E402
from so101_showharness.recorder.trajectory import TrajectoryRecorder  # noqa: E402
from so101_showharness.recorder.video import EpisodeVideo  # noqa: E402
from so101_showharness.sim.scene import VialsRackScene  # noqa: E402
from so101_showharness.sim.workshop_backend import So101WorkshopBackend, make_env  # noqa: E402

REPO = "/workspace/so101-showharness"
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


def _cam(be, key):
    cam = be.u.scene[key]
    k = cam.data.intrinsic_matrices[0].cpu().numpy()
    c = cam.data.pos_w[0].cpu().numpy() - be.robot.data.root_pos_w[0].cpu().numpy()  # task frame
    r = quat_wxyz_to_mat(cam.data.quat_w_ros[0].cpu().numpy())
    return k, c, r


def _px(k, c, r, p):
    pc = r.T @ (np.asarray(p) - c)
    uv = k @ pc
    return uv[:2] / uv[2]


def direction_table(be) -> str:
    """For the CURRENT pose: where each MV_* token moves the gripper in each image."""
    tcp = be.tcp_pos()
    ke, ce, re_ = _cam(be, "camera_external_D455")
    kw, cw, rw = _cam(be, "camera_ego")
    ahead = tcp + be.tcp_rotmat() @ (be.kin.tcp_offset / np.linalg.norm(be.kin.tcp_offset)) * 0.05
    lines = []
    for tok, d in MOVE_DIRS.items():
        d = d * 0.02
        ext = _px(ke, ce, re_, tcp + d) - _px(ke, ce, re_, tcp)
        # wrist camera moves with the hand: a static point shifts the OTHER way; the gripper
        # therefore heads toward image direction (-shift)
        shift = _px(kw, cw + d, rw, ahead) - _px(kw, cw, rw, ahead)
        lines.append(f"- {tok}: in image 1 the gripper moves {_dir_words(ext)}; "
                     f"in image 2 the gripper heads toward the {_dir_words(-shift)} of the image")
    return "\n".join(lines)


def png_b64(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img[..., :3]).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ask(payload: dict) -> dict:
    """GPT に1手聞く。サーバ不通・codex 側のエラー（レート制限など）は待って同じ手を聞き直す
    （手数の上限を消費しない）。待ちの合計が --max_wait_s を超えたら諦めて None。"""
    waited, k = 0.0, 0
    while True:
        try:
            r = urllib.request.Request(f"{args.server}/decide", data=json.dumps(payload).encode(),
                                       headers={"Content-Type": "application/json"})
            res = json.loads(urllib.request.urlopen(r, timeout=300).read())
            if not res["error"] or res["token"]:
                return res
            why = res["error"]
        except (OSError, ValueError) as e:  # URLError / RemoteDisconnected / 壊れた JSON
            why = repr(e)
        if waited >= args.max_wait_s:
            return None
        d = min(30 * 2 ** k, 900)
        print(f"[GPT収集] 問い合わせ失敗（{why[:200]}）→ {d}s 待って再試行", flush=True)
        time.sleep(d)
        waited, k = waited + d, k + 1


def strict_in_hole(scene, v) -> bool:
    """dm05_eval_sim.py の in_hole と同じ（立っていて、どれかの穴の中心から 1.5 cm 以内、底が穴の縁より 1 cm 下）。"""
    if v.up_z <= 0.85:
        return False
    bottom = v.pos[2] - scene.vial_length / 2
    return any(np.linalg.norm((v.pos - s)[:2]) < 0.015 and bottom < s[2] - 0.010 for s in scene.slots())


def ask_vlm(task: str, recent: list, gripper_closed: bool) -> dict:
    """FT 済み VLM に1手聞く。入力は学習データと同じ（Show-Harness v5 の文面、履歴、prepared_pair 済みの2画像）。"""
    av, wr = be_ref.grab_frames()
    r = urllib.request.Request(f"{args.vlm_server}/act", headers={"Content-Type": "application/json"},
                               data=json.dumps({"task": task, "recent_moves": recent, "gripper_closed": gripper_closed,
                                                "agentview_png_b64": png_b64(av), "wrist_png_b64": png_b64(wr)}).encode())
    res = json.loads(urllib.request.urlopen(r, timeout=300).read())
    return {"token": res["token"] if res["in_vocab"] else None, "raw": res["raw"],
            "latency_s": res["latency_s"], "error": None}


def main():
    global be_ref
    os.makedirs(args.out, exist_ok=True)
    crit = yaml.safe_load(open(f"{REPO}/configs/motion_criteria.yaml"))
    p = OracleParams()
    env = make_env(args.task, seed=args.seed0, fixed_look=True, preplaced_vial=False, look="white_arm")
    be = So101WorkshopBackend(env, jaw_open_rad=p.jaw_open_rad)
    ex = So101AtomicExec(be, step_m=0.02)
    be_ref = be
    scene = None
    log_path = os.path.join(args.out, "episodes.jsonl")
    for i in range(args.episodes):
        seed = args.seed0 + i
        name = f"ep_{seed:06d}"
        t0 = time.time()
        be.reset(seed)
        ex.lattice_origin, ex.hold_target, ex.grip_cmd = None, None, 1.0
        scene = scene or VialsRackScene(be)
        be.hand_floor_task_z = scene.table_top + crit["hand_floor_above_table_m"]
        be.workspace = {"min": [0.05, -0.30, scene.table_top + 0.002], "max": [0.45, 0.25, 0.32]}
        pos = TARGET_CYCLE[seed % 3] if args.target == "cycle" else args.target
        instruction, target_words = TARGETS[pos]
        task_dir = instruction.lower().replace(" ", "_")
        # 特権情報は記録用だけ（GPT には渡さない）: その位置にあるバイアルの名前
        rack_c = scene.rack_box()[0] / 2 + scene.rack_box()[1] / 2
        by_dist = sorted(scene.vials(), key=lambda v: float(np.linalg.norm((v.pos - rack_c)[:2])))
        target_vial = by_dist[TARGET_CYCLE.index(pos)].name
        rollout_dir = os.path.join(args.out, task_dir, name)
        for d in (rollout_dir, os.path.join(args.out, "traj", name)):  # 途中で落ちた前回の残骸
            shutil.rmtree(d, ignore_errors=True)
        writer = RolloutWriter(rollout_dir)
        traj = TrajectoryRecorder(os.path.join(args.out, "traj", name))
        vid = EpisodeVideo(os.path.join(args.out, "video", f"{name}.mp4"), task=instruction)
        be.step_hooks = [traj.on_step, vid.on_step]
        begin = np.asarray(p.begin_world) - be.robot.data.root_pos_w[0].cpu().numpy()
        ex.goto_pose(begin, np.radians(p.grasp_pitch_deg), roll=ROLL_CLOSING_HORIZONTAL)
        be.decision_id = None

        recent, calls, last_result, reason, gpt_log = [], 0, "none (first step)", None, []
        missed_since_move = False
        while writer.step < args.max_tokens and calls < args.max_calls:
            ext_raw, wr_raw = be.raw_frames()
            prompt = PROMPT.format(task=instruction, target=target_words, table=direction_table(be), step=writer.step + 1,
                                   max_tokens=args.max_tokens,
                                   gripper=gripper_state(ex, be), tilt=tilt_state(be, p),
                                   height=f"{100 * (be.commanded_tcp()[2] - scene.table_top):.1f} cm",
                                   recent=", ".join(recent[:8]) or "none", last_result=last_result)
            res = ask_vlm(instruction, recent, ex.gripper_closed) if args.vlm_server else ask({"model": args.model, "effort": args.effort, "prompt": prompt, "episode": name,
                       "step": writer.step, "external_png_b64": png_b64(ext_raw), "wrist_png_b64": png_b64(wr_raw)})
            if res is None:
                reason = "gpt_error"
                break
            calls += 1
            tok = res["token"]
            gpt_log.append({"call": calls, "step": writer.step, "token": tok, "raw": res["raw"],
                            "latency_s": res["latency_s"], "error": res["error"],
                            **({"prompt": prompt} if calls in (1, 20, 40) else {})})
            if not tok:
                last_result = "your answer had no valid action token; answer with one token on the last line"
                continue
            if tok == "DONE":
                av, wr = be.grab_frames()
                be.decision_id = writer.step
                writer.add_step(token="DONE", kind="done", agentview=av, wrist=wr,
                                gripper_closed=ex.gripper_closed, ee_pose=be.tcp_pose7(), width=be.gripper_width())
                break
            if tok == "GRASP" and missed_since_move:
                last_result = ("GRASP was REFUSED: the last grasp missed at this same spot; move (MV_UP, then sideways "
                               "to centre the vial between the fingers) before grasping again")
                continue
            ok, why = ex.check(tok)
            if not ok:
                last_result = f"{tok} was REFUSED and not executed ({why}); choose a different action"
                continue
            av, wr = be.grab_frames()  # frame BEFORE the token, deployment contract
            be.decision_id = writer.step
            kind = "move" if tok in MOVE_DIRS else ("rotate" if tok.startswith("RT_") else "gripper")
            writer.add_step(token=tok, kind=kind, agentview=av, wrist=wr, gripper_closed=ex.gripper_closed,
                            ee_pose=be.tcp_pose7(), width=be.gripper_width())
            ex.run(tok)
            if ex.last_blocked:  # aborted mid-way (contact / stall): the label no longer holds
                reason = f"aborted:{tok}:{ex.last_blocked}"
                break
            last_result = f"{tok} executed"
            if tok == "GRASP":
                missed_since_move = be.gripper_width() < HOLD_MIN_WIDTH_M
                if missed_since_move:
                    last_result = "GRASP executed but CLOSED ON NOTHING (missed)"
            elif tok in HISTORY:
                missed_since_move = False
            if tok in HISTORY:
                recent.insert(0, tok)
        ex.hold(60)
        success = be.success()
        if reason is None and not success:
            reason = "not_placed" if writer.step < args.max_tokens and calls < args.max_calls else "budget"
        valid = bool(success and reason is None)
        upright = [v.name for v in scene.vials() if v.up_z > 0.7]  # 記録用（人の判定の手がかり）
        # 記録用（挙動は変えない）：評価（dm05_eval_sim.py）と同じ厳しい判定。立って穴に 1 cm 以上刺さった vial
        in_hole = [v.name for v in scene.vials() if strict_in_hole(scene, v)]
        meta = {"instruction": instruction, "task": instruction, "task_text": instruction, "seed": seed,
                "target_position": pos, "target_vial": target_vial, "upright_vials_at_end": upright,
                "operator": f"vlm:{args.model}" if args.vlm_server else f"gpt:{args.model}:{args.effort}", "embodiment": "so101",
                "sim": "Sim-to-Real-SO-101-Workshop v1.0", "env": args.task, "success": success,
                "valid": valid, "reason": reason}
        writer.close(meta)
        traj.close({"episode": name, **meta})
        vid.close()
        be.step_hooks = []
        with open(os.path.join(args.out, f"{name}.gpt.jsonl"), "w") as f:
            for g in gpt_log:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        row = {"episode": name, "seed": seed, "target_position": pos, "target_vial": target_vial,
               "target_upright_at_end": target_vial in upright, "instruction": instruction,
               "vials_in_hole": in_hole, "target_in_hole": target_vial in in_hole,
               "model": args.model, "effort": args.effort,
               "success": success, "valid": valid, "reason": reason, "tokens": writer.step, "calls": calls,
               "refused": sum(1 for g in gpt_log if g["token"]) - writer.step,
               "mean_latency_s": round(float(np.mean([g["latency_s"] for g in gpt_log])), 2) if gpt_log else None,
               "seconds": round(time.time() - t0, 1)}
        with open(log_path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[GPT収集] {args.model} {name} {pos}({target_vial}) 成功={success} 採用={valid} 理由={reason} "
              f"トークン={writer.step} 問い合わせ={calls} 平均{row['mean_latency_s']}s 計{row['seconds']}s", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
