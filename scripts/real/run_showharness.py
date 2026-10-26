"""Show-Harness（GPT-6 Astra が 1 手ずつ選ぶ）で実機 SO-101 を動かす（ローカル PC 側）。

sim の収集（scripts/collect/gpt_collect_v150.py）と同じ流れ・同じ指示文・同じ実行部（So101AtomicExec）で、
入出力だけを実機（So101RealBackend + SafeArm + OpenCV カメラ）に差し替えたもの。GPT は AWS の
gpt_operator_server.py（Codex のログインがある方）に SSH トンネル越しで頼む。

    # AWS 側：bash scripts/real/aws_serve.sh showharness
    # ローカル：ssh -N -L 8770:127.0.0.1:8770 <AWS>
    python scripts/real/run_showharness.py --port /dev/ttyACM0 --robot_id follower_arm_1 --front 0 --wrist 2 \\
        --position near --confirm      # 1 手ごとに Enter で承認してから動かす（最初はこれで）

- 「指先のマットからの高さ」は configs/real_setup.json の table_top_task_z_m（マット上面の高さ、基部が原点）から出す。
  実機の机の高さが sim と違うなら実測して書き換える。
- 実機には力センサが無いので、接触は「関節の追従ずれ + 進まない」から推定する（So101RealBackend）。
- 記録：各手の GPT の応答と状態（jsonl）、各手の直前の front/wrist 画像を --out に残す。
"""
import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
SHOW_HARNESS = os.environ.get("SHOW_HARNESS_ROOT", str(REPO.parent / "Show-Harness"))
sys.path[:0] = [str(REPO / "src"), SHOW_HARNESS]

from so101_showharness.controller import ROLL_CLOSING_HORIZONTAL  # noqa: E402
from so101_showharness.exec import So101AtomicExec  # noqa: E402
from so101_showharness.operator_prompt import (  # noqa: E402
    HISTORY, HOLD_MIN_WIDTH_M, MOVE_DIRS, PROMPT, TARGETS, cameras_from_setup, direction_table, gripper_state,
    tilt_state)
from so101_showharness.oracle.vials_to_rack import OracleParams  # noqa: E402
from so101_showharness.real.robot_io import MockArm, SafeArm, apply_camera_fix, connect_so101, open_cameras  # noqa: E402
from so101_showharness.real.so101_robot import So101RealBackend  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--server", default="http://127.0.0.1:8770")
ap.add_argument("--model", default="gpt-6-astra")
ap.add_argument("--effort", default="low")
ap.add_argument("--port", help="SO-101 のシリアルポート")
ap.add_argument("--robot_id", default="follower_arm_1")
ap.add_argument("--front", default="0")
ap.add_argument("--wrist", default="2")
ap.add_argument("--position", default="near", choices=list(TARGETS))
ap.add_argument("--max_tokens", type=int, default=70)
ap.add_argument("--max_calls", type=int, default=110)
ap.add_argument("--max_step_rad", type=float, default=0.08)
ap.add_argument("--setup", default=str(REPO / "configs/real_setup.json"))
ap.add_argument("--camera_fix", default=None, help="camera_calib.py align が作る camera_fix.json（front を sim のカメラの形に直す）")
ap.add_argument("--out", default=str(REPO / "outputs/real_showharness"))
ap.add_argument("--confirm", action="store_true", help="1 手ごとに Enter で承認してから動かす（s で飛ばす、q で終了）")
ap.add_argument("--dry_run", action="store_true", help="腕には送らない（GPT との往復と指示文の確認だけ）")
ap.add_argument("--mock", action="store_true", help="実機なし。--front/--wrist に動画ファイル")
args = ap.parse_args()


def png_b64(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img[..., :3]).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ask(payload: dict) -> dict:
    r = urllib.request.Request(f"{args.server}/decide", data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=300).read())


def main():
    setup = json.load(open(args.setup))
    crit = yaml.safe_load(open(REPO / "configs/motion_criteria.yaml"))
    p = OracleParams()
    q_start = np.asarray(setup["start_q_rad"])
    robot = MockArm(q_start) if args.mock else connect_so101(args.port, args.robot_id)
    arm = SafeArm(robot, max_step_rad=args.max_step_rad, dry_run=args.dry_run)
    cams = apply_camera_fix(open_cameras(args.front, args.wrist, mock=args.mock), args.camera_fix, setup)
    be = So101RealBackend(arm, cameras={"external": cams["front"], "wrist": cams["wrist"]},
                          jaw_open_rad=p.jaw_open_rad)
    table_top = float(setup["table_top_task_z_m"])
    be.hand_floor_task_z = table_top + crit["hand_floor_above_table_m"]
    be.workspace = {"min": [0.05, -0.30, table_top + 0.002], "max": [0.45, 0.25, 0.32]}
    ex = So101AtomicExec(be, step_m=0.02)
    instruction, target_words = TARGETS[args.position]
    name = time.strftime("%Y%m%d_%H%M%S") + f"_{args.position}"
    out = Path(args.out) / name
    (out / "frames").mkdir(parents=True, exist_ok=True)
    log = open(out / "steps.jsonl", "w")
    print(f"[real_sh] {instruction}  server={args.server} dry_run={args.dry_run} mock={args.mock}", flush=True)
    try:
        if not args.dry_run:
            arm.move_to(q_start, duration_s=3.0)
        be.reset()
        ex.lattice_origin, ex.hold_target, ex.grip_cmd = None, None, 1.0
        if not args.mock:
            input("[real_sh] vial を並べたら Enter（収集の開始姿勢へ動きます）")
        ex.goto_pose(np.asarray(setup["begin_task_m"]), np.radians(p.grasp_pitch_deg), roll=ROLL_CLOSING_HORIZONTAL)
        recent, calls, step, last_result, missed = [], 0, 0, "none (first step)", False
        while step < args.max_tokens and calls < args.max_calls:
            ext_raw, wr_raw = be.raw_frames()
            prompt = PROMPT.format(task=instruction, target=target_words,
                                   table=direction_table(be, *cameras_from_setup(setup, be)), step=step + 1,
                                   max_tokens=args.max_tokens, gripper=gripper_state(ex, be), tilt=tilt_state(be, p),
                                   height=f"{100 * (be.commanded_tcp()[2] - table_top):.1f} cm",
                                   recent=", ".join(recent[:8]) or "none", last_result=last_result)
            res = ask({"model": args.model, "effort": args.effort, "prompt": prompt, "episode": name, "step": step,
                       "external_png_b64": png_b64(ext_raw), "wrist_png_b64": png_b64(wr_raw)})
            calls += 1
            tok = res["token"]
            Image.fromarray(ext_raw).save(out / "frames" / f"{calls:03d}_front.jpg")
            Image.fromarray(wr_raw).save(out / "frames" / f"{calls:03d}_wrist.jpg")
            row = {"call": calls, "step": step, "token": tok, "raw": res["raw"], "latency_s": res["latency_s"],
                   "error": res["error"], "q": be.q_obs().round(4).tolist(), "prompt": prompt if calls == 1 else None}
            print(f"[real_sh] #{calls} step {step + 1}: {tok or '(なし)'}  ← {res['raw'][-160:]!r}", flush=True)
            if not tok:
                last_result = "your answer had no valid action token; answer with one token on the last line"
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            if tok == "DONE":
                log.write(json.dumps({**row, "result": "done"}, ensure_ascii=False) + "\n")
                break
            if tok == "GRASP" and missed:
                last_result = ("GRASP was REFUSED: the last grasp missed at this same spot; move (MV_UP, then sideways "
                               "to centre the vial between the fingers) before grasping again")
                log.write(json.dumps({**row, "result": "refused_regrasp"}, ensure_ascii=False) + "\n")
                continue
            ok, why = ex.check(tok)
            if not ok:
                last_result = f"{tok} was REFUSED and not executed ({why}); choose a different action"
                log.write(json.dumps({**row, "result": f"refused:{why}"}, ensure_ascii=False) + "\n")
                continue
            if args.confirm:
                c = input(f"[real_sh] {tok} を実行？ [Enter=実行 / s=飛ばす / q=終了] ").strip().lower()
                if c == "q":
                    break
                if c == "s":
                    last_result = f"{tok} was REFUSED by the operator; choose a different action"
                    log.write(json.dumps({**row, "result": "skipped_by_operator"}, ensure_ascii=False) + "\n")
                    continue
            ex.run(tok)
            step += 1
            if ex.last_blocked:
                log.write(json.dumps({**row, "result": f"aborted:{ex.last_blocked}"}, ensure_ascii=False) + "\n")
                print(f"[real_sh] {tok} が途中で止まった（{ex.last_blocked}）。終了", flush=True)
                break
            last_result = f"{tok} executed"
            if tok == "GRASP":
                missed = be.gripper_width() < HOLD_MIN_WIDTH_M
                if missed:
                    last_result = "GRASP executed but CLOSED ON NOTHING (missed)"
            elif tok in HISTORY:
                missed = False
            if tok in HISTORY:
                recent.insert(0, tok)
            log.write(json.dumps({**row, "result": "executed"}, ensure_ascii=False) + "\n")
        ex.hold(30)
    except KeyboardInterrupt:
        print("\n[real_sh] 中止：その場で止める", flush=True)
        arm.hold()
    finally:
        log.close()
        if not args.dry_run and not args.mock:
            input("[real_sh] グリッパを開いて開始姿勢へ戻します。手を離したら Enter")
            q = arm.q()
            q[5] = p.jaw_open_rad
            arm.move_to(q, duration_s=1.0)
            arm.move_to(q_start, duration_s=4.0)
        for c in cams.values():
            c.close()
        arm.disconnect()
        print(f"[real_sh] 記録: {out}", flush=True)


if __name__ == "__main__":
    main()
