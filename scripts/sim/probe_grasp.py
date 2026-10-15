"""把持条件を実験で決める: ピッチ × 開口内の位置 × 軸からの深さ を総当たり。

各試行: 同じシーンに reset → 把持点の 6 cm 上へ（記録しない移動）→ まっすぐ降りる → 指を閉じる
→ 4 cm 持ち上げる → バイアルが上がったか・手の中で動いていないかを測る。
結果は probe_grasp.json と日本語の表（probe_grasp_ja.md）、全試行を1本の閉ループ動画に。
"""

import argparse
import itertools
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Lerobot-So101-Teleop-Vials-To-Rack-DR")
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--vial", default="vial_2")
parser.add_argument("--pitches_deg", default="47,60")
parser.add_argument("--centers_mm", default="25,35,44",
                    help="バイアル中心を TCP（固定指）から閉じ方向に何 mm の所に置くか")
parser.add_argument("--below_axis_mm", default="0,4,8", help="TCP をバイアル軸より何 mm 下にするか")
parser.add_argument("--jaw_open", type=float, default=0.45,
                    help="把持前の指の開き [rad]（人のデモは 0.36〜0.46、0.9 だと上から押し出す）")
parser.add_argument("--out", default="/workspace/out/probe_grasp")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402

from so101_showharness.controller import ROLL_CLOSING_HORIZONTAL  # noqa: E402
from so101_showharness.exec import So101AtomicExec  # noqa: E402
from so101_showharness.recorder.video import EpisodeVideo  # noqa: E402
from so101_showharness.sim.scene import VialsRackScene  # noqa: E402
from so101_showharness.sim.workshop_backend import So101WorkshopBackend, make_env  # noqa: E402


def main():
    os.makedirs(args.out, exist_ok=True)
    env = make_env(args.task, seed=args.seed)
    be = So101WorkshopBackend(env, jaw_open_rad=args.jaw_open)
    ex = So101AtomicExec(be, step_m=0.02)
    video = EpisodeVideo(os.path.join(args.out, "probe_grasp.mp4"))
    results = []
    combos = list(itertools.product([float(x) for x in args.pitches_deg.split(",")],
                                    [float(x) for x in args.centers_mm.split(",")],
                                    [float(x) for x in args.below_axis_mm.split(",")]))
    scene = None
    for pitch_deg, center_mm, below_mm in combos:
        be.reset(args.seed)
        be.step_hooks = [video.on_step]
        ex.lattice_origin, ex.hold_target, ex.grip_cmd = None, None, 1.0
        scene = scene or VialsRackScene(be)
        v0 = scene.vial(args.vial)
        pitch = np.radians(pitch_deg)
        tag = f"pitch{pitch_deg:.0f}_c{center_mm:.0f}_b{below_mm:.0f}"
        be.annotation = {"token": "-", "phase": tag}
        ex.goto_pose(v0.pos + [0, 0, 0.09], pitch, roll=ROLL_CLOSING_HORIZONTAL)
        c = be.tcp_rotmat()[:, 0]
        target = v0.pos - c * (center_mm / 1000) - [0, 0, below_mm / 1000]
        r = {"pitch_deg": pitch_deg, "center_mm": center_mm, "below_axis_mm": below_mm,
             "closing_vs_vial_axis_deg": float(np.degrees(np.arccos(min(1.0, abs(c @ v0.axis)))))}
        r["above_ok"] = ex.goto_pose(target + [0, 0, 0.06], pitch, roll=ROLL_CLOSING_HORIZONTAL)
        r["down_ok"] = ex._servo(target)
        r["down_contact"] = getattr(be, "last_contact", None) if ex.last_contact else None
        r["tcp_err_mm"] = float(np.linalg.norm(be.tcp_pos() - target) * 1000)
        ex.grasp()
        r["jaw_after_close_rad"] = float(be.q_obs()[5])
        v1 = scene.vial(args.vial)
        r["vial_moved_on_close_mm"] = float(np.linalg.norm(v1.pos - v0.pos) * 1000)
        rel0 = be.tcp_rotmat().T @ (v1.pos - be.tcp_pos())
        lift_ok = ex._servo(be.tcp_pos() + [0, 0, 0.04])
        v2 = scene.vial(args.vial)
        rel1 = be.tcp_rotmat().T @ (v2.pos - be.tcp_pos())
        r.update({
            "lift_ok": lift_ok,
            "vial_rise_mm": float((v2.pos[2] - v0.pos[2]) * 1000),
            "vial_shift_in_hand_mm": float(np.linalg.norm(rel1 - rel0) * 1000),
            "grasped_flag": be.grasped(),
            "contacts_after_lift": be.contact_report(),
        })
        r["held"] = bool(r["vial_rise_mm"] > 30 and r["vial_shift_in_hand_mm"] < 5)
        results.append(r)
        print(f"[把持試験] {tag} 保持={r['held']} 上昇={r['vial_rise_mm']:.0f}mm "
              f"手の中のずれ={r['vial_shift_in_hand_mm']:.1f}mm 閉後の指={r['jaw_after_close_rad']:.2f}rad", flush=True)
        ex.release()
    video.close()
    with open(os.path.join(args.out, "probe_grasp.json"), "w") as f:
        json.dump(results, f, indent=1, default=float, ensure_ascii=False)
    lines = ["# 把持条件の実験結果", "", f"シード {args.seed}、対象 {args.vial}、把持前の指の開き {args.jaw_open} rad", "",
             "| ピッチ | 中心位置 [mm] | 軸より下 [mm] | 保持 | 上昇 [mm] | 手の中のずれ [mm] | 閉後の指 [rad] | 降下中の接触 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        lines.append(f"| {r['pitch_deg']:.0f}° | {r['center_mm']:.0f} | {r['below_axis_mm']:.0f} | "
                     f"{'✅' if r['held'] else '❌'} | {r['vial_rise_mm']:.0f} | {r['vial_shift_in_hand_mm']:.1f} | "
                     f"{r['jaw_after_close_rad']:.2f} | {r['down_contact'] or '-'} |")
    with open(os.path.join(args.out, "probe_grasp_ja.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
