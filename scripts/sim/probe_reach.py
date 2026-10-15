"""把持の届きやすさを、位置（near / middle / far）× 把持時の手の傾き（ピッチ）で測る（GPT 不使用）。

オラクルの _choose_side（把持点まで降りる全行程が実行可能か＝IK・関節限界・手の床、と、手の衝突形状と
ほかのバイアル・ラックとの最小隙間）をそのまま使う。far が +60° で手首ピッチの限界に当たる件と、
middle で降りる途中に隣へ当たる件を、ピッチを変えて測る。

    bash scripts/sim/run_in_teleop_docker.sh scripts/sim/probe_reach.py --seeds 5000 5019 --out /workspace/out/probe_reach
"""
import argparse
import json
import os

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--seeds", type=int, nargs=2, default=(5000, 5019), help="最初と最後のシード（両端含む）")
ap.add_argument("--pitches", type=float, nargs="+", default=[30, 40, 50, 60, 70, 80])
ap.add_argument("--lift_m", type=float, default=0.10, help="把持点からこの高さまで持ち上げられるかも見る")
ap.add_argument("--out", default="/workspace/out/probe_reach")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402

from so101_showharness.exec import So101AtomicExec  # noqa: E402
from so101_showharness.oracle.vials_to_rack import EpisodeInvalid, VialsToRackOracle  # noqa: E402
from so101_showharness.sim.scene import VialsRackScene  # noqa: E402
from so101_showharness.sim.workshop_backend import So101WorkshopBackend, make_env  # noqa: E402

POSITIONS = ("near", "middle", "far")


def main():
    os.makedirs(args.out, exist_ok=True)
    env = make_env(seed=args.seeds[0], fixed_look=True, preplaced_vial=False)
    be = So101WorkshopBackend(env, jaw_open_rad=0.45)
    ex = So101AtomicExec(be)
    scene, rows = None, []
    for seed in range(args.seeds[0], args.seeds[1] + 1):
        be.reset(seed)
        scene = scene or VialsRackScene(be)
        be.hand_floor_task_z = scene.table_top + 0.003
        be.workspace = {"min": [0.05, -0.30, scene.table_top + 0.002], "max": [0.45, 0.25, 0.32]}
        rack_c = scene.rack_box()[0] / 2 + scene.rack_box()[1] / 2
        by_dist = sorted(scene.vials(), key=lambda v: float(np.linalg.norm((v.pos - rack_c)[:2])))
        for pos, vial in zip(POSITIONS, by_dist):
            for pitch_deg in args.pitches:
                orc = VialsToRackOracle(be, ex, scene, writer=None, rng=np.random.default_rng(seed))
                pitch = np.radians(pitch_deg)
                row = {"seed": seed, "position": pos, "vial": vial.name, "pitch_deg": pitch_deg,
                       "reach_xy_m": round(float(np.linalg.norm(vial.pos[:2])), 3)}
                try:
                    roll, tcp, gap = orc._choose_side(vial, vial.pos, pitch)
                    row.update(ok=True, gap_mm=round(gap * 1000, 1),
                               lift_ok=all(be.feasible(tcp + [0, 0, dz], pitch, None, use_yaw_ref=False, roll=roll)[0]
                                           for dz in np.linspace(0.02, args.lift_m, 5)))
                except EpisodeInvalid as e:
                    row.update(ok=False, why=str(e))
                row["sides"] = orc.log.events  # 両側（ロール ±90°）の隙間・拒否理由・最寄りの障害物
                rows.append(row)
        print(f"[probe_reach] seed {seed} done", flush=True)
    with open(os.path.join(args.out, "reach.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")
    # 集計: 位置 × ピッチ ごとの「掴める（全行程実行可能・隙間 5 mm 以上）」率と持ち上げ可
    print("\nposition pitch  graspable  lift_ok  median_gap_mm", flush=True)
    for pos in POSITIONS:
        for pd in args.pitches:
            rs = [r for r in rows if r["position"] == pos and r["pitch_deg"] == pd]
            ok = [r for r in rs if r["ok"]]
            gaps = [r["gap_mm"] for r in ok]
            print(f"{pos:7s} {pd:5.0f}  {len(ok):3d}/{len(rs):<3d}   {sum(r['lift_ok'] for r in ok):3d}     "
                  f"{np.median(gaps) if gaps else float('nan'):6.1f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        os._exit(0)
