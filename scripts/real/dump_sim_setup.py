"""sim（Workshop の Vials-To-Rack、収集と同じ設定）から、実機パイプラインの既定値を書き出す。

- 外部カメラ（camera_external_D455）：内部パラメータ K、タスク座標での位置と向き（ROS 規約：+z が前、+y が下）
- 手首カメラ（camera_ego）：sim では姿勢が更新されず基部に固定。v150 の GPT への指示と同じにするため、その固定値を持つ
- マットの上面の高さ（タスク座標、基部が原点）と、収集の開始姿勢

実機ではこの値を初期値にして、カメラを sim と同じ位置に置く（または実測した値で上書きする）。

    bash scripts/sim/run_in_teleop_docker.sh scripts/real/dump_sim_setup.py --out /workspace/out/real_setup_sim.json
"""
import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Lerobot-So101-Teleop-Vials-To-Rack-DR")
parser.add_argument("--seed", type=int, default=100000)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402

from so101_showharness.kinematics import quat_wxyz_to_mat  # noqa: E402
from so101_showharness.oracle.vials_to_rack import OracleParams  # noqa: E402
from so101_showharness.sim.scene import VialsRackScene  # noqa: E402
from so101_showharness.sim.workshop_backend import So101WorkshopBackend, make_env  # noqa: E402


def main():
    p = OracleParams()
    env = make_env(args.task, seed=args.seed, fixed_look=True, preplaced_vial=False, look="white_arm")
    be = So101WorkshopBackend(env, jaw_open_rad=p.jaw_open_rad)
    be.reset(args.seed)
    scene = VialsRackScene(be)
    root = be.robot.data.root_pos_w[0].cpu().numpy()

    def cam(key):
        c = be.u.scene[key]
        return (c.data.intrinsic_matrices[0].cpu().numpy(), c.data.pos_w[0].cpu().numpy() - root,
                quat_wxyz_to_mat(c.data.quat_w_ros[0].cpu().numpy()))

    ke, ce, re_ = cam("camera_external_D455")
    kw, cw, rw = cam("camera_ego")
    out = {
        "source": f"sim {args.task} seed {args.seed}（収集 v150 と同じ設定）",
        "image_size": [640, 480],
        "external": {"K": ke.tolist(), "pos_task_m": ce.tolist(), "R_task_cam_ros": re_.tolist()},
        # sim の camera_ego は pos_w / quat_w_ros がアームと一緒に更新されず、基部に対して固定のまま（2026-10-20 確認）。
        # v150 の収集で GPT に渡した「画像 2 の向き」はこの固定値から計算されていたので、実機でも同じ値を使う。
        "wrist": {"K": kw.tolist(), "mode": "v150_fixed", "pos_task_m": cw.tolist(), "R_task_cam_ros": rw.tolist()},
        "table_top_task_z_m": float(scene.table_top),
        "begin_task_m": (np.asarray(p.begin_world) - root).tolist(),
        "grasp_pitch_deg": float(p.grasp_pitch_deg),
        "release_pitch_target_deg": float(p.release_pitch_target_deg),
        "jaw_open_rad": float(p.jaw_open_rad),
    }
    # 実機で物を置く位置（タスク座標、基部が原点、+X 前方・+Y 左）。vial の並びは DR でシードごとに揺れるので幅で出す
    lay = []
    for sd in range(args.seed, args.seed + 20):
        be.reset(sd)
        rb = scene.rack_box()
        lay.append({"rack_min": rb[0][:2].tolist(), "rack_max": rb[1][:2].tolist(),
                    "vials": {v.name: v.pos[:2].tolist() for v in scene.vials()}})
    a = lambda k: np.asarray([x[k] for x in lay])  # noqa: E731
    vp = {n: np.asarray([x["vials"][n] for x in lay]) for n in lay[0]["vials"]}
    out["layout_20_seeds"] = {
        "rack_xy_min_mean": a("rack_min").mean(0).round(4).tolist(), "rack_xy_max_mean": a("rack_max").mean(0).round(4).tolist(),
        "rack_xy_center_range": [(a("rack_min") / 2 + a("rack_max") / 2).min(0).round(4).tolist(),
                                 (a("rack_min") / 2 + a("rack_max") / 2).max(0).round(4).tolist()],
        "vials_xy_mean": {n: v.mean(0).round(4).tolist() for n, v in vp.items()},
        "vials_xy_min": {n: v.min(0).round(4).tolist() for n, v in vp.items()},
        "vials_xy_max": {n: v.max(0).round(4).tolist() for n, v in vp.items()},
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
