"""収集した 30 Hz 軌道（traj/）を LeRobot v3 データセットへ変換する（VLA 学習用）。

Workshop / sreetz-nv のデータと同じ形にする:
  - lerobot 0.4.3（teleop-docker 内。sreetz-nv の v3.0 データを書いた版）で書く
  - 特徴量名・形状・fps・robot_type を sreetz-nv/so101_teleop_vials_rack_left と一致させる
  - 単位は Workshop の実機単位（腕 −100〜100、グリッパー 0〜100。utils/lerobot_interface.py と同じ変換）
  - observation.state = 観測関節（q_obs）、action = 指令関節（q_cmd = IK 解 + 重力補正）
採用（valid）エピソードだけを変換し、最後に読み戻して元データと突き合わせる。

    bash scripts/sim/run_in_teleop_docker.sh scripts/convert/traj_to_lerobot.py \\
        --src /workspace/out/collect_test4 --out /workspace/out/lerobot/so101_showharness_oracle_test
"""

import argparse
import json
import os
import shutil

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--src", required=True, help="oracle_collect.py の出力フォルダ")
parser.add_argument("--out", required=True, help="LeRobot データセットのルート（新規）")
parser.add_argument("--repo_id", default="Isk5434/so101_showharness_vials_oracle")
parser.add_argument("--headless", action="store_true")  # run_in_teleop_docker.sh が付ける
args = parser.parse_args()

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from so101_showharness.units import LEROBOT_JOINTS as JOINTS  # noqa: E402
from so101_showharness.units import rad_to_raw as _rad_to_raw  # noqa: E402


def rad_to_raw(q) -> np.ndarray:
    """sim の rad → Workshop 実機単位（units.py、Workshop の get_raw_actions_from_radians と同じ式）。"""
    return _rad_to_raw(q).astype(np.float32)


FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (6,), "names": [f"{j}.pos" for j in JOINTS]},
    "action": {"dtype": "float32", "shape": (6,), "names": [f"{j}.pos" for j in JOINTS]},
    "observation.images.ego": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
    "observation.images.external_D455": {"dtype": "video", "shape": (480, 640, 3),
                                         "names": ["height", "width", "channels"]},
}


def read_video(path: str) -> list:
    import imageio.v3 as iio

    return [f for f in iio.imiter(path)]


def main():
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    episodes = [json.loads(line) for line in open(os.path.join(args.src, "episodes.jsonl"))]
    valid = [e for e in episodes if e.get("valid")]
    print(f"[変換] 採用 {len(valid)} / 全 {len(episodes)} エピソード", flush=True)
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=30, features=FEATURES, root=args.out,
                               robot_type="so101_follower", use_videos=True)
    expect = []
    for e in valid:
        d = os.path.join(args.src, "traj", e["episode"])
        rows = [json.loads(line) for line in open(os.path.join(d, "frames.jsonl"))]
        ext = read_video(os.path.join(d, "video", "external.mp4"))
        wri = read_video(os.path.join(d, "video", "wrist.mp4"))
        n = min(len(rows), len(ext), len(wri))
        if abs(len(rows) - len(ext)) > 1 or abs(len(rows) - len(wri)) > 1:
            raise RuntimeError(f"{e['episode']}: フレーム数が合わない rows={len(rows)} ext={len(ext)} wrist={len(wri)}")
        for i in range(n):
            ds.add_frame({
                "observation.state": rad_to_raw(rows[i]["q_obs"]),
                "action": rad_to_raw(rows[i]["q_cmd"]),
                "observation.images.ego": wri[i][..., :3],
                "observation.images.external_D455": ext[i][..., :3],
                "task": e["instruction"],
            })
        ds.save_episode()
        expect.append((e["episode"], n, rad_to_raw(rows[0]["q_obs"]), rad_to_raw(rows[n - 1]["q_cmd"])))
        print(f"[変換] {e['episode']} {n} フレーム", flush=True)
    ds.finalize()

    # 読み戻して突き合わせ（フレーム数・最初の state・最後の action）
    back = LeRobotDataset(args.repo_id, root=args.out)
    assert back.meta.total_episodes == len(expect), (back.meta.total_episodes, len(expect))
    start = 0
    for name, n, s0, a_last in expect:
        f0 = back[start]
        fl = back[start + n - 1]
        assert np.allclose(f0["observation.state"].numpy(), s0, atol=1e-4), name
        assert np.allclose(fl["action"].numpy(), a_last, atol=1e-4), name
        start += n
    print(f"[変換] 読み戻し確認 OK: {back.meta.total_episodes} エピソード / {back.meta.total_frames} フレーム "
          f"→ {args.out}", flush=True)


if __name__ == "__main__":
    main()
