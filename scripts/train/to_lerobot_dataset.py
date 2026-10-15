"""GPT 収集のエピソード（30 Hz 軌道 + 生の外部/手首動画）を LeRobotDataset v3.0 に変換する。

DM05（lerobot/dm05_base）などの LeRobot 方策の学習用。専用 env（lerobot main 2577da0e）で動かす:

    /home/ubuntu/so101-showharness-deps/lerobot-dm05/.venv/bin/python scripts/train/to_lerobot_dataset.py \\
        --list outputs/dm05_success_list.json --root outputs/lerobot/so101_gpt_success54

- observation.state: 観測した関節角 6 次元（rad, Workshop 順 Rotation..Wrist_Roll, Jaw）= frames.jsonl の q_obs
- action: その tick に送った関節指令 6 次元（rad）= q_cmd
- observation.images.front / .wrist: 外部カメラ / 手首カメラの生フレーム 640x480（回転しない。
  Show-Harness の画像規約は Show-Harness 形式のデータにだけ適用する）
- task: エピソードの指示文
30 Hz（1 制御ステップ = 1 フレーム）。軌道と動画のフレーム数が一致しないエピソードは飛ばす。
"""
import argparse
import json
from pathlib import Path

import av
import numpy as np

from lerobot.datasets import LeRobotDataset

JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
OUT = Path(__file__).resolve().parents[2] / "outputs"


def frames(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", required=True, help='[{"dir": ..., "episode": ..., "instruction": ...}]（outputs からの相対）')
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo_id", default="local/so101_gpt_success")
    args = ap.parse_args()
    eps = json.load(open(args.list))
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINTS},
        "action": {"dtype": "float32", "shape": (6,), "names": JOINTS},
        "observation.images.front": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
        "observation.images.wrist": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
    }
    ds = LeRobotDataset.create(args.repo_id, fps=30, features=features, root=args.root, robot_type="so101",
                               use_videos=True)
    kept, skipped = 0, []
    for e in eps:
        t = OUT / e["dir"] / "traj" / e["episode"]
        rows = [json.loads(line) for line in open(t / "frames.jsonl")]
        ext, wr = frames(t / "video" / "external.mp4"), frames(t / "video" / "wrist.mp4")
        n = min(len(rows), len(ext), len(wr))
        if max(len(rows), len(ext), len(wr)) - n > 2:
            skipped.append((e["dir"], e["episode"], len(rows), len(ext), len(wr)))
            continue
        for i in range(n):
            ds.add_frame({
                "observation.state": np.asarray(rows[i]["q_obs"], dtype=np.float32),
                "action": np.asarray(rows[i]["q_cmd"], dtype=np.float32),
                "observation.images.front": ext[i],
                "observation.images.wrist": wr[i],
                "task": e["instruction"],
            })
        ds.save_episode()
        kept += 1
        print(f"[to_lerobot] {kept}/{len(eps)} {e['dir']} {e['episode']} frames={n}", flush=True)
    ds.finalize()
    print(f"[to_lerobot] kept {kept}, skipped {len(skipped)}: {skipped}", flush=True)


if __name__ == "__main__":
    main()
