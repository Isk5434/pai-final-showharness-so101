"""camera_check.py 用の参照画像を作る（AWS 側）。学習データの話の最初のコマ（開始姿勢・vial が 3 本並んだ状態）を書き出す。

    python3 scripts/real/make_reference.py [--episode near200/w0/gpt-6-astra/ep_100001] [--out outputs/real_reference]
"""
import argparse
import os
from pathlib import Path

import av
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--episode", default="near200/w0/gpt-6-astra/ep_100001")
ap.add_argument("--out", default=str(REPO / "outputs/real_reference"))
args = ap.parse_args()
d, ep = os.path.split(args.episode)
os.makedirs(args.out, exist_ok=True)
for key, f in (("front", "external.mp4"), ("wrist", "wrist.mp4")):
    with av.open(str(REPO / "outputs" / d / "traj" / ep / "video" / f)) as c:
        frame = next(c.decode(video=0)).to_ndarray(format="rgb24")
    Image.fromarray(frame).save(os.path.join(args.out, f"{key}.png"))
    print(os.path.join(args.out, f"{key}.png"), frame.shape)
