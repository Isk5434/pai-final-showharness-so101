"""学習に使ったお手本（dm05_success50_list.json）で、Astra が実際につかんだ vial と、指示の vial が一致するかを数える。

つかんだ vial は軌道データ（traj/<ep>/frames.jsonl）の接触記録から決める：つかんでいる間（grasped、または
gripper と jaw の両方が接触）に最も多く接触した vial。レイアウトは評価と同じで、手前 = vial_2、真ん中 = vial_1、一番奥 = vial_3。

    python3 scripts/analysis/count_grasped_vial.py [--list outputs/dm05_success50_list.json]
"""
import argparse
import collections
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
POS = {"closest": "near", "middle": "middle", "farthest": "far"}
VIAL = {"near": "vial_2", "middle": "vial_1", "far": "vial_3"}


def grasped_vial(frames):
    c = collections.Counter()
    for line in open(frames):
        f = json.loads(line)
        links = {k["link"] for k in f.get("contacts") or []}
        if f.get("grasped") or {"gripper", "jaw"} <= links:
            for k in f.get("contacts") or []:
                for name in k.get("with", {}):
                    c[name.lower()] += 1
    return c.most_common(1)[0][0] if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default=str(REPO / "outputs/dm05_success50_list.json"))
    args = ap.parse_args()
    out = REPO / "outputs"
    per = collections.defaultdict(collections.Counter)
    for x in json.load(open(args.list)):
        pos = next(v for k, v in POS.items() if k in x["instruction"])
        meta = {}
        eps = out / x["dir"] / "episodes.jsonl"
        if eps.exists():
            meta = next((json.loads(l) for l in open(eps) if json.loads(l)["episode"] == x["episode"]), {})
        target = meta.get("target_vial", VIAL[pos])
        frames = out / x["dir"] / "traj" / x["episode"] / "frames.jsonl"
        g = grasped_vial(frames) if os.path.exists(frames) else None
        per[pos]["target" if g == target else f"other:{g}"] += 1
    for pos in ("near", "middle", "far"):
        print(pos, sum(per[pos].values()), dict(per[pos]))


if __name__ == "__main__":
    main()
