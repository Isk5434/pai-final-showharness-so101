"""採用エピソードだけを Show-Harness の学習レイアウトに並べる（コピーせずシンボリックリンク）。

Show-Harness の train/scripts/prepare_dataset.sh は
    SRC/<タスク>/rollout_000, rollout_001, ...（各 metadata.json に task_text）
を前提にする。収集フォルダ（ep_XXXXXX、_rejected/ に不採用）からこの形を作る。
複数の収集フォルダをまとめられる。古い収集で task_text が無ければ metadata.json に足す。

    python scripts/train/export_rollouts.py --src outputs/collect_a outputs/collect_b \\
        --out data/train_view/oracle_v1
    # VLA と同じエピソード一覧（指示を付け替えたものを含む）から作る。指示は一覧の値を使い、
    # タスクごとの task_text.txt に書く（prepare_dataset.sh は rollout_000 の metadata より優先する）
    python scripts/train/export_rollouts.py --list outputs/dm05_near_mix_list.json --out data/train_view/near_mix57
"""
import argparse
import json
import os
import shutil
from pathlib import Path


def entries(args):
    if args.list:
        for e in json.load(open(args.list)):
            src = Path("outputs") / e["dir"]
            meta = json.loads((find_rollout(src, e["episode"]) / "metadata.json").read_text())
            yield src, {**e, "seed": meta.get("seed")}
        return
    for src in map(Path, args.src):
        for line in open(src / "episodes.jsonl"):
            e = json.loads(line)
            if e.get("valid"):
                yield src, e


def find_rollout(src: Path, episode: str) -> Path:
    """収集時の指示のフォルダ（_rejected を除く）にある rollout。付け替えた指示とは別のフォルダのことがある。"""
    hits = [p for p in src.glob(f"*/{episode}") if p.parent.name not in ("traj", "video", "_rejected")]
    if len(hits) != 1:
        raise SystemExit(f"{src}/*/{episode}: {len(hits)} 件")
    return hits[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--src", nargs="+")
    g.add_argument("--list", help='[{"dir": ..., "episode": ..., "instruction": ...}]（outputs からの相対）')
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    n_by_task: dict[str, int] = {}
    manifest = []
    for src, e in entries(args):
        task_dir = e["instruction"].lower().replace(" ", "_")
        ep = find_rollout(src, e["episode"]).resolve()
        meta_p = ep / "metadata.json"
        meta = json.loads(meta_p.read_text())
        if "task_text" not in meta:  # 古い収集の補完
            meta["task_text"] = e["instruction"]
            meta_p.write_text(json.dumps(meta, indent=2, sort_keys=True))
        k = n_by_task.get(task_dir, 0)
        dst = out / task_dir / f"rollout_{k:03d}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(ep, dst)
        n_by_task[task_dir] = k + 1
        (out / task_dir / "task_text.txt").write_text(e["instruction"] + "\n")
        manifest.append({"rollout": str(dst.relative_to(out)), "source": str(ep),
                         "seed": e["seed"], "tokens": e.get("tokens")})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(f"[書き出し] {sum(n_by_task.values())} エピソード → {out}  {n_by_task}")


if __name__ == "__main__":
    main()
