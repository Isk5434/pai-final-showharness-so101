"""採用エピソードだけを Show-Harness の学習レイアウトに並べる（コピーせずシンボリックリンク）。

Show-Harness の train/scripts/prepare_dataset.sh は
    SRC/<タスク>/rollout_000, rollout_001, ...（各 metadata.json に task_text）
を前提にする。収集フォルダ（ep_XXXXXX、_rejected/ に不採用）からこの形を作る。
複数の収集フォルダをまとめられる。古い収集で task_text が無ければ metadata.json に足す。

    python scripts/train/export_rollouts.py --src outputs/collect_a outputs/collect_b \\
        --out data/train_view/oracle_v1
"""
import argparse
import json
import os
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    n_by_task: dict[str, int] = {}
    manifest = []
    for src in map(Path, args.src):
        for line in open(src / "episodes.jsonl"):
            e = json.loads(line)
            if not e.get("valid"):
                continue
            task_dir = e["instruction"].lower().replace(" ", "_")
            ep = (src / task_dir / e["episode"]).resolve()
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
            manifest.append({"rollout": str(dst.relative_to(out)), "source": str(ep),
                             "seed": e["seed"], "tokens": e.get("tokens")})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(f"[書き出し] {sum(n_by_task.values())} エピソード → {out}  {n_by_task}")


if __name__ == "__main__":
    main()
