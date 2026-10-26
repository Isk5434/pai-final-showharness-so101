#!/usr/bin/env bash
# 並列収集の状況：ワーカーごとの試行数・成功数と、全体の進み具合。
REPO=$(cd "$(dirname "$0")/../../.." && pwd); RUN=${RUN:-near200}
python3 - "$REPO/outputs/$RUN" <<'PY'
import glob, json, os, sys, time
root = sys.argv[1]
tot_n = tot_s = 0; secs = []
for w in sorted(glob.glob(os.path.join(root, "w*/"))):
    rows = [json.loads(l) for f in glob.glob(os.path.join(w, "*/episodes.jsonl")) for l in open(f) if l.strip()]
    n = len(rows); s = sum(r.get("target_in_hole", False) for r in rows); sim = sum(r.get("success", False) for r in rows)
    tot_n += n; tot_s += s; secs += [r.get("seconds", 0) for r in rows]
    last = open(os.path.join(w, "worker.log")).read().strip().splitlines()[-1] if os.path.exists(os.path.join(w, "worker.log")) else ""
    print(f"{os.path.basename(w.rstrip('/'))}: 試行 {n:4d}  厳しい成功 {s:3d}  sim 判定 {sim:3d}   {last[-70:]}")
rate = tot_s / tot_n if tot_n else 0
print(f"合計: 試行 {tot_n}  厳しい成功 {tot_s}（{rate:.1%}）  1 話平均 {sum(secs)/max(len(secs),1):.0f} s")
PY
docker ps --format '{{.Names}} {{.Status}}' | grep '^so101sh_w' || echo "（動いているコンテナなし）"
