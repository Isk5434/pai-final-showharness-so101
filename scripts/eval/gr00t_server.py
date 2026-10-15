"""GR00T N1.7（SO-101 データで FT したチェックポイント）の推論サーバ。dm05_server.py と同じ /act の約束で、
評価側（dm05_eval_sim.py）をそのまま使えるようにする。専用 env（Isaac-GR00T d2b7e75、uv.lock どおり）で動かす:

    /home/ubuntu/so101-showharness-deps/Isaac-GR00T-n17/.venv/bin/python scripts/eval/gr00t_server.py \\
        --ckpt outputs/train_gr00t_success50/checkpoint-10000 --ensemble 5 --port 8780

POST /act   {"reset": bool, "task": str, "state": [6], "front_png_b64": ..., "wrist_png_b64": ...}
         -> {"action": [6], "infer_s": float}
状態と行動は 腕 5 関節（single_arm）+ グリッパ 1（gripper）。行動は絶対関節角（configs/gr00t/so101_config.py）。
--ensemble K で dm05_server.py と同じ temporal ensemble（K ステップごとに予測し、重なりを exp(-m·i) で平均）。
0 なら予測チャンクの先頭から n_action_steps ずつ使って切り替える。
"""
import argparse
import base64
import importlib.util
import io
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]


def img(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"), dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=str(REPO / "configs/gr00t/so101_config.py"))
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--ensemble", type=int, default=0)
    ap.add_argument("--ens_m", type=float, default=0.01)
    ap.add_argument("--n_action_steps", type=int, default=8, help="--ensemble 0 のとき 1 チャンクから使う手数")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("so101_config", args.config)  # NEW_EMBODIMENT に登録される
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy import Gr00tPolicy

    policy = Gr00tPolicy(model_path=args.ckpt, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT, device="cuda:0", strict=True)
    horizon = len(policy.get_modality_config()["action"].delta_indices)
    print(f"[gr00t_server] loaded {args.ckpt} horizon={horizon} ensemble={args.ensemble} m={args.ens_m} "
          f"on :{args.port}", flush=True)
    ens = {"t": 0, "preds": {}}
    plan = {"steps": []}

    def chunk(req) -> list[np.ndarray]:
        s = np.asarray(req["state"], dtype=np.float32)
        obs = {"video": {"front": img(req["front_png_b64"])[None, None], "wrist": img(req["wrist_png_b64"])[None, None]},
               "state": {"single_arm": s[None, None, :5], "gripper": s[None, None, 5:]},
               "language": {"annotation.human.task_description": [[req["task"]]]}}
        act, _ = policy.get_action(obs)
        a = np.concatenate([act["single_arm"][0], act["gripper"][0]], axis=-1)  # (horizon, 6)
        return list(a.astype(np.float64))

    def ensembled(req) -> np.ndarray:
        t = ens["t"]
        if t % args.ensemble == 0 or t not in ens["preds"]:
            for i, a in enumerate(chunk(req)):
                ens["preds"].setdefault(t + i, []).append(a)
        ps = ens["preds"].pop(t)
        w = np.exp(-args.ens_m * np.arange(len(ps)))
        ens["t"] = t + 1
        return (w[:, None] * np.asarray(ps)).sum(0) / w.sum()

    def chunked(req) -> np.ndarray:
        if not plan["steps"]:
            plan["steps"] = chunk(req)[: args.n_action_steps]
        return plan["steps"].pop(0)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            t0 = time.time()
            if req.get("reset"):
                policy.reset()
                ens["t"], ens["preds"], plan["steps"] = 0, {}, []
            out = (ensembled(req) if args.ensemble > 0 else chunked(req)).tolist()
            body = json.dumps({"action": out, "infer_s": round(time.time() - t0, 3)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    HTTPServer(("0.0.0.0", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
