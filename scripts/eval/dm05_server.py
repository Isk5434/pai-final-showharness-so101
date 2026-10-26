"""DM05（LoRA で FT したチェックポイント）の推論サーバ。ホストの専用 env で動かし、sim（docker）から呼ぶ。

    /home/ubuntu/so101-showharness-deps/lerobot-dm05/.venv/bin/python scripts/eval/dm05_server.py \\
        --ckpt outputs/train_dm05_success50/run/checkpoints/last/pretrained_model \\
        --dataset outputs/lerobot/so101_gpt_success50 --port 8780

POST /act   {"reset": bool, "task": str, "state": [6], "front_png_b64": ..., "wrist_png_b64": ...}
         -> {"action": [6], "infer_s": float}
POST /chunk 同じ入力 -> {"actions": [[6] x n], "infer_s": float}（実機用：1 回の往復で n_action_steps 手をまとめて返す。
         クラウド越しの往復を毎ステップ払わないため。--blend / --ensemble は効かない。画像は PNG でも JPEG でもよい）
読み込みと前後処理は lerobot-eval と同じ道筋（make_policy + make_pre_post_processors）。行動チャンクは
policy.select_action が内部のキューで n_action_steps ずつ出すので、sim は制御ステップ（30 Hz）ごとに 1 回呼ぶ。

--blend K（> 0）でチャンクの継ぎ目をならす: n_action_steps ごとに predict_action_chunk で新しいチャンクを作り、
前のチャンクの未実行の残り（同じ時刻を指す）と先頭 K ステップを線形にクロスフェードする（新しい側の重みが
1/(K+1) → K/(K+1) と増える）。sim は推論を待つ同期実行なので、残りと新チャンクの時刻はずれない。
0 なら select_action そのまま（継ぎ目で指令が飛び、0.5 秒周期でかくかくする）。

--ensemble K（> 0）で ACT 式の temporal ensemble: K ステップごとにチャンク（chunk_size 先まで）を予測し、各時刻に
ついて、その時刻を含むこれまでの全予測を重み exp(-m·i)（i = 古い順の番号、--ens_m = m）で平均して出す。
推論 1 回 約 1.5 秒なので K=1 は遅すぎる（1 話 45 分）。--blend とは同時に使えない。
"""
import argparse
import base64
import io
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch
from PIL import Image

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies import make_policy, make_pre_post_processors


def img(b64: str) -> torch.Tensor:
    a = np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W) in [0, 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True, help="学習に使った LeRobot データの root（特徴と統計）")
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--host", default="0.0.0.0", help="実機用は 127.0.0.1（SSH トンネル経由だけで届く）")
    ap.add_argument("--blend", type=int, default=0, help="チャンク継ぎ目のクロスフェード長（ステップ）。0 で無効")
    ap.add_argument("--ensemble", type=int, default=0, help="temporal ensemble の予測間隔（ステップ）。0 で無効")
    ap.add_argument("--ens_m", type=float, default=0.01, help="temporal ensemble の重み exp(-m·i) の m（ACT 既定 0.01）")
    args = ap.parse_args()

    cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    cfg.pretrained_path = args.ckpt
    cfg.device = "cuda"
    meta = LeRobotDatasetMetadata("local/so101_gpt_success", root=args.dataset)
    policy = make_policy(cfg, ds_meta=meta)
    policy.eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=args.ckpt)
    if args.blend and args.ensemble:
        raise SystemExit("--blend と --ensemble は同時に使えない")
    print(f"[dm05_server] loaded {args.ckpt} (use_peft={cfg.use_peft}, blend={args.blend}, "
          f"ensemble={args.ensemble}, m={args.ens_m}, n_action_steps={cfg.n_action_steps}) on {args.host}:{args.port}", flush=True)
    ens = {"t": 0, "preds": {}}  # preds: 時刻 -> その時刻を指す予測のリスト（古い順）

    def ensembled(batch) -> np.ndarray:
        t = ens["t"]
        if t % args.ensemble == 0 or t not in ens["preds"]:
            for i, a in enumerate(chunk(batch)):
                ens["preds"].setdefault(t + i, []).append(a)
        ps = ens["preds"].pop(t)
        w = np.exp(-args.ens_m * np.arange(len(ps)))
        ens["t"] = t + 1
        return (w[:, None] * np.asarray(ps)).sum(0) / w.sum()
    st = {"plan": [], "used": 0}  # plan: これから実行する絶対関節指令（--blend 時）

    def chunk(batch) -> list[np.ndarray]:
        c = policy.predict_action_chunk(pre(batch))[0]  # (chunk_size, 6) 正規化済み
        return list(post(c).float().cpu().numpy())  # 時刻をバッチ次元として逆正規化

    def blended(batch) -> np.ndarray:
        if not st["plan"] or st["used"] >= cfg.n_action_steps:
            new, old = chunk(batch), st["plan"]
            for i in range(min(args.blend, len(old))):
                w = (i + 1) / (args.blend + 1)
                new[i] = (1 - w) * old[i] + w * new[i]
            st["plan"], st["used"] = new, 0
        st["used"] += 1
        return st["plan"].pop(0)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            t0 = time.time()
            if req.get("reset"):
                policy.reset()
                st["plan"], st["used"] = [], 0
                ens["t"], ens["preds"] = 0, {}
            batch = {
                "observation.state": torch.tensor(req["state"], dtype=torch.float32).unsqueeze(0),
                "observation.images.front": img(req["front_png_b64"]),
                "observation.images.wrist": img(req["wrist_png_b64"]),
                "task": [req["task"]],
            }
            if self.path == "/chunk":
                with torch.inference_mode():
                    acts = [a.tolist() for a in chunk(batch)[: cfg.n_action_steps]]
                body = json.dumps({"actions": acts, "infer_s": round(time.time() - t0, 3)}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            with torch.inference_mode():
                if args.ensemble > 0:
                    out = ensembled(batch).tolist()
                elif args.blend > 0:
                    out = blended(batch).tolist()
                else:
                    out = post(policy.select_action(pre(batch))).squeeze(0).float().cpu().numpy().tolist()
            body = json.dumps({"action": out, "infer_s": round(time.time() - t0, 3)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    HTTPServer((args.host, args.port), H).serve_forever()


if __name__ == "__main__":
    main()
