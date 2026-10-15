"""FT 済み LoRA を配信する最小サーバ（学習 venv で動かす）。sim / 実機から1手ずつ問い合わせる。

学習と推論で入力がずれないことが最優先なので、LLaMA-Factory 自身の ChatModel（学習と同じ
テンプレート qwen3_5_nothink・同じ画像処理プラグイン）で推論し、プロンプト本文も学習データを
作った Show-Harness のテンプレート（prompts/<version>/mvtoken_generator_lite.txt）を同じ引数で
埋める。

POST /act  JSON:
  {"task": str, "recent_moves": [newest, ...], "gripper_closed": bool,
   "agentview_png_b64": str, "wrist_png_b64": str}       画像は prepared_pair 済み 256x256
→ {"token": str, "raw": str, "in_vocab": bool, "latency_s": float}

    FLA_TILELANG=0 python scripts/eval/policy_server.py --adapter outputs/train/smoke --port 8765
"""
import argparse
import base64
import io
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from PIL import Image

SH = os.environ.get("SHOW_HARNESS_ROOT", "/home/ubuntu/so101-showharness-deps/Show-Harness")
VOCAB = ("MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN", "RT_ROLL_LEFT", "RT_ROLL_RIGHT",
         "RT_PITCH_FWD", "RT_PITCH_BACK", "RT_YAW_CW", "RT_YAW_CCW", "GRASP", "RELEASE", "DONE")
RECENT_WINDOW = 5  # rollouts_to_alpaca.RECENT_WINDOW


def build_prompt(template: str, task: str, recent_moves: list, gripper_closed: bool) -> str:
    """rollouts_to_alpaca.py の lite モードと同じ埋め方（履歴は新しい順・最大 5、無ければ none）。"""
    recent = ", ".join(recent_moves[:RECENT_WINDOW]) or "none"
    return template.format(task=task, gripper_state="closed" if gripper_closed else "open",
                           recent_moves=recent)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--version", default="v5")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    from llamafactory.chat import ChatModel

    template = open(os.path.join(SH, "prompts", args.version, "mvtoken_generator_lite.txt")).read()
    model = ChatModel(dict(
        model_name_or_path=args.base, adapter_name_or_path=args.adapter, finetuning_type="lora",
        template="qwen3_5_nothink", image_max_pixels=65536, trust_remote_code=True,
        infer_backend="huggingface", do_sample=False, max_new_tokens=8,
    ))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_POST(self):
            if self.path != "/act":
                self.send_error(404)
                return
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            t0 = time.time()
            imgs = [Image.open(io.BytesIO(base64.b64decode(req[k]))).convert("RGB")
                    for k in ("agentview_png_b64", "wrist_png_b64")]
            prompt = build_prompt(template, req["task"], req.get("recent_moves", []),
                                  bool(req.get("gripper_closed", False)))
            out = model.chat([{"role": "user", "content": "<image><image>" + prompt}], images=imgs)
            raw = out[0].response_text.strip()
            token = raw.split()[0].strip(",.") if raw else ""
            body = json.dumps({"token": token, "raw": raw, "in_vocab": token in VOCAB,
                               "latency_s": round(time.time() - t0, 3)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    print(f"[配信] {args.adapter} を :{args.port} で待ち受け", flush=True)
    HTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
