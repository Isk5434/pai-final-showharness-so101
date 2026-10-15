"""サブスクの GPT（Codex CLI の ChatGPT ログイン）に1手ずつ決めさせる、ホスト側の小さなサーバ。

sim（teleop-docker、--network=host）の gpt_collect.py から毎ステップ呼ばれる:
  POST /decide  {"model", "effort", "prompt", "external_png_b64", "wrist_png_b64"}
  → {"token", "raw", "latency_s", "error"}
`codex exec` を「読み取り専用・空の作業フォルダ・一時セッション」で起動し、最後の発言から
語彙のトークンを1つ取り出す。生の応答は返り値と --log に残す（何を見て何を言ったかを後で追える）。

    python3 scripts/collect/gpt_operator_server.py --port 8770 --log outputs/gpt_server.jsonl
"""
import argparse
import base64
import json
import re
import subprocess
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VOCAB = ("MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN", "RT_PITCH_FWD",
         "RT_PITCH_BACK", "RT_YAW_CW", "RT_YAW_CCW", "GRASP", "RELEASE", "DONE")
TOKEN_RE = re.compile(r"\b(" + "|".join(sorted(VOCAB, key=len, reverse=True)) + r")\b")


def decide(model: str, effort: str, prompt: str, ext_png: bytes, wrist_png: bytes,
           timeout_s: int = 240) -> dict:
    with tempfile.TemporaryDirectory(prefix="gptop_") as tmp:
        tmp = Path(tmp)
        (tmp / "external.png").write_bytes(ext_png)
        (tmp / "wrist.png").write_bytes(wrist_png)
        out = tmp / "last.txt"
        cmd = ["codex", "exec", "-m", model, "-c", f'model_reasoning_effort="{effort}"',
               "-s", "read-only", "--skip-git-repo-check", "--ephemeral", "-C", str(tmp),
               "-i", str(tmp / "external.png"), "-i", str(tmp / "wrist.png"),
               "-o", str(out), prompt]
        t0 = time.time()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, stdin=subprocess.DEVNULL)
            raw = out.read_text().strip() if out.exists() else ""
            err = None if p.returncode == 0 else f"rc={p.returncode}: {p.stderr[-400:]}"
        except subprocess.TimeoutExpired:
            raw, err = "", f"timeout>{timeout_s}s"
        found = TOKEN_RE.findall(raw)
        # the answer is the LAST vocabulary token said (the prompt asks for it on the last line)
        return {"token": found[-1] if found else "", "raw": raw[-2000:],
                "latency_s": round(time.time() - t0, 2), "error": err}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--log", default="outputs/gpt_server.jsonl")
    args = ap.parse_args()
    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            if self.path != "/decide":
                self.send_error(404)
                return
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            res = decide(req["model"], req.get("effort", "medium"), req["prompt"],
                         base64.b64decode(req["external_png_b64"]), base64.b64decode(req["wrist_png_b64"]))
            with open(log, "a") as f:
                f.write(json.dumps({"t": time.time(), "model": req["model"], "effort": req.get("effort"),
                                    "episode": req.get("episode"), "step": req.get("step"), **res},
                                   ensure_ascii=False) + "\n")
            body = json.dumps(res).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    print(f"[GPT] :{args.port} で待ち受け（codex exec、読み取り専用）", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
