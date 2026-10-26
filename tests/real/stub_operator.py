"""gpt_operator_server.py の代わり（GPT を呼ばない）。決まったトークン列を順に返す。実機クライアントの通し試験用。

    python3 tests/real/stub_operator.py --port 8799 --tokens MV_DOWN,MV_DOWN,GRASP,MV_UP,DONE --save_prompt /tmp/p.txt
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8799)
ap.add_argument("--tokens", required=True)
ap.add_argument("--save_prompt", default=None)
args = ap.parse_args()
seq = args.tokens.split(",")
state = {"i": 0}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if args.save_prompt and state["i"] == 0:
            open(args.save_prompt, "w").write(req["prompt"])
        tok = seq[min(state["i"], len(seq) - 1)]
        state["i"] += 1
        body = json.dumps({"token": tok, "raw": f"stub: {tok}", "latency_s": 0.0, "error": None}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


HTTPServer(("127.0.0.1", args.port), H).serve_forever()
