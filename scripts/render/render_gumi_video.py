"""保存済みエピソードを GUMI 風パネル付きで描き直す（ホストで動く。Isaac 不要）。

入力は収集が残した 30 Hz 軌道（traj/<ep>/frames.jsonl + video/external.mp4 + video/wrist.mp4）。

    python scripts/render/render_gumi_video.py outputs/collect_test8/traj/ep_001004 \\
        --out outputs/collect_test8/video_gumi/ep_001004.mp4 [--speed 1.0]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import imageio.v3 as iio

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from so101_showharness.recorder.gumi_video import GumiState, compose, make_panel  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("traj_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--speed", type=float, default=1.0, help="再生速度（2.0 で 2 倍速）")
    args = ap.parse_args()
    d = Path(args.traj_dir)
    rows = [json.loads(line) for line in open(d / "frames.jsonl")]
    meta = json.loads((d / "frames_meta.json").read_text()) if (d / "frames_meta.json").exists() else {}
    task = meta.get("instruction") or meta.get("task_text") or "Pick up the vial and place it in the rack"
    fps = int(meta.get("record_hz", 30))
    ext = iio.imiter(d / "video" / "external.mp4")
    wri = iio.imiter(d / "video" / "wrist.mp4")
    first = None
    state = GumiState(task)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = None
    for row, e, w in zip(rows, ext, wri):
        if first is None:
            first = e
            panel = make_panel(2 * e.shape[1])
        frame = compose(e, w, panel.render(**state.update(row)))
        if proc is None:
            h, wd = frame.shape[:2]
            proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{wd}x{h}", "-r", str(fps * args.speed), "-i", "-",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", str(out)],
                stdin=subprocess.PIPE)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"[描画] {out}  {len(rows)} フレーム")


if __name__ == "__main__":
    main()
