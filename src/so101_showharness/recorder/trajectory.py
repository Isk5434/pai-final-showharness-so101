"""Continuous trajectory recorder, synchronized with the discrete Show-Harness decisions.

One episode directory holds two streams that share a clock (env step) and a key
(``decision_id`` = the Show-Harness rollout step index):

  decisions  -> Show-Harness ``RolloutWriter`` layout (agentview/ wrist/ actions.jsonl,
                frame = observation BEFORE the token). Written by the caller.
  frames     -> this recorder, every ``60 / record_hz`` env steps, also while the VLM is
                thinking or the arm is settling:
                  frames.jsonl          one row per recorded step
                  video/external.mp4    raw 640x480 external_D455, record_hz fps
                  video/wrist.mp4       raw 640x480 ego
                Row ``i`` of frames.jsonl is frame ``i`` of both videos.

Columns of frames.jsonl (joint order: Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw; rad):
  frame, env_step, t, ctrl_step, decision_id, token, phase, blocked,
  q_obs      measured joints            -> VLA state
  q_cmd      joints SENT to the env      -> VLA action label (q_des + gravity bias)
  q_des      IK solution before the bias
  tcp_obs    [x y z qw qx qy qz], task frame
  tcp_target commanded TCP position this control step (null when holding)
  pitch_ref, yaw_ref, gripper_width, grasped, success

Units and the sim->LeRobot conversion (degrees, gripper 0-100) are NOT applied here; the
converter owns that contract (configs/), so the raw log stays lossless.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from so101_showharness.recorder.video import _ffmpeg


class _RawVideo:
    def __init__(self, path: Path, fps: int, size=(640, 480), crf: int = 18) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            [_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
             "-g", str(fps), str(path)],
            stdin=subprocess.PIPE,
        )

    def write(self, img: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()


def _f(x, nd=6):
    return None if x is None else [round(float(v), nd) for v in np.asarray(x).reshape(-1)]


class TrajectoryRecorder:
    def __init__(self, episode_dir: str | Path, record_hz: int = 30, env_hz: int = 60) -> None:
        self.dir = Path(episode_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.every = max(1, round(env_hz / record_hz))
        self.record_hz = record_hz
        self._rows = (self.dir / "frames.jsonl").open("w", encoding="utf-8")
        self._ext = _RawVideo(self.dir / "video" / "external.mp4", record_hz)
        self._wrist = _RawVideo(self.dir / "video" / "wrist.mp4", record_hz)
        self.frame = 0
        self.closed = False

    def on_step(self, rec: dict) -> None:
        if self.closed or rec["env_step"] % self.every:
            return
        be = rec["backend"]
        ext, wrist = be.raw_frames()
        self._ext.write(ext)
        self._wrist.write(wrist)
        ann = rec.get("annotation") or {}
        row = {
            "frame": self.frame,
            "env_step": rec["env_step"],
            "t": round(rec["t"], 4),
            "ctrl_step": rec["ctrl_step"],
            "decision_id": rec["decision_id"],
            "token": ann.get("token"),
            "phase": ann.get("phase"),
            "blocked": ann.get("blocked"),
            "q_obs": _f(rec["q_obs"]),
            "q_cmd": _f(rec["q_cmd"]),
            "q_des": _f(rec["q_des"]),
            "tcp_obs": _f(rec["tcp_obs"]),
            "tcp_target": _f(rec["tcp_target"]),
            "pitch_ref": round(float(rec["pitch_ref"]), 6),
            "yaw_ref": None if rec["yaw_ref"] is None else round(float(rec["yaw_ref"]), 6),
            "gripper_width": round(be.gripper_width(), 5),
            "grasped": be.grasped(),
            "contacts": rec.get("contacts") or [],
            "success": be.success(),
        }
        self._rows.write(json.dumps(row) + "\n")
        self.frame += 1

    def close(self, metadata: Optional[dict] = None) -> None:
        if self.closed:
            return
        self.closed = True
        self._rows.close()
        self._ext.close()
        self._wrist.close()
        meta = {"record_hz": self.record_hz, "num_frames": self.frame,
                "joint_names": ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"],
                "joint_units": "rad", "tcp_frame": "task (world orientation, robot base origin)",
                "cameras": {"external": "external_D455 raw 640x480", "wrist": "ego raw 640x480"}}
        meta.update(metadata or {})
        (self.dir / "frames_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
