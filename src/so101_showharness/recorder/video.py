"""Closed-loop episode video: external | wrist side by side, with the live control state burned in.

Fed from the backend's per-env-step hook, so the video shows exactly what the controller
did between decisions (not just the decision frames). One mp4 per episode, real time at
``fps`` (env runs at 60 Hz; every ``60 / fps``-th step is written).

Encoding goes through an ffmpeg pipe: the libx264 build bundled with imageio-ffmpeg when it
is importable (present in the Isaac kit python), else the system ffmpeg. No frames are
buffered in memory.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

PANEL_H = 142


def _wrap_pi(a: float) -> float:
    """Closing axis is a line: show yaw in [-90, 90) deg."""
    return float((a + np.pi / 2) % np.pi - np.pi / 2)


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise RuntimeError("no ffmpeg found (imageio-ffmpeg or system)")
        return exe


class EpisodeVideo:
    """Live closed-loop video: cameras on top, the GUMI-style control panel underneath with the
    executing action's button lit (recorder/gumi_panel.py)."""

    def __init__(self, path: str | Path, fps: int = 30, env_hz: int = 60,
                 size: tuple[int, int] = (640, 480), crf: int = 20,
                 task: str = "Pick up the vial and place it in the rack") -> None:
        from so101_showharness.recorder.gumi_video import PANEL_H, GumiState, make_panel

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.every = max(1, round(env_hz / fps))
        self.w, self.h = size[0] * 2, size[1] + PANEL_H
        self.frames = 0
        self.state = GumiState(task)
        self.panel = make_panel(self.w)
        self._proc: Optional[subprocess.Popen] = subprocess.Popen(
            [_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{self.w}x{self.h}", "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
             str(self.path)],
            stdin=subprocess.PIPE,
        )

    def on_step(self, rec: dict) -> None:
        """Backend step hook. ``rec`` carries ``backend`` plus the control state."""
        if self._proc is None or rec["env_step"] % self.every:
            return
        from so101_showharness.recorder.gumi_video import compose

        be = rec["backend"]
        ext, wrist = be.raw_frames()
        ann = rec.get("annotation") or {}
        row = {"t": rec["t"], "decision_id": rec["decision_id"], "token": ann.get("token"),
               "phase": ann.get("phase"), "blocked": ann.get("blocked"),
               "q_cmd": list(rec["q_cmd"]), "tcp_obs": rec["tcp_obs"], "pitch_ref": rec["pitch_ref"],
               "gripper_width": be.gripper_width(), "grasped": be.grasped(), "success": be.success(),
               "contacts": rec.get("contacts") or []}
        frame = compose(ext, wrist, self.panel.render(**self.state.update(row)))
        self._proc.stdin.write(frame.tobytes())
        self.frames += 1

    def close(self) -> None:
        if self._proc is not None:
            self._proc.stdin.close()
            self._proc.wait()
            self._proc = None
