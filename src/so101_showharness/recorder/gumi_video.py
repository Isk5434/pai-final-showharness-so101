"""GUMI 風パネル付きの閉ループ動画（上: 外部カメラ | 手首カメラ、下: GUMI 操作パネル）。

収集中のライブ描画（EpisodeVideo）と、保存済み 30 Hz 軌道からの描き直し
（scripts/render/render_gumi_video.py）で同じ状態追跡・同じ描画を使う。
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from so101_showharness.recorder.gumi_panel import GumiPanel

HISTORY_TOKENS = {"MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN",
                  "RT_PITCH_FWD", "RT_PITCH_BACK", "RT_YAW_CCW", "RT_YAW_CW"}
PANEL_H = 286


class GumiState:
    """1 フレームごとの記録（frames.jsonl の行と同じ形）から、パネルに出す状態を追う。"""

    def __init__(self, task: str) -> None:
        self.task = task
        self.recent: list[str] = []
        self._decision = None
        self._token = None
        self._blocked = None

    def update(self, row: dict) -> dict:
        dec, tok, blk = row.get("decision_id"), row.get("token"), row.get("blocked")
        if dec != self._decision:
            # 前の判断が終わった: 実行された（拒否されていない）移動・回転は履歴へ
            if self._token in HISTORY_TOKENS and not self._blocked:
                self.recent.insert(0, self._token)
                del self.recent[5:]
            self._decision = dec
        self._token, self._blocked = tok, blk
        q_cmd = row.get("q_cmd") or [0] * 6
        tcp = row.get("tcp_obs") or [0] * 7
        contacts = row.get("contacts") or []
        contact = "contact: " + ("; ".join(f"{c['link']} {c['force_n']}N {c['with']}" for c in contacts) or "-")
        pitch = row.get("pitch_ref")
        telemetry = (f"t={row.get('t', 0):5.1f}s   tcp=[{tcp[0]:+.3f} {tcp[1]:+.3f} {tcp[2]:+.3f}] m   "
                     + (f"pitch ref {np.degrees(pitch):+.1f}°   " if pitch is not None else "")
                     + f"jaw width {1000 * (row.get('gripper_width') or 0):.1f} mm")
        return dict(task=self.task, token=tok if dec is not None or tok == "DONE" else None,
                    phase=row.get("phase"), blocked=blk, step=dec, recent=list(self.recent),
                    gripper_closed=q_cmd[5] < 0.2, grasped=bool(row.get("grasped")),
                    success=bool(row.get("success")), telemetry=telemetry, contact=contact)


def compose(ext: np.ndarray, wrist: np.ndarray, panel: Image.Image) -> np.ndarray:
    h, w = ext.shape[:2]
    canvas = Image.new("RGB", (2 * w, h + panel.height), (255, 255, 255))
    canvas.paste(Image.fromarray(ext[..., :3]), (0, 0))
    canvas.paste(Image.fromarray(wrist[..., :3]), (w, 0))
    canvas.paste(panel, (0, h))
    return np.asarray(canvas)


def make_panel(width: int) -> GumiPanel:
    return GumiPanel(width=width, height=PANEL_H)
