"""カメラ合わせ用の印刷物（A4、PDF）を作る（AWS 側）。印刷は必ず「実際のサイズ（100%）」で。

    python3 scripts/real/make_targets.py   → outputs/real_targets/{aruco_sheet.pdf, checkerboard.pdf}

- aruco_sheet.pdf：A4 横。マーカー 4 つ（一辺 50 mm）。紙の中心の十字を、ロボットの基部（肩の回転軸の真下）から
  前へ 30.0 cm・右へ 8.0 cm に置き、"FAR" の辺をロボットから遠い側、紙の辺をロボットの正面方向とそろえる。
- checkerboard.pdf：A4 縦。内側の角 9x6、一辺 25 mm。レンズの内部パラメータを測るときに使う（板に貼って平らに）。
印刷後、定規でマーカーの一辺が 50 mm、マスが 25 mm になっているか確かめる。
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from so101_showharness.real.camera_fix import ARUCO_DICT, MARKER_M, MARKER_OFFSETS, SHEET_CENTER  # noqa: E402

DPI = 300
MM = DPI / 25.4
OUT = REPO / "outputs/real_targets"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def font(pt):
    return ImageFont.truetype(FONT, int(pt * DPI / 72), index=0)


def aruco_sheet():
    w, h = int(297 * MM), int(210 * MM)
    im = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(im)
    cx, cy = w / 2, h / 2
    d.line([(cx - 15 * MM, cy), (cx + 15 * MM, cy)], fill=0, width=3)
    d.line([(cx, cy - 15 * MM), (cx, cy + 15 * MM)], fill=0, width=3)
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT))
    side = int(round(MARKER_M * 1000 * MM))
    for i, (dx, dy) in MARKER_OFFSETS.items():
        # 紙の上 = +X（遠い側）、紙の右 = −Y（ロボットから見て右）
        u, v = cx + (-dy * 1000) * MM, cy - (dx * 1000) * MM
        m = cv2.aruco.generateImageMarker(dic, i, side)
        im.paste(Image.fromarray(m), (int(round(u - side / 2)), int(round(v - side / 2))))
        d.text((u - side / 2, v + side / 2 + 2 * MM), f"id {i}", font=font(9), fill=0)
    d.text((cx - 25 * MM, 4 * MM), "FAR（ロボットから遠い側）", font=font(12), fill=0)
    d.text((cx - 30 * MM, h - 14 * MM), "NEAR（ロボット側）", font=font(12), fill=0)
    d.text((8 * MM, cy - 30 * MM),
           f"十字 = 基部から前へ {SHEET_CENTER[0] * 100:.1f} cm、\n右へ {-SHEET_CENTER[1] * 100:.1f} cm\nマーカー一辺 {MARKER_M * 1000:.0f} mm",
           font=font(10), fill=0)
    d.line([(10 * MM, h - 6 * MM), (60 * MM, h - 6 * MM)], fill=0, width=4)
    d.text((10 * MM, h - 13 * MM), "50 mm", font=font(8), fill=0)
    return im.convert("RGB")


def checkerboard(cols=10, rows=7, sq_mm=25):
    w, h = int(210 * MM), int(297 * MM)
    im = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(im)
    s = sq_mm * MM
    x0, y0 = (w - cols * s) / 2, (h - rows * s) / 2
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                d.rectangle([x0 + c * s, y0 + r * s, x0 + (c + 1) * s, y0 + (r + 1) * s], fill=0)
    d.text((x0, y0 - 12 * MM), f"内側の角 {cols - 1}x{rows - 1}、一辺 {sq_mm} mm（印刷後に定規で確認）", font=font(10), fill=0)
    im.convert("RGB").save(OUT / "checkerboard.pdf", resolution=DPI)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    aruco_sheet().save(OUT / "aruco_sheet.pdf", resolution=DPI)
    checkerboard()
    print(OUT / "aruco_sheet.pdf", OUT / "checkerboard.pdf")
