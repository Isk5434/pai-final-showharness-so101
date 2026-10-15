"""Show-Harness GUMI（gumi/web_teleop/static/index.html）の見た目を真似た操作パネルを描く。

閉ループ動画の下半分に置き、いま実行しているアクションのボタンを GUMI の「押下中」と同じ
琥珀色で点灯させる。配色・並びは GUMI の CSS をそのまま写した:
  MOVE（W/A/S/D の十字: FWD が上、LEFT/BACK/RIGHT が下）、LIFT（UP/DOWN）、ROTATE、Grasp。
ROTATE は SO-101 で使う 4 つ（PITCH 前/後・YAW 反時計/時計）。GUMI 本家は CCW/CW の 2 つ。
拒否されたトークンは赤（GUMI の --bad 系）、DONE は緑の帯（--ok 系）。
"""
from __future__ import annotations

from typing import Optional

from PIL import Image, ImageDraw, ImageFont

# GUMI :root の色
INK, INK2, DIM = "#16191d", "#464c54", "#8b929b"
LINE, SOFT, SOFT2 = "#e5e8ec", "#f7f9fb", "#eef1f5"
OK, OK_BG, OK_LINE = "#4a7a38", "#eef5ea", "#d6e6cd"
HOT, HOT_BG = "#c47c21", "#fdf4e8"
BAD, BAD_BG, BAD_LINE = "#a8413f", "#fdf5f5", "#f0cfce"
BTN, BTN_LINE, BTN_LINE_HOVER = "#eef1f6", "#8695a7", "#6d7d90"
PRESSED_BG, PRESSED_INK, PRESSED_KK = "#fadfb8", "#6f4310", "#96631d"
KEY_INK = "#232930"

KW, KH, GAP = 96, 46, 7  # キー幅・高さ・間隔（GUMI の 108x58 を動画幅に合わせて縮小）

# (トークン, ラベル, キーヒント)
MOVE_KEYS = {"MV_FWD": ("FWD", "W"), "MV_LEFT": ("LEFT", "A"), "MV_BACK": ("BACK", "S"),
             "MV_RIGHT": ("RIGHT", "D")}
LIFT_KEYS = {"MV_UP": ("UP", "R/↑"), "MV_DOWN": ("DOWN", "F/↓")}
ROT_KEYS = {"RT_PITCH_FWD": ("PITCH FWD", ""), "RT_PITCH_BACK": ("PITCH BACK", ""),
            "RT_YAW_CCW": ("↺ CCW", "Z"), "RT_YAW_CW": ("↻ CW", "X")}


def _font(size: int, bold: bool = False):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1: scalable default
    except TypeError:
        return ImageFont.load_default()


class GumiPanel:
    def __init__(self, width: int = 1280, height: int = 300) -> None:
        self.w, self.h = width, height
        self.f_key = _font(15, bold=True)
        self.f_kk = _font(10, bold=True)
        self.f_h3 = _font(11, bold=True)
        self.f_task = _font(15, bold=True)
        self.f_small = _font(11)
        self.f_grasp = _font(20, bold=True)
        self.f_banner = _font(14, bold=True)

    # ------------------------------------------------------------ primitives
    def _key(self, d, x, y, w, h, label, hint, state: Optional[str]):
        if state == "pressed":
            bg, line, ink, kk = PRESSED_BG, HOT, PRESSED_INK, PRESSED_KK
        elif state == "blocked":
            bg, line, ink, kk = BAD_BG, BAD, BAD, BAD
        else:
            bg, line, ink, kk = BTN, BTN_LINE, KEY_INK, BTN_LINE_HOVER
        d.rounded_rectangle([x, y, x + w, y + h], radius=9, fill=bg, outline=line, width=2 if state else 1)
        tw = d.textlength(label, font=self.f_key)
        d.text((x + (w - tw) / 2, y + h / 2 - 10), label, font=self.f_key, fill=ink)
        if hint:
            d.text((x + 6, y + h - 14), hint, font=self.f_kk, fill=kk)

    def _cluster(self, d, x, y, w, h, title, sub=""):
        d.rounded_rectangle([x, y, x + w, y + h], radius=11, fill="#ffffff", outline=LINE)
        d.text((x + 11, y + 8), title, font=self.f_h3, fill=INK2)
        if sub:
            d.text((x + w - 11 - d.textlength(sub, font=self.f_small), y + 8), sub, font=self.f_small, fill=DIM)

    # ------------------------------------------------------------ the panel
    def render(self, *, task: str, token: Optional[str], phase: Optional[str], blocked: Optional[str],
               step: Optional[int], recent: list, gripper_closed: bool, grasped: bool, success: bool,
               telemetry: str, contact: str) -> Image.Image:
        img = Image.new("RGB", (self.w, self.h), "#ffffff")
        d = ImageDraw.Draw(img)

        def st(tok):
            if tok != token:
                return None
            return "blocked" if blocked else "pressed"

        # top bar: TASK + step / state pills
        d.rounded_rectangle([8, 6, self.w - 8, 40], radius=11, fill=SOFT, outline=LINE)
        d.text((20, 15), "TASK", font=self.f_h3, fill=DIM)
        d.text((62, 13), task, font=self.f_task, fill=INK)
        pills = [(f"STEP {step if step is not None else '-'}", SOFT2, DIM, LINE)]
        pills.append(("● GRASPED" if grasped else "○ open", HOT_BG if grasped else SOFT2,
                      HOT if grasped else DIM, LINE))
        if success:
            pills.append(("✓ SUCCESS", OK_BG, OK, OK_LINE))
        px = self.w - 18
        for text, bg, ink, line in reversed(pills):
            tw = d.textlength(text, font=self.f_h3) + 22
            d.rounded_rectangle([px - tw, 11, px, 35], radius=12, fill=bg, outline=line)
            d.text((px - tw + 11, 16), text, font=self.f_h3, fill=ink)
            px -= tw + 8

        # deck
        y0 = 48
        x = 8
        # MOVE: 3 x 2 cross
        cw = 3 * KW + 2 * GAP + 22
        ch = 2 * KH + GAP + 34
        self._cluster(d, x, y0, cw, ch, "MOVE", "2 cm")
        kx, ky = x + 11, y0 + 28
        self._key(d, kx + KW + GAP, ky, KW, KH, *MOVE_KEYS["MV_FWD"], st("MV_FWD"))
        for i, t in enumerate(("MV_LEFT", "MV_BACK", "MV_RIGHT")):
            self._key(d, kx + i * (KW + GAP), ky + KH + GAP, KW, KH, *MOVE_KEYS[t], st(t))
        x += cw + 8
        # LIFT
        lw = KW + 22
        self._cluster(d, x, y0, lw, ch, "LIFT")
        for i, t in enumerate(("MV_UP", "MV_DOWN")):
            self._key(d, x + 11, y0 + 28 + i * (KH + GAP), KW, KH, *LIFT_KEYS[t], st(t))
        x += lw + 8
        # ROTATE (SO-101: pitch + yaw)
        rw = 2 * (KW + 18) + GAP + 22
        self._cluster(d, x, y0, rw, ch, "ROTATE", "10°")
        for i, t in enumerate(("RT_PITCH_FWD", "RT_PITCH_BACK")):
            self._key(d, x + 11, y0 + 28 + i * (KH + GAP), KW + 18, KH, *ROT_KEYS[t], st(t))
        for i, t in enumerate(("RT_YAW_CCW", "RT_YAW_CW")):
            self._key(d, x + 11 + KW + 18 + GAP, y0 + 28 + i * (KH + GAP), KW + 18, KH, *ROT_KEYS[t], st(t))
        x += rw + 8
        # centre: Grasp / Release (GUMI: #btn-grasp, amber while closed)
        gx1 = self.w - 8
        g_active = token in ("GRASP", "RELEASE")
        closed_look = gripper_closed or g_active
        bg = PRESSED_BG if closed_look else BTN
        line = BAD if (g_active and blocked) else (HOT if closed_look else BTN_LINE)
        d.rounded_rectangle([x, y0, gx1, y0 + ch], radius=12, fill=bg, outline=line, width=3 if g_active else 2)
        label = token.title() if g_active else ("Release" if gripper_closed else "Grasp")
        tw = d.textlength(label, font=self.f_grasp)
        d.text((x + (gx1 - x - tw) / 2, y0 + ch / 2 - 22), label, font=self.f_grasp,
               fill=PRESSED_INK if closed_look else KEY_INK)
        sub = "closed" if gripper_closed else "Space"
        tw = d.textlength(sub, font=self.f_small)
        d.text((x + (gx1 - x - tw) / 2, y0 + ch / 2 + 8), sub, font=self.f_small, fill=INK2)

        # status strip / banners
        y1 = y0 + ch + 8
        if token == "DONE":
            d.rounded_rectangle([8, y1, self.w - 8, y1 + 30], radius=11, fill=OK_BG, outline=OK_LINE)
            d.text((22, y1 + 7), "✓ TASK DONE", font=self.f_banner, fill=OK)
        elif blocked:
            d.rounded_rectangle([8, y1, self.w - 8, y1 + 30], radius=11, fill=BAD_BG, outline=BAD_LINE)
            d.text((22, y1 + 7), f"REFUSED {token}: {blocked}", font=self.f_banner, fill=BAD)
        else:
            d.rounded_rectangle([8, y1, self.w - 8, y1 + 30], radius=11, fill=SOFT, outline=LINE)
            now = token or "-"
            if phase:
                now += f"  ({phase})"
            d.text((22, y1 + 7), f"NOW  {now}", font=self.f_banner, fill=INK)
            rs = "Recent moves: " + (", ".join(recent[:5]) if recent else "none")
            d.text((330, y1 + 9), rs, font=self.f_small, fill=INK2)
        d.text((14, y1 + 38), telemetry, font=self.f_small, fill=DIM)
        d.text((14, y1 + 54), contact, font=self.f_small, fill=DIM)
        return img
