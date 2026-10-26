"""Notion のまとめの流れに数値スライドを挟んだ 3 分以内のまとめ動画（無音、1920x1080、30fps）を作る。

素材: docs/notion_export/（Notion のエクスポート）。スライドは PIL で描き、ffmpeg で倍速クリップと繋ぐ。
倍速にしたクリップには右下に「×N 再生」を入れる（倍率は自由。長いクリップは倍率を上げて 1 話を通して見せる）。
画面の下には目次の帯を常に出し、今の章を強調する。先頭と末尾に講座の OP / ED を繋ぎ、全体で 3 分以内に収める。

    python3 scripts/render/make_summary_video.py [--out outputs/video/summary.mp4]
"""
import argparse
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "docs/notion_export"
SHOW = Path("/home/ubuntu/Show-Harness-so101/assets")  # showlab/Show-Harness（Apache-2.0）の公式素材
COURSE = Path("/home/ubuntu/sim2real/Sim-to-Real-SO-101-Workshop-v1.0")  # 講座から配布された OP / ED
OP, ED = COURSE / "PAI応用1_最終課題_OP.mp4", COURSE / "PAI応用1_最終課題_ED.mp4"
REAL = REPO / "docs/real_video"  # 実機 SO-101 をスマホで撮った映像（縦撮り）
SIM_FRAME = REPO / "outputs/eval_dm05_near_mix57_te5/full/video/near_7002.mp4"  # Workshop タスクの sim 画面の切り出し元
W, H, FPS = 1920, 1080, 30
BG, PANEL, FG, MUTED = "#FFFFFF", "#F3F4F6", "#111827", "#6B7280"
BLUE, GREEN, AMBER, RED = "#0369A1", "#047857", "#B45309", "#B91C1C"
ON = "#FFFFFF"  # 色付きの札の上の文字
BAR = 120          # クリップ画面の上の帯
TOC_H = 56         # 画面の下の目次の帯
BOX = (30, BAR + 20, W - 60, H - BAR - 40 - TOC_H)  # クリップを収める枠 x, y, w, h
SIDE_BOX = (60, BOX[1], 880, BOX[3])  # 縦撮りのクリップは左に寄せ、右に要点を書く
TOP_BOX = (30, BAR + 10, W - 60, 480)  # 横長のクリップは上に置き、下に説明のカードを並べる
CHAPTERS = ["背景と動機", "0.1 ManiSkill", "0.2 SO-101", "0.3 Sim to Real", "番外編", "まとめ"]


def font(size, bold=False):
    f = "NotoSansCJK-Bold.ttc" if bold else "NotoSansCJK-Regular.ttc"
    return ImageFont.truetype(f"/usr/share/fonts/opentype/noto/{f}", size, index=0)


def canvas(alpha=False):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0)) if alpha else Image.new("RGB", (W, H), BG)
    return im, ImageDraw.Draw(im)


def pill(d, xy, text, size, fg, bg, bold=True, pad=(22, 10)):
    f = font(size, bold)
    l, t, r, b = d.textbbox((0, 0), text, font=f)
    x, y = xy
    d.rounded_rectangle((x, y, x + r - l + 2 * pad[0], y + b - t + 2 * pad[1]), radius=(b - t) // 2 + pad[1], fill=bg)
    d.text((x + pad[0] - l, y + pad[1] - t), text, font=f, fill=fg)
    return x + r - l + 2 * pad[0]


def header(d, tag, title, sub=None):
    x = pill(d, (80, 70), tag, 34, ON, BLUE)
    d.text((x + 30, 62), title, font=font(54, True), fill=FG)
    if sub:
        d.text((82, 160), sub, font=font(32), fill=MUTED)


def table(d, x, y, cols, rows, widths, row_h=78, size=36, hi=None):
    """cols: 見出し, rows: 行（各セルは str か (str, color)）, hi: 強調する列。"""
    d.rounded_rectangle((x, y, x + sum(widths), y + row_h * (len(rows) + 1)), radius=16, fill=PANEL)
    cx = x
    for j, c in enumerate(cols):
        d.text((cx + 24, y + row_h / 2), c, font=font(size - 6, True), fill=MUTED, anchor="lm")
        cx += widths[j]
    for i, r in enumerate(rows):
        yy = y + row_h * (i + 1)
        d.line((x + 16, yy, x + sum(widths) - 16, yy), fill="#E5E7EB", width=2)
        cx = x
        for j, c in enumerate(r):
            txt, col = c if isinstance(c, tuple) else (c, FG)
            d.text((cx + 24, yy + row_h / 2), txt, font=font(size, j == 0 or j == hi), fill=col, anchor="lm")
            cx += widths[j]


def bullets(d, x, y, items, size=36, gap=70, width=None):
    for it in items:
        txt, col = it if isinstance(it, tuple) else (it, FG)
        d.text((x, y), "•", font=font(size, True), fill=BLUE)
        d.text((x + 40, y), txt, font=font(size), fill=col)
        y += gap
    return y


def toc(d, chapter):
    """画面の下に目次の帯を描く。今の章は青の札、済んだ章は濃く、これからの章は薄く。"""
    y0 = H - TOC_H
    d.rectangle((0, y0, W, H), fill=PANEL)
    d.line((0, y0, W, y0), fill="#E5E7EB", width=2)
    cur = CHAPTERS.index(chapter) if chapter in CHAPTERS else -1
    f, cw = font(26, True), W / len(CHAPTERS)
    for i, c in enumerate(CHAPTERS):
        cx, cy = cw * (i + 0.5), y0 + TOC_H / 2
        if i == cur:
            l, t, r, b = d.textbbox((0, 0), c, font=f)
            d.rounded_rectangle((cx - (r - l) / 2 - 22, cy - 20, cx + (r - l) / 2 + 22, cy + 20), radius=20, fill=BLUE)
        d.text((cx, cy), c, font=f, fill=ON if i == cur else FG if i < cur else MUTED, anchor="mm")
        if i:
            d.text((cw * i, cy), "›", font=f, fill="#D1D5DB", anchor="mm")


def paste_fit(im, path, box):
    p = (path if isinstance(path, Image.Image) else Image.open(path)).convert("RGB")
    x, y, w, h = box
    s = min(w / p.width, h / p.height)
    p = p.resize((int(p.width * s), int(p.height * s)), Image.LANCZOS)
    px, py = x + (w - p.width) // 2, y + (h - p.height) // 2
    im.paste(p, (px, py))
    ImageDraw.Draw(im).rectangle((px - 2, py - 2, px + p.width + 1, py + p.height + 1), outline="#D1D5DB", width=2)
    return px


# ------------------------------------------------------------------ slides
def ease(u):
    return 1 - (1 - min(max(u, 0), 1)) ** 3


class Motion:
    """濃紺の背景で、要素ごとの絵（RGBA）を時刻に合わせて浮かび上がらせる動くスライドの共通部分。"""
    NAVY0, NAVY1 = (8, 15, 30), (14, 42, 72)
    CYAN, MINT, GOLD, SOFT = "#38BDF8", "#34D399", "#FBBF24", "#CBD5E1"

    def __init__(self):
        g = Image.linear_gradient("L").resize((W, H))
        self.bg = Image.composite(Image.new("RGB", (W, H), self.NAVY1), Image.new("RGB", (W, H), self.NAVY0), g)
        d = ImageDraw.Draw(self.bg)
        for x in range(0, W, 48):  # うっすらした点の格子
            for y in range(0, H, 48):
                d.point((x, y), fill=(40, 62, 92))
        self.glows = [self.glow(1000, (56, 189, 248), 70), self.glow(800, (52, 211, 153), 50)]
        self.items = []  # (絵, x, y, 出始め, 出る長さ, 下からずらす px, 横から寄せる px)
        self.lines = []  # (中心 x, y, 幅, 出始め, 出る長さ, 色)

    @staticmethod
    def glow(size, rgb, alpha):
        g = Image.new("RGBA", (size, size), rgb + (0,))
        ImageDraw.Draw(g).ellipse((size // 4, size // 4, size * 3 // 4, size * 3 // 4), fill=rgb + (alpha,))
        return g.filter(ImageFilter.GaussianBlur(size // 8))

    @staticmethod
    def text(txt, size, col, bold=False, spacing=0):
        f = font(size, bold)
        w = sum(f.getlength(c) + spacing for c in txt) if spacing else f.getlength(txt)
        im = Image.new("RGBA", (int(w) + 8, int(size * 1.5)), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        if spacing:
            x = 0
            for c in txt:
                d.text((x, 0), c, font=f, fill=col)
                x += f.getlength(c) + spacing
        else:
            d.text((0, 0), txt, font=f, fill=col)
        return im

    @staticmethod
    def pill_img(txt, size, fg, bg, outline=None):
        f = font(size, True)
        l, t, r, b = f.getbbox(txt)
        w, h = r - l + 52, b - t + 26
        im = Image.new("RGBA", (w + 4, h + 4), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((2, 2, w, h), radius=h // 2, fill=bg, outline=outline, width=3 if outline else 0)
        d.text((28 - l, 15 - t), txt, font=f, fill=fg)
        return im

    def add(self, im, x, y, t0, dt, rise=0, dx=0):
        self.items.append((im, int(x), int(y), t0, dt, rise, dx))

    def centered(self, parts, y, t0, step, rise=16):
        """[(文字, 大きさ, 色, 太字)] を中央そろえで 1 行に並べ、左から順に出す。"""
        ims = [self.text(t, sz, c, b) for t, sz, c, b in parts]
        x = W / 2 - sum(font(sz, b).getlength(t) for t, sz, c, b in parts) / 2
        for k, ((t, sz, c, b), im) in enumerate(zip(parts, ims)):
            self.add(im, x, y, t0 + step * k, 0.45, rise)
            x += font(sz, b).getlength(t)

    def tracked(self, parts, y, t0, step=0.03, dt=0.8, spread=0.35):
        """大きな文字を 1 文字ずつ、広い字間から締めながら中央に出す。"""
        chars = [(ch, sz, c, b) for t, sz, c, b in parts for ch in t]
        x = W / 2 - sum(font(sz, b).getlength(ch) for ch, sz, c, b in chars) / 2
        for k, (ch, sz, c, b) in enumerate(chars):
            w = font(sz, b).getlength(ch)
            self.add(self.text(ch, sz, c, b), x, y, t0 + step * k, dt, 0, (x + w / 2 - W / 2) * spread)
            x += w

    def frame(self, t):
        im = self.bg.copy().convert("RGBA")
        im.alpha_composite(self.glows[0], (1150 - int(40 * t), -250 + int(15 * t)))  # ゆっくり流れる光
        im.alpha_composite(self.glows[1], (1350 + int(30 * t), 450 - int(20 * t)))
        d = ImageDraw.Draw(im)
        for cx, y, w, t0, dt, col in self.lines:  # 中央から左右に伸びる線
            u = ease((t - t0) / dt)
            if u > 0:
                d.rounded_rectangle((cx - w * u / 2, y, cx + w * u / 2, y + 6), radius=3, fill=col)
        for sp, x, y, t0, dt, rise, dx in self.items:
            u = ease((t - t0) / dt)
            if u <= 0:
                continue
            a = sp.copy()
            a.putalpha(a.getchannel("A").point(lambda v: int(v * u)))
            im.alpha_composite(a, (int(x + dx * (1 - u)), y + int(rise * (1 - u))))
        return im.convert("RGB")


class Title(Motion):
    def __init__(self):
        super().__init__()
        self.tracked([("Teleopless ", 150, "#FFFFFF", False), ("Sim2Real", 150, self.CYAN, True)], 330, 0.15)
        self.lines.append((W / 2, 585, 760, 0.9, 0.7, self.CYAN))
        self.centered([("～  ", 44, self.SOFT, False), ("Frontier VLM", 44, "#FFFFFF", True), ("  と  ", 44, self.SOFT, False),
                       ("VLA", 44, self.MINT, True), ("  と  ", 44, self.SOFT, False), ("Show-Harness", 44, self.GOLD, True),
                       ("  ～", 44, self.SOFT, False)], 615, 1.2, 0.08)
        a, b = self.pill_img("SO-101", 28, self.GOLD, None, self.GOLD), self.pill_img("OSS：Show-Harness 本家にマージ（PR #6）", 28, self.MINT, None, self.MINT)
        x = W / 2 - (a.width + 24 + b.width) / 2
        self.add(a, x, 840, 2.0, 0.45, 12)
        self.add(b, x + a.width + 24, 840, 2.15, 0.45, 12)
        self.centered([("時間の都合で、映像は倍速・一部抜粋のものがあります（右下に表示）", 24, "#94A3B8", False)], 940, 2.4, 0, 0)


class Extra(Motion):
    """番外編の区切り。"""
    def __init__(self):
        super().__init__()
        self.centered([("E X T R A", 30, self.CYAN, True)], 330, 0.0, 0, 0)
        self.tracked([("番外編", 140, "#FFFFFF", True)], 380, 0.1, 0.06, 0.6, 0.6)
        self.lines.append((W / 2, 610, 560, 0.5, 0.6, self.CYAN))
        self.centered([("世界モデル Cosmos Transfer で、見た目の Sim to Real のずれを埋める", 40, self.SOFT, False)], 640, 0.7, 0)


class Build:
    """白いスライドの一部（枠の中身）を順に出す。imgs[0] は中身なしの土台、imgs[k] は段階 k まで描いた絵。
    pieces = [(段階, 枠, 出始め, 出る長さ, "fade" か "wipe")]。wipe は左から伸ばす。"""
    def __init__(self, fn, chapter, stages, pieces):
        self.imgs = [fn(k) for k in range(stages + 1)]
        for im in self.imgs:
            toc(ImageDraw.Draw(im), chapter)
        self.pieces = pieces

    def frame(self, t):
        im = self.imgs[0].copy()
        for k, box, t0, dt, kind in self.pieces:
            u = ease((t - t0) / dt)
            if u <= 0:
                continue
            x0, y0, x1, y1 = box
            src = self.imgs[k]
            if kind == "wipe":
                xw = int(x0 + (x1 - x0) * u)
                if xw > x0:
                    im.paste(src.crop((x0, y0, xw, y1)), (x0, y0))
            else:
                rise = int(14 * (1 - u))
                region = im.crop((x0, y0 + rise, x1, y1 + rise))
                im.paste(Image.blend(region, src.crop((x0, y0, x1, y1)), u), (x0, y0 + rise))
        return im


def a_hyp():
    rows = [(1, (850, 630 + 76 * r, 1840, 684 + 76 * r), 1.1 + 0.3 * r, 0.6, "wipe") for r in range(4)]
    return Build(s_hyp, "0.2 SO-101", 1, [
        (1, (110, 525, 420, 600), 0.3, 0.4, "fade"),
        *[(1, (110, 615 + 100 * k, 800, 712 + 100 * k), 0.6 + 0.4 * k, 0.45, "fade") for k in range(3)],
        (1, (1040, 550, 1840, 612), 0.8, 0.4, "fade"),
        *rows,
        (1, (100, 950, W - 100, 995), 2.6, 0.5, "fade")])


def a_dm05():
    return Build(s_dm05, "0.2 SO-101", 2, [
        (1, (130, 735, 900, 812), 0.4, 0.4, "fade"),
        (1, (130, 822, 1300, 882), 0.9, 0.45, "fade"),
        *[(1, (1320, 756 + 52 * k, 1810, 798 + 52 * k), 1.4 + 0.35 * k, 0.5, "wipe") for k in range(3)],
        (1, (130, 884, 1300, 946), 2.6, 0.45, "fade"),
        (2, (1380, 745, 1815, 960), 3.0, 0.5, "fade"),
        (1, (130, 950, 1300, 996), 3.4, 0.45, "fade")])


def s_bg():
    im, d = canvas()
    header(d, "背景と動機", "人が教えずに、VLM の操作ログだけでロボットを育てたい")
    for i, (tag, c, items) in enumerate([
            ("背景", MUTED, [
                "ロボットの学習データは、人がテレオペで\n集めるのが一般的で、時間と手間がかかる",
                "Frontier VLM はロボットを操作できるが、\nモデルが大きく、毎回の推論が重い",
                ("Show-Harness：VLM の操作をそのまま\nデータとして残し、小さなモデルに学ばせる", BLUE)]),
            ("確かめたいこと", BLUE, [
                "テレオペなしで、Frontier VLM の操作ログ\nだけで小さなモデルを育てられるか",
                "VLM の離散的な動作（1 語ずつ）と VLA の\n連続的な関節角、細かい作業に向くのは？",
                "成功率だけでなく、行動の癖（往復・取り違え・\n停止）まで測って、何が効くかを切り分ける"])]):
        x = 100 + i * 880
        d.rounded_rectangle((x, 220, x + 840, 960), radius=24, fill=PANEL)
        pill(d, (x + 40, 250), tag, 34, ON, c)
        y = 370
        for it in items:
            txt, col = it if isinstance(it, tuple) else (it, FG)
            d.text((x + 40, y), "•", font=font(34, True), fill=BLUE)
            d.multiline_text((x + 80, y), txt, font=font(34), fill=col, spacing=14)
            y += 190
    return im


def s_show():
    im, d = canvas()
    header(d, "Show-Harness", "VLM がロボットを「操作」するための harness",
           "showlab（シンガポール国立大学）、2026。arXiv 2609.10522、github.com/showlab/Show-Harness")
    paste_fit(im, SHOW / "overview.png", (60, 230, 1000, 700))
    bullets(d, 1110, 260, [
        "VLM は意味のある動作単位を 1 語ずつ選ぶ",
        ("MV_FWD・MV_UP・GRASP・RELEASE …", BLUE),
        "interpreter が各単位を実際の関節の動きに変換",
        "2 つの使い方：",
        ("① Frontier VLM がゼロショットで操作", GREEN),
        ("② 小型 VLM を FT して 1 手 1 トークンで動かす", GREEN),
        "GUMI：ブラウザでロボットを操作してデモを収集",
        "Franka・Piper・ManiSkill・LIBERO などで共通",
    ], size=32, gap=78)
    d.text((70, 960), "図：Show-Harness 公式リポジトリ（Apache-2.0）より", font=font(26), fill=MUTED)
    return im


def s_01():
    im, d = canvas()
    header(d, "0.1", "Show-Harness：VLM で集めて、小型 VLM に学ばせる",
           "Show-Harness の考え方：Frontier VLM がロボットを操作してデータを集める → 小さなモデルが学ぶ")
    steps = [("GPT-6 Astra（VLM）", "8 語の動作で操作\nデータ収集", BLUE),
             ("Qwen3.5-4B", "Astra の成功 100 話で\nSFT（LoRA）", GREEN),
             ("強化学習", "REINFORCE / PPO\nで判断を磨く", AMBER)]
    x = 140
    for i, (t, s, c) in enumerate(steps):
        d.rounded_rectangle((x, 380, x + 460, 700), radius=24, fill=PANEL, outline=c, width=4)
        d.text((x + 230, 450), t, font=font(40, True), fill=c, anchor="mm")
        d.multiline_text((x + 230, 590), s, font=font(36), fill=FG, anchor="mm", align="center", spacing=14)
        if i < 2:
            d.text((x + 520, 540), "→", font=font(80, True), fill=MUTED, anchor="mm")
        x += 580
    d.text((140, 800), "動作は「前・後・左・右・上・下・つかむ・離す」の 1 語ずつ（離散動作）", font=font(34), fill=MUTED)
    return im


def s_rl():
    # 数値は ZeroTeleop/zero-teleop-rl の検証 100 配置（seed 31–99, 200–230、同じ seed）を rl_report.py で集計したもの
    im, d = canvas()
    header(d, "0.1", "強化学習：成功率 59% → 81%、往復が減ったのは PPO だけ",
           "Qwen3.5-4B をシミュレータで練習させた（検証 100 配置）。往復 = 逆向きの 2 手が続く（左 → 右 など）")
    table(d, 80, 215, ["", "成功", "往復 / 100 手", "成功した話の中の往復", "堂々巡りで失敗"], [
        ["お手本だけ（SFT）", "59 / 100", "1.2", "0.2", "31"],
        ["REINFORCE", ("78 / 100", GREEN), ("6.0（5 倍）", RED), ("3.1", RED), "6"],
        ["GRPO", "71 / 100", ("4.9", RED), ("3.9", RED), "11"],
        ["PPO", ("81 / 100", GREEN), ("0.5", GREEN), ("0.2", GREEN), "5"],
    ], [420, 280, 340, 400, 320], row_h=64, size=30)
    for i, (tag, c, items, foot) in enumerate([
            ("REINFORCE：なぜ往復が 5 倍に？", RED, [
                "報酬は最後に成功したか（1 / 0）だけ",
                "成功した話では、どの手も同じだけ褒められる",
                "途中の「左 → 右」の無駄な往復まで強化される"],
             "往復 351 回のうち 335 回が左右\n   1 話に同じ点を付ける GRPO も 4.9"),
            ("PPO：なぜ往復が増えない？", GREEN, [
                "critic が「今の状態から成功できそうか」V(s) を見積もる",
                "1 手の点 = その手で V がどれだけ上がったか（GAE）",
                "左 → 右で元に戻ると V は上がらない → 褒められない"],
             "成功に近づいた手だけが強化され、\n   往復は 1.2 → 0.5 に減った")]):
        x = 80 + i * 900
        d.rounded_rectangle((x, 560, x + 860, 1000), radius=24, fill=PANEL)
        pill(d, (x + 30, 582), tag, 30, ON, c)
        y = bullets(d, x + 34, 670, items, size=29, gap=66)
        d.multiline_text((x + 34, y + 10), "→ " + foot, font=font(26, True), fill=c, spacing=10)
    return im


def sim_frame():
    """評価動画の 1 秒目から、上段左の外部カメラ（640x478）だけを切り出す。"""
    png = REPO / "outputs/video/sim_frame.png"
    run(["ffmpeg", "-y", "-ss", "1", "-i", str(SIM_FRAME), "-frames:v", "1", "-vf", "crop=640:478:0:0", str(png)])
    return Image.open(png)


def real_frame():
    """実機の撮影（DM0.5、縦撮り）の 29 秒目から、アーム・vial・rack が入る範囲を切り出す。"""
    png = REPO / "outputs/video/real_frame.png"
    run(["ffmpeg", "-y", "-ss", "29", "-i", str(REAL / "1024.mp4"), "-frames:v", "1", "-vf", "crop=1080:900:0:560", str(png)])
    return Image.open(png)


def s_ws():
    im, d = canvas()
    header(d, "SO-101 Workshop", "SO-101 で vial をつかみ、rack の穴に差し込む",
           "NVIDIA Isaac「Sim-to-Real SO-101 Workshop」のタスク（Lerobot-So101-Teleop-Vials-To-Rack）")
    for i, (img, lab, c) in enumerate([
            (sim_frame(), "シミュレータ（Isaac Lab）", BLUE),
            (real_frame(), "実機 SO-101", GREEN)]):
        px = paste_fit(im, img, (100 + i * 960, 300, 760, 520))
        pill(d, (px, 240), lab, 32, ON, c)
    d.text((W / 2, 540), "→", font=font(90, True), fill=BLUE, anchor="mm")
    d.text((W / 2, 610), "Sim to Real", font=font(28, True), fill=BLUE, anchor="mm")
    bullets(d, 110, 860, [
        "机の上の vial（手前・真ん中・一番奥）から指定の 1 本をつかみ、rack の穴に立てて差し込む",
        ("Workshop の流れ：シミュレータで集めたデータで学習し、そのまま実機で動かす（Sim to Real）", BLUE),
    ], size=32, gap=60)
    return im


def s_road():
    im, d = canvas()
    header(d, "SO-101 Workshop", "人のテレオペを VLM に置き換え、Sim to Real まで",
           "上：Workshop 本来の流れ　下：この課題の流れ（0.2 でシミュレータ、0.3 で実機）")
    xs, bw = [110, 560, 1010, 1460], 380
    d.text((110, 270), "Workshop", font=font(30, True), fill=MUTED)
    for x, t in zip(xs, ["SO-101 のシミュレータ", "人がテレオペで\nデータを集める", "VLA を学習", "実機で動かす"]):
        d.rounded_rectangle((x, 320, x + bw, 440), radius=18, fill=PANEL)
        d.multiline_text((x + bw / 2, 380), t, font=font(32), fill=MUTED, anchor="mm", align="center", spacing=8)
    d.text((110, 470), "この課題", font=font(30, True), fill=BLUE)
    steps = [("Show-Harness を\nSO-101 に対応", "本家にマージ（OSS）", GREEN),
             ("GPT-6 Astra が操作して\nデータを集める", "人のテレオペの代わり", BLUE),
             ("VLA（DM0.5）を学習\nシミュレータで評価", "成功 50 話から学ぶ", BLUE),
             ("実機 SO-101 で推論", "組み立てから実施", AMBER)]
    for i, (x, (t, sub, c)) in enumerate(zip(xs, steps)):
        d.rounded_rectangle((x, 570, x + bw, 790), radius=22, fill=BG, outline=c, width=5)
        pill(d, (x + 24, 548), f"{i + 1}", 28, ON, c, pad=(16, 6))
        d.multiline_text((x + bw / 2, 670), t, font=font(32, True), fill=FG, anchor="mm", align="center", spacing=10)
        d.text((x + bw / 2, 760), sub, font=font(26), fill=MUTED, anchor="mm")
        if i < 3:
            d.text((x + bw + 35, 680), "→", font=font(56, True), fill=MUTED, anchor="mm")
    d.line((xs[1] + bw / 2, 440, xs[1] + bw / 2, 570), fill=BLUE, width=4)
    d.text((xs[1] + bw / 2 + 16, 505), "ここを置き換え", font=font(28, True), fill=BLUE, anchor="lm")
    for x0, x1, lab, c in [(xs[0], xs[2] + bw, "0.2 シミュレータ", BLUE), (xs[3], xs[3] + bw, "0.3 Sim to Real", AMBER)]:
        d.line((x0, 830, x0, 850, x1, 850, x1, 830), fill=c, width=4)
        d.text(((x0 + x1) / 2, 895), lab, font=font(34, True), fill=c, anchor="mm")
    return im


def s_02():
    im, d = canvas()
    header(d, "0.2 OSS 貢献", "Show-Harness を SO-101 に対応し、本家にマージ",
           "showlab/Show-Harness への Pull Request #6「SO-101 support（sim）」")
    paste_fit(im, SRC / "image 3.png", (50, 240, 1100, 780))
    x = 1200
    pill(d, (x, 300), "✓ Merged  2026-10-10", 44, ON, GREEN)
    bullets(d, x, 430, [
        "本家 showlab（★533）に取り込み",
        "5 自由度アームの運動学・実行部",
        "Isaac Lab Workshop 環境に接続",
        "設定・テスト・文書まで実装",
        ("issue でメンテナと範囲を事前合意", BLUE),
    ], size=34, gap=96)
    return im


def s_hyp(stage=1):
    im, d = canvas()
    header(d, "0.2", "GPT-6 Astra では vial がなかなか穴に入らない → VLA に学ばせる")
    bullets(d, 110, 220, [
        "仮説：VLM の動作は離散的（1 語ずつ）なので、VLA に比べて細かい作業が苦手",
        "→ 連続的な関節角を出す VLA に、Astra の成功データを学ばせる",
        "データ：Astra は 134 試行で 46 成功（34%）＋別の試行の 4 本 → 成功 50 話",
        "評価：手前 / 真ん中 / 一番奥 × 各 30 話。成功 = 狙った 1 本が立って穴に 1 cm 以上刺さる",
    ], size=32, gap=66)
    # なぜ DM0.5 か：RoboDojo-Sim の総合スコア（Dexmal OpenDM README の表、2026-08-25 時点）
    d.rounded_rectangle((80, 510, W - 80, 1000), radius=24, fill=PANEL)
    if not stage:
        return im
    pill(d, (120, 535), "なぜ DM0.5？", 32, ON, BLUE)
    bullets(d, 120, 625, [
        "RoboDojo：汎用のロボット操作方策を\n同じ条件で比べるベンチマーク",
        "sim 42 タスク・実機 18 タスク、\n30 方策を評価（arXiv 2607.04434）",
        ("DM0.5（Dexmal）は RoboDojo-Sim で 1 位\nLeRobot に公式対応 → lerobot/dm05_base", BLUE),
    ], size=28, gap=100)
    rows = [("DM0.5", 24.90, 19.34, GREEN), ("π0.5", 11.41, 6.91, MUTED), ("π0", 3.48, 1.53, MUTED), ("GR00T N1.7", 2.85, 1.31, MUTED)]
    x0, bw, y = 1050, 500, 560
    d.text((x0, y), "RoboDojo-Sim 総合スコア（2026-08-25 時点）", font=font(26, True), fill=FG)
    for name, score, sr, c in rows:
        y += 76
        d.text((x0 - 20, y + 20), name, font=font(28, True), fill=FG if c == GREEN else MUTED, anchor="rm")
        d.rounded_rectangle((x0, y, x0 + max(bw * score / 25, 12), y + 40), radius=8, fill=c)
        d.text((x0 + bw * score / 25 + 14, y + 20), f"{score:.2f}（成功 {sr:.1f}%）", font=font(24), fill=FG, anchor="lm")
    d.text((W / 2, 970), "出典：robodojo-benchmark.com ・ arXiv 2607.04434 ・ github.com/shideqin/opendm（Dexmal）・ huggingface.co/lerobot/dm05_base",
           font=font(22), fill=MUTED, anchor="mm")
    return im


def s_dm05(stage=2):
    im, d = canvas()
    header(d, "0.2", "DM0.5 の結果：狙った vial を 3 つの位置すべてで穴に挿入",
           "成功 = 狙った vial が穴に 1 cm 以上。取り違え = 別の vial を入れた（成功には数えない）")
    table(d, 260, 290, ["位置", "成功", "取り違え"], [
        ["手前", ("4 / 30（13%）", GREEN), "0"],
        ["真ん中", ("2 / 30（7%）", GREEN), "5（手前を入れた）"],
        ["一番奥", ("2 / 30（7%）", GREEN), "8（手前 5、真ん中 3）"],
        ["合計", ("8 / 90（9%）", GREEN), ("13", AMBER)],
    ], [300, 520, 580], row_h=78, size=38, hi=1)
    # 次のクリップで足す Temporal Ensemble の説明（scripts/eval/dm05_server.py --ensemble 5）
    d.rounded_rectangle((100, 720, W - 100, 1000), radius=24, fill=PANEL)
    if not stage:
        return im
    pill(d, (140, 745), "次に試す：Temporal Ensemble（TE）", 30, ON, BLUE)
    bullets(d, 140, 830, [
        "TE なし：30 手先まで予測し 15 手使って予測し直す → 継ぎ目で関節が飛び、カクカク",
        ("TE（K=5）：5 手ごとに予測し、同じ時刻を指す予測をすべて重み付きで平均", BLUE),
    ], size=28, gap=62)
    d.text((140, 960), "ACT（Zhao et al., 2023）と同じ方式。重みは古い予測ほど大きい exp(−0.01·i)", font=font(24), fill=MUTED)
    x0, px, y0 = 1400, 10, 760  # 1 手 = 10 px の時間軸
    for k, c in enumerate([BLUE, GREEN, AMBER]):
        y = y0 + k * 52
        d.rounded_rectangle((x0 + 5 * k * px, y, x0 + (5 * k + 30) * px, y + 34), radius=8, fill=c)
        d.text((x0 - 14, y + 17), f"t={5 * k}", font=font(22, True), fill=MUTED, anchor="rm")
    cx = x0 + 18 * px
    if stage < 2:
        return im
    d.rectangle((cx - 6, y0 - 10, cx + 6, y0 + 2 * 52 + 44), outline=FG, width=3)
    d.text((cx, y0 + 3 * 52 + 20), "この時刻は 3 つの予測を平均", font=font(24, True), fill=FG, anchor="mm")
    return im


def s_cmp():
    im, d = canvas()
    header(d, "0.2", "DM0.5 は手前の vial を入れがち、TE を足すと取り違えが増えた",
           "上：お手本の GPT-6 Astra の収集　下：学習したモデル（各 90 回）　TE = Temporal Ensemble（予測の重ね平均）")
    # Astra の成功は outputs/gpt_astra150.log の集計（シミュレータ標準の判定：どれかの vial が rack に置かれた）
    rows = [("GPT-6 Astra（収集時）", 134, [(46, BLUE, "成功 46 / 134（34%）"), (88, "#D1D5DB", "失敗 88")],
             "（手前 12/44・真ん中 25/45・一番奥 9/45）"),
            ("DM0.5", 90, [(8, GREEN, "狙い通り 8"), (13, AMBER, "別の vial 13"), (69, "#D1D5DB", "入らず 69")], None),
            ("DM0.5 + TE", 90, [(8, GREEN, "狙い通り 8"), (20, AMBER, "別の vial 20"), (62, "#D1D5DB", "入らず 62")], None),
            ("（参考）GR00T N1.7 + TE", 90, [(1, GREEN, "狙い通り 1"), (2, AMBER, "別の vial 2"), (87, "#D1D5DB", "入らず 87")], None)]
    x0, bw, y = 620, 1140, 230
    for i, (name, total, segs, note) in enumerate(rows):
        ref = name.startswith("（参考）")
        if i == 0:
            d.text((x0 - 30, y + 80), "判定：rack に置けたか", font=font(24), fill=MUTED, anchor="rm")
        d.text((x0 - 30, y + 32 if i == 0 else y + 35), name, font=font(32 if ref or i == 0 else 40, True), fill=MUTED if ref else BLUE if i == 0 else FG, anchor="rm")
        x = x0
        for n, c, _ in segs:
            w = bw * n / total
            d.rectangle((x, y, x + w, y + 64), fill=c)
            x += w
        d.text((x0 + 12, y + 80), "   ".join(l for _, _, l in segs) + (f"　{note}" if note else ""), font=font(28), fill=FG)
        if i == 0:
            d.line((80, y + 138, W - 80, y + 138), fill="#E5E7EB", width=2)
            d.text((x0 + 12, y + 152), "↓ 学習後：狙った vial が穴に 1 cm 以上刺さったか", font=font(26, True), fill=MUTED)
            y += 70
        y += 150
    bullets(d, 170, 915, [
        ("TE を足すと入れる回数は増えたが、増えた分はすべて「別の vial」", AMBER),
        ("参考として、同じ条件で GR00T N1.7 + TE とも比較した", MUTED),
    ], size=28, gap=46)
    return im


def s_03():
    im, d = canvas()
    header(d, "0.3", "実機 SO-101 で Sim to Real：2 つのやり方を試す",
           "どちらも実機のデータは 0。推論は AWS で行い、SSH トンネル越しに手元の SO-101 を動かす")
    for i, (tag, c, items) in enumerate([
            ("Show-Harness（GPT-6 Astra）", BLUE, [
                "カメラ画像を見て、Astra が\n1 手ずつ動作の語を選ぶ",
                "0.2 で本家に入れた SO-101 対応を\nそのまま実機に繋いだ"]),
            ("DM0.5（VLA）", GREEN, [
                "シミュレータで Astra が集めた\nデモだけで学習した VLA",
                "関節角をまとめて出し、\n実機をそのまま動かす"])]):
        x = 100 + i * 880
        d.rounded_rectangle((x, 260, x + 840, 760), radius=24, fill=PANEL)
        pill(d, (x + 40, 290), tag, 34, ON, c)
        y = 410
        for txt in items:
            d.text((x + 40, y), "•", font=font(34, True), fill=BLUE)
            d.multiline_text((x + 80, y), txt, font=font(34), fill=FG, spacing=14)
            y += 170
    d.text((W / 2, 850), "カメラの位置は ArUco のシートで測り、シミュレータのカメラに合わせた", font=font(34), fill=MUTED, anchor="mm")
    return im


def s_end():
    im, d = canvas()
    d.text((W / 2, 170), "まとめ", font=font(72, True), fill=FG, anchor="mm")
    y = bullets(d, 220, 300, [
        ("0.1 Show-Harness：Qwen3.5-4B SFT 68% → PPO で 81 / 100、往復も減少", GREEN),
        ("0.2 SO-101：Show-Harness に SO-101 対応をマージ（PR #6、OSS 貢献）", GREEN),
        "0.2 DM0.5 + TE：vial を穴に入れた 28 / 90（31.1%）、うち狙い通り 8 / 90",
        ("課題：狙いとは別の vial を入れる取り違えが多い。TE でも改善せず", AMBER),
        ("0.3 実機：Astra・DM0.5 とも vial をつかんで運べたが、穴への挿入はまだ", AMBER),
    ], size=38, gap=96)
    d.text((W / 2, 960), "github.com/showlab/Show-Harness", font=font(30), fill=MUTED, anchor="mm")
    return im


# ------------------------------------------------------------------ clips
def src_duration(src):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(src)],
                         capture_output=True, text=True, check=True).stdout
    return float(out)


def overlay(chapter, tag, caption, speed, note=None, excerpt=False, side=None, approx=False, cards=None):
    im, d = canvas(alpha=True)
    d.rectangle((0, 0, W, BAR), fill=BG)
    toc(d, chapter)
    x = pill(d, (40, 30), tag, 30, ON, BLUE)
    d.text((x + 26, BAR / 2), caption, font=font(44, True), fill=FG, anchor="lm")
    if cards:
        y0, cw = TOP_BOX[1] + TOP_BOX[3] + 20, (W - 160 - 40 * (len(cards) - 1)) / len(cards)
        for k, (lab, col, txt) in enumerate(cards):
            x = 80 + k * (cw + 40)
            d.rounded_rectangle((x, y0, x + cw, y0 + 270), radius=20, fill=PANEL)
            pill(d, (x + 28, y0 + 22), lab, 28, ON, col)
            d.multiline_text((x + 28, y0 + 96), txt, font=font(30), fill=FG, spacing=12)
    if side:
        x0 = SIDE_BOX[0] + SIDE_BOX[2] + 60
        d.rectangle((x0 - 30, BAR, W, H - TOC_H), fill=BG)
        y = BAR + 120
        for it in side:
            txt, col = it if isinstance(it, tuple) else (it, FG)
            d.text((x0, y), "•", font=font(38, True), fill=BLUE)
            d.multiline_text((x0 + 44, y), txt, font=font(38), fill=col, spacing=14)
            y += 190
    if speed != 1 or note or excerpt:
        parts = ([f"{'約 ' if approx else ''}×{speed:g} 再生"] if speed != 1 else []) + (["一部抜粋"] if excerpt else [])
        txt = "▶ " + "・".join(parts) + (f"（{note}）" if note else "")
        f = font(34, True)
        l, t, r, b = d.textbbox((0, 0), txt, font=f)
        pill(d, (W - (r - l) - 44 - 40, H - TOC_H - (b - t) - 20 - 24), txt, 34, ON, AMBER)
    return im


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def fades(dur):
    return f"fade=in:st=0:d=0.25,fade=out:st={dur - 0.3:.2f}:d=0.3"


def render_anim(frames, dur, out):
    """frames(t) -> PIL 画像 を 30fps で ffmpeg に流し込む。"""
    p = subprocess.Popen(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-vf", f"{fades(dur)},format=yuv420p", "-c:v", "libx264", "-crf", "18", str(out)],
                         stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for n in range(int(dur * FPS)):
        p.stdin.write(frames(n / FPS).tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError(f"ffmpeg failed: {out}")
    frames(dur).save(out.with_suffix(".png"))  # 説明資料の図に使う


def render_still(img, dur, out):
    png = out.with_suffix(".png")
    img.save(png)
    run(["ffmpeg", "-y", "-loop", "1", "-framerate", str(FPS), "-i", str(png), "-t", f"{dur}",
         "-vf", f"{fades(dur)},format=yuv420p", "-r", str(FPS), "-c:v", "libx264", "-crf", "18", str(out)])


# 元の映像がすでに実時間より速いクリップ（右下の倍率は実時間比で出す）
# データ収集の録画は 1 手 約 0.56 秒のリプレイ。収集時の Astra は 1 手 約 7.7 秒（outputs/gpt_astra150 の latency 中央値）
BASE = {"レコーディング_2026-10-04_221344.mp4": 7.7 / 0.56}

# 下に操作パネル（ボタン）が写っているクリップは、上のカメラ映像だけを使う（GUMI は使っていないので誤解を避ける）
CROP = {"20261014-0430-50.5295639.mp4": "crop=iw:292:0:0,"}


def render_bumper(src, out):
    """講座の OP / ED（29.97fps）を本編と同じ 30fps・同じ符号化に揃える。"""
    run(["ffmpeg", "-y", "-i", str(src), "-vf", f"fps={FPS},scale={W}:{H},format=yuv420p", "-an", "-r", str(FPS),
         "-c:v", "libx264", "-crf", "18", str(out)])
    return src_duration(out)


def render_clip(src, start, length, speed, ov, out, box=BOX):
    png = out.with_suffix(".png")
    ov.save(png)
    dur = length / speed
    x, y, w, h = box
    vf = (f"[0:v]{CROP.get(str(src), '')}setpts=(PTS-STARTPTS)/{speed},fps={FPS},scale=w={w}:h={h}:force_original_aspect_ratio=decrease,"
          f"pad={W}:{H}:{x}+({w}-iw)/2:{y}+({h}-ih)/2:color={BG}[v];[v][1:v]overlay=0:0,{fades(dur)},format=yuv420p")
    run(["ffmpeg", "-y", "-ss", f"{start}", "-t", f"{length}", "-i", str(src if isinstance(src, Path) else SRC / src), "-loop", "1", "-i", str(png),
         "-filter_complex", vf, "-t", f"{dur:.3f}", "-an", "-r", str(FPS), "-c:v", "libx264", "-crf", "18", str(out)])
    return dur


# bumper = (ファイル) / still = (章, slide 関数, 秒) / clip = (章, ファイル, 開始, 長さ, 倍速, タグ, 説明, 注記[, 右の要点])
# 右の要点を付けたクリップ（縦撮り）は左に寄せて置く
# 章は目次の帯で強調する項目（CHAPTERS）。None なら強調なし
PLAN = [
    ("bumper", OP),
    ("anim", Title, 3.5),
    ("still", "背景と動機", s_bg, 8),
    ("still", "背景と動機", s_show, 7),
    ("clip", "背景と動機", SHOW / "show-harness-demo.mp4", 8.0, 12.0, 2, "Show-Harness", "公式デモ：VLM が考え、ロボットが動く", "出典：showlab/Show-Harness"),
    ("still", "0.1 ManiSkill", s_01, 5),
    ("clip", "0.1 ManiSkill", "20260930-0202-21.5847085.mp4", 0.0, 60.0, 6, "0.1", "GPT-6 Astra が ManiSkill の Franka を操作：10 / 10 成功", None),
    ("clip", "0.1 ManiSkill", "20261007-0147-45.8821166.mp4", 0.0, 14.0, 2, "0.1", "GPT-6 Astra の成功 100 話で Qwen3.5-4B を SFT：34 / 50（68%）成功", "判断 5 回/秒のリプレイ"),
    ("still", "0.1 ManiSkill", s_rl, 13),
    ("still", "0.2 SO-101", s_ws, 7),
    ("still", "0.2 SO-101", s_road, 8),
    ("still", "0.2 SO-101", s_02, 7),
    ("clip", "0.2 SO-101", "レコーディング_2026-10-04_221344.mp4", 3.9, 18.0, 2, "0.2", "GPT-6 Astra で SO-101 のデータ収集：134 試行で 46 成功（34%）", "実時間比。実際は 1 手 約 8 秒"),
    ("anim", a_hyp, 9),
    ("clip", "0.2 SO-101", "レコーディング_2026-10-10_110325.mp4", 10.4, 20.8, 2, "0.2", "DM0.5（Astra の成功 50 話で学習）：動きがカクカク", None),
    ("anim", a_dm05, 9),
    ("clip", "0.2 SO-101", "20261014-0430-50.5295639.mp4", 5.0, 18.0, 2, "0.2", "DM0.5 + Temporal Ensemble（K=5）", None),
    ("still", "0.2 SO-101", s_cmp, 10),
    ("still", "0.3 Sim to Real", s_03, 3.5),
    ("clip", "0.3 Sim to Real", REAL / "showharness1025.mp4", 0.0, 420.0, 50, "0.3", "Show-Harness で GPT-6 Astra が実機 SO-101 を操作", None, [
        "Astra が 1 手ずつ選んで\n手前の vial に近づく",
        ("vial をつかんで\n持ち上げることができた", GREEN),
        ("穴には入れられず、\n離した vial は机に落ちた", AMBER),
        "1 手に数秒かかるので\n1 話で約 7 分"]),
    ("clip", "0.3 Sim to Real", REAL / "1024.mp4", 0.0, 44.0, 6, "0.3", "DM0.5 の Sim to Real：シミュレータのデモだけで学習", None, [
        "実機のデータなしで\nそのまま実機を動かした",
        ("vial をつかんで\nrack の手前まで運べた", GREEN),
        ("穴への挿入はまだできていない", AMBER),
        ("途中から別の角度で撮影", MUTED)]),
    ("anim", Extra, 2),
    ("clip", "番外編", "4b2beedf-ddc4-4270-9e08-656c7be75f70.mp4", 0.0, 16.0, 2, "番外編", "世界モデル Cosmos Transfer：シミュレータの映像を実写風に変換", None, {"cards": [
        ("目的", BLUE, "シミュレータの映像（左）を、\n世界モデルで実写風の見た目（右）に\n変換できるかを試した"),
        ("動機", AMBER, "sim だけで学んだ VLA は、実機の\n照明・質感・背景の違いでつまずく\n（見た目の Sim to Real のずれ）"),
        ("最終的な目標", GREEN, "変換した映像を学習データに足し、\n実機のデータなしでも実機で動く\nVLA を育てる")]}),
    ("still", "まとめ", s_end, 6),
    ("bumper", ED),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "outputs/video/summary.mp4"))
    ap.add_argument("--only_slides", action="store_true", help="スライドの PNG だけ書き出す（確認用）")
    args = ap.parse_args()
    out = Path(args.out)
    work = out.parent / "parts"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    parts, total = [], 0.0
    for i, p in enumerate(PLAN):
        seg = work / f"{i:02d}.mp4"
        if p[0] == "bumper":
            if args.only_slides:
                continue
            dur, name = render_bumper(p[1], seg), p[1].name
        elif p[0] == "anim":  # 動くスライド（目次の帯は各クラスが描く）
            _, cls, dur = p
            obj, name = cls(), cls.__name__
            if args.only_slides:
                for t in (0.6, 1.5, 2.5, dur):
                    obj.frame(t).save(work / f"{i:02d}_{name}_{t:.1f}.png")
                continue
            render_anim(obj.frame, dur, seg)
        elif p[0] == "still":
            _, chap, fn, dur = p
            img = fn()
            toc(ImageDraw.Draw(img), chap)
            name = fn.__name__
            if args.only_slides:
                img.save(work / f"{i:02d}_{name}.png")
                continue
            render_still(img, dur, seg)
        else:
            _, chap, src, start, length, speed, tag, cap, note, *side = p
            extra = side[0] if side else None  # list = 右の要点、dict = 下のカード
            cards = extra.get("cards") if isinstance(extra, dict) else None
            side = extra if isinstance(extra, list) else None
            name = Path(src).name
            if args.only_slides:
                overlay(chap, tag, cap, speed, note, side=side, cards=cards).save(work / f"{i:02d}_overlay.png")
                continue
            path = src if isinstance(src, Path) else SRC / src
            excerpt = length < src_duration(path) - 0.5  # 時間の都合で一部だけ使ったクリップ
            base = BASE.get(name, 1)
            shown = round(speed * base, -1) if base != 1 else speed
            dur = render_clip(src, start, length, speed, overlay(chap, tag, cap, shown, note, excerpt, side, base != 1, cards), seg,
                              SIDE_BOX if side else TOP_BOX if cards else BOX)
        parts.append(seg)
        total += dur
        print(f"{i:02d} {dur:5.1f}s  計 {total:6.1f}s  {name}", flush=True)
    if args.only_slides:
        return
    lst = work / "list.txt"
    lst.write_text("".join(f"file '{p.name}'\n" for p in parts))
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(out)])
    print(f"{out}  {total:.1f}s")


if __name__ == "__main__":
    main()
