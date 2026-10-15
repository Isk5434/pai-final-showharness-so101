"""Notion のまとめの流れに数値スライドを挟んだ 3 分以内のまとめ動画（無音、1920x1080、30fps）を作る。

素材: docs/notion_export/（Notion のエクスポート）。スライドは PIL で描き、ffmpeg で倍速クリップと繋ぐ。
倍速にしたクリップには右下に「×N 再生」を入れる（講座の規定で 2 倍速まで）。

    python3 scripts/render/make_summary_video.py [--out outputs/video/summary.mp4]
"""
import argparse
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "docs/notion_export"
SHOW = Path("/home/ubuntu/Show-Harness-so101/assets")  # showlab/Show-Harness（Apache-2.0）の公式素材
W, H, FPS = 1920, 1080, 30
BG, PANEL, FG, MUTED = "#FFFFFF", "#F3F4F6", "#111827", "#6B7280"
BLUE, GREEN, AMBER, RED = "#0369A1", "#047857", "#B45309", "#B91C1C"
ON = "#FFFFFF"  # 色付きの札の上の文字
BAR = 120          # クリップ画面の上の帯
BOX = (30, BAR + 20, W - 60, H - BAR - 50)  # クリップを収める枠 x, y, w, h


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


def paste_fit(im, path, box):
    p = Image.open(path).convert("RGB")
    x, y, w, h = box
    s = min(w / p.width, h / p.height)
    p = p.resize((int(p.width * s), int(p.height * s)), Image.LANCZOS)
    px, py = x + (w - p.width) // 2, y + (h - p.height) // 2
    im.paste(p, (px, py))
    ImageDraw.Draw(im).rectangle((px - 2, py - 2, px + p.width + 1, py + p.height + 1), outline="#D1D5DB", width=2)


# ------------------------------------------------------------------ slides
def s_title():
    im, d = canvas()
    d.text((W / 2, 360), "VLM が集めたデータで VLA を育てる：SO-101", font=font(80, True), fill=FG, anchor="mm")
    d.text((W / 2, 500), "Frontier VLM が操作して集めたデータで、小型 VLM・VLA にロボットを動かさせる", font=font(40), fill=FG, anchor="mm")
    pill(d, (W / 2 - 400, 660), "OSS 貢献：Show-Harness 本家に SO-101 対応をマージ", 34, ON, GREEN)
    d.text((W / 2, 580), "0.1 ManiSkill（Franka）  →  0.2 SO-101（Show-Harness）  →  0.3 Sim to Real", font=font(36), fill=BLUE, anchor="mm")
    pill(d, (W / 2 - 520, 820), "時間の都合で、映像は 2 倍速・一部抜粋のものがあります（右下に表示）", 32, ON, AMBER)
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
    d.text((70, 990), "図：Show-Harness 公式リポジトリ（Apache-2.0）より", font=font(26), fill=MUTED)
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
    im, d = canvas()
    header(d, "0.1", "強化学習：練習を重ねると、成功率 59% → 81%",
           "Qwen3.5-4B。成功したら「その判断を増やす」ようにシミュレータで練習させた（100 通りの配置で評価）")
    d.text((560, 330), "59%", font=font(150, True), fill=MUTED, anchor="mm")
    d.text((560, 440), "お手本で学んだだけ", font=font(34), fill=FG, anchor="mm")
    d.text((W / 2, 330), "→", font=font(120, True), fill=MUTED, anchor="mm")
    d.text((1360, 330), "81%", font=font(150, True), fill=GREEN, anchor="mm")
    d.text((1360, 440), "強化学習（PPO）のあと", font=font(34), fill=FG, anchor="mm")
    for i, (name, rate, note, c) in enumerate([
            ("やり方 A：REINFORCE", "成功 78%", "でも、無駄な行ったり来たりが 5 倍に", RED),
            ("やり方 B：PPO", "成功 81%", "無駄な行ったり来たりも減った", GREEN)]):
        x = 160 + i * 820
        d.rounded_rectangle((x, 560, x + 760, 840), radius=24, fill=PANEL)
        d.text((x + 40, 600), name, font=font(36, True), fill=FG)
        d.text((x + 40, 670), rate, font=font(56, True), fill=FG)
        d.text((x + 40, 770), note, font=font(34, True), fill=c)
    d.text((160, 900), "PPO は「成功に近づいた一手」だけを褒めるので、余計な動きまで覚えない", font=font(34), fill=BLUE)
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


def s_hyp():
    im, d = canvas()
    header(d, "0.2", "GPT-6 Astra では vial がなかなか穴に入らない")
    bullets(d, 150, 330, [
        "仮説：VLM の動作は離散的（1 語ずつ）なので、VLA に比べて細かい作業が苦手",
        "→ 連続的な関節角を出す VLA に、Astra の成功データを学ばせる",
        ("DM0.5（lerobot/dm05_base）：RoboDojo の LeRobot 対応モデルで sim の精度が最も高い", BLUE),
        "データ：Astra は 134 試行で 46 成功（34%）＋別の試行の 4 本 → 成功 50 話",
        "評価：手前 / 真ん中 / 一番奥 × 各 30 話。成功 = 狙った 1 本が立って穴に 1 cm 以上刺さる",
    ], size=40, gap=120)
    return im


def s_dm05():
    im, d = canvas()
    header(d, "0.2", "DM0.5 の結果：狙った vial を 3 つの位置すべてで穴に挿入",
           "成功 = 狙った vial が穴に 1 cm 以上。取り違え = 別の vial を入れた（成功には数えない）")
    table(d, 260, 290, ["位置", "成功", "取り違え"], [
        ["手前", ("4 / 30（13%）", GREEN), "0"],
        ["真ん中", ("2 / 30（7%）", GREEN), "5（手前を入れた）"],
        ["一番奥", ("2 / 30（7%）", GREEN), "8（手前 5、真ん中 3）"],
        ["合計", ("8 / 90（9%）", GREEN), ("13", AMBER)],
    ], [300, 520, 580], row_h=96, size=42, hi=1)
    return im


def s_cmp():
    im, d = canvas()
    header(d, "0.2", "DM0.5 は手前の vial を入れがち、GR00T はほぼ入らない",
           "各 90 回（手前・真ん中・一番奥 × 30 回）の内訳。TE = 動きを滑らかにする工夫")
    rows = [("DM0.5", 8, 13), ("DM0.5 + TE", 8, 20), ("GR00T N1.7 + TE", 1, 2)]
    x0, bw, y = 520, 1240, 270
    for name, ok, wrong in rows:
        d.text((x0 - 30, y + 45), name, font=font(40, True), fill=FG, anchor="rm")
        segs = [(ok, GREEN, f"狙い通り {ok}"), (wrong, AMBER, f"別の vial {wrong}"), (90 - ok - wrong, "#D1D5DB", f"入らず {90 - ok - wrong}")]
        x = x0
        for n, c, lab in segs:
            w = bw * n / 90
            d.rectangle((x, y, x + w, y + 90), fill=c)
            x += w
        d.text((x0 + 12, y + 130), "   ".join(l for _, _, l in segs), font=font(32), fill=FG)
        y += 210
    bullets(d, 170, 900, [
        ("TE を足すと入れる回数は増えたが、増えた分はすべて「別の vial」", AMBER),
    ], size=34, gap=66)
    return im


def s_gr00t():
    im, d = canvas()
    header(d, "0.2", "GR00T N1.7：練習ではほぼ完璧なのに、本番ではほぼ失敗")
    cards = [("練習問題", "学習に使ったお手本を見せると", "ほぼ正解", "動きのズレは約 2°", GREEN),
             ("本番", "シミュレータで実際に動かすと", "90 回中 1 回", "しか成功しない", RED)]
    for i, (tag, cond, big, small, c) in enumerate(cards):
        x = 160 + i * 820
        d.rounded_rectangle((x, 220, x + 760, 640), radius=24, fill=PANEL)
        pill(d, (x + 40, 250), tag, 32, ON, c)
        d.text((x + 380, 370), cond, font=font(34), fill=FG, anchor="mm")
        d.text((x + 380, 480), big, font=font(96, True), fill=c, anchor="mm")
        d.text((x + 380, 580), small, font=font(36), fill=FG, anchor="mm")
    bullets(d, 170, 700, [
        "プログラムの誤りではないことは確認済み（同じ入力で練習問題を解かせ直した）",
        "本番では vial の上で止まって迷う → お手本 50 回分では「丸暗記」に近く、少し違う場面に対応できない",
        ("考察：お手本の数と多様さが足りないと、大きなモデルでも本番で動けないと考えられる", BLUE),
    ], size=34, gap=78)
    return im


def s_03():
    im, d = canvas()
    header(d, "0.3", "GR00T N1.7 で Sim to Real", "SO-101 の組み立て → キャリブレーション → 実機で推論")
    for i, f in enumerate(["image 6.png", "image 7.png", "image 8.png"]):
        paste_fit(im, SRC / f, (100 + i * 580, 250, 560, 760))
    return im


def s_end():
    im, d = canvas()
    d.text((W / 2, 170), "まとめ", font=font(72, True), fill=FG, anchor="mm")
    y = bullets(d, 220, 300, [
        ("0.1 Show-Harness：Qwen3.5-4B SFT 68% → PPO で 81 / 100、往復も減少", GREEN),
        ("0.2 SO-101：Show-Harness に SO-101 対応をマージ（PR #6、OSS 貢献）", GREEN),
        "0.2 DM0.5 + TE：vial を穴に入れた 28 / 90（31.1%）、うち狙い通り 8 / 90",
        ("課題：狙いとは別の vial を入れる取り違えが多い。TE・GR00T でも改善せず", AMBER),
        ("0.3 Sim to Real は実機の時間が取れず途中まで。次は取り違えの解消と実機での検証", AMBER),
    ], size=38, gap=96)
    d.text((W / 2, 980), "github.com/showlab/Show-Harness", font=font(30), fill=MUTED, anchor="mm")
    return im


# ------------------------------------------------------------------ clips
def src_duration(src):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(src)],
                         capture_output=True, text=True, check=True).stdout
    return float(out)


def overlay(tag, caption, speed, note=None, excerpt=False):
    im, d = canvas(alpha=True)
    d.rectangle((0, 0, W, BAR), fill=BG)
    x = pill(d, (40, 30), tag, 30, ON, BLUE)
    d.text((x + 26, BAR / 2), caption, font=font(44, True), fill=FG, anchor="lm")
    if speed != 1 or note or excerpt:
        parts = ([f"×{speed:g} 再生"] if speed != 1 else []) + (["一部抜粋"] if excerpt else [])
        txt = "▶ " + "・".join(parts) + (f"（{note}）" if note else "")
        f = font(34, True)
        l, t, r, b = d.textbbox((0, 0), txt, font=f)
        pill(d, (W - (r - l) - 44 - 40, H - (b - t) - 20 - 30), txt, 34, ON, AMBER)
    return im


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def fades(dur):
    return f"fade=in:st=0:d=0.25,fade=out:st={dur - 0.3:.2f}:d=0.3"


def render_still(img, dur, out):
    png = out.with_suffix(".png")
    img.save(png)
    run(["ffmpeg", "-y", "-loop", "1", "-framerate", str(FPS), "-i", str(png), "-t", f"{dur}",
         "-vf", f"{fades(dur)},format=yuv420p", "-r", str(FPS), "-c:v", "libx264", "-crf", "18", str(out)])


# 下に操作パネル（ボタン）が写っているクリップは、上のカメラ映像だけを使う（GUMI は使っていないので誤解を避ける）
CROP = {"20261014-0430-50.5295639.mp4": "crop=iw:292:0:0,"}


def render_clip(src, start, length, speed, ov, out):
    png = out.with_suffix(".png")
    ov.save(png)
    dur = length / speed
    x, y, w, h = BOX
    vf = (f"[0:v]{CROP.get(str(src), '')}setpts=(PTS-STARTPTS)/{speed},fps={FPS},scale=w={w}:h={h}:force_original_aspect_ratio=decrease,"
          f"pad={W}:{H}:{x}+({w}-iw)/2:{y}+({h}-ih)/2:color={BG}[v];[v][1:v]overlay=0:0,{fades(dur)},format=yuv420p")
    run(["ffmpeg", "-y", "-ss", f"{start}", "-t", f"{length}", "-i", str(src if isinstance(src, Path) else SRC / src), "-loop", "1", "-i", str(png),
         "-filter_complex", vf, "-t", f"{dur:.3f}", "-an", "-r", str(FPS), "-c:v", "libx264", "-crf", "18", str(out)])
    return dur


# (種類, ...) : still = (slide 関数, 秒) / clip = (ファイル, 開始, 長さ, 倍速, タグ, 説明, 注記)
PLAN = [
    ("still", s_title, 4),
    ("still", s_show, 10),
    ("clip", SHOW / "show-harness-demo.mp4", 8.0, 12.0, 2, "Show-Harness", "公式デモ：VLM が考え、ロボットが動く", "出典：showlab/Show-Harness"),
    ("still", s_01, 5),
    ("clip", "20260930-0202-21.5847085.mp4", 39.2, 24.0, 2, "0.1", "GPT-6 Astra が ManiSkill の Franka を操作：10 / 10 成功", None),
    ("clip", "20261007-0147-45.8821166.mp4", 0.0, 18.0, 2, "0.1", "GPT-6 Astra の成功 100 話で Qwen3.5-4B を SFT：34 / 50（68%）成功", "判断 5 回/秒のリプレイ"),
    ("still", s_rl, 11),
    ("still", s_02, 8),
    ("clip", "レコーディング_2026-10-04_221344.mp4", 3.9, 18.0, 2, "0.2", "GPT-6 Astra で SO-101 のデータ収集：134 試行で 46 成功（34%）", None),
    ("still", s_hyp, 7),
    ("clip", "レコーディング_2026-10-10_110325.mp4", 10.4, 20.8, 2, "0.2", "DM0.5（Astra の成功 50 話で学習）：動きがカクカク", None),
    ("still", s_dm05, 10),
    ("clip", "20261014-0430-50.5295639.mp4", 5.0, 18.0, 2, "0.2", "DM0.5 + Temporal Ensemble（K=5）", None),
    ("still", s_cmp, 12),
    ("clip", "20261015-0353-30.8463886.mp4", 10.0, 20.7, 2, "0.2", "GR00T N1.7 + TE：90 話で唯一の成功（一番奥）", None),
    ("still", s_gr00t, 10),
    ("still", s_03, 5),
    ("clip", "8f46e37a-ecc6-41d0-a09b-1a9a1c1cc2c6.mp4", 3.4, 14.0, 2, "0.3", "キャリブレーション", None),
    ("clip", "7820073b-9601-4665-8196-0dcac996abab.mp4", 0.0, 19.5, 2, "0.3", "実機で推論：実機の時間が取れず、検証は途中まで", None),
    ("clip", "4b2beedf-ddc4-4270-9e08-656c7be75f70.mp4", 0.0, 12.0, 2, "番外編", "世界モデル Cosmos Transfer で見た目を変換", None),
    ("still", s_end, 7),
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
        if p[0] == "still":
            img = p[1]()
            if args.only_slides:
                img.save(work / f"{i:02d}_{p[1].__name__}.png")
                continue
            render_still(img, p[2], seg)
            dur = p[2]
        else:
            _, src, start, length, speed, tag, cap, note = p
            if args.only_slides:
                overlay(tag, cap, speed, note).save(work / f"{i:02d}_overlay.png")
                continue
            path = src if isinstance(src, Path) else SRC / src
            excerpt = length < src_duration(path) - 0.5  # 時間の都合で一部だけ使ったクリップ
            dur = render_clip(src, start, length, speed, overlay(tag, cap, speed, note, excerpt), seg)
        parts.append(seg)
        total += dur
        print(f"{i:02d} {dur:5.1f}s  計 {total:6.1f}s  {Path(p[1]).name if p[0] == 'clip' else p[1].__name__}", flush=True)
    if args.only_slides:
        return
    lst = work / "list.txt"
    lst.write_text("".join(f"file '{p.name}'\n" for p in parts))
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(out)])
    print(f"{out}  {total:.1f}s")


if __name__ == "__main__":
    main()
