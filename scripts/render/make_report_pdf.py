"""PAI 最終課題の説明資料（A4、PDF）を作る。PIL でページを描いて 1 つの PDF にまとめる（追加の依存なし）。

図は make_summary_video.py が書き出したスライド（outputs/video/parts/*.png）を使うので、先にそちらを実行する。
個人情報（氏名・所属・メール）は入れない。

    python3 scripts/render/make_report_pdf.py --account <omnicampus アカウント名>
"""
import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
PARTS = REPO / "outputs/video/parts"
DPI, PW, PH, M = 200, 1654, 2339, 140          # A4 @200dpi、余白
INK, MUTED, ACCENT, RULE, HEAD_BG = "#111827", "#4B5563", "#0369A1", "#D1D5DB", "#F3F4F6"


def font(size, bold=False):
    f = "NotoSansCJK-Bold.ttc" if bold else "NotoSansCJK-Regular.ttc"
    return ImageFont.truetype(f"/usr/share/fonts/opentype/noto/{f}", size, index=0)


NO_START = set("。、，．）」』】〕)]」・ー")  # 行頭に置かない文字（ぶら下げる）


def tokens(text):
    # 英数字の語は分けずに、それ以外は 1 文字ずつ折り返す
    return re.findall(r"[A-Za-z0-9_.\-/%+=:#()（）]+|\s|.", text)


def wrap(text, f, width):
    lines, cur = [], ""
    for t in tokens(text):
        if t == "\n":
            lines.append(cur)
            cur = ""
            continue
        if f.getlength(cur + t) > width and cur and t not in NO_START:
            lines.append(cur.rstrip())
            cur = t.lstrip()
        else:
            cur += t
    lines.append(cur)
    return lines


class Doc:
    def __init__(self, footer):
        self.pages, self.footer = [], footer
        self.new_page()

    def new_page(self):
        self.im = Image.new("RGB", (PW, PH), "white")
        self.d = ImageDraw.Draw(self.im)
        self.pages.append(self.im)
        self.y = M

    def need(self, h):
        if self.y + h > PH - M:
            self.new_page()

    def text(self, s, size=30, bold=False, color=INK, indent=0, gap=14, after=10):
        f = font(size, bold)
        lh = int(size * 1.55)
        for line in wrap(s, f, PW - 2 * M - indent):
            self.need(lh)
            self.d.text((M + indent, self.y), line, font=f, fill=color)
            self.y += lh
        self.y += after

    def title(self, s, sub):
        self.text(s, 50, True, after=6)
        self.text(sub, 30, color=MUTED, after=20)
        self.d.line((M, self.y, PW - M, self.y), fill=ACCENT, width=4)
        self.y += 30

    def h(self, s):
        self.need(300)  # 見出しだけがページ末に残らないように
        self.y += 18
        self.text(s, 38, True, ACCENT, after=4)
        self.d.line((M, self.y, PW - M, self.y), fill=RULE, width=2)
        self.y += 18

    def h3(self, s):
        self.need(90)
        self.y += 6
        self.text(s, 32, True, after=2)

    def ul(self, items, size=29):
        for it in items:
            f = font(size)
            lh = int(size * 1.55)
            lines = wrap(it, f, PW - 2 * M - 44)
            self.need(lh)
            self.d.text((M + 10, self.y), "•", font=font(size, True), fill=ACCENT)
            for line in lines:
                self.need(lh)
                self.d.text((M + 44, self.y), line, font=f, fill=INK)
                self.y += lh
            self.y += 4
        self.y += 8

    def table(self, cols, rows, widths, size=26):
        rh = int(size * 2.0)
        scale = (PW - 2 * M) / sum(widths)
        widths = [w * scale for w in widths]
        self.need(rh * (len(rows) + 1) + 20)
        x0 = M
        self.d.rectangle((x0, self.y, PW - M, self.y + rh), fill=HEAD_BG)
        for k, row in enumerate([cols] + rows):
            x = x0
            for j, c in enumerate(row):
                self.d.text((x + 14, self.y + rh / 2), c, font=font(size, k == 0 or j == 0), fill=INK, anchor="lm")
                x += widths[j]
            self.y += rh
            self.d.line((x0, self.y, PW - M, self.y), fill=RULE, width=2)
        self.y += 24

    def image(self, path, caption, width=None):
        p = Image.open(path).convert("RGB")
        w = int(width or (PW - 2 * M))
        h = int(p.height * w / p.width)
        self.need(h + 70)
        self.im.paste(p.resize((w, h), Image.LANCZOS), (M + (PW - 2 * M - w) // 2, self.y))
        self.y += h + 10
        self.text(caption, 24, color=MUTED, after=16)

    def save(self, out):
        n = len(self.pages)
        for i, pg in enumerate(self.pages):
            ImageDraw.Draw(pg).text((PW / 2, PH - 70), f"{self.footer}  ·  {i + 1} / {n}", font=font(22), fill=MUTED, anchor="mm")
        self.pages[0].save(out, save_all=True, append_images=self.pages[1:], resolution=DPI)


def build(doc, github, youtube):
    doc.title("VLM が集めたデータで VLA を育てる：SO-101",
              "Show-Harness を使い、Frontier VLM が操作して集めたデータで小型 VLM・VLA にロボットを動かさせる")

    doc.h("1. 概要")
    doc.text("Show-Harness（showlab）は、VLM が「前へ・上へ・つかむ」といった意味のある動作単位を 1 語ずつ選び、"
             "それを各ロボットの動きに変換して実行させる仕組みです。本課題では、これを使って Frontier VLM の GPT-6 Astra（OpenAI、モデル ID gpt-6-astra、Codex CLI 経由、推論 effort は low。以下 Astra）に"
             "シミュレータ上のロボットを操作させ、成功したエピソードだけを学習データにしました。そのデータで、"
             "より小さなモデルが同じタスクをこなせるかを、3 段階で確かめました。")
    doc.image(Path("/home/ubuntu/Show-Harness-so101/assets/overview.png"),
              "図 0：Show-Harness の概要（showlab/Show-Harness 公式リポジトリ、Apache-2.0 より）", width=1100)
    doc.ul([
        "0.1 ManiSkill（Franka、ブロックをコースターへ）：Astra のデータで Qwen3.5-4B を SFT し、強化学習（REINFORCE / GRPO / PPO）で改善",
        "0.2 SO-101（Isaac Lab の Workshop タスク、vial を rack へ）：Show-Harness を SO-101 に対応させ、本家にマージ（OSS 貢献）。"
        "Astra のデータで VLA（DM0.5、GR00T N1.7）を学習し、同じ条件で比較",
        "0.3 Sim to Real：SO-101 を組み立て、キャリブレーションし、実機での推論を開始（実機を使う時間が取れず、検証は途中まで）",
    ])

    doc.h("2. 目的")
    doc.ul([
        "人がテレオペで教えなくても、Frontier VLM の操作ログだけで小さなモデルを育てられるかを確かめる",
        "VLM の離散的な動作（1 語ずつ）と、VLA の連続的な関節角のどちらが細かい作業に向くかを比べる",
        "成功率だけでなく、行動の癖（往復・取り違え・停止）まで測って、何が効いて何が効かないかを切り分ける",
    ])

    doc.h("3. 使用したツール・環境")
    doc.table(["区分", "使用したもの"], [
        ["フレームワーク", "Show-Harness（showlab、Apache-2.0）"],
        ["シミュレータ", "ManiSkill（ZtBlockPlace-v1、Franka）、Isaac Lab（SO-101 Workshop）"],
        ["データ収集", "GPT-6 Astra（gpt-6-astra）が GUI で操作。成功エピソードのみ使用"],
        ["小型 VLM", "Qwen3.5-4B（LoRA で SFT）＋ REINFORCE / GRPO / PPO"],
        ["VLA", "DM0.5（lerobot/dm05_base）、GR00T N1.7（Isaac-GR00T）"],
        ["実機", "SO-101（組み立て・キャリブレーションから実施）"],
        ["その他", "Cosmos Transfer（世界モデルによる見た目の変換）、GPU は L40S 1 枚"],
    ], [300, 1074])

    doc.h("4. 取り組みと結果")
    doc.h3("0.1 ManiSkill：小型 VLM の SFT と強化学習")
    doc.text("Astra はこのタスクを 10 / 10 で成功し、その成功 100 エピソードで Qwen3.5-4B を SFT すると、"
             "閉ループで 34 / 50（68%）成功しました。失敗の多くは同じ語を繰り返して時間切れになる形だったので、"
             "成功だけに報酬を与える強化学習で改善を試みました（検証用 100 配置、基準と同じ seed で比較）。")
    doc.table(["モデル", "成功", "基準との比較", "往復 / 100 判断"], [
        ["基準（SFT）", "59 / 100", "—", "1.2"],
        ["REINFORCE", "78 / 100", "+32 / −13、p = 0.007", "6.0"],
        ["GRPO", "71 / 100", "+24 / −12、p = 0.065", "4.9"],
        ["PPO（critic + GAE）", "81 / 100", "+27 / −5、p = 0.0001", "0.5"],
    ], [380, 240, 460, 294])
    doc.text("評価の前に条件を記録しておいた最終テスト（別の 100 配置）でも、REINFORCE は 87 / 100 で、"
             "基準より有意に成功が多くなりました（正確 McNemar 検定、p = 0.020）。", 28, color=MUTED)

    doc.h3("0.2 SO-101：Show-Harness の拡張と VLA の比較")
    doc.text("Show-Harness は SO-101 に対応していなかったため、5 自由度アーム向けの運動学・実行部と、"
             "Isaac Lab の Workshop 環境への接続を実装し、本家の showlab/Show-Harness に Pull Request（#6）を出して"
             "マージされました。そのうえで Astra に vial を rack に入れるタスクを操作させましたが、vial はなかなか穴に入りませんでした。"
             "VLM の動作は 1 語ずつの離散的な動きなので細かい作業が苦手だと考え、Astra の成功 50 エピソードで VLA を学習しました。")
    doc.table(["位置", "Astra の試行", "成功（sim の判定）", "成功率"], [
        ["手前", "44", "12", "27%"], ["真ん中", "45", "25", "56%"], ["一番奥", "45", "9", "20%"], ["合計", "134", "46", "34.3%"],
    ], [300, 340, 400, 334])
    doc.text("sim の判定は「どれかの vial が rack に置かれた」です。失敗 88 本のうち 79 本は動作が収束せず打ち切り。学習にはこの 46 本と、別の試行で集めた成功 4 本を使いました。", 26, color=MUTED)
    doc.text("評価は vial の位置（手前・真ん中・一番奥）ごとに 30 エピソード、合計 90 エピソードです。"
             "成功は「狙った 1 本が立って、穴に 1 cm 以上刺さった」場合だけとし、別の vial を入れた場合は取り違えとして分けて数えました。")
    doc.image(PARTS / "13.png", "図 1：3 条件の比較（各位置 30 エピソード、同じ seed）。TE = temporal ensembling（K=5）")
    doc.ul([
        "DM0.5 は、3 つの位置すべてで狙った vial を穴に入れることができました（合計 8 / 90）。",
        "temporal ensembling を足すと、何かしらの vial を入れられた数は 21 → 28 / 90 に増えましたが、増えた分はすべて取り違えで、"
        "差も有意ではありません（Fisher の正確検定、p ≈ 0.32）。",
        "GR00T N1.7 は 1 / 90 でした。学習データ上の予測誤差は 0.037 rad（約 2°）と小さく、推論サーバの入力も学習データと一致していることを確認しました。"
        "閉ループでは vial の上で数十秒止まる動きが多く、パイプラインの不具合ではなく汎化の問題と考えています。",
    ])

    doc.h3("0.3 Sim to Real")
    doc.text("SO-101 を組み立ててキャリブレーションし、GR00T N1.7 で実機での推論を始めましたが、実機を使える時間が取れず、検証は途中までとなりました。"
             "番外編として、世界モデル Cosmos Transfer で sim の映像を実写風に変換し、見た目の差を埋めるデータ拡張の可能性も試しました。")

    doc.h("5. 工夫した点")
    doc.ul([
        "OSS への貢献：Show-Harness を SO-101 に対応させ、本家に PR を出してマージされた。事前に issue でメンテナと範囲（sim のみ、"
        "共通の語彙は変えない、未対応の回転単位は明示的に失敗させる）を合意してから実装した。",
        "テレオペなしのデータ収集：人ではなく Frontier VLM がブラウザの GUI でロボットを操作し、成功したエピソードだけを学習に使った。",
        "評価を厳しくした：シミュレータ標準の成功判定は、どの vial でも一度置けば成功のまま残ることに気づいた。"
        "そこで「狙った vial が刺さっているか」で判定し直し、取り違えを別に数えた。",
        "公平な比較：同じ seed・同じ位置で比べ、Fisher や McNemar の検定で差が偶然かどうかを確かめた。",
        "成功率以外も測った：RL では往復の頻度、VLA では取り違え・停止・推論時間を記録し、「成功率は同じでも中身が違う」ことを示した。",
        "原因の切り分け：GR00T が失敗したとき、学習データでの open-loop 評価と、推論サーバと同じ経路での再現を行い、"
        "実装の誤りではないことを確認してから考察した。",
    ])

    doc.h("6. 考察")
    doc.h3("6.1 Temporal ensembling は「迷い」ではなく「動き」をならす")
    doc.text("TE は、少しずつずれた複数の予測を平均して、動きを滑らかにします。そのため vial を穴まで運び切る回数は 21 → 28 回に増えました。"
             "一方で、どの vial に向かうかという判断の誤りは、予測を平均しても直りません。増えた 7 回がすべて取り違えだったのは、このためと考えています。")
    doc.h3("6.2 GR00T：練習問題は解けても、本番では自分のずれから戻れない")
    doc.text("学習データを見せたときの予測のずれは約 2° でした。一方、実際に動かすと、自分の小さなずれが積み重なり、"
             "学習データにない状態（vial の真上で止まった状態など）に入ります。50 話のお手本で 3B のモデルを学習したため、"
             "そうした状態から立て直す動きを学べなかったと考えています。DM0.5 との差が事前学習の違いによるものかは、まだ切り分けられていません。")
    doc.h3("6.3 強化学習：何を褒めるかで、身につく癖が変わる")
    doc.text("REINFORCE は成功した話の判断をすべて同じだけ強めるため、成功率は上がった一方で無駄な往復も約 5 倍に増えました。"
             "PPO は critic が「成功に近づいた判断」だけを強めるので、成功率を上げつつ往復を減らせました。"
             "成功率だけを見ていたら、この違いには気づけませんでした。")

    doc.h("7. 反省点と今後")
    doc.ul([
        "お手本の数と多様さを増やす：位置ごとに十分な数の成功例をそろえ、言語の指示がきちんと効くかを確かめる。",
        "判定をそろえる：「狙った vial が穴に 1 cm 以上刺さった」という厳しい基準を、最初から使う。",
        "GR00T：temporal ensembling なし、途中のチェックポイントで評価し、学習のしすぎか、TE との相性かを切り分ける。",
        "Sim to Real：実機の時間を確保し、キャリブレーション・照明・カメラ位置の差を詰めたうえで、Cosmos Transfer による見た目の拡張も学習に入れて検証する。",
    ])

    doc.h("8. 実用性")
    doc.text("人がテレオペの機材を用意してデモを取らなくても、Frontier VLM にデモを作らせれば、小さなモデルを安く育てられる可能性があります。"
             "一方で、狙いと違う vial を入れる取り違えが残り、言語の指示を確実に効かせることが実用化の鍵になると考えます。")

    doc.h("9. 提出物")
    doc.ul([
        f"コード：{github}",
        f"動画：{youtube}（約 3 分。ロボットの映像の一部は 2 倍速で、動画内の右下に「×2 再生」と表示）",
        "参考：Show-Harness（github.com/showlab/Show-Harness、arXiv 2609.10522）、LeRobot、Isaac-GR00T、ManiSkill、Isaac Lab",
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", default="isk54_34")
    ap.add_argument("--github", default="https://github.com/Isk5434/pai-final-showharness-so101")
    ap.add_argument("--youtube", default="（YouTube の URL を記入）")
    args = ap.parse_args()
    out = REPO / f"outputs/report/説明資料_PAI最終課題_{args.account}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = Doc(f"PAI最終課題_{args.account}")
    build(doc, args.github, args.youtube)
    doc.save(out)
    print(out, len(doc.pages), "pages")


if __name__ == "__main__":
    main()
