# VLM が集めたデータで VLA を育てる：SO-101 — Show-Harness × SO-101

Physical AI 応用1講座 最終課題（omnicampus: isk54_34）のコード一式。

- **動画**：（YouTube の URL）— 約 3 分。ロボットの映像の一部は 2 倍速（動画内の右下に表示）
- **説明資料**：`説明資料_PAI最終課題_isk54_34.pdf`（omnicampus に提出）
- **OSS 貢献**：このリポジトリの SO-101 対応を [showlab/Show-Harness](https://github.com/showlab/Show-Harness) に
  Pull Request（[#6](https://github.com/showlab/Show-Harness/pull/6)）として提出し、2026-10-10 にマージされた

| 段階 | 内容 | 主なコード |
| --- | --- | --- |
| データ収集 | GPT-6 Astra（gpt-6-astra）が Show-Harness の動作単位で SO-101（Isaac Lab sim）を操作 | `scripts/collect/` |
| VLA の学習 | 成功 50 話を LeRobot 形式に変換し、DM0.5 と GR00T N1.7 を fine-tune | `scripts/train/` |
| 評価 | 手前 / 真ん中 / 一番奥 × 各 30 話。狙った vial が穴に 1 cm 以上刺さったら成功 | `scripts/eval/` |
| 実機 | SO-101 実機 backend（LeRobot） | `src/so101_showharness/real/`, `scripts/real/` |
| 動画・資料 | まとめ動画（PIL + ffmpeg）と説明資料の PDF | `scripts/render/` |

データ・学習済み重み・評価動画はサイズの都合でリポジトリに含めていない。

---

# so101-showharness

[Show-Harness](https://github.com/showlab/Show-Harness)（VLM が離散アクショントークン
`MV_FWD` / `MV_DOWN` / `GRASP` … を1ステップ1個出してロボットを動かす枠組み）を **SO-101** に載せ、

1. **データ収集**: SO-101 の Isaac Lab sim（NVIDIA Sim-to-Real SO-101 Workshop の環境。
   sreetz-nv の vials / orange 系データと同じ環境）で Show-Harness 形式のデモを新規収集
2. **FT**: 収集データで小型 VLM（Qwen3.5-2B LoRA, Show-Harness の `train/` パイプライン）を fine-tune
3. **Sim-to-Real**: 同じトークン語彙・同じ画像契約のまま、実機 SO-101 で閉ループ実行

までを行うための開発リポジトリ。
同じ成功デモを判断単位（離散アクション → VLM 学習）と 30Hz の連続軌道（指令関節角 → VLA 学習）の両方で記録し、
VLM と VLA を比較できるようにする。

## このリポジトリに入れるもの / 入れないもの

| 入れる（git） | 入れない（`.gitignore`） |
| --- | --- |
| SO-101 用 interpreter・sim/real backend・収集/学習/評価スクリプト・設定・テスト・ドキュメント | 収集データ（`data/`）、学習済み重み（`models/`）、実験出力・動画（`outputs/`）、秘密情報 |

データと重みは Hugging Face Hub（`Isk5434/…`, private）に置き、ここには参照だけ書く。
上流コード（Show-Harness, workshop）は**複製せず**、固定コミットを [docs/UPSTREAM.md](docs/UPSTREAM.md) に記録して参照する。

## レイアウト

```
src/so101_showharness/
  interpreters/   # SO-101 用 atomic controller（トークン → EE 微小移動 → IK → 関節目標）
  sim/            # Isaac Lab workshop env へのアダプタ（観測・カメラ・成功判定）
  real/           # 実機 SO-101 backend（Show-Harness の robot duck type を LeRobot で実装）
  recorder/       # 共通軌道レコーダー（30Hz: 画像・観測関節・指令関節・判断ID）→ Show-Harness形式 / LeRobot形式へ変換
configs/          # robot_so101_sim.yaml / robot_so101_real.yaml / primitives_so101.yaml
scripts/
  collect/        # GUMI(ブラウザ操作) / 特権オラクル / zero-shot VLM による収集
  train/          # rollouts → LLaMA-Factory 形式変換・LoRA 学習起動
  eval/           # sim 閉ループ評価（成功率・動画）
  real/           # 実機ブリングアップ・キャリブ・閉ループ実行（ローカル PC 側）
train/            # LoRA 設定（Show-Harness train/configs を SO-101 用に派生）
docs/             # 計画・上流の固定版・環境記録
data/ models/ outputs/   # ローカルのみ（git 管理外）
```

## 進め方

詳細な段階計画と判定基準は [docs/PLAN.md](docs/PLAN.md)。
