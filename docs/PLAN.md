# 計画（2026-10-03 作成・同日改訂）

## 構成

| 役割 | 使うもの | 用途 |
| --- | --- | --- |
| SO-101 sim 環境 | isaac-sim/Sim-to-Real-SO-101-Workshop **v1.0** | バイアル→黄色ラック、カメラ、DR、sim 評価 |
| 操作・収集 | showlab/Show-Harness | VLM / GUMI(ブラウザ) / 特権オラクルで離散アクションを出す |
| VLM の FT | Show-Harness `train/`（LLaMA-Factory） | 離散アクションを出す小型 VLM（Qwen3.5-2B LoRA） |
| VLA の FT（比較用） | Workshop の GR00T 学習ルート（LeRobot 形式） | 連続した関節指令を出す VLA |
| つなぎ（本リポジトリ） | SO-101 アダプター + 共通軌道レコーダー | トークン → IK → 関節指令、sim / 実機の両方 |

**Workshop を環境として固定し、Show-Harness を接続する。** 同じ成功デモから
「離散アクションを出す VLM」と「連続動作を出す VLA」の両方を学習・比較できるようにする。

## 前提として確認済みの事実

- Show-Harness の interpreter は franka / piper / maniskill / robolab(Isaac Lab, Franka) / libero のみ。**SO-101 用は無い**。
  雛形は `interpreters/robolab_atomic_controller.py`（Isaac Lab, 差分 IK）と `real_atomic_controller.py`（実機の duck type）。
- workshop env の action は **6 関節の絶対位置**（`Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw`）。
  EE 指令は無いので、トークン → EE Δ → **IK（位置 + 手首ピッチ保持。5DoF なので姿勢は全拘束できない）** → 関節目標、をアダプター側で持つ。
- カメラは `ego`（手首, -45°）と `external_D455`（外部）。
- 動く sim 実行構成は `teleop-docker:latest` + workshop v1.0。
- このマシン（AWS L40S）には USB が無い → **実機は必ずローカル PC 側**。VLM 推論だけ AWS に置き SSH トンネルで繋ぐ。
- 論文の sim LoRA は 1 トークン ≒ 20 mm 契約。SO-101 は Franka より小さいので刻み幅は実測で決め直し、契約として固定・記録する。

## 記録設計（判断の頻度と軌道の頻度を分ける）

Show-Harness が判断するのは 1 トークンごと（数百 ms〜数秒）。その 1 トークンを実行している間も、
**sim の制御ループ側で共通の時刻に同期して連続記録する**（初期値 30 Hz、負荷と使う VLA に合わせて調整）。
VLM の応答待ち中も記録は止めない（ロボットは静止指令を保持している、という事実ごと残す）。

| 保存するデータ | 頻度 | Show-Harness 型 VLM の学習 | VLA の学習 |
| --- | --- | --- | --- |
| タスク指示 | エピソード | 入力 | 入力 |
| カメラ画像（ego / external）+ 時刻 | 30 Hz | 判断時点の画像 | 各時点の画像 |
| **観測**した関節角 + グリッパー状態 | 30 Hz | 必要に応じて入力 | 状態入力 |
| **指令**した目標関節角 + グリッパー指令 | 30 Hz | 通常は不要 | **行動の教師ラベル** |
| EE pose（観測 / 目標） | 30 Hz | 移動量検証 | 補助 |
| 離散アクション（トークン）・判断 ID | 判断ごと | **教師ラベル** | 補助情報 |
| 成功判定・人の介入・操作元（oracle/GUMI/VLM）・エピソード ID | エピソード / 判断 | データ選別 | データ選別 |

- **観測関節角と指令関節角は別の列にする**（動作中は一致しない。VLA の教師は指令側）。
- 30 Hz の各フレームに「その時実行中の判断 ID」を持たせ、判断単位と軌道を後から結合できるようにする。
- 正本は 1 つの生ログ（エピソードごとのフレーム列 + 判断列）。そこから 2 形式へ変換する:
  - 判断単位 → Show-Harness rollouts 形式（steps.jsonl + 画像）→ `train/`
  - 連続軌道 → LeRobot 形式（workshop / GR00T と同じ feature 名・関節順）
- **関節の順番・単位（rad/deg/正規化）・グリッパー範囲・カメラ名は sim と実機でそろえ**、`configs/` に契約として 1 か所に書く。

## Phase 0: Workshop 単体の起動確認

- `teleop-docker` + v1.0 で vials-to-rack(DR) env を headless 起動し、reset → step、関節を動かせること、
  ego / external_D455 の画像が取れることを確認する。
- 新しい依存の導入（Show-Harness のクリーンなチェックアウト、LLaMA-Factory 環境）は repro-env-check の後で行う。
- **GO 判定**: env を外部スクリプトから reset / step でき、2 カメラの画像と関節状態を取得できる。

## Phase 1: Show-Harness → SO-101 sim の接続

- `src/so101_showharness/sim/`: workshop env のラッパー（reset / step / 画像 / 関節 / EE pose / 物体 pose / 成功判定）。
- `src/so101_showharness/interpreters/so101_atomic_controller.py`: 移動・把持・開放のトークン → IK → 関節指令。
- `src/so101_showharness/recorder/`: 上記の共通軌道レコーダー（sim / 実機で同じ）。
- `scripts/eval/probe_axes.py`: 各トークンの実移動量を実測 → `configs/primitives_so101.yaml` に符号・刻みを記録。
- **GO 判定**: 全トークンで符号が正しく、1 トークンの移動量のばらつきが ±20% 以内。

## Phase 2: 成功デモの新規収集

1. **特権オラクル**（Show-Harness `scripts/trajectory/real2sim/atomic_tokenizer.py` に SO-101 backend を追加）— 量産の主力。
2. **GUMI**（ブラウザでトークン操作。port 8600 を SSH トンネル）— 回復動作・多様性。
3. zero-shot VLM（任意）— 失敗例の収集。
- DR 版 env で見た目・配置を振る。評価用に**学習に使わない配置**を別に取り分けておく。
- データは `data/` に置き、HF Hub に private で push。

## Phase 3: FT

- VLM: rollouts → `train/data_preparation/rollouts_to_alpaca.py` → LLaMA-Factory 登録 → Qwen3.5-2B LoRA（`qwen3_5_2b_sim.yaml` から派生）。
  学習契約（プロンプト版・画像変換・トークン表記）は Show-Harness `docs/finetuned.md` に従い、変えた点はここに記録。
- VLA（比較）: LeRobot 形式 → workshop の GR00T 学習ルート。

## Phase 4: sim 閉ループ評価

- 未学習の配置で N エピソード → 成功率・動画。公開 `qwen3_5_2b_sim` を zero-shot 基準に、VLM と VLA を同じ評価で比較。

## Phase 5: Sim-to-Real

- `src/so101_showharness/real/so101_robot.py`: Show-Harness の duck type（`get_ee_pose` / `get_gripper_position` /
  `control_gripper` / `update_desired_ee_pose`）を LeRobot `SO101Follower` + 同じ IK で実装 → `real_atomic_controller.py` に載せる。
- 同じアクション定義・同じレコーダーを使う。カメラ位置（D455 位置・手首 -45°）、座標系、1 トークンの移動量を sim と合わせる。
- Z 床・作業空間制限はキャリブで取得する（config を手で書かない）。
- 構成: AWS = vLLM 推論サーバ / ローカル PC = SO-101 + カメラ + harness + レコーダー。
- 必要なら実機で GUMI 収集を少量追加して co-train。
