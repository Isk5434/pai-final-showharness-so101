# 環境記録（repro-env-check, 2026-10-03）

マシン: AWS g6e, NVIDIA L40S 46 GB, driver 580.126.09, 4 vCPU / 31 GB RAM, host に nvcc 無し, uv 0.11.18。

## 3 つの環境（混ぜない）

| env | 場所 | 用途 | 状態 |
| --- | --- | --- | --- |
| harness `.venv` | `/home/ubuntu/so101-showharness-deps/Show-Harness/.venv` | GUMI、runner、変換スクリプト、VLM へのリクエスト | 構築済み。固定 = `configs/env.lock.harness.txt` |
| sim（Isaac コンテナ） | `teleop-docker:latest` の kit python | Workshop env + SO-101 アダプター + オラクル収集 | イメージ無改造。Show-Harness は読み取り専用マウント + PYTHONPATH |
| 学習（LLaMA-Factory） | `/home/ubuntu/so101-showharness-deps/Show-Harness/third_party/LlamaFactory/.venv` | Qwen3.5-2B LoRA | 構築済み（2026-10-03）。固定 = `configs/env.lock.train.txt` |

## 想定版 ↔ 実版

| 層 | 想定（出典） | 実版 | 一致 | リスク |
| --- | --- | --- | --- | --- |
| harness Python | 3.11（`.github/workflows/ci.yml:12`, `scripts/setup.sh` の `PYTHON_VERSION`） | 3.11.15（uv） | ✅ | — |
| numpy | `>=1.23.5,<2`（`requirements/requirements.txt`） | 1.26.4 | ✅ | — |
| opencv | `>=4.10,<4.12` | 4.11.0 | ✅ | — |
| 上流テスト | CI: `pytest tests/` | 45 passed / 1 skipped | ✅ | — |
| sim 側 Python | 未指定（harness は `>=3.10`） | kit python 3.11.13, numpy 1.26.0, scipy 1.15.3, torch 2.7.0+cu128 | ✅ | `zerorpc` 無し → Franka 通信層（session）は import しない設計にする |
| sim から import | — | `core.action_units` / `interpreters.sim_state` / `atomic_tokenizer` / `core.record.images` すべて ok（`scripts/sim/check_harness_import.py`） | ✅ | — |
| LLaMA-Factory | commit `9ce6b66`（`train/scripts/setup_llamafactory.sh`） | `9ce6b66`（0.9.6.dev0） | ✅ | — |
| 学習 Python / torch | 3.12 / torch 2.8.0 cu129 | 3.12 / torch 2.8.0+cu129、GPU 認識 | ✅ | — |
| transformers | 5.7.0（qwen3_5 用。`setup_llamafactory.sh` の `TRANSFORMERS_MAIN`） | 5.7.0 | ✅ | — |
| flash-attn | 最新を `--no-build-isolation` で入れる | 2.8.3.post1（配布 wheel） | ✅ | 公式スクリプトのままだと nvcc 無しでソースビルドに入り `CUDA_HOME` エラーで停止した。`FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE` で配布 wheel を取得し、残り手順は同じ順で手動実行（`outputs/setup_rest.sh` の内容をこの表に記録） |
| fla / tilelang | fla 0.5.1, tilelang 0.1.11 | — | ⚠️ | tilelang は Hopper 向けの回避策。L40S は Ada なので要否を Phase 3 で確認 |
| VRAM | 2B bf16 ≈ 4 GB、bs 4、256 px、cutoff 2048（`train/configs/qwen3_5_2b_sim.yaml`） | 46 GB | ✅ | OOM の心配は小さい |

## 判定

- harness: **GO**（構築・テスト済み）
- sim 側から harness を使う: **GO**（イメージに何も入れない）
- 学習: **GO**（構築済み。flash-attn は配布 wheel、他は公式スクリプトと同じ版）
  - **deepspeed は外した**: import 時に CUDA_HOME（nvcc）を要求して学習開始前に落ちる（`MissingCUDAException`）。
    この機は driver のみで CUDA toolkit が無い。pip の `nvidia-cuda-nvcc-cu12==12.9` は ptxas しか含まず代替にならなかった。
    単 GPU（L40S 1 枚）の LoRA では使わない（公式設定でも deepspeed 行はコメントアウト）。複数 GPU の ZeRO が必要になったら toolkit を入れて戻す
  - **`FLA_TILELANG=0` で学習する**（`scripts/train/train.sh` が設定）: fla の逆伝播が tilelang を選ぶと、tilelang も
    CUDA toolkit を要求して落ちる。tilelang は Hopper(sm90) 向けの回避策で、L40S(Ada) は Triton 経路でよい
  - 煙テスト（10 ステップ、バッチ 2）: 10 秒・損失 6.3 → 0.03〜2.1・LoRA アダプタ保存まで確認（2026-10-03）
