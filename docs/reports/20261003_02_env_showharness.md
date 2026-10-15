# レポート 02: Show-Harness の環境構築（repro-env-check）

日付: 2026-10-03

## 何を実装したか

- repro-env-check で Show-Harness の想定版と実機の版を突き合わせた（結果は `docs/ENV.md`）。
- Show-Harness を main 先端 `e9867e9` でクリーンに clone し、harness 用 `.venv` を作った。
  上流テストは 45 passed / 1 skipped。版は `configs/env.lock.harness.txt` に固定した。
- `scripts/sim/run_in_teleop_docker.sh` に Show-Harness の読み取り専用マウントと PYTHONPATH を追加した。
- `scripts/sim/check_harness_import.py`: Isaac コンテナの kit python から harness のモジュールを import できるか確認するスクリプト。
  → 全モジュール ok。アクション語彙は `MV_FWD/BACK/LEFT/RIGHT/UP/DOWN`。

## 詰まった点

| 症状 | 原因 | 解決 |
| --- | --- | --- |
| UPSTREAM.md に「先端との差は README だけ」と誤記していた | `git log` だけ見て `git diff --stat` を見ていなかった | 実際は 29 ファイル +4134 行（オラクルの `atomic_tokenizer.py` 改修を含む）。固定先を `e9867e9` に変えて訂正 |
| entrypoint を外して `docker run` するとコンテナ内で numpy が見つからない | kit python の環境設定は entrypoint 側で行われる | 版の確認も実行も必ず entrypoint 経由（`run_in_teleop_docker.sh`）で行う |

## 改善点・次への宿題

- Isaac イメージに pip で足さずに済んだので、イメージは無改造のまま使える。今後も sim 側で必要なものは PYTHONPATH で足す方針にする。
- 学習環境は flash-attn のソースビルドが詰まりどころになりそう（nvcc 無し・4 コア）→ Phase 3 の最初に wheel を確認する。
- 次: Phase 1 本体（SO-101 sim ラッパー + IK + atomic controller + 移動量の実測）。
