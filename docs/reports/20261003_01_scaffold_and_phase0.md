# レポート 01: リポジトリ骨組み + Phase 0（Workshop 単体起動）

日付: 2026-10-03 / commit: `57276ee` → `ed2e39f`

## 何を実装したか

- 開発フォルダ `/home/ubuntu/so101-showharness` と private リポジトリ `Isk5434/so101-showharness` を作成した。
  `.gitignore` でデータ・重み・出力・秘密情報を除外し、コードだけが入るようにした。
- 上流は複製せず、固定するコミットを `docs/UPSTREAM.md` に記録した（Show-Harness `137d571`、Workshop は公式 `v1.0`）。
- 判断（トークン）と 30Hz の軌道を分けて記録する設計を `docs/PLAN.md` に書いた。
- `scripts/sim/run_in_teleop_docker.sh`: このリポジトリのスクリプトを teleop-docker 内で headless 実行するためのラッパー。
- `scripts/sim/smoke_workshop_env.py`: env の reset / step、関節の追従、2 カメラ画像の取得を確認するスクリプト。
  → **GO**（詳細は `docs/PHASE0_RESULT.md`）。

## 詰まった点

| 症状 | 原因 | 解決 |
| --- | --- | --- |
| `bash: python: command not found` | コンテナに `python` という名前のコマンドが無い。Isaac の kit python は `python3` | ラッパーで `python3` を使う |
| rc=0 なのに結果の `print` がログに出ない | Isaac の終了処理で stdout が flush されないまま消える | 結果は `summary.json` に書き、ログに頼らない |
| （回避済み）公式 main を使うと import で落ちる | 公式 main は lerobot 0.5.1 API、docker 内は 0.4.3 | 公式タグ `v1.0` に固定 |

## 改善点・次への宿題

- DR env の 1 エピソードが 5 秒しかない → Phase 1 のラッパーで延長する（トークン収集には 25 秒以上必要）。
- 関節の単位が sim は rad、Workshop の LeRobot 側は度 + グリッパー 0–100 → 単位の契約を `configs/` に 1 か所で定める。
- 手先位置（`ee_frame_state`）の基準座標系と四元数の並び順は未確認 → 移動量の実測で確定させる。
- 既存の Show-Harness チェックアウトは LIBERO 作業で変更が残っている → Phase 1 ではクリーンな clone を使う（repro-env-check の後）。
