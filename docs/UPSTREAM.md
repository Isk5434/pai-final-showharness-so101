# 上流の固定版

コードは複製せず、ここに書いたコミットを参照する。版を上げたら必ずここを更新する。

| 上流 | URL | 固定コミット | ライセンス | 備考 |
| --- | --- | --- | --- | --- |
| Show-Harness | https://github.com/showlab/Show-Harness | `e9867e971c3a8c5da352e151006cac859923e85e`（2026-10-03 の main 先端） | Apache-2.0 | クリーン clone: `/home/ubuntu/so101-showharness-deps/Show-Harness`。初版で固定しかけた `137d571` からは 29 ファイル +4134 行の差（回転のアクション単位、guided decoding、`atomic_tokenizer.py` 改修、LIBERO 統合）があり、オラクル収集の土台が変わるため先端を採用。既存の `ZeroTeleop/.../Show-Harness` は LIBERO 作業で dirty なので使わない |
| Sim-to-Real SO-101 Workshop | https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop | 公式タグ `v1.0` = `d327be41fbd9cb5e278c4a92cefa47da161dbd44` | Apache-2.0 | 公式 main（`ce807d9`）は lerobot 0.5.1 API に更新済みで、`teleop-docker:latest` 内の lerobot 0.4.3 と噛み合わない（`so_leader` が import できない）→ **v1.0 で固定**。ローカル `/home/ubuntu/sim2real/Sim-to-Real-SO-101-Workshop-v1.0`（Isk5434 fork 経由で同一コミット。`lerobot_eval.py` に過去実験の未 commit 変更あり） |

## 参照データ（比較用, 学習には直接使わない予定）

- https://huggingface.co/sreetz-nv — 同じ workshop 環境の SO-101 データ・GR00T/pi05 checkpoint 群
  （例: `so101_teleop_vials_rack_left`, `so101_newton_dr500_20260923`, `so101_real_manual`）。
