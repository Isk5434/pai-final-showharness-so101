# Phase 0 結果（2026-10-03）: Workshop v1.0 単体起動 → GO

実行: `bash scripts/sim/run_in_teleop_docker.sh scripts/sim/smoke_workshop_env.py`
（teleop-docker:latest, Isaac Sim 5.1, isaaclab 0.54.2, lerobot 0.4.3, L40S, headless。起動〜終了 約1分）

| 確認項目 | 結果 |
| --- | --- |
| env | `Lerobot-So101-Teleop-Vials-To-Rack-DR` を外部スクリプトから reset / step できた |
| 制御周期 | physics 120 Hz × decimation 2 → **env.step = 60 Hz**（30 Hz 記録 = 2 step に 1 回） |
| action | `Box(1, 6)` = 6 関節の**絶対位置 [rad]**、順番 `Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw` |
| 保持 | reset 姿勢を 2 s 指令し続けたときの最大ずれ 0.0135 rad（重力で沈む分） |
| 追従 | `Rotation` を +0.3 rad → 2 s で誤差 1.3e-6 rad、他関節の変化 6e-7 rad |
| 観測 | `policy.joint_pos_obs`(6) / `joint_pos_rel`(6) / `ee_frame_state`(7 = 位置 + 四元数) |
| カメラ | `rgb_ego`, `rgb_external_D455` とも 640×480 uint8（depth / instance seg もあり） |
| 成功判定 | `subtask_terms.vial_grasped` / `vial_placed` が観測に出ている → 収集時の成功ラベルにそのまま使える |

## Phase 1 に効く発見

- **DR env の `episode_length_s` は 5 s**（Eval 版でも 7.5 s）。トークン 1 個 ≈ 0.5 s として 50 手 ≈ 25 s かかるので、
  ラッパー側で延長するか time_out を外す必要がある。
- sim の関節は rad、workshop の LeRobot/GR00T 側は度 + グリッパー 0–100（`utils/lerobot_interface.py` の `get_raw_actions_from_radians`）。
  LeRobot 形式へ変換するときは workshop の既存データ（sreetz-nv）と同じ単位にそろえる → 契約として `configs/` に固定する。
- `ee_frame_state` の基準座標系と四元数の並び（xyzw か wxyz か）は未確認。IK と移動量の実測（probe_axes）で確定させる。
