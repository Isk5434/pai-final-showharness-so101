# 実機 SO-101 で動かす（Sim to Real）

推論は AWS（GPU）、実機とカメラはローカル PC。間は SSH トンネルだけでつなぎ、サーバはインターネットに開かない。
構成は 2026-07 に GR00T で実機まで通したもの（`~/sim2real/LOCAL_REAL_ROBOT_RUNBOOK.md`）と同じ。

```
[ローカル PC] SO-101(USB) + front/wrist カメラ        [AWS]
  run_dm05.py        ──状態・画像──▶  :8780 dm05_server.py（DM0.5）
  run_showharness.py ──プロンプト・画像──▶ :8770 gpt_operator_server.py（GPT-6 Astra、Codex のログイン）
                     ◀──行動（関節角 / トークン）──
```

| 方式 | 何が決める | 1 往復で返るもの | ローカルで動くもの |
| --- | --- | --- | --- |
| DM0.5 | 学習した VLA | 関節角 15 手（0.5 秒分） | 30 Hz で順に送るだけ |
| Show-Harness | GPT-6 Astra | 動作トークン 1 つ | sim と同じ実行部（IK・2 cm の移動・接触で中止） |

## 0. 一度だけ：ローカル PC の準備（torch なし、約 300MB）

AWS で `bash scripts/real/make_local_bundle.sh` → `outputs/real_bundle.tgz`（約 400KB。実機クライアント、必要な部品、
Show-Harness の atomic_tokenizer.py だけ、参照画像、印刷物）。ローカルでは git も Show-Harness 全体も要らない。

```powershell
scp ri:/home/ubuntu/so101-showharness/outputs/real_bundle.tgz . ; tar -xzf real_bundle.tgz ; cd so101_real\so101-showharness
uv venv --python 3.10 .venv
uv pip install --python .venv --no-deps lerobot==0.4.1
uv pip install --python .venv -r scripts\real\requirements-local.txt
.venv\Scripts\python scripts\real\preflight.py
.venv\Scripts\python scripts\real\calibrate.py --robot.type=so101_follower --robot.port=COM3 --robot.id=follower_arm_1
```
lerobot は依存なしで入れ、腕の制御に要るものだけ足す。lerobot 0.4.1 は腕の制御の経路で torch・accelerate・datasets を
import するが使わないので、`robot_io.lerobot_lite()` が名前だけの代用品を差し込む（本物が入っていればそちらを使う）。
`lerobot-calibrate` の代わりに `scripts/real/calibrate.py`（同じ引数）を使う。

## 1. カメラを sim に合わせる（DM0.5 では特に大事）

印刷物は束の `print/`（必ず「実際のサイズ」で印刷し、定規で 50 mm・25 mm を確認）。

1. レンズを測る（1 回だけ）：`checkerboard.pdf` を板に貼り、いろいろな角度で 12 枚以上
   `python scripts/real/camera_calib.py lens --front 0`（c で撮る、k で計算）→ `camera_front_lens.json`
2. 位置を合わせる：`aruco_sheet.pdf` の十字を基部（肩の回転軸の真下）から前へ 30.0 cm・右へ 8.0 cm、FAR の辺を奥に置く
   `python scripts/real/camera_calib.py align --front 0` → 画面の「あと何 cm」に従ってカメラを動かし、位置が 0.5 cm 以内（緑）で s
   → `camera_fix.json`。向きの残りとレンズの違いはソフトで消すので、位置を優先する。
3. 実行時に `--camera_fix camera_fix.json` を付ける（front の画像を sim のカメラで撮った形に直して送る）。
4. 目で確かめるなら `camera_check.py`（sim の参照画像を重ねる）。

合成試験（`tests/real/check_camera_fix.py`）：姿勢の推定誤差 0.1 cm・0.1°、向き 4°・レンズ −15%・歪みのあるカメラを補正すると
マーカーのずれ 30 px → 0.7 px。

## 2. AWS でサーバを起動

```bash
# DM0.5（学習名と学習データ。学習データは推論時の正規化に使う）
bash scripts/real/aws_serve.sh dm05 dm05_near_mix57 outputs/lerobot/so101_near_mix57
# Show-Harness（Codex のクレジットが必要）
bash scripts/real/aws_serve.sh showharness
```
どちらも 127.0.0.1 だけで待ち受ける。sim の評価・学習と GPU を取り合うので、実機の間は止めておくのが安全。

## 3. ローカルからトンネルを張る（別の端末で張りっぱなし）

```bash
ssh -N -L 8780:127.0.0.1:8780 -L 8770:127.0.0.1:8770 ri
```

## 4. 実機で動かす（必ず dry run → 本番の順）

**DM0.5**
```bash
# 腕は動かさず、推論と記録だけ（指令が制限に当たる回数や往復時間を見る）
.venv\\Scripts\\python scripts/real/run_dm05.py --port <PORT> --front 0 --wrist 2 --position near --dry_run
# 本番
.venv\\Scripts\\python scripts/real/run_dm05.py --port <PORT> --front 0 --wrist 2 --position near --episodes 5
```

**Show-Harness（GPT-6 Astra）**
```bash
# 1 手ごとに Enter で承認してから動かす（s で飛ばす、q で終了）
.venv\\Scripts\\python scripts/real/run_showharness.py --port <PORT> --front 0 --wrist 2 --position near --confirm
```

記録は `outputs/real_dm05/`・`outputs/real_showharness/` に残る（関節・指令・推論時間の jsonl と画像）。

## 安全装置

- すべての指令は `SafeArm` を通る：関節の可動範囲で切り、1 回の指令で動く量を `--max_step_rad`（既定 0.08 rad ≈ 4.6°）に制限。
  さらに LeRobot 側の `max_relative_target=10` も効く。制限に当たった回数は各話の終わりに出る。
- Ctrl-C でその場に止め、開始姿勢へゆっくり（4 秒）戻る。
- Show-Harness は sim と同じ実行部なので、作業範囲の外・指が机より下に行く手は実行前に拒否され、押し付け（関節の追従ずれ＋進まない）を検知すると途中で止まる。
- 最初は E-stop（電源を抜ける位置）の近くで、周りを空けてから。

## 知っておくこと

- **DM0.5 は 0.5 秒動いて、推論の間（AWS 上で約 1.2 秒＋往復）止まる**。sim の評価も推論を待つ同期実行だったので同じ動き方。
  推論中は最後の指令の位置で保持する。
- **画像は JPEG（品質 95）で送る**。学習は無圧縮なので、気になるなら `--jpeg 0`（PNG、重くなる）。
- **Show-Harness の「画像のどちらへ動くか」の表**は、v150 の収集で GPT が見たものと同じになるよう、sim のカメラ値（`configs/real_setup.json`）
  から計算する。sim の手首カメラは姿勢が更新されず基部に固定されていたので、その固定値をそのまま使っている（`wrist.mode = v150_fixed`）。
  sim で 3 シード × 6 姿勢、元の計算と完全に一致することを確認済み（`tests/sim/check_direction_table.py`）。
- **指先の高さ**は `table_top_task_z_m`（マット上面の高さ、基部が原点、sim は 3.5 cm）から出す。実機の机が違えば実測して書き換える。
- プロンプトの文面は sim の収集と同じ（"in simulation" も含む）。変えると収集時と条件がずれるため。

## 実機なしの試験（AWS だけで通せる）

```bash
E=outputs/near200/w0/gpt-6-astra/traj/ep_100001/video
# DM0.5：仮のサーバ（別ポート）＋模擬アーム＋ sim の動画をカメラ代わりに
PORT=8790 bash scripts/real/aws_serve.sh dm05 dm05_success50 outputs/lerobot/so101_gpt_success50 &
<dm05 env の python> scripts/real/run_dm05.py --mock --server http://127.0.0.1:8790 --front $E/external.mp4 --wrist $E/wrist.mp4 --max_s 6
# Show-Harness：GPT の代わりに決まったトークン列を返すサーバ
python3 tests/real/stub_operator.py --port 8799 --tokens MV_DOWN,GRASP,MV_UP,DONE &
SHOW_HARNESS_ROOT=<Show-Harness> <python> scripts/real/run_showharness.py --mock --server http://127.0.0.1:8799 --front $E/external.mp4 --wrist $E/wrist.mp4
```
2026-10-20：どちらも通しで動作を確認（DM0.5 は 30 手を送信・記録、Show-Harness は 8 手を実行して DONE で終了）。
