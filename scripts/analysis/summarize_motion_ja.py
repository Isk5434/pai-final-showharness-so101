"""verify_motion.py の motion_report.json から日本語の要約（motion_report_ja.md）を作る。

    python scripts/analysis/summarize_motion_ja.py outputs/verify_motion_p47_v3
"""
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def ok(flag):
    return "✅ 合格" if flag else "❌ 不合格"


def fmt(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, (list, tuple)):
        return "〜".join(fmt(x, nd) for x in v)
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def main(out_dir: Path) -> Path:
    r = json.loads((out_dir / "motion_report.json").read_text())
    c = yaml.safe_load((REPO / "configs/motion_criteria.yaml").read_text())
    v = r["verdict"]
    a = r["args"]
    lines = [
        f"# 動作検証の結果（{out_dir.name}）",
        "",
        (f"- 作業姿勢: 手先ピッチ {a['pitch_deg']}° / 手首ロール固定 {a.get('roll_deg')}°（指は常に水平に閉じる）"
         if a.get("orient", "yaw") == "roll" else
         f"- 作業姿勢: 手先ピッチ {a['pitch_deg']}° / 閉じ方向の方位 {a['yaw_deg']}°")
        + f"、ホーム（ワールド座標）{a['home_world']}",
        "- 判定基準: `configs/motion_criteria.yaml`（結果を見る前に固定）",
        "- 閉ループ動画: 同じフォルダの `*.mp4`（左=外部カメラ、右=手首カメラ、下=制御状態）",
        "",
    ]
    if "grid" in v:
        g = v["grid"]
        lines += [
            f"## 格子検証 {ok(g['PASS'])}",
            "",
            "| 項目 | 結果 | 基準 |",
            "| --- | --- | --- |",
            f"| 始点に到達 | {g['starts_reached']}（障害物内で除外 {g.get('skipped_starts', 0)}） | — |",
            f"| 実行トークン / 拒否 | {g['executed']} / {g['blocked']} | — |",
            f"| 空間中での接触・未収束 | {g.get('contacts_in_free_space', '—')} | 0 |",
            f"| 1トークンの移動量 [mm] | {fmt(g['travel_mm_min_max'])} | {fmt(c['travel_mm'], 1)} |",
            f"| 軸からのずれ 最大 [mm] | {fmt(g['off_axis_mm_max'])} | ≤ {c['off_axis_mm_max']} |",
            f"| 行き過ぎ 最大 [mm] | {fmt(g['overshoot_mm_max'])} | ≤ {c['overshoot_mm_max']} |",
            f"| 移動中のピッチ変化 最大 [°] | {fmt(g['dpitch_deg_absmax'])} | ≤ {c['orientation_deg_max']} |",
            f"| 移動中の{'手首ロール' if a.get('orient', 'yaw') == 'roll' else 'ヨー'}変化 最大 [°] | {fmt(g['dyaw_deg_absmax'])} | ≤ {c['orientation_deg_max']} |",
            f"| 指の閉じる方向の傾き 最大 [°] | {fmt(g.get('closing_tilt_deg_absmax'))} | （参考） |",
            "",
        ]
    if "walk" in v:
        w = v["walk"]
        lines += [
            f"## ランダム往復 {ok(w['PASS'])}",
            "",
            f"- {w['tokens']} 手進んでから逆順に戻る（拒否 {w['refused_attempts']} 回 {w.get('refusal_reasons', {})}、"
            f"障害物を避けて選び直し {w.get('avoided_obstacle_tokens', 0)} 回、最大到達距離 {fmt(w['max_excursion_mm'], 1)} mm）",
            f"- 戻り誤差: 位置 {fmt(w['closure_error_mm'], 3)} mm（基準 ≤ {c['walk_closure_mm_max']}）、"
            f"ピッチ {fmt(w['closure_dpitch_deg'], 3)}°、ヨー {fmt(w['closure_dyaw_deg'], 3)}°",
            "",
        ]
    if "rotate" in v:
        ro = v["rotate"]
        lines += [
            f"## 回転トークン {ok(ro['PASS'])}",
            "",
            f"- 実行 {ro['executed']} / 拒否 {ro['blocked']}",
            f"- 1トークンの回転角: {fmt(ro['angle_deg_min_max'])}°（基準 {fmt(c['rot_step_deg'], 1)}°）",
            f"- 回転中の手先位置のずれ 最大: {fmt(ro['tcp_drift_mm_max'])} mm（基準 ≤ {c['rot_tcp_drift_mm_max']}）",
            "",
        ]
    if "grasp" in v:
        gr = v["grasp"]
        lines += [
            f"## 把持 {ok(gr['PASS'])}",
            "",
            f"- 把持目標に対する手先の誤差 [mm]: {fmt(gr.get('tcp_minus_grasp_target_mm', gr.get('tcp_minus_vial_mm')), 1)}",
            f"- 閉じたときの指の幅: {fmt(gr.get('closed_width_mm'), 1)} mm（0 付近なら何も挟めていない）",
            f"- 持ち上げ中も把持判定が続いた: {'はい' if gr['grasped_while_lifted'] else 'いいえ'}（持ち上がったステップ {gr['lifted_steps']}）",
            f"- 手の中でのバイアルのずれ 最大: {fmt(gr['vial_in_hand_shift_mm_max'], 1)} mm（基準 ≤ {c['grasp_slip_mm_max']}）",
            f"- 手の中でのバイアルの回転 最大: {fmt(gr['vial_in_hand_turn_deg_max'], 1)}°（基準 ≤ {c['grasp_turn_deg_max']}）",
            f"- 離したあと把持判定が消えた: {'はい' if gr['released'] else 'いいえ'}",
            "",
        ]
    out = out_dir / "motion_report_ja.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(main(Path(sys.argv[1])))
