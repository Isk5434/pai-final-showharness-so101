"""Show-Harness の rollouts_to_alpaca.py を、回転トークン（RT_*）込みで動かす薄いラッパー。

上流（e9867e9）の変換器は MV_* と GRASP/RELEASE しかラベルにせず、RT_* を黙って捨てる。
一方、実行時ランナー（core/runners/mvtoken.py）は「直近の手」の履歴に RT_* を入れ、
コメントで変換器の ``--rotation-tokens`` と揃えると書いているが、その引数は公開版に無い。
学習と実行で入力がずれないよう、ここで上流モジュールの集合を差し替えてから main() を呼ぶ:
  - ラベル: MV_* + RT_* + GRASP/RELEASE（実行時の語彙 = prompts/v5 と同じ）
  - 履歴:   MV_* + RT_*（実行時の HISTORY_ATOMS と同じ）
上流のコードは変更しない。引数は上流の rollouts_to_alpaca.py と同じ。
"""
import os
import sys

SH = os.environ.get("SHOW_HARNESS_ROOT", "/home/ubuntu/so101-showharness-deps/Show-Harness")
sys.path.insert(0, SH)
sys.path.insert(0, os.path.join(SH, "train", "data_preparation"))

from core.action_units import RT_ATOMS  # noqa: E402

import rollouts_to_alpaca as conv  # noqa: E402

conv.MV_TOKENS = set(conv.MV_TOKENS) | set(RT_ATOMS)          # 履歴（Recent moves）に回転も
conv.ACTION_TOKENS = conv.MV_TOKENS | set(conv.GRIPPER_TOKENS)  # ラベルに回転も

if __name__ == "__main__":
    sys.argv[0] = conv.__file__
    conv.main()
