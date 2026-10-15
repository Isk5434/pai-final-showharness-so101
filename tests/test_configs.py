from pathlib import Path

import yaml

from scripts_compat import MOVE_DIRS

CFG = Path(__file__).resolve().parents[1] / "configs"


def test_move_vectors_match_show_harness_contract():
    prim = yaml.safe_load((CFG / "primitives_so101.yaml").read_text())
    for tok, vec in MOVE_DIRS.items():
        assert prim["move_vectors"][tok] == [float(x) for x in vec], tok
