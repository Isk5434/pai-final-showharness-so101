"""Check that the Show-Harness modules the SO-101 sim adapter builds on import under the
Isaac kit python, without installing anything into the image. Writes import_check.json."""

import argparse
import importlib
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")  # passed by run_in_teleop_docker.sh
parser.add_argument("--out", default="/workspace/out/import_check")
args = parser.parse_args()

MODULES = [
    "core.action_units",
    "interpreters.sim_state",
    "scripts.trajectory.real2sim.atomic_tokenizer",
    "core.record.images",
    "so101_showharness",
]
result = {"python": sys.version.split()[0], "modules": {}}
for name in MODULES:
    try:
        importlib.import_module(name)
        result["modules"][name] = "ok"
    except Exception as e:  # report every failure, not just the first
        result["modules"][name] = f"{type(e).__name__}: {e}"
from core.action_units import MOVE_ATOMS  # noqa: E402

result["move_atoms"] = sorted(MOVE_ATOMS) if not isinstance(MOVE_ATOMS, dict) else sorted(MOVE_ATOMS.keys())
os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "import_check.json"), "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result), flush=True)
