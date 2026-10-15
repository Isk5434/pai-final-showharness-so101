import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
_sh = os.environ.get("SHOW_HARNESS_ROOT", "/home/ubuntu/so101-showharness-deps/Show-Harness")
if Path(_sh).is_dir():
    sys.path.insert(0, _sh)
