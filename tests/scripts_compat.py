"""Import Show-Harness's token contract when it is on the path; fall back to a frozen copy."""
try:
    from scripts.trajectory.real2sim.atomic_tokenizer import MOVE_DIRS  # noqa: F401
except ImportError:  # Show-Harness not on PYTHONPATH
    MOVE_DIRS = {
        "MV_FWD": [1.0, 0.0, 0.0], "MV_BACK": [-1.0, 0.0, 0.0],
        "MV_LEFT": [0.0, -1.0, 0.0], "MV_RIGHT": [0.0, 1.0, 0.0],
        "MV_UP": [0.0, 0.0, 1.0], "MV_DOWN": [0.0, 0.0, -1.0],
    }
