"""lerobot-calibrate と同じ（軽量環境でも動くよう、先に lerobot_lite() を通す）。

    python scripts/real/calibrate.py --robot.type=so101_follower --robot.port=COM3 --robot.id=follower_arm_1
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from so101_showharness.real.robot_io import lerobot_lite  # noqa: E402

lerobot_lite()
from lerobot.scripts.lerobot_calibrate import main  # noqa: E402

main()
