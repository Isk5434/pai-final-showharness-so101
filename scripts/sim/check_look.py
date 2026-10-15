"""固定外観（アーム白・爪黒）と初期配置を目で確かめる: 数シードで外部カメラ・手首カメラの画像を保存。

    bash scripts/sim/run_in_teleop_docker.sh scripts/sim/check_look.py --seeds 0 1 2
"""
import argparse
import os

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--out", default="/workspace/out/check_look")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from so101_showharness.sim.workshop_backend import So101WorkshopBackend, make_env  # noqa: E402

os.makedirs(args.out, exist_ok=True)
env = make_env(seed=args.seeds[0], fixed_look=True, preplaced_vial=False)
be = So101WorkshopBackend(env)
for s in args.seeds:
    be.reset(s)
    ext, wr = be.raw_frames()
    Image.fromarray(np.concatenate([ext, wr], axis=1)).save(f"{args.out}/seed{s}.png")
    print(f"[check_look] seed {s} 保存", flush=True)
env.close()
