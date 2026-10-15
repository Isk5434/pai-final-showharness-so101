Implements the sim-only part of #5. This adds the SO-101 (LeRobot, 5-DoF) as an embodiment on the NVIDIA Sim-to-Real SO-101 Workshop v1.0 (Isaac Lab). It follows the layout agreed there:

- execution and kinematics in `interpreters/`
- the Workshop backend in `scripts/trajectory/real2sim/`
- configs in `configs/`
- no changes to `core/` or the shared vocabulary

## What's in it

| Path | What |
| --- | --- |
| `interpreters/so101_kinematics.py` | numpy FK/IK on the USD's own joint chain: TCP position + in-plane tool pitch (+ optional closing-axis yaw) |
| `interpreters/so101_atomic_controller.py` | `So101ArmController` (I/O-agnostic motion core) and `So101AtomicExec` (token executor). Imports no simulator and nothing from `scripts/` |
| `configs/primitives_so101.yaml` | MV_* vectors (same contract as the tokenizer), rotation units, measured numbers |
| `configs/robot_so101_workshop.yaml` | task, scene variants, camera contract |
| `scripts/trajectory/real2sim/backends/so101_workshop.py` | `AtomicSimEnv` backend: joints, cameras, measured contact, success; registered as `so101_workshop` |
| `scripts/trajectory/real2sim/so101_workshop/` | `run_in_docker.sh`, `check_views.py`, `verify_motion.py`, USD extraction scripts, privileged `scene.py` |
| `tests/test_so101_*.py` | 21 hardware- and simulator-free tests |
| `docs/simulators.md` | SO-101 section |

## RT_* on 5 DoF

The vocabulary is unchanged, and the interpreter states what the arm can realise:

- `RT_PITCH_*`: tool pitch in the arm plane, with the TCP held. The axis is the arm-plane normal, which equals the shared world axis only while the arm points along +X.
- `RT_YAW_*`: the hand turned about the vertical through Wrist_Roll, with the TCP held.
- `RT_ROLL_*`: **refused deterministically** as `unsupported_on_5dof`. `check()` and `run()` agree, the arm never moves, and a test covers it.

The backend does not claim `supports_rotation`, since that contract means an arbitrary world-axis rotation. Tokens run through `So101AtomicExec` instead.

## Checks

- **Whole-token refusal.** Every token is checked before it runs: end-point and mid-point IK, joint-limit margin, the whole hand's collision hull above the table, and the workspace box. `check()` answers without moving, so a generator never records a refused token.
- **`verify_motion.py` (in the Workshop image, 2026-10-08): PASS.**
  - 96 MV tokens from 8 start poses: 19.80–20.12 mm each, ≤ 0.76 mm off-axis, ≤ 0.65 mm overshoot, 0 contacts.
  - An 80-token random walk and its reverse close within 0.095 mm.
  - RT tokens turn 9.90–10.12° with ≤ 1.50 mm of TCP drift.
- **`check_views.py`: PASS.** The camera contract is measured, not assumed:
  - agentview: by TCP projection.
  - wrist: on the rendered frames. Each token is executed and the scene shift measured; the fingers are found as the pixels that ride with the camera.
  - Every point goes through `rotate_and_flip` itself.
- **FK** matches PhysX link poses (40 Workshop samples, in `tests/fixtures/`).
- `pytest tests/`, `ruff check .` and `scripts/leakcheck.sh` pass locally.

## Points to review

1. **Camera contract.** Raw, neither Workshop camera matches the shared prompt geometry. I used proper rotations only: agentview 90°, wrist 180°. As a result the agentview is portrait, so its 256 letterbox bars are left/right instead of top/bottom. Is that acceptable, or would you prefer a crop?
2. **Success.** The Workshop's own `vial_placed` is lenient: it latches at release and doesn't check the hole or which vial it was. I documented it rather than replacing it. Happy to add a stricter predicate if you'd like one for evaluation.
3. **Not included, as follow-ups:**
   - the real-arm backend (LeRobot follower behind the same controller), after hardware validation
   - a scripted vials → rack demonstration generator

Running anything in sim requires the Workshop image. `docs/simulators.md` describes setup, and the unit tests need neither the image nor hardware.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
