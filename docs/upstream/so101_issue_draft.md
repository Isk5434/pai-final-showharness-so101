# Proposal: SO-101 (LeRobot) support, including RT_* rotation units on a 5-DoF arm

Hi, and thanks for releasing Show-Harness. I have been running it on the **SO-101** (the LeRobot
low-cost arm) in simulation and would like to contribute that support upstream, if it fits your plans.

## What exists

Built on top of `scripts/trajectory/real2sim/atomic_tokenizer.py` (`AtomicSimEnv` / `AtomicExec` /
`RolloutWriter`), unchanged vocabulary and prompt v5:

- **SO-101 kinematics + IK** (numpy only). The chain is read from the official Isaac Sim SO-101
  USD (NVIDIA *Sim-to-Real SO-101 Workshop* v1.0, Apache-2.0). It holds TCP position, in-plane tool
  pitch and either the wrist-roll angle or the closing-axis yaw.
- **RT_* on a 5-DoF arm.** As far as I can tell, RT_* units are currently executed only in the LIBERO path.
  The SO-101 can't realise arbitrary world-axis rotations, so the interpreter maps:
  - `RT_PITCH_FWD/BACK` → in-plane tool pitch (fingertips tilt away from or toward the base)
  - `RT_YAW_CW/CCW` → closing-axis yaw through Wrist_Roll
  - `RT_ROLL_*` → refused (not realisable on this arm)

  The TCP is held during a rotation.
- **Executor for a low-gain arm.** Every token is checked before it runs (end/mid-point IK feasibility,
  joint limits, hand–table clearance) and refused whole if infeasible. It then servos to a fixed target
  and returns only when the arm is at rest. Measured on the sim: one MV token = 19.8–20.1 mm,
  ≤ 0.8 mm off-axis, ≤ 0.7 mm overshoot. An 80-token random walk returns within 0.1 mm. TCP drift
  during RT_* is ≤ 1.5 mm.
- **Isaac Lab backend** for the Workshop "vials → rack" task (`Lerobot-So101-Teleop-Vials-To-Rack-DR`).
  It covers the camera contract (external D455 + wrist), contact sensing on the distal links, and
  GUMI-format recording.
- **Demonstrations.**
  - A scripted oracle using only these tokens: 68/100 episodes accepted by the sim's placement check.
  - Zero-shot frontier-VLM operation with a per-step camera-derived direction table: this works but
    with a low success rate so far, and a larger run is in progress.

Real-arm support (LeRobot follower `get_observation`/`send_action` behind the same controller)
exists too. It is only tested against a mock so far, so I'd keep it out of a first PR and add it
after hardware validation.

## Proposed first PR (sim only)

- `scripts/trajectory/real2sim/so101/`: kinematics, executor, Workshop backend, chain JSON
- `configs/primitives_so101.yaml` and `configs/robot_so101_workshop.yaml`
- a "SO-101 (Isaac Lab Workshop)" section in `docs/simulators.md`
- hardware-free unit tests: FK/IK, token feasibility and refusal, RT_* mapping on a kinematic stand-in arm

No changes to `core/` or to the vocabulary.

## Questions

1. Is SO-101 support something you'd accept? If so, is this layout right, or would you prefer it under
   `interpreters/` / `core/so101/` like Piper?
2. Is refusing `RT_ROLL_*` on a 5-DoF arm OK, or should such embodiments advertise a reduced
   RT vocabulary to the prompt instead?
3. The Workshop env runs in its own Isaac Sim 5.1 / Isaac Lab Docker image. Is a documented
   "run inside that image" workflow acceptable, as with RoboLab?

Happy to adjust to whatever fits the repo best.
