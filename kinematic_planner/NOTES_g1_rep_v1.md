# g1_rep_v1 — design + the fidelity fix

`g1_rep_v1` is a HumanML3D-style kinematic motion representation for the G1,
built from Bones-SEED GT data, used as the generation target of a text→motion
MDM. Decode is **root + 29 qpos angles → G1 forward kinematics** (executable on
the real robot). `ric_data` (joint positions) is a redundant, exact target.

## Per-frame layout (142-D)

| block | dims | meaning |
|---|---|---|
| rot_velocity | 1 | per-frame root yaw delta Δψ (canonical) |
| lin_velocity | 2 | root planar velocity in the heading frame (dx, dz) |
| root_height | 1 | pelvis world height (Y, kimodo Y-up) |
| **root_orient_6d** | **6** | **full root orientation, heading-removed (the fix)** |
| joint_angles | 29 | G1 actuated qpos angles (radians, qpos order) |
| ric_data | 99 | joints 1..33 positions, heading-removed, root-relative XZ + world Y |
| foot_contacts | 4 | [L_ankle, L_toe, R_ankle, R_toe] |

The rep is **heading-canonical** (frame 0 → origin, zero heading) and
**heading-invariant** (a global Y-rotation of the input leaves features
unchanged).

## What was wrong (the first 136-D version)

The first version used a **yaw-only root**: it stored only the root *delta yaw*
(+ height + planar velocity), with no pelvis pitch/roll. So the executable
decode (`root + qpos → FK`) forced the pelvis upright and approximated body
lean with the waist joints only.

Round-trip error vs the mocap reference, `decode(encode(M))` joint position max:

| motion | pelvis tilt | [B] error (yaw-only) |
|---|---:|---:|
| warm_up_neck | 6.7° | 0.092 m |
| sit_on_heels | 26.3° | **0.207 m** |
| praying | 9.3° | 0.179 m |
| shadow_boxing | 17.5° | 0.235 m |

The error tracked **pelvis tilt** — e.g. `sit_on_heels` (26° tilt) lost ~18 cm
purely from dropping the pelvis orientation. That was the dominant, *fixable*
error.

## What was fixed

Added the **full root orientation** as a heading-removed 6D rotation
(`root_orient_6d`), recovered from the unified NPZ's `local_rot_mats[:, 0]`.
Decode reconstructs `R_root = R_y(heading) @ from6d(root_orient_6d)`, so the
pelvis pitch/roll/yaw come back exactly. Rep grew 136 → 142-D.

After the fix:

| metric | result |
|---|---|
| `ric` position round-trip (encode→decode_positions) | **0.000000 m — exact** |
| executable decode mean error vs mocap | 0.001–0.033 m (1–3 cm) |
| rep vs **GT executable pose** (qpos→FK, what the G1 can do) | **0.000000 m — exact** |

So the rep is **lossless** both for joint positions and for the real-robot-
executable pose.

## The residual that is NOT rep error

`decode vs mocap` still shows ~0.18–0.22 m max on extreme arm motions
(shadow-boxing, praying), concentrated on the wrist/hand-tip joints. This is
**not** the rep's error — it's the intrinsic gap between full-rotation mocap and
the G1's **1-DOF hardware**: the robot physically cannot roll its hands
(`*_hand_roll` are non-actuated) or twist wrists off-axis, so any
robot-executable representation (including the GT qpos itself) carries this gap.
Measured against the executable target it is 0 m (table above). Mean over all
joints/frames is ~1 cm; only hand tips at peak extension differ, which a
downstream whole-body tracker does not need exactly.

## v2 options (not needed for v1)

- Store 6D *joint* rotations instead of 1-DOF angles → eliminates the hand/wrist
  gap, but the output is no longer directly G1-executable qpos (would need a
  1-DOF projection at execution, re-introducing the same gap). Kept 1-DOF
  angles on purpose: the planner's output **is** robot-executable.
