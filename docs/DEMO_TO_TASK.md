# Retargeted demo → simulation task

This pipeline turns one retargeted do-as-i-do RGB-D demo into a residual-RL task. It runs in about 3 minutes on one RTX 4080.

- **Input:** an episode of the HF dataset `ytung/pipette-tasks`.
- **Output:** a residual-RL task in this repo's format: scene, kinematic reference, contact labels, valid starts and press prior. It also writes checks and training configs.
- **Robot placement:** the robot stands square to the table.
- **Holders:** the stand and the rack are placed at ground-truth poses.
- **Tube:** it goes back into the hole it was picked from.

```bash
.venv-rl-head/bin/python -I tracker/demo_to_task.py pipette_depth_02_tapir d02_straight      # all steps
.venv-rl-head/bin/python -I tracker/demo_to_task.py pipette_depth_02_tapir d02_straight --steps check   # one step
.venv-rl-head/bin/python -I tracker/demo_to_task.py <episode> <name> --force -- --fwd 0.35 --hole r0c2  # rebuild with builder options
```

| Step | Script | Output |
|---|---|---|
| download | `demo_to_task.py` | `~/Projects/datasets/pipette-tasks/<episode>`: only the files used below, about 30 MB |
| build | `tracker/build_demo_task.py` | `tasks/task_<name>_noprior/`: `scene.xml`, `reference.npz`, `task.json`, `build_report.json`, `wrist_targets.npz` |
| labels | `tracker/build_kin_contact_ref.py` | `contact_ref.npz`, built by distance on the reference (there is no physics demo to label) |
| audit | `rl/audit_starts.py` | `valid_starts.npy` |
| press | `tracker/add_press_prior.py` | `tasks/task_<name>/`: thumb press force in the press window, plus the eval contact labels |
| check | `tracker/check_demo_task.py` | `tasks/task_<name>/check/`: `check.json`, `montage.png`, `overlay_start.png`, `overlay_press.png` |
| config | `demo_to_task.py` | `tracker/configs/kin_<name>_s5.json` (1 GPU) and `kin_<name>_s5_g3.json` (3 GPUs) |

The configs reuse KA's config and reward unchanged. The only change is the hole pairs: every rack hole goes back into itself.

## What the raw retargeted motion gets wrong, and the fix

Every number below is from `pipette_depth_02_tapir`, and every number appears in `build_report.json`.

| # | Raw motion | Measured | Fix (stage of `build_demo_task.py`) |
|---|---|---|---|
| 1 | Everything is in the camera frame. The episode has no table, holder or robot frame, and each hand's retargeting uses its own "world". | The camera is 63 cm above the table, pitched 47° down, and turned 5.9° against the far table edge. | **1 frames.** Fit the table plane by RANSAC on the first depth frame. Set x perpendicular to the far table edge. The robot base gets yaw 180°, square to the table, at the template's distance (base front 12 cm from the table edge). The demo origin is placed `--fwd` (0.30 m) ahead of the base. |
| 2 | Holder poses are not reconstructed. The object poses inside the holders are noisy. | The tube reads 13.8° tilted in the rack, and its centre is 1.5 cm too low. Start and end disagree by 3–4 mm. | **2 holders.** The stand moves rigidly with its seated pipette: demo xy and spin, template seat height, vertical. The rack is translated so that hole `--hole` lies under the demo tube's rim centre. The tube goes back into the same hole. The env reads `task.json` `ref_holes`. |
| 3 | The reconstruction's tube mesh is not the sim tube. | Ø 32.6 × 137 mm against Ø 28.8 × 116 mm. The sim rack holes are sized for the sim tube. The pipette meshes are identical up to a 90° spin. | Keep the sim objects. Map the data poses onto them (`DATA_TO_SIM`): the pipette by its Rz(−90°), the tube with the rims aligned. |
| 4 | The retargeted grasps change while the object is held, and they cut through the holders. | The pipette slides about 10 cm in the hand between 5 and 7 s. The finger posture flips at 9.8 s. The left hand turns about 60° around the tube. Wrist and fingers go up to 26–31 mm into the stand and rack. | **4 grasps.** By default, use the template (KA) latched grasps. They are rigid within 2 mm / 0.4°, have the thumb on the cap, and are known to work with these holders. `--grasp demo` keeps the retargeting's grasps. |
| 5 | Hands are tracked only part of the time. | The right hand is valid 3.7–16.9 s and the left 8.0–18.4 s. | **6 wrists.** Synthesise the rest: hover, approach along the palm normal, grasp at lift − 0.8 s, and after the seat open and retreat. |
| 6 | With a rigid grasp, the human's mid-air object poses are out of reach for the right arm. | Right-arm IK error is 80–140 mm for the whole carry. | **5 + 7.** Away from the holders, keep the demo's position and **axis direction**. Add a constant retarget: shift, tilt scale, and spin about the object's own axis (free for a latched grasp; meaningless for the tube). The retarget is searched with batched continuous-path IK. Result for d02: down 10 cm, forward 10 cm, 10 cm to the robot's left, tilt ×0.6, pipette spin 30°. |
| 7 | Holder-to-air transitions are far too fast for the arm. | The first version reached object rotations of 300–660°/s and arm joint speeds of 7–16 rad/s, against the 2.5 rad/s command clamp. In zero-residual rollouts every start failed within 0.5 s of a lift. | **5 objects.** Extraction and insertion run straight along the hole axis in the seated orientation until the object is clear (insertion depth + 1 cm). Then a 1.5 s min-jerk blend reaches the mid-air pose. Final values: objects ≤ 64°/s, arm joints ≤ 1.4 rad/s. |
| 7b | The extraction height first followed the demo's own height, forced to 0 up to the lift / seat events. Those events are the frames where the object is already 15 mm up, so the height jumped there. | One-frame jumps of 17.7 / 13.9 mm (pipette) and 6.2 / 6.0 mm (tube): 177 m/s². The 8–24 Hz content of the right wrist path was 0.33 mm (KA: 0.02 mm), and the teacher trained on it showed the same 0.40 mm as wrist shake in the carry at every checkpoint. | **5 objects** (v2, 11-11). The extraction / insertion height is a min-jerk profile 0 → depth + 3 cm with peak speed 0.3 m/s (`EXTRACT`), starting 0.15 s before the lift event and ending exactly at the seat event. Max acceleration is now 4.9 m/s², and 8–24 Hz content is 0.02 mm / 0.005° (arm). A slower profile (0.2 m/s) shifts the clear-of-holder time by 0.4 s, and the pinned retarget then misses by 7.5 mm. |
| 7c | The pipette tip never went into the tube. The builder used the pipette frame origin's axis as the tip, (0, 0, −127) mm; the mesh tip is at (−7.7, 0, −127) mm. It also blended the tip's position linearly onto the tube axis, while the demo's tracked tip was already ~7 cm below the rim when it came within 3 cm of the axis. | The reference brought the tip in sideways, through the tube wall (45 frames at 11.43–11.56 / 13.58–13.72 / 14.14–14.29 s). The pipette collides with the tube walls in physics, so the teacher and the student pressed with the tip ~17 mm off the axis, beside the tube (outer radius 14.4 mm), in 16/16 recorded worlds. The press rule only checked the plunger. | **5 objects** (v3, 11-12). `PIP_TIP` is the mesh tip (checked against the mesh at build time). `tip_insert`: approach to 15 mm above the rim on the axis (height first), straight down the axis to the demo depth, stay in, straight back up, then back to the demo path. `tip_path_check`: 0 frames with the lower pipette inside the tube wall, tip in the bore for the whole press. The reference stores the tip's path in the tube frame (`tip_in_tube_ref`, `gate_tip_align`) for the teacher's `tip_align` reward / observation. |
| 7d | Insertions too fast (user, 11-12). | Pipette into the stand 0.30 m/s, tube into the rack 0.26 m/s, tip into the tube ~0.29 m/s. | **8b retime** (`RETIME`): the reference runs on a clock slowed 2x through the three insertions (same paths): 0.15 / 0.13 / 0.145 m/s; the reference grows 16.7 → 19.6 s. `reference.npz` `demo_time` maps each frame to the demo clock (the checker's overlays use it). |
| 8 | "Reachable at every frame" does not mean "reachable along a path". | Per-frame multi-seed IK found every frame feasible. The continuous solve still missed by up to 49 mm, because of arm branch switches. | **7 arms.** Score each candidate retarget by continuous IK along the whole path: warm-started frame to frame, 3 collision-free start postures. The final arm path is also continuous IK at 100 Hz. Final error is 0.09 mm. |
| 9 | Retargeted objects can pass through the other holder or the table. | The pipette passed 7.5 mm into the stand after the shift. | **5 objects.** Once clear of its own holder, an object's bottom stays at least 1.5 cm above any holder top and 2 cm above the table. |
| 10 | The open hand's thumb sweeps through the pipette on approach. | 10.6 mm. | **8 hands.** Replace an open thumb that cuts its object by more than 2 mm with the nearest posture that has at least 5 mm clearance. |
| 11 | Pipette tip into the tube: the sim pipette shaft is thick. | Shaft radius 8.6–12.6 mm against the tube's 12.9 mm inner radius. The demo goes 88 mm deep. | **5 objects.** Put the tip on the tube axis with the axes aligned. Limit the depth to 75 mm (shaft ≤ 11.9 mm at the rim). The press window is the tip-in-tube window. The plunger goes 0 → 13.4 mm. The right thumb tip sits on the template's cap point. |

### Left to RL on purpose

The reference does not need to be perfect. The residual policy fixes millimetre-level contact and force. It cannot fix centimetre-level errors, joint speeds above the command clamp, or objects routed through holders:

- The policy's residual is bounded: 0.04 rad per unit on the arms.
- The reward tracks the reference.
- An episode ends when an object is more than 5 cm off.

So the fixes above stop where RL takes over. What is left for d02:

- Contact penetrations of 4.3 mm (tip on the tube rim, one moment) and 3.3 mm (left ring finger).
- No grip squeeze and no press force at zero residual. The tube slips at VOC 0, as it did for KA, which was also built without a squeeze prior.

## Checks (`check/check.json`)

| Check | Pass |
|---|---|
| Zero-residual rollouts at VOC 1, holders nominal, 9 starts × 32 worlds | 0 failures. A failure means the reference itself cannot be executed. |
| Zero-residual rollouts at VOC 0 | Reported only. Grip and press come from RL. d02: 89 % fail, all as the tube slipping in the hand. |
| `build_report.json` → `ik` | Path error ≤ 2 mm / 2°, joint speed ≤ 2 rad/s, no self-collision start |
| `build_report.json` → `smoothness` | `pass`: max acceleration of the object and wrist paths ≤ 10 m/s² (a one-frame jump shows as ≥ 50 m/s²), 8–24 Hz content ≤ 0.1 mm, worst arm joint ≤ 0.03° |
| `build_report.json` → `tip_path` | `pass`: 0 frames with the lower pipette (tip cone and shaft meshes) inside the tube wall; the tip in the bore during the whole press |
| `build_report.json` → `reference_penetrations_gt3mm` | A few mm at most, and nothing through a holder |
| `overlay_start.png` | The sim rack, stand and objects lie on the real ones in the demo's first frame |
| `montage.png` | Front and top views of the whole reference |

## Knobs

- `--fwd`, `--lat`, `--base-x`, `--base-y`: where the demo sits relative to the robot. **The real robot must stand the same way.** For d02 that is square to the table, base front 12 cm from the edge, holders about 0.6 m ahead of the base.
- `--base-z` (11-12): the robot base height above the floor, i.e. the robot vs the table height (template table top 0.72 m; + = the robot sits higher relative to the table). Holders, objects and the demo stay on the table, and the IK re-solves the arms. `--base-z 0` reproduces v3 exactly. Test builds on d02: −0.015 → the right-arm IK misses by up to 5.2 mm (17 frames) and the hand cuts the pipette by up to 6 mm, so the retarget needs adjusting; +0.036 → IK within 0.42 mm, no new penetrations.
- `tracker/move_table.py SRC DST --near-dx DX` (11-12, after the build): copies a task and moves only the table's near (robot-side) edge and the near legs by DX towards the robot. The far edge, holders, objects and reference stay put; `ego/` is kept. `task_d02_straight_v3_desk` = v3 with `--near-dx 0.133` (the desk edge the real head ZED sees).
- `--hole`: the rack hole the tube was picked from (and returns to). Check it on `overlay_start.png`.
- `--grasp template|demo`, `--midair DZ DFWD ALPHA SPIN_P SPIN_U DLAT` (skips the search), `--blend-s`, `--smooth-s`.
- The search grid is `SEARCH` in `build_demo_task.py`. If no candidate is feasible, the build stops with a message.
- The search scores reachability only. It does not check arm self-collision along the path: on d02 v2 the unpinned search picked a retarget whose left arm passes 55 mm through the torso. The check flags this (`reference_penetrations_gt3mm`, zero-residual failures). Pin a retarget with `--midair` when that happens; d02 v2 uses `--midair -0.10 0.10 0.6 30 0 0.10`, the v1 search result.

## Assumptions and limits

- The episode uses the same objects as the template: the pipette, a 50 ml-type tube, the blue 2×5 rack and the black stand. Only one rack layout is supported. A different object needs its mesh and seat relation in the template.
- Event detection assumes this order: the pipette is lifted first, the tube is lifted from and returned to the rack, the pipette goes back to the stand, and the tip goes into the tube in between.
- The template grasps come from KA's demo. They are not the human's grasps in the new video.
