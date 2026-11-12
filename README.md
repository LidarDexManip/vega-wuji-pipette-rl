# Vega-U + Wuji Hand: residual RL for the pipette / tube task

Bimanual task, learned from one human video. The **left** Wuji hand picks a tube from a rack. The **right** hand takes a pipette from its stand. The pipette tip goes into the tube and the right thumb presses the plunger. Both objects go back: the tube into another hole, the pipette into its stand.

Robot: Dexmate Vega-U, two 7-DoF arms, each with a 20-DoF Wuji hand2 beta1. Fixed base, lift held.

Everything here is **simulation**. Nothing has run on the real robot yet. Each number below names the file it comes from. Robot, hand, lab and pipette meshes are **not included** (see [Assets](#9-how-to-run)).

## Status (2026-11-12)

The current work is the **d02 demo line** on branch `d02-v3-tip-insertion` ([below](#current-work-d02-demo-line)). The task is built from one retargeted RGB-D demo, `pipette_depth_02_tapir`.

- **Task `task_d02_straight_v3`.** The pipette tip really goes into the tube, and the three insertions are 2× slower than in v2.
  - `task_d02_straight_v3_desk` has the same task with the table's near edge matched to the real desk (seen by the head ZED).
- **Teacher `kin_d02v3_tip_g3`.** From frame 0: 62/64 tubes seated and 61/64 presses with the tip in the tube. Hand pushes do not change this.
- **Deployable students**, distilled with DAgger. They see wrist depth cameras plus a holder/tube estimate made once from the head camera at the start.
  - All four seat 59–60/64 tubes from start.
  - `student_d02v3_ego_ftmount_g3` is the deploy candidate. It is also trained with ±5 mm / ±3° wrist-camera mount errors and keeps 59/64 under them, where the student without that training drops to 37/64.
- **Open:**
  - the real table height vs the robot (the measurements conflict; see [Open items](#open-items));
  - the scan-network student does not follow a commanded place hole;
  - nothing has run on hardware.
- **To run any of this on another machine**, see [Running on a new machine](#running-on-a-new-machine).

Sections 1–9 below document the earlier T1k / KA line (rounds R1–R4, up to 11-10). The d02 line reuses its action space, reward, DR and PPO setup except where noted.

**Earlier line, status as of 2026-11-10:**

- **Latest rounds** (§4):
  - The **tc line** keeps the MPC reference. It replaces the demo's contact labels in the reward with contact terms computed from the policy's own contacts. tcs1 adds CAPS action smoothness on the policy mean.
  - **KA / KB** use the same reward on the **kinematic reference**, with no MPC anywhere. They train with the tracker's VOC curriculum.
- **Whole task from frame 0**, 64 runs with all randomisation on. Tubes seated:

  | tc | R3B | KA | tcs1 | KB |
  |---|---|---|---|---|
  | 47 | 42 | 40 | 34 | 29 |

- **Smoothest runs: tcs1 and KA.** In carry, CAPS halves the 3–8 Hz wrist shake of the right wrist (R3B 0.51 mm → 0.25–0.28 mm). It also brings the arm torque rate down to the zero-residual level.
- **KA full-task success** (tube seated + pipette seated + press), from local sweeps:

  | Sweep | Full success |
  |---|---|
  | Training distribution | 66 % |
  | All 100 hole pairs, holders at nominal | 76 % |
  | Stand yaw ±30° or more (trained ±10°) | 0 % |

  Failure causes and the proposed next round are in §5.
- **Not done yet (on that line):** no student was distilled from R3B or KA.

## Current work: d02 demo line

Configs carry a `_note` field with the full rationale of each run. The task builder is documented in [docs/DEMO_TO_TASK.md](docs/DEMO_TO_TASK.md), and the head-camera estimate in [docs/EGO_HOLDERS.md](docs/EGO_HOLDERS.md).

### Tasks

| Task | What changed |
|---|---|
| `task_d02_straight` (v1) | Built from HF `ytung/pipette-tasks` episode `pipette_depth_02_tapir` by `tracker/demo_to_task.py`. The robot stands square to the table, the holders are at ground-truth poses, and the tube returns to the hole it came from (r1c1). |
| `task_d02_straight_v2` | Smooth extraction and insertion: a min-jerk profile at 0.3 m/s replaces one-frame height jumps of 6–18 mm. Built with the retarget pinned, `-- --midair -0.10 0.10 0.6 30 0 0.10`. |
| **`task_d02_straight_v3`** | The pipette tip goes into the tube. `PIP_TIP` is the mesh tip, and `tip_insert` makes a peg-in-hole path along the tube axis; in v1/v2 the tip went in sideways through the tube wall, so in physics it pressed beside the tube. The three insertions run 2× slower: 0.13–0.15 m/s, and the reference grows from 16.7 to 19.6 s. Same build command as v2. |
| **`task_d02_straight_v3_desk`** | v3 with the table's near edge 13.3 cm closer to the robot. Made with `tracker/move_table.py ... --near-dx 0.133` so that the edge sits where the real head ZED sees it (33.5 cm ahead of the camera in sim, 33.3 cm on the real robot). The holders, objects and reference are unchanged. **Use this task for new runs.** |

### Teacher and students

All runs use 3 B200 GPUs with data-parallel training (`node_tracker_ddp.sh`, `node_distill.sh`).

| Run (results dir) | Config | What it is |
|---|---|---|
| **Teacher** `kin_d02v3_tip_g3` | `tracker/configs/kin_d02v3_tip_g3.json` | Warm start from the v2 teacher, with the checkpoint padded for 22 new tip observations (`tracker/expand_obs_ckpt.py`). Trained 3.5e8 steps (~2 h) on v3. Random pushes on both wrists (`extras.perturb`: 2–15 N, 0.1–0.4 s) are ramped in over 2e7–1.2e8 steps. |
| Student `student_d02v3_ego_g3` | `rl/configs/student_d02v3_ego.json` | DAgger from the teacher, 680 iterations (~2.5 h). Pushes are ramped in over iterations 50–250. |
| Student `student_d02v3_ego_ft_g3` | `rl/configs/student_d02v3_ego_ft.json` | Continues the student from iteration 680 to 880 with student-only rollouts (β 0) and lr 5e-5. |
| **Student `student_d02v3_ego_ftmount_g3`** | `rl/configs/student_d02v3_ego_ft_mount.json` | Continues `ft` from 880 to 1080 with wrist-camera mount errors: ±5 mm along each wrist axis and ±3° about each camera axis, per camera and episode (`extras.student.depth.mount_rand`). **Deploy candidate.** |
| Student `student_d02v3_ego_scan_g3` | `rl/configs/student_d02v3_ego_scan.json` | As `student_d02v3_ego_g3`, from scratch, except for the head-camera input. The holder/tube estimate is replaced by the 64-d embedding of **ScanNet**, a CNN on the initial head-depth crop (`perception/ego_scan_bank.py`, `perception/train_scan_net.py`, weights in `results/ego_scan/`). The arm is re-anchored with ScanNet's own pose prediction. |

**Student observations** come from `obs_mode='deployable'` plus `extras.student`: a 1027-d vector plus two wrist depth images, read by a small CNN in `rl/distill.py`. The layout is in `state.json` → `actor_layout`.

- 3-step history of the measured joint q, q̇ and the previous action;
- the reference command at 0 / 40 / 100 / 240 ms, the plunger reference, the phase and the clip progress;
- two wrist depth images (2×72×96) from D405-like cameras on the yaw-23° mount, with a stereo noise model (`perception/wristcam_sim.py`);
- the holder poses, pick hole and object poses, estimated **once** at the start from the head camera. The error model comes from `perception/ego_holders.py` (`tasks/*/ego/errors.npz`), ×1.25 plus extra noise.

With ScanNet the observation is 991-d, because the 64-d embedding replaces 100 dims of the estimate.

### Results

**How to read the tables.**

- Evals use `rl/eval_rr.py`, seed 123, the deterministic actor mean and full DR. Pushes are off unless the column says so.
- Each cell is **tubes seated / press with the tip in the tube**.
- The press needs the plunger rule and the tip inside the tube bore (`press_and_tip_ok`). It is counted out of 64 from start and out of the 40 starts that include the press otherwise.
- Files are in `results/<run>/eval/`.
- Columns:
  - "From start": the whole task from frame 0.
  - Train starts: every 1 s × 4.
  - Cross-hole: place hole ≠ pick hole, never trained.
  - Held-out placements: holders at 65 mm / 12°, outside the trained ±5 cm / ±10°.

| Run | From start (64) | From start, pushes (64) | Train starts (72) | Cross-hole (72) | Held-out placements (72) |
|---|---|---|---|---|---|
| Teacher `kin_d02v3_tip_g3` | **62 / 61** | 62 / 61 | 71 / 40 | **72 / 39** | **70 / 39** |
| `student_d02v3_ego_g3` | 59 / 61 | 59 / 61 | 70 / 39 | 68 / 39 | 62 / 35 |
| `student_d02v3_ego_ft_g3` | 60 / 61 | 61 / 63 | 70 / 39 | 66 / 38 | 65 / 37 |
| `student_d02v3_ego_ftmount_g3` † | 59 / 62 (nominal mount: 59 / 60) | 58 / 62 | 66 / 39 | 67 / 38 | 61 / 35 |
| `student_d02v3_ego_scan_g3` | 59 / 60 | 59 / 60 | 71 / 40 | **8** / 39 | not run ‡ |

† `ftmount` is evaluated **with** ±5 mm / ±3° mount errors, except the "nominal mount" figure.
‡ The held-out placements lie outside the scan bank's ±5 cm range.

**Wrist-camera mount errors** (from start, tubes seated of 64; `eval_rr.py --mount-rand XYZ_MM RPY_DEG`):

| Student | Nominal | ±5 mm / ±3° | ±10 mm / ±6° |
|---|---|---|---|
| `ft` (never trained with mount errors) | 60 | 37 | 2 |
| `ftmount` | 59 | 59 | 29 |

**Desk task** (`task_d02_straight_v3_desk`). The same checkpoints were evaluated with no retraining (`eval_rr.py --task`; `results/desk_v3/eval/`).

| Run | From start | From start, pushes | Train starts |
|---|---|---|---|
| Teacher | 60 / 60 (v3: 62 / 61) | 61 / 60 (v3: 62 / 61) | – |
| `ftmount`, mount ±5 mm / ±3° | 57 / 62 (v3: 59 / 62) | 58 / 62 (v3: 58 / 62) | 66 / 39 (v3: 66 / 39) |
| `ftmount`, nominal mount | 59 / 61 (v3: 59 / 60) | – | – |

No new failure kinds appeared, so the desk move needs no retraining. The left forearm's clearance to the table drops from 5.8 to 4.0 cm at the place (~17.9 s).

**Readings.**

- The students' main failure is "tube jammed tilted ~5 mm high", plus a few tubes released high on a lagging place descent (`cause_table` in each eval json).
- Tube slip in the hand (max over worlds): teacher 21–37 mm, `student_d02v3_ego_g3` 49–55 mm, `ft` 33–43 mm, `ftmount` up to 50 mm.
- ScanNet locates things better than the hand-written estimator in sim. Position std: stand 1.1 / 0.6 mm, rack 0.8 / 0.8 mm, tube 2.0 / 1.3 / 0.4 mm, against 2.6–3.2 mm for the hand-written estimator. Pick-hole accuracy is 99.9 % (`results/ego_scan/scan_eval.json`).
  - The ScanNet student fails cross-hole because the place hole is a task command that the scan cannot show. Every training pair is same-hole, and in the other students only the hole items of the estimate told them where to place.
  - Proposed fix (not done): a place-hole command input, plus random pick/place pairs in training.

### Real robot vs simulation (head camera, desk)

- **Camera.** The sim head camera `ego` is the URDF's `zed_left_camera`, i.e. the ZED's **left eye**. Like the ZED, it renders z-depth.
- **One screenshot of the real ZED viewer (11-12).** Depth was recovered from the viewer's colour bar (64 levels), then a RANSAC table-plane fit:

  | | Real | Sim |
  |---|---|---|
  | Camera above the table | 60.4 cm (±2) | 56.8 cm |
  | Tilt | 28.0° | 28.2° |
  | Roll | −1.1° | −0.9° |
  | Near table edge, ahead of the camera | 33.3 cm | 46.65 cm (v3); 33.5 cm in the desk task |

  The fit was validated on a sim render to 0.05 cm. The screenshot and the ad-hoc scripts are not in the repo.
- **Holders.** The real frame shows clutter, and the real holders differ from the sim models: a blue rack in a foam box, and a U-frame stand. ScanNet and `ego_holders.estimate` have not seen real depth yet.

### Open items

1. **Table height vs the robot. This is unresolved.**
   - A tape measurement gives a real table height of 0.735 m (sim 0.720).
   - The screenshot puts the real ZED ~5 cm higher above the floor than the URDF (≈1.339 vs 1.288 m). So either the real robot or lift is taller, or the neck was not at the sim pose (0.878, 0.017, 0.386).
   - What matters is the arm relative to the table. In sim the shoulder (arm j1) axis is 49.9 cm above the table top, i.e. 1.219 m above the floor. **Measure the real one.**
   - `tracker/build_demo_task.py --base-z` rebuilds the task with the robot higher (+) or lower (−) relative to the table. `--base-z 0` reproduces v3 exactly; a build takes about 1 min.
   - Test builds:
     - −0.015: the right-arm IK misses by up to 5.2 mm and the hand cuts the pipette by up to 6 mm, so the retarget needs adjusting.
     - +0.036: fine.
   - If the real lift can be set so that the shoulder axis is 49.9 cm above the table (1.234 m above the floor for a 0.735 m table), no sim change is needed.
2. **Horizontal placement.** The user measured 0.34 m from the robot's "CoM" to the table edge; which point was meant is still unknown. In the desk task:
   - robot CoM at the start pose: 32.0 cm;
   - shoulder line: 35.0 cm;
   - base origin: 37.4 cm.

   A 13 cm move did not change the results, so this is low-stakes. `tracker/move_table.py --near-dx` sets it.
3. **ScanNet student:** add a place-hole command (see Readings).
4. **Next distillation round:**
   - distil without pushes, then fine-tune with pushes. `rl/configs/student_d02v3_ego_ft_push.json` (it 680 → 980, pushes ramped 680–780) is ready, but its base must be a no-push distillation.
   - `action_delay` DR;
   - a `deploy_obs` runtime for the real robot;
   - a raw ZED depth capture of the real setup, to test the estimators and calibrate the noise model.

### Commands (d02 line)

Run from the repo root. Set `WARP_CACHE_PATH` on a node. GPU lists are examples; check `nvidia-smi` first.

```bash
# task: build v3, then the desk variant
.venv-rl-head/bin/python -I tracker/demo_to_task.py pipette_depth_02_tapir d02_straight_v3 -- --midair -0.10 0.10 0.6 30 0 0.10
.venv-rl-head/bin/python -I tracker/move_table.py tasks/task_d02_straight_v3 tasks/task_d02_straight_v3_desk --near-dx 0.133
#   other robot height vs the table: add  --base-z DZ  after the "--" (m; + = robot higher)

# teacher (3 GPUs, resumes from <out>/latest.pt)
GPUS="0 3 4" ./node_tracker_ddp.sh tracker/configs/kin_d02v3_tip_g3.json
# student: DAgger from the teacher's final.pt (the config's teacher_ckpt)
GPUS="0 3 4" ./node_distill.sh rl/configs/student_d02v3_ego.json
# fine-tune: the base run's final.pt becomes latest.pt of the new out dir, and its state.json is copied with status "paused"
# (distill.py resumes from the iteration count in it; max_iters / wall_hours of the ft config count from the base run's start)
O=rl/runs/student_d02v3_ego_ftmount_g3; B=results/student_d02v3_ego_ft_g3; mkdir -p $O && cp $B/final.pt $O/latest.pt
python3 -c "import json,sys; s=json.load(open(sys.argv[1])); s.update(status='paused', ft_from=sys.argv[1]); json.dump(s, open(sys.argv[2], 'w'), indent=1)" $B/state.json $O/state.json
GPUS="0 3 4" ./node_distill.sh rl/configs/student_d02v3_ego_ft_mount.json

# evals (from rl/; one GPU each)
cd rl
python -I eval_rr.py ../tracker/configs/kin_d02v3_tip_g3.json --ckpt ../results/kin_d02v3_tip_g3/final.pt --from-start 64 --out t_fromstart.json
python -I eval_rr.py ../tracker/configs/kin_d02v3_tip_g3.json --ckpt ../results/kin_d02v3_tip_g3/final.pt --from-start 64 --perturb --out t_push.json
python -I eval_rr.py ../tracker/configs/kin_d02v3_eval_crosshole.json --ckpt ../results/kin_d02v3_tip_g3/final.pt --repeat 4 --out t_crosshole.json
S=../results/student_d02v3_ego_ftmount_g3/final.pt
python -I eval_rr.py configs/student_d02v3_ego.json --student $S --from-start 64 --mount-rand 5 3 --out s_fromstart.json
python -I eval_rr.py configs/student_d02v3_ego.json --student $S --repeat 4 --mount-rand 5 3 --out s_train.json
python -I eval_rr.py configs/student_d02v3_ego_eval_crosshole.json --student $S --repeat 4 --mount-rand 5 3 --out s_crosshole.json
python -I eval_rr.py configs/student_d02v3_ego_eval_holdplace.json --student $S --repeat 4 --mount-rand 5 3 --out s_holdplace.json
python -I eval_rr.py configs/student_d02v3_ego.json --student $S --from-start 64 --task ../tasks/task_d02_straight_v3_desk --out s_desk.json
python -I eval_rr.py configs/student_d02v3_ego_scan_eval.json --student ../results/student_d02v3_ego_scan_g3/final.pt --from-start 64 --out scan_fromstart.json
cd ..

# ScanNet: render the scan bank (~4 GB, not committed), train (~2.5 min on an RTX 4080)
MUJOCO_GL=egl .venv-rl-head/bin/python -I perception/ego_scan_bank.py tasks/task_d02_straight_v3 rl/runs/ego_scan --n-train 40000 --procs 16
.venv-rl-head/bin/python -I perception/train_scan_net.py rl/runs/ego_scan

# pictures / videos (not committed; from rl/): see-through tube close-up of the tip, wrist-camera mount picture, policy comparison
MUJOCO_GL=egl python render_tip_closeup.py ../tasks/task_d02_straight_v3/scene.xml tip.mp4 --ref ../tasks/task_d02_straight_v3 reference
MUJOCO_GL=egl python render_wristcam_mount.py configs/student_d02v3_ego.json mount.png
python -I record_rollouts.py ../tracker/configs/kin_d02v3_tip_g3.json ../results/kin_d02v3_tip_g3/final.pt runs/videos_d02v3/teacher --worlds ../results/d02v2_videos/specs.json
```

## Running on a new machine

1. **Clone the branch:**

   ```bash
   git clone -b d02-v3-tip-insertion git@github.com:hyding1/vega-wuji-pipette-rl.git
   ```

   HTTPS works for a public clone; pushing needs SSH keys.
2. **Copy the meshes.** Copy `assets/` from a machine that has them; they are not in git (see §9 Assets). Every `tasks/*/scene.xml` loads `../../assets/...`.
3. **Make the environment:** `./node_setup.sh`.
   - It needs `uv` and internet.
   - It builds `.venv-rl-head` from `requirements_rl_head.txt`, with CUDA 13 wheels (driver ≥ 580), and runs the CPU unit tests.
   - On a GPU that cannot use the cu130 torch wheel, install a matching torch build first.
4. **Put the committed checkpoints where the configs expect them.** The `runs/` dirs are gitignored. Configs point at `tracker/runs/...` for the teacher and at `rl/runs/ego_scan/` for the scan bank (task-relative path).

   ```bash
   mkdir -p tracker/runs/kin_d02v3_tip_g3 rl/runs/ego_scan
   cp results/kin_d02v3_tip_g3/final.pt tracker/runs/kin_d02v3_tip_g3/      # needed to distil students
   cp results/ego_scan/{scan_net.pt,policy_bank.npz,eval_bank.npz} rl/runs/ego_scan/   # needed for the ScanNet student
   ```
5. **Check it works.** Run one eval from the commands above. For example, the teacher from start should give 62 tubes seated of 64.
6. **The demo download.** `demo_to_task.py` fetches from `https://huggingface.co/datasets/ytung/pipette-tasks`, about 30 MB, into `~/Projects/datasets/pipette-tasks`. It is needed only to rebuild a task: the built tasks are committed.

**What is committed per run** (`results/<run>/`):

- `final.pt`;
- `state.json`, which holds the resolved config;
- `progress.jsonl`;
- `eval/*.json`.

**What is not committed:** intermediate checkpoints, logs, videos and figures, the scan bank's training renders (`train.npz` and `val.npz`, regenerated by `ego_scan_bank.py`), and the v1/v2 d02 checkpoints, which are still on the node.

## 1. Pipeline layers

| # | Layer | What it is | Artifacts here |
|---|---|---|---|
| 1 | Human video → **kinematic reference** | Reconstruction, retargeting and IK to Vega-U + Wuji. No physics. | Not included. `tasks/task_kin_*` are built from it (tracker line only). |
| 2 | **Physics demo** (MuJoCo) | DAID ("do-as-I-do") sampling MPC tracks layer 1 in MuJoCo. Its executed controls are a physically valid demo ("T1_k", seed 0). | Not included. Its executed states and ctrl are exported into `tasks/task_T1k_*/reference.npz` (`rl/export_task.py`). Contact labels come from it via `rl/contact_ref.py`. |
| 3 | **Residual-PPO teacher** (MuJoCo Warp, GPU) | The policy adds a joint-space residual on top of layer 2's executed ctrl, under heavy randomisation. | `rl/`, `rl/configs/`, `results/<run>/final.pt`. Rounds R1–R3 (§3), tc line (§4). |
| 4 | **DAgger student** (deployable observations) | `rl/distill.py`, data-parallel via `node_distill.sh`. On the T1k line only plumbing-tested. On the d02 line: students with wrist depth cameras and the head-camera estimate (see [Current work](#current-work-d02-demo-line)). | `results/student_plumb/`, `results/student_d02v3_ego_*` |
| — | Tracker (WIP) | One RL policy for any kinematic reference, with no per-video MPC. | `tracker/`. KA / KB (§4) are residual teachers on the kinematic reference, with no MPC base: `results/kin_tc*_s5/`. |
| 1′ | Retargeted RGB-D demo (HF `ytung/pipette-tasks`) → **task** | One command: robot square to the table, ground-truth holders, reachable continuous arm paths, checks, training configs. See [docs/DEMO_TO_TASK.md](docs/DEMO_TO_TASK.md). | `tracker/demo_to_task.py`, `tracker/build_demo_task.py` (`--base-z`: robot height vs the table), `tracker/move_table.py` (desk edge), `tasks/task_d02_straight{,_v2,_v3,_v3_desk}/`. Teacher `results/kin_d02v3_tip_g3/`. |
| 1″ | Head-camera (ego) estimate of the holders and the tube | One depth frame at the start → stand / rack pose, tube position and pick hole; its error model drives the student's holder measurement. See [docs/EGO_HOLDERS.md](docs/EGO_HOLDERS.md). Learned alternative: ScanNet on the same depth crop. | `perception/ego_holders.py`, `tasks/task_d02_straight_v3/ego/`; `perception/ego_scan_bank.py`, `perception/train_scan_net.py`, `results/ego_scan/`. |

## 2. RL setup (teacher, layer 3)

Sources: `rl/vega_res_env.py`, `rl/vega_extras.py`, `common/rewards.py`, `rl/configs/rr_T1k_r3B_s5.json`, `deps_from_g1ps/tools/rl/gpu_train.py`.

### Action space

The policy acts **in joint space on all 54 actuators**, not in end-effector space. The actor outputs the residual directly. There is no IK on the policy output.

| Item | Value |
|---|---|
| Action | 54-d residual, one per actuator: 2×7 arm joints + 2×20 finger joints. Clamped to [-1, 1] and multiplied by a per-group scale (rad). |
| Scales used in every round (`group_scale`) | right fingers 0.10, left fingers 0.05, right arm 0.04, left arm 0.04 rad per unit. The env defaults are arm 0.08 / fingers 0.25 (YS's), but every config overrides them. |
| Smoothing | EMA on the residual, α 0.5 (`ema_alpha`). |
| Base command | The physics demo's **executed** ctrl at the sim rate (`ctrl_sim_verbatim`). The residual is added to it. |
| Rates | Policy 50 Hz. Physics dt 2.5 ms (8 substeps). Fingers: base + residual at every substep. |
| Arm path (`arm_command`, `cmd_hz` 100) | The residual is linearly upsampled to a 100 Hz target stream, with velocity feed-forward (kv/kp × the residual's slope). Targets stay ≥ 0.1 rad inside the joint limits and move ≤ 2.5 rad/s. R3 adds `clamp_after_ff`: the limit clamp also applies after the feed-forward. |
| Holder re-anchoring | When the rack or stand moves, an IK delta re-anchors the demo's arm command. It uses the **measured** holder pose and is added to the base command. The policy does not output it. |

### Simulated servo gains

Source: `tasks/*/scene.xml`; the R2/R3 finger damping comes from the DR files.

| Joints | kp | kd | Force limit |
|---|---|---|---|
| Arm j1–j2 | 500 | 35 | 150 |
| Arm j3–j4 | 400 | 20 | 80 |
| Arm j5–j7 | 250 | 12 | 25 |
| Wuji finger joints | 2.0 | 0.06 nominal, ×3 in R2/R3 (`kv_hand` DR 2.1–4.2; tracker fixed ×3) | 0.2–2 |

The arm uses gravity compensation. NODE_PLAN_R3 checked the arm damping: nominal ζ is 0.74–1.05.

### Teacher observations

Privileged, 1140-d, with the actor and critic sharing them:

- joint q, joint q̇, previous residual (54 each);
- per hand, the held object's six axis points (±5 cm) in the wrist frame;
- plunger depth and velocity;
- at future offsets of 0 / 40 / 100 / 240 ms:
  - the reference command (54);
  - per side, the reference object axis points and the 21 link positions in the wrist frame;
  - the holder-adjusted reference wrist pose relative to the current wrist;
  - the plunger reference.

### Reward

The env returns `0.01 × (tracking + extras)`.

| Term | Definition / weight (R3B) |
|---|---|
| Tracking (per hand) | 8·exp(−(axis err/4 cm)²) + links exp(−(err/2.5 cm)²) + finger joints exp(−(err/0.1 range)²) + 2·wrist pose + 2·object world pose |
| Plunger | 2·exp(−((depth−ref)/3 mm)²) in the press window |
| Smoothness | −0.01·(Δr² + Δ²r²) on the residual |
| **T1 contact set** (0.5, penalty) | Missed labelled links / labelled links, plus unintended links / 6 (capped). Per hand, over 42 hand-link bodies. |
| **T2 contact graph** (0.5) | exp(−mean\|d_ij − d_ij,ref\| / 3 mm) over pairwise distances between contact points, in the object frame |
| **T3 wrench support** (0.5) | tanh of friction-cone support (8 edges, μ 1.5) along the required object wrench direction |
| T1–T3 labels | `rl/contact_ref.py` runs on the physics demo's states. A link is in contact when its normal force on its own object exceeds 0.1 × the object's weight. The required wrench comes from the demo's object acceleration. |
| Current contacts | Read from the MJWarp contact list (`d.contact` + constraint force), with the same 0.1 × weight threshold |
| Naturalness | `finger_still` 1.0, `finger_band` 3.0 (B) / 2.0 (A), `arm_band` 1.0, `free_finger_acc` 1.0, `finger_jerk` 0.2, `action_rate` 0.5, `thumb_prior` 0.5, `posture` 0.1, `joint_margin` 1.0, `elbow_height` 50 |
| Naturalness definitions | `finger_still`: finger speed where the reference finger is still, all 40 joints. `*_band`: causal 3–10 Hz band (MA2−MA8) of the deviation from the reference. |
| Task | `tube_tilt` 1.0: tube-axis tilt in the hand vs the reference, 1 = 2.9°. `pipette_seat` 1.0: pipette pose vs its seated reference around the put-back, 1 = 5 mm / 2.9°. |
| Not used | `press_progress` (tracker only). Tube spin is not penalised. |
| tc line, KA / KB (§4) | T1–T3 and `free_finger_acc` set to 0. Replaced by contact terms computed from the policy's own contacts, plus CAPS on the policy mean. |

### Termination and success

Termination, per object: relative position > 5 cm or orientation > 45°, axis points > 5 cm, world position > 5 cm or orientation > 45°, height < 6 cm, velocity blow-up, or non-finite state.

Success means reaching the end of the 24 s reference. Each episode starts at a random one of 1199 audited start frames (reference state initialisation).

### Randomisation, per episode

Sources: `rl/configs/dr_rr1_kv3_zeta.json` and `extras.holder`.

| Item | Range |
|---|---|
| Rack and stand placement | ±5 cm xy, ±10° yaw each (uniform) |
| Robot's holder estimate (drives re-anchoring) | true + N(5 mm, 2°) |
| Hole pairs (pick → place) | 88 training pairs. Held-out pairs r0c1→r1c4 and r1c2→r0c0. |
| Arm gains | kp ×0.5–2. R3: kv = √(kp factor) × U(0.95, 1.4), so ζ ≥ 0.70. R1/R2: kv ×0.7–1.4. |
| Finger gains | kp ×0.7–1.4. kv ×2.1–4.2 (R2/R3) or ×0.7–1.4 (R1). |
| Friction | hand–object ×0.7–1.3, pipette–tube ×0.7–1.3 |
| Object mass | ×0.8–1.2 |
| Encoder offsets | arm ±0.01 rad, fingers ±0.02 rad |
| Object start in holder | ±2 mm, ±3° |
| **Held-out eval placements** | 8 fixed placements at 65 mm radius, ±12° yaw (`eval_fixed_placements`). These lie outside the training range. |

### PPO

rsl_rl 5.5.2:

- 4096 envs on 1 B200 per run, 24 steps per env per iteration;
- 5 epochs × 4 minibatches;
- learning rate 3e-4, adaptive (KL 0.01);
- initial std 0.1;
- no value clipping;
- actor MLP 512-256-128, critic MLP 512-512-256-128;
- the actor's last layer is zero-initialised, so the first policy equals the demo (zero residual).

## 3. Rounds

### What changed between rounds

| Round (run dir) | Base task | Change vs previous | Steps / wall (from `progress.jsonl`) |
|---|---|---|---|
| R1 (`rr_T1k_full_lp_s0`) | `task_T1k_lp` | First full config: wide placement, hole pairs, T1–T3, naturalness terms, 100 Hz arm path | 1.18e8 / 1.75 h |
| R2 s2 (`rr_T1k_r2_s2`) | `task_T1k_lp_noff` | No finger velocity feed-forward in the base; finger kd ×3; `finger_band` 1.0; from scratch | 1.46e8 / 2.15 h |
| R2b (`rr_T1k_r2w_s3`) | `task_T1k_lp_noff` | R2 settings, **warm-started from R1 final**. The press collapsed, because the R1 residual does not transfer to a new base. The config file was not kept; per `docs/RL_PLAN.md` §8 it was R2 with seed 3. | 6.0e7 / 0.88 h |
| R3A (`rr_T1k_r3A_s4`) | `task_T1k_still3_noff` | Base fingers hold still between events. Added `clamp_after_ff`, `tube_tilt`, `arm_band`, `finger_still`, `pipette_seat`, band on deviation, coupled arm-damping DR. `finger_band` 2.0. From scratch. | 1.97e8 / 2.85 h |
| **R3B** (`rr_T1k_r3B_s5`) | `task_T1k_still3_noff` | Same as R3A, with `finger_band` 3.0. **Recommended teacher.** | 1.97e8 / 2.85 h |

### Results: policy vs zero residual

Both are run on **identical** perturbations (`eval_rr.py`, seed 123, deterministic actor mean). Each run is judged under its own config and DR.

Each cell is policy | zero residual. Counts are reach end / tube seated / press OK.

| Candidate | From start (64) | Train perturbations (92) | Held-out holes (92) | Held-out placements (92) |
|---|---|---|---|---|
| R1 final | 42 / 22 / 42 \| 2 / 0 / 15 | 85 / 70 / 28 of 36 \| 23 / 18 / 25 | 80 / 57 / 27 \| 25 / 23 / 23 | 57 / 42 / 26 \| 18 / 13 / 23 |
| R2 s2 final | 44 / 40 / 54 \| 2 / 1 / 14 | 83 / 72 / 33 \| 23 / 19 / 23 | 79 / 68 / 34 \| 24 / 19 / 22 | 57 / 49 / 30 \| 18 / 14 / 19 |
| R2b final | 42 / 19 / 27 \| 2 / 1 / 14 | 85 / 74 / 13 \| 22 / 18 / 22 | 87 / 61 / 14 \| 25 / 19 / 22 | 62 / 45 / 12 \| 16 / 12 / 19 |
| R3A final | 48 / 38 / 42 \| 3 / 1 / 16 | 89 / 83 / 29 \| 30 / 24 / 29 | 87 / 78 / 29 \| 29 / 19 / 25 | **65 / 56 / 26** \| 13 / 10 / 24 |
| **R3B final** | **52 / 42 / 59** \| 1 / 0 / 18 | 89 / 83 / 34 \| 29 / 24 / 29 | **89 / 76 / 35** \| 28 / 20 / 25 | 58 / 53 / 29 \| 14 / 10 / 23 |

Notes on the table:

- "From start" means the whole task starting at frame 0, with press counted out of 64. The other three columns start every 1 s across the clip (92 starts), with press counted out of the 36 starts that include the press.
- R2 s2 / R3A / R3B rows are in `results/final_table.json`. R1 / R2b rows were computed from `results/<run>/eval/*.json` with the same `rl/final_table.py` logic.
- "Pipette seated" is omitted from the table. It equals reach-end within 1 in every row.

### Naturalness and other metrics

From `results/final_table.json` and the eval jsons. Proposed thresholds are marked PROPOSED and are not yet approved by the user.

| Metric | R2 s2 | R3A | R3B |
|---|---|---|---|
| Tube slip in hand from start, p50 / p90 mm (zero residual p90 ≈ 47) | 14.3 / 27.3 | 13.5 / 24.2 | 11.9 / 18.1 |
| Finger jitter cells > 0.25° (3–10 Hz, of 25) | 6 | 0 | 0 |
| Commanded arm margin, deg (rule ≥ 5.73) | 5.23 (fail) | 5.73 | 5.73 |
| Wrist shake, carry, 3–8 Hz, R / L mm (PROPOSED ≤ 0.3; zero residual on R3 base 0.28 / 0.55) | 0.50 / 0.74 | 0.41 / 0.54 | 0.51 / 0.59 |
| Idle-finger cells over PROPOSED (≤ 2° / 3°/s) | 20 | 29 | 25 |
| Pipette at placement, p50 / p90 mm | 6.1 / 11.9 | 4.7 / 10.7 | 5.1 / 9.6 |

### R3B failure causes from start

From `cause_table` in `final_table.json`. Out of 64 episodes, 42 succeed. The failures:

| Cause | Count |
|---|---|
| Tube jammed, tilted, about 5 mm high | 10 |
| Pipette tips over after placement while the fingers open | 8 |
| Tube lost during carry or press | 2 |
| Extraction from the stand | 1 |
| Before grasp | 1 |

### Known failing items for "delivered"

Source: `docs/NODE_PLAN_R3.md`.

1. **Idle fingers.** The policy's finger residuals move fingers that should be still, up to 9.3° on the right thumb during the hold. Zero residual has only 3 such cells. `finger_still` at weight 1 is too weak.
2. **Wrist shake.** The 3–8 Hz wrist shake is above the proposed 0.3 mm. Part of it is already present in the zero residual (base plus arm DR).
3. **Tube jam (10/64).** This is an open base and measurement-noise question.
4. **Pipette tipping after placement (8/64).** An ordered-release base failed its replay, so it was not used.
5. **No R3B student yet.**

## 4. Round 4 (tc line) and the kinematic-reference runs (KA / KB)

**Shared setup.** Unless a row below says otherwise, every run uses:

- seed 5, 4096 envs on one B200;
- R3B's DR (`dr_rr1_kv3_zeta.json`), holder randomisation (±5 cm / ±10°, 88 hole pairs), residual scales and PPO settings.

Configs are `rl/configs/rr_T1k_tc*_s5.json` and `tracker/configs/kin_tc*_s5.json`. The `_note` field of each config holds the full rationale.

### What changed

| Run (results dir) | Reference and base | Change | Steps / wall (`progress.jsonl`) |
|---|---|---|---|
| **tc** (`rr_T1k_tc_s5`) | MPC demo (`task_T1k_still3_noff`) | R3B with the demo's contact labels removed from the reward: T1, T2, T3 and `free_finger_acc` set to 0. Four new contact terms come from the policy's own contacts (next table). | 2.00e8 / 3.16 h |
| tcq (`rr_T1k_tcq_s5`) | MPC demo | tc + `torque_rate_arm` 2.0 and `torque_rate_finger` 1.0. `action_rate` (weight 0.5) is normalised so that it actually acts. | 2.00e8 / 2.98 h |
| **tcs1** (`rr_T1k_tcs1_s5`) | MPC demo | tc + CAPS 0.1: a smoothness loss on the policy *mean*, added to the PPO loss (`rl/caps.py`). Replaces tcq's reward-side rate terms. | 2.00e8 / 3.20 h |
| tcs5 (`rr_T1k_tcs5_s5`) | MPC demo | tc + CAPS 0.5 | 2.00e8 / 3.13 h |
| **KA** (`kin_tc_s5`) | **Kinematic reference** (`task_kin_noprior_press`: IK arms, finger reference and an analytic press prior). No MPC in the base, the reward reference or the contact terms. | See the KA notes below the table. | 3.00e8 / 4.79 h |
| KB (`kin_tcw_s5`) | Kinematic reference | KA, with the hand-tracking multipliers (`track_w` link, joint and wrist) lowered from 1 to 0.5; the object terms are kept. The four contact terms are doubled, 0.5 → 1.0. | 3.00e8 / 4.69 h |

**KA notes.** KA uses tcs1's reward and CAPS 0.1, trained with `tracker/train_tracker.py`:

- **VOC** (virtual object controller, §7) goes from 1 to 0 on success. A linear ramp also forces it to 0 between 5e7 and 1.5e8 steps, so the last 1.5e8 steps have no VOC.
- Press aids from tracker r3: `press_progress_w` 2, `plunger_weight` 8, `thumb_press_scale` 2.
- Holders move ±5 cm / ±10° over 88 hole pairs. The earlier tracker runs used ±2 cm / ±5°.

**Why tcq failed.** A rate penalty in the reward also penalises PPO's exploration noise.

- With zero residual plus noise 0.1, the penalty is −68 per step, against about +21 for tracking.
- The action std collapsed from 0.10 to 0.03.
- Training success at 1e8 steps was 0.42, against 0.88 for tc.

**CAPS.** CAPS acts on the policy mean only, so it avoids this. `rl/caps_calib.py` calibrated its coefficient on R3B final:

- coeff 0.1 gives about 0.16 × the PPO surrogate's actor gradient;
- coeff 0.5 gives about 0.8 ×.

### New reward and loss terms

Sources: the config notes, `rl/vega_extras.py`, `rl/caps.py`.

| Term | Definition |
|---|---|
| `contact_keep` | Fraction of the previous control step's hand–object link contacts that are lost. Hysteresis: a contact starts above 0.1 × the object's weight and ends below 0.05 ×. |
| `contact_drift` | Mean change of the pairwise distances between contact points on the object, 1 = 1 mm per step. The plunger stroke is removed. |
| `contact_slip` | Mean motion of each contact point on the object, 1 = 1 mm per step. |
| `wrench_margin` (replaces T3) | Checks whether the current contacts can supply the wrench that the reference object motion needs. The required wrench gets ±0.3 N on each axis. Limits: ≤ 3 N per contact, circular cones with μ 1.5. Solved every 4 control steps, only while the reference holds the object. Unlike T3, it accounts for torque balance. |
| Gating | The three contact-change terms are off during the deliberate in-hand regrasps (`temporal_off_gates`). |
| `track_w` | Per-term multipliers on the tracking reward: object, link, joint, wrist, world. The defaults are 8 / 1 / 1 / 1 / 1, which equals R3B. |
| `torque_rate_*` (tcq only) | Squared change of the control-step mean actuator torque between control steps, 1 = 1 % of the torque limit per step, cap 25. |
| CAPS (tcs1, tcs5, KA, KB) | Temporal smoothness of the policy mean as a PPO loss term (Mysore et al. 2021). It is logged as rsl_rl's `symmetry` loss. 1 = a change of 0.01 normalised action on every actuator per control step. Cap 25, 8192 pairs per update. |

### Results

**Setup.**

- Evals: `eval_rr.py`, seed 123, deterministic actor mean. Each run is judged under its own config and DR.
- Run by `node_eval.sh`; tables built by `rl/compare_runs.py`.
- Files:
  - `results/compare_tc_rounds.json`, `results/compare_kin.json`;
  - per run, `results/<run>/eval/`.
- "Zero residual" means the R3 base (`task_T1k_still3_noff`) with no policy.
- Counts are reach end / tube seated / press OK. The columns are as in §3.

| Run | From start (64) | Train perturbations (92) | Held-out holes (92) | Held-out placements (92) |
|---|---|---|---|---|
| Zero residual (R3 base) | 1 / 0 / 18 | 29 / 24 / 29 | 28 / 20 / 25 | 14 / 10 / 23 |
| R3B | **52** / 42 / **59** | 89 / 83 / 34 | **89** / 76 / 35 | 58 / 53 / 29 |
| **tc** | 50 / **47** / **59** | **90 / 86 / 36** | 86 / 80 / 35 | **63 / 60 / 30** |
| tcq | 4 / 1 / 21 | 43 / 30 / 30 | 37 / 23 / 29 | 24 / 19 / 24 |
| tcs1 | 37 / 34 / 47 | 87 / 83 / 34 | 84 / 77 / 34 | 61 / 55 / 28 |
| tcs5 | 26 / 19 / 33 | 73 / 61 / 29 | 66 / 57 / 28 | 41 / 36 / 23 |
| **KA** | 42 / 40 / 54 | 83 / 80 / 35 | 84 / **81** / 34 | 61 / **60** / 30 |
| KB | 30 / 29 / 53 | 82 / 79 / **36** | 79 / 76 / **36** | 58 / 57 / 29 |

**Smoothness and contacts.** These come from the same from-start evals:

- the `natural` block of `eval_rr.py`;
- `eval_contact_stability.py`;
- `reward_breakdown.py`.

Limits marked PROPOSED are not approved yet. The idle-finger count here is over 30 cells, so it is not comparable with the 25-cell count in §3.

| Metric | Zero res. | R3B | tc | tcq | tcs1 | tcs5 | KA | KB |
|---|---|---|---|---|---|---|---|---|
| Wrist 3–8 Hz, carry, R / L mm (PROPOSED ≤ 0.3) | 0.28 / 0.55 | 0.51 / 0.59 | 0.46 / 0.56 | 0.44 / 0.56 | 0.28 / 0.46 | 0.28 / 0.52 | **0.25 / 0.39** | 0.25 / 0.40 |
| Wrist 8–24 Hz, carry, R / L mm (PROPOSED ≤ 0.1) | 0.022 / 0.070 | 0.098 / 0.119 | 0.091 / 0.107 | 0.032 / 0.075 | 0.021 / 0.067 | 0.020 / 0.067 | 0.033 / 0.067 | 0.033 / 0.071 |
| Wrist 8–24 Hz, after release, R / L mm | 0.046 / 0.053 | 0.065 / 0.065 | 0.061 / 0.067 | 0.044 / 0.043 | 0.044 / 0.054 | 0.041 / 0.051 | 0.012 / 0.026 | 0.012 / 0.025 |
| Idle-finger cells over limits (of 30) | 3 | 25 | 30 | 5 | 14 | 9 | 21 | 14 |
| Action rate, mean square | 0 | 6.3e-4 | 6.3e-4 | 1.1e-4 | 1.0e-4 | 1.9e-4 | 7.5e-5 | 5.6e-5 |
| Arm torque rate (1 = 1 % of the limit per step) | 0.19 | 0.54 | 0.47 | 0.20 | 0.20 | 0.19 | 0.22 | 0.22 |
| Finger torque rate | 1.69 | 2.17 | 2.19 | 1.54 | 1.57 | 1.74 | 1.41 | 1.41 |
| Contacts lost per step | 0.022 | 0.026 | 0.029 | 0.027 | 0.022 | 0.020 | 0.031 | 0.032 |
| Contact drift / slip, mm per step | 0.065 / 0.092 | 0.065 / 0.098 | 0.067 / 0.098 | 0.063 / 0.094 | 0.055 / 0.084 | 0.059 / 0.087 | 0.060 / 0.083 | 0.060 / 0.090 |
| Share of reference-held steps with the required wrench, pipette / tube | 0.84 / 0.93 | 0.86 / 0.95 | 0.85 / 0.94 | 0.88 / 0.92 | 0.88 / 0.95 | 0.83 / 0.94 | 0.87 / 0.88 | 0.86 / 0.89 |

**Readings.**

- **tc vs R3B.** tc seats more tubes from start than R3B (47 vs 42) without using the demo's contact labels.
- **Reward-side rate terms vs CAPS.** tcq's reward-side rate terms break training; CAPS on the mean does not.
- **CAPS 0.1 (tcs1).** It brings the arm torque rate and the 3–8 Hz right-wrist shake down to the zero-residual level. The cost is 13 tubes seated from start (47 → 34). CAPS 0.5 (tcs5) costs more.
- **KA.** With no MPC in the loop, KA matches tc on the restart evals: held-out holes 81 vs 80 tubes seated, held-out placements 60 vs 60. It is the smoothest run. From start it seats 40 tubes, against tc's 47.
- **KB.** Lowering hand tracking and doubling the contact terms did not help (29 from start). Idle fingers drop from 21 to 14.
- **Training success is not task success.** The "success" in `progress.jsonl` (KA 0.84) starts mid-clip and does not check seating. It is therefore far above the from-start numbers.

## 5. KA failure analysis and the proposed next round

### Variation study

**Setup.**

- Rollouts: `rl/record_rollouts.py` with KA final, deterministic mean, DR on, on a local RTX 4080.
- Files: spec files and per-world outcomes are in `results/kin_tc_s5/vary/`.
- Summary command: `python3 rl/vary_summary.py results/kin_tc_s5/vary`.
- **Full success** means tube seated, pipette seated and press OK (the `eval_rr.py` press rule) at the end of the clip.
- Each cell has 16–20 worlds, so differences under about ±12 % are noise.

| Sweep | Worlds | Full success |
|---|---|---|
| `rand64`: training distribution. Random hole pair from the 88; rack and stand ±5 cm / ±10°. | 64 | 42 (66 %) |
| `holes`: all 100 hole pairs × 2, holders at nominal (`kin_tc_s5_eval_allholes.json`) | 200 | 151 (76 %): trained pairs 139/176 (79 %), never-trained 12/24 (50 %) |
| `standyaw`: stand yaw −45° … +45°, 16 worlds each, pair r0c3 → r1c3, holders otherwise nominal | 176 | 89 (51 %) |

The 12 never-trained pairs are every pick from r1c0 (10 pairs) plus the held-out r0c1 → r1c4 and r1c2 → r0c0.

**By hole** (`holes`, successes out of 20):

| Hole | r0c0 | r0c1 | r0c2 | r0c3 | r0c4 | r1c0 | r1c1 | r1c2 | r1c3 | r1c4 |
|---|---|---|---|---|---|---|---|---|---|---|
| Pick | 16 | 15 | 12 | 13 | 19 | **9** | 16 | 16 | 18 | 17 |
| Place | **11** | 20 | 17 | 17 | 13 | 15 | 15 | 16 | 16 | **11** |

**By stand yaw** (`standyaw`, successes out of 16; the training range is ±10°):

| Yaw | −45° | −30° | −20° | −15° | −10° | 0° | +10° | +15° | +20° | +30° | +45° |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Full success | 0 | 0 | 5 | 10 | 13 | 9 | 14 | 14 | 14 | 10 | 0 |

**First-failure causes**, by object and by time window of the reference:

| Cause | rand64 | holes | standyaw |
|---|---|---|---|
| Pipette: extraction from the stand | 5 | 0 | 53 |
| Pipette: carry / press | 0 | 0 | 3 |
| Pipette: return into the stand | 9 | 6 | 7 |
| Pipette: after the release | 0 | 1 | 1 |
| Tube: grasp / lift from the rack | 1 | 7 | 3 |
| Tube: carry / press | 1 | 1 | 0 |
| Tube: carry back | 1 | 14 | 2 |
| Tube: entering the rack | 2 | 10 | 9 |
| No termination, but not seated or no press at the end | 3 | 10 | 9 |

### Mechanisms

1. **Pipette return into the stand.** This is the largest cause in the training distribution, and it is not a joint limit.
   - The pipette is already off in the hand by 10 s: 6.4 mm / 3.2° in failing worlds, against 2.6 mm / 1.5° in successes.
   - Above the stand, the lateral error is 7–15 mm, against 3–5 mm.
   - Pipette failures (return + extraction) follow the stand offset: 10 of 24 worlds with an offset ≥ 45 mm fail, against 4 of 40 below.
   - The object-tracking term (axis points, σ 4 cm) is too loose to hold a mm-level in-hand pose.
2. **Left middle finger opens at 18.6–19.5 s.** The kinematic reference (the retargeted human demo) opens it while the tube is still held. The MPC reference keeps it closed until 21.8 s.
   - The left-finger residual is capped at ±0.05 rad (±2.9°), so the policy cannot keep the finger closed.
   - The tube then slides axially out of the remaining two-finger pinch. This accounts for 11 of the 14 "carry back" failures in `holes`, all at 18.5–19.3 s.
   - Opening the finger is not needed for clearance at the rack: the clearance is 38 mm.
3. **Pick from r1c0** (never trained): 9 of 20 succeed. The tube is lost while lifting out, or jams tilted.
4. **Stand yaw beyond the training range.**
   - **Negative yaw:** the right wrist joint `R_arm_j7` runs into its limit during the extraction. Its margin is 18.7° at 0°, 5° at −20° and 0° at −30°.
   - **+45°:** the reset re-anchoring IK (4 iterations) is off by 66 mm / 29°. The episode fails at 3.05 s, before the grasp.
   - **+30°:** 4 failures at the pipette return.
5. **Tube entering the rack**, around 20.6 s: 10 failures in `holes`, 9 in `standyaw`. The tube is lost or jams tilted.

Clips of each case can be rendered from the specs in `results/kin_tc_s5/vary/` (§8; videos are not committed).

### Proposed next round (not implemented)

Start from KA final and train 1–1.5e8 more steps. Run an ablation alongside with change 2 only.

1. Add a pipette in-hand pose penalty on the right hand, like `tube_tilt` (σ 5 mm / 3°). Also apply `pipette_seat` over the extraction window (3.8–5.4 s).
2. Keep the left middle finger closed in the reference until 21.8 s, as the MPC reference does.
3. Train on all 100 hole pairs.
4. Train with stand yaw ±15°. Raise the re-anchoring IK iterations from 4 to 10.
5. Keep the hand-tracking weights (KB showed that lowering them does not help), CAPS 0.1 and the DR.

Deferred: failure-weighted start sampling, and relaxing the relative-pose termination once the object is seated.

## 6. Teacher → student (DAgger)

`rl/distill.py`, using `obs_mode='deployable'` plus `extras.student`.

**Student observations (955-d):**

- 3-step history of the measured joint q, measured q̇, and previous action:
  - the measurements carry encoder offsets and obs noise;
  - finger q̇ is a first difference, because the Wuji drivers report position only;
- reference q and base command at 0 / 40 / 100 / 240 ms;
- plunger reference;
- phase one-hot;
- clip progress;
- both objects' poses measured **once** at the episode start (5 mm / 2° noise);
- the measured rack and stand offsets;
- the re-anchoring arm-command delta.

The student sees no object state after the start and no contacts.

**Training:**

- Each episode runs the teacher's action with probability β, otherwise the student's. β goes from 1 to 0 over `beta_iters`.
- The label is always the teacher's clamped action. Loss is MSE.
- Student: MLP 512-256-128 (ELU) with a running normaliser that freezes after 5e6 samples.
- Optimiser: Adam, learning rate 1e-3 → 2e-4.

**Status on the T1k line:** only a plumbing test, on an early task E1 with a smoke teacher (`rl/configs/rr_student_plumb.json`; task E1 not included). In `results/student_plumb/eval_student_plumb.json` the student reached the end in 14 of 20 runs. No student was distilled from R3B.

**The d02 line** has full students: wrist depth cameras, the head-camera estimate, hand pushes, data-parallel on 3 GPUs. See [Current work](#current-work-d02-demo-line).

## 7. Tracker (work in progress)

**Goal.** Train one RL "optimizer" that turns *any* kinematic reference (layer 1) of this task family into physically executable, natural motion, so new videos need no per-video MPC.

**Design** (`tracker/DESIGN.md`, in Chinese):

- The base is IK plus the finger reference, not an MPC demo. The residual uses the same scales as rrline.
- A virtual object controller (VOC) pulls each object to its reference. A curriculum lowers it 1 → 0 on success; 10 % of worlds always run at VOC 0.
- Playback speed varies per episode (time-warp 0.8–1.2). Holders move ±2 cm / ±5°, with hole pairs.
- Rewards are the R3 set plus `press_progress`.
- Contact labels are still the T1_k physics demo's (a deviation noted for Stage 0).

`tracker/tracker_env.py` subclasses `rl/vega_res_env.py`.

**Evals:** VOC 0, 64 runs from the start. "DR" here means only finger kd ×3.

| Checkpoint (config / task) | Reach end | Tube seated | Press | T1 miss (go rule ≤ 0.1) | File |
|---|---|---|---|---|---|
| Zero residual, kinematic base + squeeze prior (`stage0`, `task_kin_prior`), VOC 1 / 0.5 / 0 | 39 / 17 / 0 | 27 / 2 / 0 | 0 | – | `tracker/baselines/_summary.txt` |
| 50M (`stage0_noprior`) | 50 | 50 | 0/64 | 0.222 | `tracker/evals/go_50M_voc0.json` |
| **75M** (`stage0_noprior_r2`, `task_kin_noprior`) | **54** | **53** | 0/64 | 0.224 | `tracker/evals/go_75M_voc0.json`; checkpoint `tracker/evals/ckpt_75M.pt` |
| 100M (`stage0_noprior_r2`) | 26 | 26 | 0/64 | 0.243 | `tracker/evals/go_100M_voc0.json` |
| 152M (`stage0_noprior_r3`, press-prior base) | 34 | 15 | 56/64 | 0.276 | `tracker/evals/go_152M_voc0_T1.json` |

At VOC 0, carry, grasp and seating are learned, but there is no press without the press prior. With the press prior, presses work but seating drops. A run warm-started from 75M on the press-prior base, with an idle-finger mask, was in progress at the time (not included).

**Since then: KA / KB (§4).** These runs move the tracker trainer onto the full teacher setup, on the press-prior kinematic task:

- holders ±5 cm / ±10°, 88 hole pairs;
- R3 DR;
- reference-free contact terms (no T1_k labels);
- CAPS;
- VOC forced to 0 for the last half of training.

Their evals use `eval_rr.py` with the full DR. They are not comparable with the VOC-0 table above, where DR is only finger kd ×3.

## 8. Videos, rendering and perception

**Videos and figures are not in the repo.** Since 11-12, `.gitignore` excludes videos, images and logs. The earlier clips were removed from the tree in commit `65bc26e`, but git history still holds them. Regenerate any of them with the scripts.

They are MuJoCo renders of recorded rollouts (`rl/record_rollouts.py` → `rl/render_compare.py`) at 50 fps:

- Each row shows one policy: overview | right hand (pipette) | left hand (tube).
- The close-up cameras follow a smoothed wrist-object midpoint, so shake shows against a steady camera.
- Every compared policy gets the same world spec and DR seed.

The kinds that were made:

- the R2 / R3 rounds;
- layer 1 vs layer 2 references (`render_compare.py --ref`);
- CAPS rounds and wrist paths (`rl/ee_path.py`);
- KA / KB vs tcs1;
- the 12 KA failure cases of §5;
- a Blender replay (below);
- for d02, the v3 reference and teacher;
- a see-through tube close-up of the tip (`rl/render_tip_closeup.py`);
- the wrist-camera mount (`rl/render_wristcam_mount.py`).

`.blend` files are never committed, because they embed the robot and lab meshes.

**Rendering in Blender** (render only, nothing is simulated):

- `render/mj_to_blender.py` exports a recorded rollout for Blender: visual meshes, per-frame body poses, furniture anchors and the head camera frame. It handles only worlds with zero holder offsets.
- `render/blender_replay.py` runs inside Blender 5.2. It places the exported robot and objects in a lab scene ("LIDAR Environment.blend", not included) and renders the head and overview cameras. The head camera's field of view is a placeholder until the real intrinsics are known.

**Perception used by the d02 students** (simulation):

- `perception/wristcam_sim.py`: the wrist depth cameras and their stereo noise model;
- `perception/ego_holders.py`: the head-camera holder/tube estimate and its error model;
- `perception/ego_scan_bank.py` and `train_scan_net.py`: ScanNet.

**Perception bench tools:**

- `perception/record_realsense.py` records a RealSense (D435i, D405): colour video, raw metric depth per frame, and optionally a rosbag.
- `perception/d405_viewer.py` is a live D405 depth viewer.
- `perception/ffs_compare.py` compares, live, the D405's on-chip depth with Fast-FoundationStereo run on the D405's stereo pair. It runs in FFS's own venv; the checkpoints are not included.
- Raw captures (`perception/snapshots/`, `perception/recordings/`) are not committed.

## 9. How to run

**Environment.** Python 3.12, from `requirements_rl_head.txt`:

| Package | Version |
|---|---|
| torch | 2.12.0 (cu130) |
| mujoco | 3.17.0 |
| warp-lang | 1.19.0 |
| mujoco-warp | git `google-deepmind/mujoco_warp@4911167e3f2d` (`pip install --no-deps`) |
| rsl-rl-lib | 5.5.2 |
| tensordict | 0.15.3 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |

**Assets (not included).** Every `tasks/*/scene.xml` loads meshes from `../../assets/...`. The full list of 112 files and their origins is in `assets/asset_list.tsv`. With the original files in place, the rewritten scene compiles to an identical model; this was checked.

| Placeholder dir | Source | Files |
|---|---|---|
| `assets/vega/generated/` | RFT `vega_wuji_isaac6/assets/generated/vega_meshes` (from the Dexmate Vega URDF) | 15 |
| `assets/vega/dexmate_vega_1_collision/` | dexmate-urdf `robots/humanoid/vega_1/meshes/collision` | 1 |
| `assets/wuji/hand2_beta1/{left,right}/` | wuji-description `hand2/hand2_beta1/body/meshes` | 10 |
| `assets/wuji/g1_wuji_retargeting_generated/` | g1-wuji-retargeting `generated/meshes` (Wuji links) | 32 |
| `assets/lidar/` | LidarDexManip/g1-pipette-sim `assets/environments/lidar/meshes` | 30 |
| `assets/pipette/` | Convex pieces of the pipette mesh (g1-pipette-sim local output) | 24 |

**Commands.** These need a CUDA GPU. Paths in configs are relative to the config file.

```bash
cd rl
python -I train.py configs/rr_T1k_r3B_s5.json --out runs/r3B          # teacher (resumable; writes final.pt)
python -I eval_rr.py configs/rr_T1k_r3B_s5.json --ckpt ../results/rr_T1k_r3B_s5/final.pt --from-start 64 --out r3B_fromstart.json
python -I eval_rr.py configs/rr_T1k_r3B_s5.json --from-start 64 --out zero_fromstart.json   # zero residual, same perturbations
python -I eval_rr.py configs/rr_T1k_r3B_eval_holdout.json  --ckpt ../results/rr_T1k_r3B_s5/final.pt --repeat 4 --out ho.json
python -I eval_rr.py configs/rr_T1k_r3B_eval_holdplace.json --ckpt ../results/rr_T1k_r3B_s5/final.pt --repeat 4 --out hp.json
python test_arm_clamp.py                                               # CPU unit test
python test_contact_terms.py; python test_rate_terms.py; python test_caps.py   # CPU unit tests of the §4 terms
```

**Node scripts** (B200, or any node with CUDA 13 and driver ≥ 580). Run them from the repo root. `node_eval.sh` takes config paths relative to `rl/`.

```bash
./node_setup.sh                                          # uv venv from requirements_rl_head.txt + CPU unit tests
GPU=0 ./node_launch.sh rl/configs/rr_T1k_tcs1_s5.json    # one rl/train.py run, detached; resumes from latest.pt
CUDA_VISIBLE_DEVICES=1 .venv-rl-head/bin/python -I tracker/train_tracker.py tracker/configs/kin_tc_s5.json   # KA (VOC curriculum)
# 6 parallel evals (from start / train starts / held-out holes / held-out placements / contacts / reward breakdown):
GPUS="0 3 4 5 6 7" ./node_eval.sh kin_tc_s5 ../tracker/configs/kin_tc_s5.json ../tracker/configs/kin_tc_s5_eval_holdout.json \
    ../tracker/configs/kin_tc_s5_eval_holdplace.json ../results/kin_tc_s5/final.pt
python3 rl/compare_runs.py tc=rl/runs/local_eval/rr_T1k_tc_s5 KA=rl/runs/local_eval/kin_tc_s5 --json cmp.json
```

**Variation study and videos** (one GPU). Run from `rl/`:

```bash
python -I record_rollouts.py ../tracker/configs/kin_tc_s5.json ../results/kin_tc_s5/final.pt runs/vary/rand64 --worlds ../results/kin_tc_s5/vary/specs_rand64.json
python -I record_rollouts.py ../tracker/configs/kin_tc_s5_eval_allholes.json ../results/kin_tc_s5/final.pt runs/vary/holes --worlds ../results/kin_tc_s5/vary/specs_holes.json
python3 vary_summary.py runs/vary
MUJOCO_GL=egl python render_compare.py ../tasks/task_kin_noprior_press/scene.xml OUT --run runs/vary/rand64 KA --world 37
```

`G1PS_ROOT` defaults to `deps_from_g1ps/`. That folder holds the few files of ours from the lab repo that the code imports: `tools/rl/gpu_train.py`, `gpu_env.py`, `gpu_replay.py`, `g1_pipette_sim.retarget.gpu.core`, `g1_pipette_sim.physics.mjw_head`.

**Not runnable from this repo:**

- `rl/export_task.py`, `tracker/build_kin_task.py`, `jitter_whole.py`, `slowmo_closeup.py`, `data_gate.py`: they need the DAID physics-demo outputs (not included). `export_task.py` also needs `tools/export_residual_ppo_task.py` from LidarDexManip/g1-pipette-sim. That file is pinned at `4b80f93195d098cbc9c23edb648b3eb6c0d33e78`, but `load_physics_stage` exists only in a later, uncommitted version.
- `rl/render_rollout.py`: needs our LIDAR-visual renderer and scene (not included).

**Credits.** The tracking reward structure, termination thresholds and default residual scales follow Yi-Shiuan Tung's residual-PPO env (g1-wuji-retargeting). The code here is our own GPU reimplementation. Jiming Ren proposed T1–T3. DAID (do-as-I-do) produced the physics demo.

**Files.** See `MANIFEST.md`, which lists every file with its source and size (its table covers the files up to the 11-10 commit; later files are listed at its end), and `AUDIT.md`. `docs/` holds the original plan and results notes, with node and host details scrubbed.
