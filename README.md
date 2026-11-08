# Vega-U + Wuji Hand: residual RL for the pipette / tube task

Bimanual task, learned from one human video. The **left** Wuji hand picks a tube from a rack. The **right** hand takes a pipette from its stand. The pipette tip goes into the tube and the right thumb presses the plunger. Both objects go back: the tube into another hole, the pipette into its stand.

Robot: Dexmate Vega-U, two 7-DoF arms, each with a 20-DoF Wuji hand2 beta1. Fixed base, lift held.

Everything here is **simulation**. Nothing has run on the real robot yet. Each number below names the file it comes from. Robot, hand, lab and pipette meshes are **not included** (see [Assets](#6-how-to-run)).

## 1. Pipeline layers

| # | Layer | What it is | Artifacts here |
|---|---|---|---|
| 1 | Human video → **kinematic reference** | Reconstruction, retargeting and IK to Vega-U + Wuji. No physics. | Not included. `tasks/task_kin_*` are built from it (tracker line only). |
| 2 | **Physics demo** (MuJoCo) | DAID ("do-as-I-do") sampling MPC tracks layer 1 in MuJoCo. Its executed controls are a physically valid demo ("T1_k", seed 0). | Not included. Its executed states and ctrl are exported into `tasks/task_T1k_*/reference.npz` (`rl/export_task.py`). Contact labels come from it via `rl/contact_ref.py`. |
| 3 | **Residual-PPO teacher** (MuJoCo Warp, GPU) | The policy adds a joint-space residual on top of layer 2's executed ctrl, under heavy randomisation. | `rl/`, `rl/configs/`, `results/<run>/final.pt`, `videos/` |
| 4 | **DAgger student** (deployable observations) | `rl/distill.py`. Only plumbing-tested so far. | `results/student_plumb/` |
| — | Tracker (WIP) | One RL policy for any kinematic reference, with no per-video MPC. | `tracker/` |

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

## 4. Teacher → student (DAgger)

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

**Status:** only a plumbing test, on an early task E1 with a smoke teacher (`rl/configs/rr_student_plumb.json`; task E1 not included). In `results/student_plumb/eval_student_plumb.json` the student reached the end in 14 of 20 runs. The deployable student has not been distilled from R3B yet.

## 5. Tracker (work in progress)

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

At VOC 0, carry, grasp and seating are learned, but there is no press without the press prior. With the press prior, presses work but seating drops. A run warm-started from 75M on the press-prior base, with an idle-finger mask, is in progress (not included).

## 6. How to run

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
```

`G1PS_ROOT` defaults to `deps_from_g1ps/`. That folder holds the few files of ours from the lab repo that the code imports: `tools/rl/gpu_train.py`, `gpu_env.py`, `gpu_replay.py`, `g1_pipette_sim.retarget.gpu.core`, `g1_pipette_sim.physics.mjw_head`.

**Not runnable from this repo:**

- `rl/export_task.py`, `tracker/build_kin_task.py`, `jitter_whole.py`, `slowmo_closeup.py`, `data_gate.py`: they need the DAID physics-demo outputs (not included). `export_task.py` also needs `tools/export_residual_ppo_task.py` from LidarDexManip/g1-pipette-sim. That file is pinned at `4b80f93195d098cbc9c23edb648b3eb6c0d33e78`, but `load_physics_stage` exists only in a later, uncommitted version.
- `rl/render_rollout.py`: needs our LIDAR-visual renderer and scene (not included).

**Credits.** The tracking reward structure, termination thresholds and default residual scales follow Yi-Shiuan Tung's residual-PPO env (g1-wuji-retargeting). The code here is our own GPU reimplementation. Jiming Ren proposed T1–T3. DAID (do-as-I-do) produced the physics demo.

**Files.** See `MANIFEST.md`, which lists every file with its source and size, and `AUDIT.md`. `docs/` holds the original plan and results notes, with node and host details scrubbed.
