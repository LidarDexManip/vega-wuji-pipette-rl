# Vega-U + Wuji Hand: residual RL for the pipette / tube task

Bimanual task, learned from one human video. The **left** Wuji hand picks a tube from a rack. The **right** hand takes a pipette from its stand. The pipette tip goes into the tube and the right thumb presses the plunger. Both objects go back: the tube into another hole, the pipette into its stand.

Robot: Dexmate Vega-U, two 7-DoF arms, each with a 20-DoF Wuji hand2 beta1. Fixed base, lift held.

Everything here is **simulation**. Nothing has run on the real robot yet. Each number below names the file it comes from. Robot, hand, lab and pipette meshes are **not included** (see [Assets](#9-how-to-run)).

## Status (2026-11-10)

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
- **Not done yet:** no new student has been distilled, and nothing has run on hardware.

## 1. Pipeline layers

| # | Layer | What it is | Artifacts here |
|---|---|---|---|
| 1 | Human video → **kinematic reference** | Reconstruction, retargeting and IK to Vega-U + Wuji. No physics. | Not included. `tasks/task_kin_*` are built from it (tracker line only). |
| 2 | **Physics demo** (MuJoCo) | DAID ("do-as-I-do") sampling MPC tracks layer 1 in MuJoCo. Its executed controls are a physically valid demo ("T1_k", seed 0). | Not included. Its executed states and ctrl are exported into `tasks/task_T1k_*/reference.npz` (`rl/export_task.py`). Contact labels come from it via `rl/contact_ref.py`. |
| 3 | **Residual-PPO teacher** (MuJoCo Warp, GPU) | The policy adds a joint-space residual on top of layer 2's executed ctrl, under heavy randomisation. | `rl/`, `rl/configs/`, `results/<run>/final.pt`, `videos/`. Rounds R1–R3 (§3), tc line (§4). |
| 4 | **DAgger student** (deployable observations) | `rl/distill.py`. Only plumbing-tested so far. | `results/student_plumb/` |
| — | Tracker (WIP) | One RL policy for any kinematic reference, with no per-video MPC. | `tracker/`. KA / KB (§4) are residual teachers on the kinematic reference, with no MPC base: `results/kin_tc*_s5/`. |
| 1′ | Retargeted RGB-D demo (HF `ytung/pipette-tasks`) → **task** | One command: robot square to the table, ground-truth holders, reachable continuous arm paths, checks, training configs. See [docs/DEMO_TO_TASK.md](docs/DEMO_TO_TASK.md). | `tracker/demo_to_task.py`, `tracker/build_demo_task.py`, `tasks/task_d02_straight_v2/` (smooth extraction; teacher `kin_d02v2_ft_g3`, fine-tuned from `kin_d02_s5_g3`). |
| 1″ | Head-camera (ego) estimate of the holders and the tube | One depth frame at the start → stand / rack pose, tube position and pick hole; its error model drives the student's holder measurement. See [docs/EGO_HOLDERS.md](docs/EGO_HOLDERS.md). | `perception/ego_holders.py`, `tasks/task_d02_straight_v2/ego/`, student config `rl/configs/student_d02v2_ego.json`. |

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

Clips of each case are in `videos/kin_failures/` (§8).

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

**Status:** only a plumbing test, on an early task E1 with a smoke teacher (`rl/configs/rr_student_plumb.json`; task E1 not included). In `results/student_plumb/eval_student_plumb.json` the student reached the end in 14 of 20 runs. The deployable student has not been distilled from R3B yet.

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

**Videos.** Unless noted, these are MuJoCo renders of recorded rollouts (`rl/record_rollouts.py` → `rl/render_compare.py`) at 50 fps.

- Each row shows one policy: overview | right hand (pipette) | left hand (tube).
- The close-up cameras follow a smoothed wrist-object midpoint, so shake shows against a steady camera.
- Every compared policy gets the same world spec and DR seed.

| Path | What |
|---|---|
| `videos/*.mp4` (top level) | R2 / R3 videos (§3) |
| `videos/reference_layers/` | Layer 1 (kinematic reference) vs layer 2 (MPC physics demo), replayed with no policy. Full clip, plus press 8–12 s and return 17–24 s at 0.5×. |
| `videos/caps_rounds/` | R3B vs tcs1 (CAPS 0.1) vs tcs5 (CAPS 0.5), worlds 2 and 3, and carry at 0.25×. `ee_path_world2.png`: wrist paths vs the reference (`rl/ee_path.py`). |
| `videos/kin_ref_final/` | tcs1 (MPC reference) vs KA vs KB (kinematic reference), final checkpoints, worlds 1–3, and press at 0.5×. Wrist-path figure. |
| `videos/kin_failures/` | The 12 KA failure cases of §5, each labelled with its world spec and fail reason, plus all 12 in one montage at 0.5×. |
| `videos/blender/KA_final_world2/` | KA final, world 2, replayed in a Blender lab scene: robot head camera, fixed overview, and both side by side. |

**Not committed:**

- per-policy single clips, which the `compare_*` videos already show side by side;
- the 175M-step midpoint renders;
- `.blend` files, which embed the robot and lab meshes.

**Rendering in Blender** (render only, nothing is simulated):

- `render/mj_to_blender.py` exports a recorded rollout for Blender: visual meshes, per-frame body poses, furniture anchors and the head camera frame. It handles only worlds with zero holder offsets.
- `render/blender_replay.py` runs inside Blender 5.2. It places the exported robot and objects in a lab scene ("LIDAR Environment.blend", not included) and renders the head and overview cameras. The head camera's field of view is a placeholder until the real intrinsics are known.

**Perception bench tools** (not used by the RL yet):

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

**Files.** See `MANIFEST.md`, which lists every file with its source and size, and `AUDIT.md`. `docs/` holds the original plan and results notes, with node and host details scrubbed.
