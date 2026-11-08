# RL plan — real-robot line (Vega-U + Wuji beta1, run01, layout T1(b))

Status 11-06 02:10. Starts the moment a T1_j physics seed passes every non-deferred item (gate rrline-t1-v2). T1_j seeds 0/1/2 running.
Layers: "physics demo" = vega_mpc / DAID MPC rollout; "RL env" = MuJoCo Warp batched env (`rrline/vega_res_env.py`), residual PPO (rsl_rl 5.5.2).

## 1. What the teacher trains on
- **Demo**: the first T1_j seed (0/1/2) whose `status_T1.json` has no non-deferred failing item. Gate rrline-t1-v2 = vega_mpc gate +
  seats + tip + IK (rolled target) + pipette-in-wrist (approved window) + T1 layout + `thumb_rest_pose` (hard).
- **RL reference** (`export_task.py STAGE task_T1j --smooth-fingers`):
  - objects, wrists, arm ctrl and grasp-finger ctrl = the physics demo, executed ctrl replayed verbatim;
  - grasp fingers lightly smoothed (zero-phase 0.1 s) in `ctrl_sim` and `qpos_ref`;
  - free fingers = the smoothed stage reference (0.2 s), not the physics jitter;
  - right thumb = the new natural start pose and the two-stage relaxed end pose; the wrist withdraws after release;
  - zero-residual replay must still reach the end on 20/20 starts with no DR (checked at export).
- **Object-relative**:
  - wrist and object references are re-anchored per episode to the sampled rack and stand placement (IK re-anchoring, holder blend weights);
  - the teacher sees the true state;
  - the re-anchoring itself uses the *measured* holder pose (5 mm / 2° noise), the same estimate the student gets once at the start.

## 2. Randomisation (per episode, same set for policy and zero residual)
| item | train range | held-out test |
|---|---|---|
| start frame | any audited valid start (T1b: 1099) | frame 0 only |
| object start in holder | ±2 mm, ±3° | same |
| rack placement (task frame) | ±5 cm xy, ±10° yaw (wide: learn the process) | ±6–7 cm and yaw outside the train set: 8 fixed placements |
| stand placement | ±5 cm xy, ±10° yaw | same scheme |
| holder measurement error | 5 mm, 2° (Gaussian) | same |
| pick / place hole | per-episode pair from the working set: 9 pick holes (all but r1c0) × 10 place holes = 90 pairs, minus the 2 held out; offsets from this scene's own bore rings, rack-tied, composed before the rack placement | r0c1→r1c4, r1c2→r0c0 |
| arm gains | kp ×0.5–2, kv ×0.7–1.4 | same |
| finger drive | kp, kv ×0.7–1.4 | same |
| friction | hand-object ×0.7–1.3, pipette-tube ×0.7–1.3 | same |
| object mass | ×0.8–1.2 | same |
| encoder offsets | arm ±0.01 rad, fingers ±0.02 rad | same |
| observation noise (student) | q 0.005 rad arm / 0.01 rad fingers, qdot 0.05 rad/s | same |

- Command path, always on:
  - 50 Hz policy, residual EMA α 0.5;
  - the residual is linearly upsampled to a 100 Hz stream with velocity feed-forward;
  - arm targets ≥0.1 rad inside every limit, ≤2.5 rad/s (`vega_extras.arm_command`, `cmd_hz` 100).
- Holder DR uses the per-world static-geom fix: `d.geom_xpos` and `geom_xmat` are written at reset.
  - Zero residual, placement only (T1_g task, no other DR, reach end): ±2 cm/±5° 63/63, ±2.5/±6 62/63, ±3.5/±7.5 62/63, ±5/±10 59/63.
  - Hole pairs (11-06, zero residual, rack nominal): 90/100 pick, lift, insert and seat; every r1c0 pick loses the tube in transport (edge hole, left out). Containment checked on the compiled env model: the tube axis starts in the intended bore ring 100/100 (≤0.01 mm) and ends in it for 90/90 (≤0.14 mm).
  - Two fixes behind that: hole centres were taken from rack_visual/profile.json, whose frame is yawed ~190° vs this scene (every pair aimed at the mirrored neighbour); and the rack blend weight is now 1 while the tube is engaged with its hole (insertion depth + 15 mm).
  - The demo's place stays the human's diagonal drop (no axial approach: coverage met without it).
  - Still to run on the T1_j task: hole pairs × placement ±2 cm/±5° and ±5 cm/±10°.

## 3. Reward (weights; tracking terms as in `gpu_env`)
| term | weight | note |
|---|---|---|
| object / wrist / joint tracking | `gpu_env` defaults | holder-adjusted references |
| Jiming T1 contact set / T2 contact graph / T3 wrench support | 0.5 / 0.5 / 0.5 | labels from the physics demo (force >0.1 × weight) |
| free-finger acceleration | 1.0 | free mask from the demo labels |
| action rate | 0.5 | plus EMA α 0.5 on the residual |
| elbow above shoulder | 50 | hard rule, also a termination check |
| joint margin (<0.1 rad) | 1.0 | |
| thumb rest-pose prior | 0.5 (new) | deviation from the reference thumb in the pre-grasp / post-release windows |
| finger jitter (velocity reversals >0.3°) | 0.2 (new) | per finger, same quantity as `jitter2.py` |
| posture prior (arm joints vs demo) | 0.1 (new) | |
| tube spin about its own axis | 0 | not penalised (user) |

## 4. Acceptance (RL env layer)
Same perturbations and seeds for the policy and for zero residual; 3 × 21 starts plus 8 held-out placements.

The policy must beat zero residual on:
- reach the end;
- pipette seated, tube seated;
- tip in tube;
- press;
- tube drift in hand, axis tilt.

It must also pass under perturbation:
- naturalness: jitter v2, thumb_rest_pose, elbow, joint margin, action rate;
- the generalisation test on the held-out placements: beat zero residual there too.

Task success without naturalness is not delivered.

| threshold | status |
|---|---|
| elbow not above the shoulder | user (hard) |
| joint margin ≥0.1 rad, ≤2.5 rad/s | user (real-robot rules) |
| pipette in wrist ≤5 mm, window grasp-closed → before release | user (11-05) |
| IK ≤10% of frames over 2 mm / 2° on the rolled target | user (11-05) |
| wrist links ≥2 cm from the rack | user (relaxed from 4 cm, 11-05) |
| tube spin not judged; thumb_pose and tube slip deferred (printed failing) | user |
| thumb_rest_pose: (b) abd ≥0.1 rad | user rule |
| thumb_rest_pose: (a) ≤20° from human, (c) ≥2 mm under ±0.15 rad | PROPOSED |
| jitter v2, deciding: 3–10 Hz band RMS ≤0.25° per finger and window (hold, post-release); pre-grasp closure cells reported separately (closure kept unsmoothed with noise on, grasp first); SG residual / reversals logged only | PROPOSED |
| tube slip (deferred, printed): tube in the left wrist vs the demo in-hand pose at the grasp-dwell end, over [dwell end, held end] (check_physics_tube_grip definition); zero residual 11.0 / 11.1 mm p50 / max | user: deferred |
| free-finger HP RMS ≤0.6° | PROPOSED |
| RL-eval press | absolute ≥1.5 s at ≥90% decides; relative 0.9 × demo PROPOSED |
| "beats zero residual" = strictly more on every count item and lower median drift / tilt | PROPOSED |

## 5. Compute
Throughput measured on the 5080 (RL env, 1024 envs):
- E1 smoke alone: about 19k steps/s;
- T1 smoke with a physics worker on the GPU: 7.5k steps/s;
- VRAM 0.6 GB.

Learning: the E1 smoke (8.5M steps, 20 min) already beat zero residual (38/60 vs 17/60 reach the end). Wide placement DR plus the extra terms need more, about 1–2 × 10⁸ steps to plateau (the G1 line's W2_s1 ran 1e8).

- **(a) Local 5080, 3 h**:
  - alone (no physics running): about 15–19k steps/s → 160–200M steps;
  - shared: about 80M.
  - Enough for the fixed-placement teacher. Marginal for wide placement plus hole changes.
- **(b) B200 node**:
  - `train.py` already has configurable env count and device, resume (latest.pt / state.json), done-marker, and wall-time and step caps.
  - **Missing**:
    - (1) multi-GPU: single process today. Either add rsl_rl `multi_gpu` under torchrun, or run independent configs/seeds, one per GPU, which works now.
    - (2) hard-coded `<home>/...` paths in `train.py`, `distill.py`, `eval_rr.py` and `vega_res_env.py`, and absolute mesh paths in the task `scene.xml`: needs a ROOT env var plus a relocatable task copy.
    - (3) the `.venv-rl-head` stack (mujoco 3.17, warp 1.19, torch 2.12, rsl_rl 5.5.2) on the node.
  - Estimate (not measured): 4096–8192 envs per B200, about 3–5× the 5080 → 50–80k steps/s per GPU → 0.5–0.9 × 10⁹ steps per GPU in 3 h.
  - Example ask: 2 GPUs × 3 h, two independent teacher configs (fixed vs wide placement) or two seeds.
  - The user decides GPUs and hours; nothing is submitted.
- **Decided (user, 11-06)**: no separate fixed-placement round. The first run is on the B200 node with the full config: wide placement, hole pairs (working set above), all reward terms, smoothed reference, 100 Hz path. GPUs and hours are the user's call.
- **Node readiness** (CPU only, nothing submitted): (2) done — `G1PS_ROOT` in train / distill / eval_rr / vega_res_env, relocatable task copy (meshes in `assets/`, model identical); (3) done — env `<node_data>/envs/rl_head` (torch 2.12+cu130, mujoco 3.17, warp 1.19, rsl_rl 5.5.2) verified on CPU, task loads; (1) one config/seed per GPU via `node_pkg/launch_node.sh` (CUDA_VISIBLE_DEVICES, 15 s apart). The package is rebuilt with the final task (task_T1j) after the data gate.

## 6. Pre-launch checklist (standing user rules)
1. **Data gate (hard)**:
   - demo seed passes every non-deferred item;
   - exported reference: zero-residual replay 20/20 with no DR;
   - contact labels sane (per-finger fractions match the demo);
   - valid-start audit kept ≥95%;
   - smoothed fingers differ from the demo by ≤1° on grasp fingers.
2. Print the merged config (env, DR, extras, PPO):
   - check LR 3e-4 adaptive (desired KL 0.01), no warmup, init std 0.1, reward scale 0.01;
   - group scales FR 0.1, AR 0.04, FL 0.05, AL 0.04.
3. zero_grad check: one PPO update on a frozen batch; gradients are zero before backward and finite after.
4. All planned items are in before any run over 20 min: wide placement, hole pairs, new reward terms, smoothed reference, 100 Hz path.
5. Training speed checked within 10 min of any other GPU job starting (and logged).
6. Never hold the gpu.lock for training; free VRAM ≥8 GB at launch; detached with a done-marker; checked every ≤20 min; local runs ≤3 h.

## 7. After the teacher: deployable student
- **Observations**:
  - joint encoders with noise, 3-step history;
  - previous actions;
  - reference tables;
  - object poses measured once at the start (5 mm, 2°) and holder estimates;
  - re-anchoring arm delta.
  - Code ready: `vega_extras.student_*`, `obs_mode` deployable.
- **Training**: DAgger (`distill.py`), β 1 → 0 over 100 iterations, MSE on clamped teacher actions, same DR plus observation noise. The plumbing test ran (4 min, 14/20 reach the end).
- **Acceptance** (same eval):
  - beats zero residual on every task item;
  - reaches the end at ≥90% of the teacher's rate on the same perturbations;
  - passes the same naturalness gates;
  - passes the held-out placement test.
- Then sim videos (labelled RL policy, student / teacher layer) go to <projects>.

## 8. Node results 11-06 (B200, GPUs 6-7, 15:58-19:12 CST; layer: RL env, policy vs zero residual on identical perturbations)
Table: node_results/final_table.json (final_table.py); checkpoints and eval JSONs under node_results/<run>/.
- R1 (rr_T1k_full_lp_s0, base task_T1k_lp, 1.18e8): task good (100M: whole task from start 40/64 reach end, tube 21, press 49/64); finger jitter = base (11 cells > 0.25).
- R2 s2 (rr_T1k_r2_s2: base without finger velocity FF, finger kd x3, finger_band 1.0; from scratch, 1.46e8) = RECOMMENDED teacher:
  from start 42/64 end, tube 36, press 53/64 (zero 2 / 1 / 14); train 84/92, tube 73, press 34/36; held-out holes 75/92, tube 68, 34/36; held-out
  placements 57/92, tube 48, 30/36; jitter cells > 0.25: 6 (left thumb / index / middle hold 0.09 / 0.09 / 0.20; ring 0.26, pinky 0.29, right thumb hold
  0.29, press 0.14, left middle post-release 0.32); slip p50 14.8 mm all, matched 15.1 vs 21.0 mm (n = 2); action rate 1.0e-3.
- R2b (warm start from R1 final with the R2 settings): press collapses (13/36): the R1 residual does not transfer to the new base.
- Still failing acceptance: 4 jitter cells slightly over 0.25; commanded arm margin 5.24 deg < 5.73 (bug: the residual velocity FF is added after the
  0.1 rad clamp in arm_command); tube seat misses = tilted tube jammed ~5 mm high (in-hand tilt ~7 deg), all holes.
