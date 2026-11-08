# Node plan — round 3 (real-robot line, Vega-U + Wuji beta1) — RAN 11-06 11:13–14:16 EST (node GPUs 6, 7); results at the end

Needs the user's GPU / hours decision. Everything below is RL-env layer (MuJoCo Warp, residual PPO), node <user>@<node>.

## What changes vs round 2 (teacher R2 s2)
| change | why (11-06 evidence) | where |
|---|---|---|
| `clamp_after_ff: true` — commanded arm target clamped ≥0.1 rad inside every limit **after** the residual's velocity feed-forward | R1/R2 commanded margin 5.24–5.45° < 5.73° rule; unit test: old path −0.075 rad, new 0.100 rad (`test_arm_clamp.py`, PASS) | `vega_extras.arm_command` (default off) |
| `tube_tilt` reward 1.0 — tube axis tilt in the left hand vs the reference in-hand pose, while held; 1 = 0.05 rad (2.9°) | every tube seat miss = tube tilted ~8° jammed ~5 mm high (lateral 1.5 mm), all holes / placements | `vega_extras.reward` (default 0) |
| `finger_band` 2.0 (A) / 3.0 (B), was 1.0 | 4 jitter cells left at 0.26–0.32° (ring / pinky hold, right thumb hold, left middle post-release) | config |
| right thumb **not** exempted from the finger damping | R2 s2 press was the best of all candidates (53/64 from start); R2b's press loss came from the warm start | — |
| from scratch | R2b: R1 residual warm-started onto the new base broke the press (13/36) | — |
| `arm_band` reward 1.0 — causal ~3–10 Hz band (MA 2 − MA 8 control steps) on the 14 arm joints; 1 = 0.002 rad (≈1 mm at the wrist) | user saw the R2 s2 videos as shaky; wrist deviation from the reference in the carry under DR: 0.5–3 Hz 2.1 / 3.2 mm, 3–8 Hz 0.45 / 0.72 mm (R / L) for the policy and the same for zero residual with DR (3.1 / 3.1, 0.52 / 0.54) vs zero residual DR off 0.27 / 0.85, 0.05 / 0.38 mm: the policy inherits the DR ringing, it must damp it | `vega_extras.reward` (default 0) |
| arm DR coupled: `kv_g1_x_sqrt_kp: [0.95, 1.4]` — arm kv factor = √(kp factor) × U, so ζ' = ζ0·U ≥ 0.70 on the weakest joint; **kp range kept 0.5–2** | arm servo (kp 500/400/250, kd 35/20/12, real link inertias + armature, pose at 9 s): j1–j4 ζ 0.74–1.05, f_n 3.3–5.3 Hz at nominal; the independent draws (kp ×0.5–2, kv ×0.7–1.4) reach ζ 0.37–0.52 = an underdamped arm ringing at 3–7 Hz. Evidence against an underdamped real arm: the Dexmate factory-PID sine benchmark (dexcontrol `examples/benchmark/arm_tracking/`, 1 Hz, 0.4 rad, pos + vel FF, 200 Hz) shows a soft, lagging response (j1/2/5/6/7 ≈77 % amplitude, ≈0.1 s lag; j3/4 ≈96 %, 0.07 s), no overshoot or ringing | `vega_res_env` DR (default absent) + `configs/dr_rr1_kv3_zeta.json` |
| video rendering at 50 fps for 50 Hz rollouts | 30 fps nearest-frame picking stepped 2, 1, 2, 2, 1 control steps per frame = visible judder (artifact) | `render_rollout.py --fps 50` |

Unchanged: base `task_T1k_lp_noff` (demo T1_k s0, latched-finger 3 Hz LP, whole-press thumb LP, no finger velocity FF), finger kd ×3 (DR 2.1–4.2),
placement ±5 cm / ±10°, 88 training hole pairs (held out r0c1→r1c4, r1c2→r0c0), DR1, all other reward terms, 100 Hz arm path, 4096 envs.

## Ask: 2 GPUs × 3.5 h
- GPU a: `rr_T1k_r3A_s4.json` (finger_band 2.0, seed 4), 2e8 steps, 3.0 h wall cap (≈20k steps/s → ~2.8 h), checkpoints every 2.5e7.
- GPU b: `rr_T1k_r3B_s5.json` (finger_band 3.0, seed 5), same budget.
- Last 0.5 h: eval on the same 2 GPUs (below), results + best checkpoint fetched to `rrline/node_results/`.
- If only 1 GPU: run A only.

## Pre-launch checklist (standing rules)
1. Local GPU smoke under the shared lock (≈1 min): one PPO iteration of each config (new terms run, finite, `_tube_tilt_deg` logged; arm_band /
   tube_tilt / finger_band penalties finite; coupled DR draws give zeta >= 0.70 on every arm joint — print the drawn kp / kv factors).
2. Zero-residual replay with `clamp_after_ff` on, no DR: still 23/23 reach the end and press 9/9 (the clamp must not cost the base).
3. Package (`make_node_pkg.sh ../task_T1k_lp_noff configs/rr_T1k_r3*.json …`), sha256 check on the node, nvidia-smi before / after launch, CUDA_VISIBLE_DEVICES per run, 15 s apart, kill by PID only.
4. Speed check within 10 min, then every ≤20 min.

## Evaluation (policy vs zero residual, identical perturbations, eval_rr seed 123)
- Whole task from the start: 64 worlds (`--from-start 64`): reach end, pipette / tube seated, press, slip / tilt (`slip_compare.py`, full-window, matched).
- Train perturbations, held-out hole pairs, held-out placements (6.5 cm, ±12°): 92 starts each (`eval_round.sh`).
- Seat misses (`seat_miss_analysis.py`): lateral / height / tilt, release timing (probe fixed), by hole and placement.
- Naturalness: jitter v2 cells (3–10 Hz, PROPOSED 0.25°) — named: left hold fingers, right press thumb, post-release; thumb rest pose (abd margin);
  **shake (user rule 11-06: no shaking under any perturbation; measured WITH DR, policy vs zero residual on the same perturbations):** wrist
  position deviation from the reference, band RMS per hand and phase (carry, press, post-release): 3–8 Hz ≤ 0.3 mm and 8–24 Hz ≤ 0.1 mm
  (PROPOSED), 0.5–3 Hz reported; arm joint deviation band RMS (worst joint) reported in the same bands (eval_rr `shake` block). The policy must be
  below the zero residual and below the thresholds (it has to damp the DR ringing, not inherit it).
  Reference values 11-06 (R2 s2, carry, DR on): 3–8 Hz 0.45 / 0.72 mm (fails), 8–24 Hz 0.07 / 0.11 mm (borderline); DR off zero residual 0.05 / 0.38 mm;
  elbow; **commanded and executed arm margin** (rule ≥5.73° on the commanded target); action rate.
- Compare against R2 s2 final (current recommended teacher) in `final_table.py`.

## Acceptance to call a teacher "delivered"
Beats zero residual on every task item; all jitter cells ≤0.25° on hold / press / post-release (PROPOSED threshold, user decides); wrist shake
3–8 Hz ≤0.3 mm and 8–24 Hz ≤0.1 mm under DR and below the zero residual (PROPOSED); commanded arm
margin ≥5.73°; elbow below the shoulder; same under the held-out sets. Then: DAgger student (`distill.py`), sim videos labelled by layer.

## Updates 11-06 evening (local, CPU + short holds on the shared GPU lock; node untouched)

### Base: fingers hold still between events (user: "fingers move when they don't need to"; both hands, all 40 joints)
`task_T1k_still3_noff` = T1_k s0 export (latched-finger 3 Hz LP, whole-press thumb LP) + `--fingers-still --still-kd-scale 3` + no finger FF.
Every finger joint holds a constant target between events and makes one minimum-jerk transition per event: pre-shape (late, ending 0.3 s before
the closure window), the left ring / pinky rack-clearance curl (70–100°) moved once EARLY at the start of the reach (a late 100° move near the rack
was the alternative; early keeps the hand still near the rack), opening after each release (starts at the release). Untouched: latched rows, the
0.6 s closure windows (joint and hand), the right-thumb press and its way onto the cap. Transition speed per joint = what the drive can follow
(damping torque ≤ half the force limit at kd ×3: 0.75–2.0 rad/s); v2 (2 rad/s for all) left the fingers catching up 7–11° after the opening.

Zero-residual replay, R3 settings (clamp_after_ff, coupled arm DR), RL env:
| condition | still v3 | still v2 | old base lp_noff |
|---|---|---|---|
| no DR, nominal (n 23): end / tube / press | 23 / 23 / 9/9 | 23 / 23 / 9/9 | 23 / 23 / 9/9 |
| holders ±2 cm, physical DR off (n 46) | 35 / 25 / 15/18 | 36 / 28 / 15/18 | 34 / 25 / 14/18 |
| DR on, nominal (n 46) | 17 / 11 / 9/18 | 19 / 12 / 10/18 | 15 / 10 / 10/18 |
Idle fingers (no DR; [max range in an idle window deg, RMS speed deg/s], PROPOSED ≤2° / ≤3°/s): pre-grasp 0.0 / 0.0 on every finger of both
hands (the user's start complaint); still over: right thumb hold 3.2° / 5.9, right thumb post-release 3.7° / 11.4, right pinky hold 2.7° / 5.5,
left middle / ring post-release 2.4° / 2.9, 2.0° / 2.1 (contact / drive lag in physics; `finger_still` targets these).
(Note: `--no-dr` also disables the holders; the holder rows use `dr: {}`.)

### Rewards added (all default 0 / off; R3 configs set them)
`finger_still` 1.0 (finger speed where the reference finger is still, all 40 joints, 1 = 0.1 rad/s), `arm_band` 1.0 and `finger_band` 2.0 / 3.0
now on the deviation from the reference (`band_on_deviation`: on raw q they charged the intended motion — smoke means 53 → 3.1 and 8.3 → 0.8),
`tube_tilt` 1.0, `clamp_after_ff`, coupled arm DR (ζ ≥ 0.70).

### Event gate (deployable: wrist FK vs its event pose from the robot's own holder estimate; default off) — NOT enabled in R3
Zero residual, whole task from frame 0 (64 worlds), still v2 base: gate (closure + release, 3 mm / 3°, cap 1 s) on vs off:
no DR nominal 23 / 23 / 9 vs same; DR on nominal end 6 / tube 2 / press 25 vs 8 / 3 / 24; holders ±5 cm physical DR off 16 / 6 / 28 vs 16 / 9 / 27;
holders ±5 cm + DR 2 / 0 / 18 vs 1 / 0 / 17. The wrist rarely gets within 3 mm / 3° (left-closure gate held to the 1 s cap in 55–64 of 64 worlds),
so the gate mostly just waits and then proceeds; no benefit for the base. It may help the policy (better tracking) — test in a later round with a
tolerance matched to the policy's tracking, not blind.

### Failure causes (eval_rr now writes `cause_table` + per-world reasons by default)
R2 s2 policy, whole task (local rerun, 64 worlds): tube 16 (jam 8, released ~46 mm high 4, fell / missed in the place 2, lost at the lift 1,
disturbed at the grasp 1), pipette 12 (falls after placement in the stand while the fingers open 9, extraction jam 3).
- Pipette after placement: placed less accurately (8–16 mm, 1.7–3.8° vs p50 5.6 mm, 1.2°) and pushed by the opening pinky / ring (6–22 N); the
  wrist pose does not separate it (3–10 mm in both). Next: open in order with the wrist still, withdraw along the axis after; "pipette seated
  before release" reward for the teacher. Sensing the seat on the robot would need finger currents or wrist F/T (ask).
- Extraction jam is a base problem: zero residual with DR fails 35 / 64 there, the policy rescues all but 3–5. Zero residual, holders ±5 cm,
  physical DR off: 22 / 64 jams with the 5 mm / 2° measurement noise vs 9 / 64 with none. **Question for the user:** is 5 mm / 2° realistic for
  the stand / rack estimate? (no change made)

### Smoke (pre-launch checklist item 1) — PASS, both configs
Env check (256 envs, 40 steps, action std 0.1): coupled arm DR ζ min 0.703 (p5 0.796), kp range 0.5–2.0; rewards finite; new terms finite
(arm_band 3.1, finger_band 0.83, finger_still 2.2, tube_tilt 0.17–0.19); commanded arm margin 5.73° (= the 0.1 rad rule, clamp holds).
PPO (train.py, 512 envs, 1 min): 32 iterations, no non-finite worlds, ≈6.3k steps/s on the shared 5080.
Zero-residual replay with clamp + coupled DR (item 2): no DR 23/23, press 9/9 (above); DR on 17 / 46 reach the end, 9/18 press.

### Status: ready to launch once the user approves GPUs / hours (configs `rr_T1k_r3A_s4.json`, `rr_T1k_r3B_s5.json`, base `task_T1k_still3_noff`).
The node package must be rebuilt with `task_T1k_still3_noff` and these configs (make_node_pkg.sh), sha-checked on the node.

### Pipette tipping after placement (largest single cause, 9 / 64) — added before launch
- `pipette_seat` reward 1.0 in both R3 configs (teacher-only, privileged): pipette pose vs its holder-adjusted seated reference, from 1 s before
  it is back in the stand to 1.5 s after (the finger opening); 1 = 5 mm or 2.9°, clipped at 50. Default 0 elsewhere.
- Right-release opening order in the base (pinky / ring first, index / middle +0.3 s, thumb +0.6 s, right arm held still 1.0 s, then the
  original withdrawal time-shifted; export `--right-open-order 1.0`, task_T1k_still3o) — **FAILED the zero-residual replay, NOT in R3**:
  no DR 2 / 23 reach the end (still3 23 / 23), holders ±2 cm 4 / 46 (35), DR on 4 / 46 (17); 20–36 worlds fail as "pipette after placement".
  The ordered opening with the wrist parked tips the pipette in this physics. R4 item: find why (contact with the opening hand while the wrist
  holds; the original reference opens and withdraws together) and design the release from the contacts, not a fixed order.
- Smoke re-run after `pipette_seat`: PASS both configs (ζ min 0.703; all terms finite, pipette_seat mean 1.03–1.06 at action std 0.1;
  commanded arm margin 5.73°; train.py 1 min at 512 envs: 32 iterations, no non-finite worlds, 6.3–6.7k steps/s).
- R3 base stays `task_T1k_still3_noff`. Event gate stays out of R3.

## Results (node run 11-06, GPUs 6 / 7, 11:13–14:04 EST training, evals done 14:15, no R3 process left 14:16)
Layer: RL env (MuJoCo Warp physics), eval_rr seed 123, each run under its own config (R3: dr_rr1_kv3_zeta; R2 s2: dr_rr1_kv3).
Training stopped at the 2.85 h wall limit: R3A 1.969e8, R3B 1.974e8 steps, about 19k steps/s each; training success flat after 1e8 (about 0.90).
final.pt ≈ model_175079424 (all items within ±2). Files: node_results/rr_T1k_r3{A_s4,B_s5}/, table node_results/final_table.json (`final_table.py --r3-only`).

| item | R2 s2 | R3A | R3B |
|---|---|---|---|
| from start ×64: end / tube / press | 44 / 40 / 54 | 48 / 38 / 42 | **52 / 42 / 59** |
| train perturbations ×92: end / tube | 83 / 72 | 89 / 83 | 89 / 83 |
| held-out holes ×92: end / tube | 79 / 68 | 87 / 78 | 89 / 76 |
| held-out placements ×92: end / tube | 57 / 49 | **65 / 56** | 58 / 53 |
| slip from start p50 / p90 mm | 14.3 / 27.3 | 13.5 / 24.2 | **11.9 / 18.1** |
| jitter cells > 0.25° (3–10 Hz) | 6 | 0 | 0 |
| commanded arm margin (rule ≥ 5.73°) | 5.23 FAIL | 5.73 | 5.73 |
| shake carry 3–8 Hz R / L mm (PROPOSED ≤ 0.3); zero residual 0.27 / 0.55 | 0.50 / 0.74 | 0.41 / 0.54 | 0.51 / 0.59 |
| shake carry 8–24 Hz R / L mm (PROPOSED ≤ 0.1) | 0.08 / 0.13 | 0.07 / 0.10 | 0.10 / 0.12 |
| idle-finger cells over (PROPOSED ≤ 2° / 3°/s); zero residual on R3 base: 3 (left pre-grasp, ≤ 2.7°) | 20 (max 6.9°) | 29 (max 9.6°) | 25 (max 9.3°) |
| pipette at placement p50 / p90 mm | 6.1 / 11.9 | 4.7 / 10.7 | 5.1 / 9.6 |
| causes from start (R3B): success 42; tube jammed tilted ~5 mm high 10; pipette after placement 8; tube lost carry/press 2; extraction 1; before grasp 1 | | (R3A: 37; 10; 6; …) | |

Reading:
- R3B final = current best teacher (from start, press, held-out holes, slip). R3A better only on held-out placements and shake.
- NOT deliverable: idle fingers (the user's "抖") are moved by the policy's finger residuals (zero residual: 3 cells over; policy: 25–29, up to
  9.3° on the right thumb during the hold); finger_still at weight 1 was too weak. Wrist shake 3–8 Hz over PROPOSED; the left wrist part is
  already in the zero residual (base / arm DR).
- Like-for-like: R2 s2 from start under the R3 DR (dr_rr1_kv3_zeta, local): end 45, tube 31, press 56 / 64, success 31 → R3B gain is not the milder DR.
- Tube jam (10 / 64) unchanged: base / measurement-noise question (open with the user).
- R4 proposal (not approved): zero the finger residual wherever the reference fingers are still (deployable: reference clock), or penalise the
  residual action there; release order from contacts (see above).
