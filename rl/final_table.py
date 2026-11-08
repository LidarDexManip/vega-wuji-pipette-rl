"""Final comparison of teacher candidates (layer: RL env rollouts, MuJoCo Warp; policy vs zero residual on identical perturbations, eval_rr seed 123).
Reads node_results/<run>/eval/*.json. Rows: whole task from the start (64 worlds), train / held-out holes / held-out placements (92 starts each),
slip compare (full-window, both reach end), jitter cells vs 0.25 deg (hold, press, post-release; PROPOSED), action rate, margins (commanded / executed).
  python final_table.py"""
import json
from pathlib import Path
import numpy as np
B = Path(__file__).resolve().parents[1]/'results'
CAND = [('R1 100M', 'rr_T1k_full_lp_s0', 'model_100073472', ('rr_T1k_full_lp_s0', 'rr_T1k_eval_holdout_lp', 'rr_T1k_eval_holdplace_lp')),
        ('R1 final', 'rr_T1k_full_lp_s0', 'final', ('rr_T1k_full_lp_s0', 'rr_T1k_eval_holdout_lp', 'rr_T1k_eval_holdplace_lp')),
        ('R2 s2 final', 'rr_T1k_r2_s2', 'final', ('rr_T1k_r2_s2', 'rr_T1k_r2_eval_holdout', 'rr_T1k_r2_eval_holdplace')),
        ('R2b final', 'rr_T1k_r2w_s3', 'final', ('rr_T1k_r2w_s3', 'rr_T1k_r2_eval_holdout', 'rr_T1k_r2_eval_holdplace')),
        ('R3A final', 'rr_T1k_r3A_s4', 'final', ('rr_T1k_r3A_s4', 'rr_T1k_r3A_eval_holdout', 'rr_T1k_r3A_eval_holdplace')),
        ('R3B final', 'rr_T1k_r3B_s5', 'final', ('rr_T1k_r3B_s5', 'rr_T1k_r3B_eval_holdout', 'rr_T1k_r3B_eval_holdplace'))]
import sys as _sys
if '--r3-only' in _sys.argv: CAND = [c for c in CAND if c[0].startswith(('R2 s2', 'R3'))]
JK = 'jitter_v2_band_3_10hz_rms_deg (median over worlds, worst joint)'
def L(p):
    try: return json.loads(p.read_text())
    except Exception: return None
def task(d): return f"{d['reach_end']}/{d['n']} end, pip {d['pipette_seated']}, tube {d['tube_seated']}, press {d['press_ok_of_press_starts']}" if d else 'n/a'
rows = []
for lab, run, ck, (cfg, ho, hp) in CAND:
    E = B/run/'eval'; fs = L(E/f'{cfg}__fromstart__{ck}.json'); fz = L(E/f'{cfg}__fromstart__zero.json'); sl = L(E/f'{cfg}__fromstart__{ck}.slip.json')
    tr, trz = L(E/f'{cfg}__{ck}.json'), L(E/f'{cfg}__zero.json'); hO, hOz = L(E/f'{ho}__{ck}.json'), L(E/f'{ho}__zero.json'); hP, hPz = L(E/f'{hp}__{ck}.json'), L(E/f'{hp}__zero.json')
    src = fs or tr; J = (src or {}).get('natural', {}).get(JK, {})
    over = {k: v for k, v in J.items() if v > .25}
    named = {k: J.get(k) for k in ('left thumb hold', 'left index hold', 'left middle hold', 'left ring hold', 'left pinky hold', 'right thumb hold', 'right thumb press', 'left middle post_release', 'right index post_release')}
    nat = (src or {}).get('natural', {})
    rows.append({'candidate': lab, 'from_start (policy | zero)': f"{task(fs)} | {task(fz)}", 'train perturbations': f"{task(tr)} | {task(trz)}",
                 'held-out holes': f"{task(hO)} | {task(hOz)}", 'held-out placements': f"{task(hP)} | {task(hPz)}",
                 'slip from start (a: all full-window p50/p90 mm; b: both reach end p50, n)': (sl or {}).get('a_all_full_window', {}).get('slip_mm_p50_p90_n'), 'slip_b': (sl or {}).get('b_both_reach_end', {}).get('slip_mm_p50_p90_n'),
                 'tilt_b': (sl or {}).get('b_both_reach_end', {}).get('tilt_deg_p50_p90_n'),
                 'jitter cells > 0.25 (of %d)' % len(J): len(over), 'jitter named': named, 'action_rate': nat.get('action_rate_mean_sq'),
                 'margin cmd / q (deg)': (nat.get('arm_joint_margin_cmd_deg_min (commanded targets; rule >= 5.73)'), nat.get('arm_joint_margin_deg_min')), 'elbow cm': nat.get('elbow_minus_shoulder_cm_max'),
                 'cause_table (from start)': (fs or {}).get('cause_table'),
                 'idle fingers over (PROPOSED <=2 deg / 3 deg/s)': nat.get('idle_fingers_over (PROPOSED)'),
                 'shake wrist 3-8 / 8-24 Hz mm carry [R, L] (PROPOSED 0.3 / 0.1)': (lambda sh: (sh.get('carry', {}).get('3-8Hz'), sh.get('carry', {}).get('8-24Hz')))(nat.get('shake (wrist / arm deviation from the reference, band RMS; PROPOSED 3-8 Hz <= 0.3 mm, 8-24 Hz <= 0.1 mm)', {}).get('wrist_dev_band_mm [R, L]', {})),
                 'shake zero residual (same)': (lambda z: (lambda sh: (sh.get('carry', {}).get('3-8Hz'), sh.get('carry', {}).get('8-24Hz')))(((z or {}).get('natural', {}).get('shake (wrist / arm deviation from the reference, band RMS; PROPOSED 3-8 Hz <= 0.3 mm, 8-24 Hz <= 0.1 mm)', {}) or {}).get('wrist_dev_band_mm [R, L]', {})))(fz),
                 'pipette at placement: pos err p50 / p90 mm': (lambda P: (round(float(np.nanmedian(P['at_release_pipette_pos_err_mm'])), 1), round(float(np.nanpercentile(P['at_release_pipette_pos_err_mm'], 90)), 1)) if P else None)((fs or {}).get('pipette_diag'))})
print(json.dumps(rows, indent=1))
(B/'final_table.json').write_text(json.dumps(rows, indent=1))
