"""Whole-hand / arm / object jitter (CPU analysis). Two measures per signal, on the 50 Hz control-step grid (what the RL env records):
  SG  = RMS of the residual after a zero-phase Savitzky-Golay fit (5 samples = 0.1 s, quadratic)
  HP8 = RMS of a zero-phase 4th-order Butterworth high-pass above 8 Hz (8-25 Hz)
Signals: wrist / palm position (3-D norm, mm) and orientation (rotation vector vs the phase's first frame, norm, deg), both hands; arm joints
(worst joint, deg); pipette and tube pose (mm, deg); finger joints (worst, deg); commanded arm targets (ctrl, worst joint, deg; RL rollouts only).
Phases (task reference gates): carry (phase 1), press (phase 2), hold still (plunger >= 90 %), post-release (both hands released),
free air L (left hand empty before the tube grasp), free air R (right hand empty after the pipette release).
Sources: RL policy rollouts, zero-residual rollouts (same perturbations; rows before the first failure only), zero residual DR off,
the T1_k physics demo (trajectory_mjwp at the same instants; plus the native 400 Hz HP8 for content above 25 Hz), the kinematic reference.
  python jitter_whole.py --task TASK --scene SCENE --demo STAGE --src label=DIR ... [--out J]"""
import argparse, glob, json
from pathlib import Path
import numpy as np, mujoco
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.spatial.transform import Rotation as Rot
p = argparse.ArgumentParser(); p.add_argument('--task', type=Path, required=True); p.add_argument('--scene', type=Path, required=True); p.add_argument('--demo', type=Path, required=True)
p.add_argument('--src', action='append', default=[]); p.add_argument('--out', type=Path); a = p.parse_args()
m = mujoco.MjModel.from_xml_path(str(a.scene)); d = mujoco.MjData(m); r = np.load(a.task/'reference.npz'); tr = r['timestamps']
B = lambda n: m.body(n).id; bodies = {'wrist_R': B('right_wrist'), 'wrist_L': B('left_wrist'), 'pipette': B('right_object'), 'tube': B('left_object')}
jq = lambda pre: [(m.joint(j).name, int(m.jnt_qposadr[j])) for j in range(m.njnt) if m.joint(j).name.startswith(pre)]
ARM = {'R': jq('R_arm'), 'L': jq('L_arm')}; FING = {'R': jq('r_'), 'L': jq('l_')}
act_arm = {s: [i for i in range(m.nu) if m.actuator(i).name.startswith(f'{s}_arm')] for s in 'RL'}
ph = r['phase']; pl = r['plunger']; gt, gp = r['gate_tube_held'].astype(bool), r['gate_pipette_held'].astype(bool)
tt0, tt1, tp1 = tr[np.flatnonzero(gt)[0]], tr[np.flatnonzero(gt)[-1]], tr[np.flatnonzero(gp)[-1]]
def phase_masks(t):
    k = np.clip(np.searchsorted(tr, t), 0, len(tr)-1)
    return {'carry': ph[k] == 1, 'press': ph[k] == 2, 'hold_still': (ph[k] == 2) & (pl[k] >= .9*pl.max()), 'post_release': t > max(tt1, tp1)+.3,
            'free_air_L': (t > tr[0]+.4) & (t < tt0-.3), 'free_air_R': t > tp1+.3}
HP = butter(4, 8/25, 'high'); HP400 = butter(4, 8/200, 'high')
def meas(X, sel, hp=HP):
    """X (T, D) on the grid; RMS over the selected rows of the per-row vector norm"""
    if sel.sum() < 15: return None
    sg = X-savgol_filter(X, 5, 2, axis=0); h = filtfilt(*hp, X, axis=0)
    return round(float(np.sqrt((sg[sel]**2).sum(1).mean())), 3), round(float(np.sqrt((h[sel]**2).sum(1).mean())), 3)
def signals(Q):
    P = {k: [] for k in bodies}; R = {k: [] for k in bodies}
    for q in Q:
        d.qpos[:] = q; mujoco.mj_kinematics(m, d)
        for k, b in bodies.items(): P[k].append(d.xpos[b].copy()); R[k].append(d.xmat[b].reshape(3, 3).copy())
    S = {}
    for k in bodies:
        S[f'{k}_pos_mm'] = np.array(P[k])*1e3
        Rs = Rot.from_matrix(np.array(R[k])); S[f'{k}_rot_deg'] = np.degrees((Rs[0].inv()*Rs).as_rotvec())
    for s in 'RL':
        S[f'arm_{s}_deg'] = np.degrees(Q[:, [a_ for _, a_ in ARM[s]]]); S[f'fing_{s}_deg'] = np.degrees(Q[:, [a_ for _, a_ in FING[s]]])
    return S
def per_phase(S, t, valid, extra=None, hp=HP):
    M = phase_masks(t); out = {}
    for ph_, sel in M.items():
        sel = sel & valid; row = {}
        for k, X in S.items():
            if k.startswith(('arm_', 'fing_', 'ctrl_')):   # worst single joint
                vals = [meas(X[:, [j]], sel, hp) for j in range(X.shape[1])]; vals = [v for v in vals if v]
                row[k] = (max(v[0] for v in vals), max(v[1] for v in vals)) if vals else None
            else: row[k] = meas(X, sel, hp)
        out[ph_] = row
    return out
res = {'layer': 'whole-hand / arm / object jitter: SG residual (0.1 s) and >8 Hz high-pass RMS, 50 Hz grid', 'sources': {}}
grid = None
for spec in a.src:
    lab, dr_ = spec.split('='); D = Path(dr_); summ = json.loads((D/'summary.json').read_text())['worlds']; per = []
    for w, o in enumerate(summ):
        z = np.load(D/f'world_{w}.npz'); Q, t = z['qpos'], z['time']; grid = t if grid is None else grid
        valid = np.ones(len(t), bool) if o['first_fail_s'] < 0 else t < o['first_fail_s']-.2
        S = signals(Q)
        if 'ctrl' in z.files:
            for s in 'RL': S[f'ctrl_{s}_deg'] = np.degrees(z['ctrl'][:, act_arm[s]])
        per.append(per_phase(S, t, valid))
    agg = {}
    for ph_ in per[0]:
        agg[ph_] = {}
        for k in per[0][ph_]:
            v = [x[ph_][k] for x in per if x[ph_][k] is not None]
            agg[ph_][k] = (round(float(np.median([x[0] for x in v])), 3), round(float(np.median([x[1] for x in v])), 3)) if v else None
    res['sources'][lab] = {'worlds': len(per), 'median_over_worlds (SG, HP8)': agg}
fs = sorted(glob.glob(str(a.demo/'official/*/*/*/0/trajectory_mjwp.npz')))[0]; zd = np.load(fs); QD = zd['qpos'].reshape(-1, m.nq); TD = zd['time'].reshape(-1)
idx = np.array([int(np.argmin(np.abs(TD-x))) for x in grid]); res['sources']['T1_k physics demo (50 Hz instants)'] = {'median_over_worlds (SG, HP8)': per_phase(signals(QD[idx]), grid, np.ones(len(grid), bool))}
sel400 = (TD >= grid[0]) & (TD <= grid[-1]); Q4, T4 = QD[sel400][::1], TD[sel400]
S4 = signals(Q4[::2]); t4 = T4[::2]   # 200 Hz: HP8 up to 100 Hz (SG window 5 samples = 25 ms here; use HP column)
res['sources']['T1_k physics demo (200 Hz, HP8 to 100 Hz)'] = {'median_over_worlds (SG, HP8)': per_phase(S4, t4, np.ones(len(t4), bool), hp=butter(4, 8/100, 'high'))}
Qr = r['qpos_ref']; kr = np.clip(np.searchsorted(tr, grid), 0, len(tr)-1); res['sources']['kinematic reference (50 Hz)'] = {'median_over_worlds (SG, HP8)': per_phase(signals(Qr[kr]), grid, np.ones(len(grid), bool))}
# compact table: per phase, the key signals for every source
keys = ['wrist_R_pos_mm', 'wrist_R_rot_deg', 'wrist_L_pos_mm', 'wrist_L_rot_deg', 'pipette_pos_mm', 'tube_pos_mm', 'arm_R_deg', 'arm_L_deg', 'fing_R_deg', 'fing_L_deg', 'ctrl_R_deg', 'ctrl_L_deg']
for ph_ in ('carry', 'press', 'hold_still', 'post_release', 'free_air_L', 'free_air_R'):
    print(f'== {ph_}  (SG / HP8)')
    for lab, v in res['sources'].items():
        row = v['median_over_worlds (SG, HP8)'][ph_]
        print(f'  {lab[:34]:34s} ' + ' '.join(f"{k.replace('_deg','').replace('_mm','')}={row.get(k)[0]:.2f}/{row.get(k)[1]:.2f}" if row.get(k) else f"{k.replace('_deg','').replace('_mm','')}=-" for k in keys))
if a.out: a.out.write_text(json.dumps(res, indent=1))
