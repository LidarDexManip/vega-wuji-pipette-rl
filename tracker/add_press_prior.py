"""TRACKER (11-06): PRESS PRIOR for a kinematic-base task, the analogue of the squeeze rule. Layer: TASK EXPORT (no physics run).

The kinematic reference puts the thumb pad ON the cap (what the human thumb did), so the base commands no press force and zero residual
presses 0 mm. In the reference press window the right-thumb command is offset by

    dq = J^T f / kp,   f = -(k_spring * depth_ref + margin) * a_plunger(world),   clipped so |kp dq| <= frac * forcerange

J = translational Jacobian of the right thumb-tip site (4 thumb dofs) at the reference posture, a_plunger = the plunger joint axis in
world (depth = -q_plunger), k_spring / springref from the scene, kp from the thumb actuators' gainprm. Ramped (cosine) over RAMP_S at the
start and the end of the press window (phase 2). General: everything comes from the scene and the reference, nothing from a physics demo
(the T1_k demo's thumb command overshoot in its deep press is printed as a cross-check only).

  python -I add_press_prior.py SRC_TASK OUT_TASK [--margin-n 1.5] [--frac 0.95] [--ramp-s 0.15]
"""
import argparse, json, shutil, time
from pathlib import Path
import numpy as np, mujoco

HERE = Path(__file__).resolve().parent; DEMO = HERE.parent/'tasks/task_T1k_still3_noff'
THUMB = ('r_thumb_cmc_flex', 'r_thumb_cmc_abd', 'r_thumb_mcp', 'r_thumb_ip')


def main():
    p = argparse.ArgumentParser(); p.add_argument('src', type=Path); p.add_argument('out', type=Path)
    p.add_argument('--margin-n', type=float, default=1.5); p.add_argument('--frac', type=float, default=.95); p.add_argument('--ramp-s', type=float, default=.15); a = p.parse_args()
    if a.out.exists(): raise SystemExit(f'{a.out} exists')
    r = dict(np.load(a.src/'reference.npz')); m = mujoco.MjModel.from_xml_path(str(a.src/'scene.xml')); d = mujoco.MjData(m)
    nf, ns = len(r['timestamps']), len(r['sim_times']); acts = [m.actuator(i).name for i in range(m.nu)]
    ai = [acts.index(n) for n in THUMB]; jd = [int(m.jnt_dofadr[m.joint(n).id]) for n in THUMB]
    kp = np.array([m.actuator_gainprm[i, 0] for i in ai]); fr = np.array([m.actuator_forcerange[i, 1] for i in ai])
    pj = m.joint('pipette_plunger').id; ks = float(m.jnt_stiffness[pj]); site = m.site('right_thumb_tip').id
    press = r['phase'] == 2; idx = np.flatnonzero(press)
    if not len(idx): raise SystemExit('no press window (phase 2) in the reference')
    w = np.zeros(nf); w[press] = 1.; nr = max(1, int(round(a.ramp_s/.01)))
    for k in range(nr):   # cosine ramps inside the window at its start and end
        s = .5-.5*np.cos(np.pi*(k+1)/(nr+1)); w[idx[0]+k] = min(w[idx[0]+k], s); w[idx[-1]-k] = min(w[idx[-1]-k], s)
    dq = np.zeros((nf, 4)); Fmag = np.zeros(nf); clipped = np.zeros(nf, bool)
    jac = np.zeros((3, m.nv))
    for k in idx:
        d.qpos[:] = r['qpos_ref'][k]; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        axis = d.xaxis[pj].copy(); depth = -float(r['qpos_ref'][k, m.jnt_qposadr[pj]])
        F = ks*max(depth, 0.)+a.margin_n; f = -F*axis/np.linalg.norm(axis)   # force the thumb applies to the cap: along -axis (q_plunger negative = deeper)
        mujoco.mj_jacSite(m, d, jac, None, site); tau = jac[:, jd].T@f
        x = tau/kp; lim = a.frac*fr/kp; c = np.clip(x, -lim, lim); clipped[k] = bool((c != x).any())
        dq[k] = c*w[k]; Fmag[k] = F
    rows = np.minimum(np.arange(ns)//4, nf-1)
    r['ctrl_sim'] = r['ctrl_sim'].copy(); r['controls'] = r['controls'].copy()
    for j, i in enumerate(ai):
        r['ctrl_sim'][:, i] += dq[rows, j]; r['controls'][:, i] += dq[:, j]
    a.out.mkdir(parents=True)
    for f_ in ('scene.xml', 'contact_ref.json', 'contact_ref.npz', 'valid_starts.npy', 'valid_starts_audit.json'): shutil.copy2(a.src/f_, a.out/f_)
    np.savez_compressed(a.out/'reference.npz', **r)
    deep = press & (r['plunger'] >= .9*r['plunger'].max())
    rep = {'press_frames': int(press.sum()), 'force_n_max': round(float(Fmag.max()), 2), 'k_spring_N_per_m': round(ks, 2), 'margin_n': a.margin_n,
           'dq_deg_mean_deep_press [cmc_flex, cmc_abd, mcp, ip]': np.round(np.degrees(dq[deep].mean(0)), 2).tolist(),
           'dq_deg_max_abs': np.round(np.degrees(np.abs(dq).max(0)), 2).tolist(), 'clipped_frames': int(clipped.sum()), 'ramp_s': a.ramp_s}
    try:   # cross-check only: the physics demo's thumb command minus its executed q in ITS deep press
        rd = np.load(DEMO/'reference.npz'); md = mujoco.MjModel.from_xml_path(str(DEMO/'scene.xml')); ad = [md.jnt_qposadr[md.joint(n).id] for n in THUMB]
        acd = [md.actuator(i).name for i in range(md.nu)]; cd = [acd.index(n) for n in THUMB]; dd = (rd['phase'] == 2) & (rd['plunger'] >= .9*rd['plunger'].max())
        rep['crosscheck_T1k_demo_cmd_minus_q_deg_deep_press'] = np.round(np.degrees((rd['controls'][dd][:, cd]-rd['qpos_ref'][dd][:, ad]).mean(0)), 2).tolist()
    except Exception as e: rep['crosscheck'] = str(e)
    tj = json.loads((a.src/'task.json').read_text()); tj.update(press_prior=rep, mode=tj.get('mode', '')+' + PRESS PRIOR', created=time.strftime('%Y-%m-%d %H:%M:%S'), press_prior_src=str(a.src))
    (a.out/'task.json').write_text(json.dumps(tj, indent=1)+'\n'); print(json.dumps(rep))


if __name__ == '__main__':
    main()
