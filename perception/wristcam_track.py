"""(11-10) Tracking the in-hand object pose over a whole hold with the forearm wrist camera (render only; wristcam_sim.py camera, noise
model and fitter), to see whether the camera can tell how far the object sits from the reference in-hand pose.

Per case (recorded rollout world + hand), every --stride control steps from the reference grasp to the end of the hold (or the world's
first failure + 0.6 s): simulated stereo depth at each noise level -> hand-aware masking -> ICP + axial render-and-compare, started
from the previous estimate (tracking; the first frame starts from the REFERENCE in-hand pose, i.e. the intended grasp) -> exponential
smoothing of the in-hand estimate (alpha). Per-episode calibration errors as in wristcam_sim.NOISE.
Logged: deviation of the true and the estimated in-hand pose from the reference in-hand pose (object origin, mm; long-axis tilt, deg),
and the estimation error. In-hand pose = wrist^-1 x object, the reference one from reference.npz qpos_ref at the same reference frame.
  MUJOCO_GL=egl python perception/wristcam_track.py SCENE.xml OUT --case RUN_DIR WORLD SIDE LABEL [--case ...] [--levels medium high]
Writes OUT/track.json and OUT/track.png.
"""
import argparse, json, math
from pathlib import Path
import numpy as np
import wristcam_sim as WS

GATE = {'right': 'gate_pipette_held', 'left': 'gate_tube_held'}


def tilt(Ra, Rb): return math.degrees(math.acos(np.clip(Ra[:, 2]@Rb[:, 2], -1, 1)))


def inhand(side):
    m, d = WS._G['m'], WS._G['d']; w = m.body(f'{side}_wrist').id; o = m.body(WS.OBJ[side]).id
    Rw = d.xmat[w].reshape(3, 3).copy(); pw = d.xpos[w].copy()
    return Rw, pw, Rw.T@(d.xpos[o]-pw), Rw.T@d.xmat[o].reshape(3, 3)


def setq(q):
    import mujoco
    m, d = WS._G['m'], WS._G['d']; d.qpos[:] = q; mujoco.mj_kinematics(m, d); mujoco.mj_camlight(m, d)


def episode(w, levels):
    import mujoco
    rng = np.random.default_rng(1000*w+7); WS._G['epi'] = {}
    for lev in levels:
        nz = WS.NOISE[lev]
        for side in ('right', 'left'):
            u = rng.normal(size=3); a = rng.normal(size=3); dq = np.zeros(4)
            mujoco.mju_axisAngle2Quat(dq, a/np.linalg.norm(a), math.radians(nz['extr'][1]))
            WS._G['epi'][(lev, side)] = dict(scale=nz['scale']*rng.normal(), cam=(nz['extr'][0]*1e-3*u/np.linalg.norm(u), dq))
    return rng


def estimate(cam, side, lev, R0, t0, rng):
    """One fit from the prior pose (R0, t0) (world). Returns (R, t) or None."""
    import mujoco
    m, d = WS._G['m'], WS._G['d']; nz = WS.NOISE[lev]
    zt, _ = (WS.render(cam, {1, 2, 3}, 'depth'), None) if tuple(nz['extr']) == (0., 0.) else WS.observe(cam, side, lev, False)
    seg = WS.render(cam, {1, 2, 3}, 'seg'); zn = WS.stereo(zt, WS.segmask(seg, WS._G['fitg'][side]), nz, rng, WS._G['epi'][(lev, side)]['scale'])
    q = d.qpos.copy(); d.qpos[WS._G['fq']] += rng.normal(size=len(WS._G['fq']))*.02; mujoco.mj_kinematics(m, d); mujoco.mj_camlight(m, d)
    zh = WS.render(cam, {1}, 'depth'); d.qpos[:] = q; mujoco.mj_kinematics(m, d); mujoco.mj_camlight(m, d)
    valid = np.isfinite(zn); zf = np.where(valid, zn, 0.); sz = WS.sigma_z(.13, nz)
    P = WS.backproject(zf, valid & ~(np.abs(zf-zh) < max(.004, 3*sz)), cam)
    if len(P): dist, _ = WS._G['T'][side].query((P-t0)@R0); P = P[dist < .02]
    if len(P) < 50: return None
    R, t, n_in = WS.icp(P, WS._G['T'][side], WS._G['M'][side], R0.copy(), t0.copy(), floor=max(.0015, 2*sz))
    if n_in < 20: return None
    return R, WS.axial_search(cam, side, R, t, zf, valid, trunc=max(.01, 3*sz))


def blend(Ra, Rb, a):
    """Rotation a of the way from Ra to Rb."""
    import mujoco
    qa, qb = np.zeros(4), np.zeros(4); mujoco.mju_mat2Quat(qa, Ra.ravel()); mujoco.mju_mat2Quat(qb, Rb.ravel())
    if qa@qb < 0: qb = -qb
    q = (1-a)*qa+a*qb; q /= np.linalg.norm(q); R = np.zeros(9); mujoco.mju_quat2Mat(R, q); return R.reshape(3, 3)


def plot(res, out, levels):
    """Needs matplotlib (system python3; the RL venv has none)."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'; COL = {'medium': '#2a78d6', 'high': '#eb6834'}
    NAME = {'medium': 'camera estimate, medium noise (FFS-like)', 'high': 'camera estimate, high noise (D405 on-chip-like)'}
    fig, axs = plt.subplots(len(res), 1, figsize=(9, 2.6*len(res)), sharex=True, facecolor=SURF)
    for ax, c in zip(np.atleast_1d(axs), res):
        R = c['rows']; t = [r['t'] for r in R]; ax.set_facecolor(SURF)
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'): ax.spines[s].set_color(GRID)
        ax.grid(axis='y', color=GRID, lw=.8); ax.tick_params(colors=MUTED, labelsize=8)
        ax.plot(t, [r['true_dev_mm'] for r in R], color=INK, lw=2.2, label='true (simulator)', zorder=3)
        for lev in levels:
            ax.plot(t, [r[f'{lev}_dev_mm'] for r in R], color=COL[lev], lw=2, label=NAME[lev], zorder=2)
            tr_ = [r['t'] for r in R if f'{lev}_raw_dev_mm' in r]; ax.scatter(tr_, [r[f'{lev}_raw_dev_mm'] for r in R if f'{lev}_raw_dev_mm' in r],
                                                                              s=9, color=COL[lev], alpha=.35, lw=0, zorder=1)
        if c['first_fail_s'] >= 0:
            ax.axvline(c['first_fail_s'], color=MUTED, ls='--', lw=1); ax.text(c['first_fail_s'], ax.get_ylim()[1]*.92, ' episode fails', color=MUTED, fontsize=8, va='top')
        ax.set_ylabel('off the reference\nin-hand pose (mm)', color=MUTED, fontsize=8); ax.set_title(c['label'], loc='left', color=INK, fontsize=10)
    np.atleast_1d(axs)[0].legend(frameon=False, fontsize=8, loc='upper right', labelcolor=INK)
    np.atleast_1d(axs)[-1].set_xlabel('time (s)  -- lines: smoothed estimate; dots: single-frame fits', color=MUTED, fontsize=8)
    fig.tight_layout(); fig.savefig(out/'track.png', dpi=130, facecolor=SURF); print('->', out/'track.png')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('scene', type=Path); p.add_argument('out', type=Path)
    p.add_argument('--case', nargs=4, action='append', required=True, metavar=('RUN', 'WORLD', 'SIDE', 'LABEL'))
    p.add_argument('--levels', nargs='+', default=['medium', 'high']); p.add_argument('--stride', type=int, default=10)
    p.add_argument('--alpha', type=float, default=.35); p.add_argument('--plot-only', action='store_true', help='redraw OUT/track.png from OUT/track.json')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    if a.plot_only: plot(json.loads((a.out/'track.json').read_text()), a.out, a.levels); raise SystemExit
    WS.init(a.scene, {'forearm': WS.MOUNTS['forearm']}, (0., 0.)); ref = np.load(a.scene.parent/'reference.npz'); Qr, tr = ref['qpos_ref'], ref['timestamps']
    res = []
    for run, w, side, lab in a.case:
        w = int(w); run = Path(run); z = np.load(run/f'world_{w}.npz'); Q, T = z['qpos'], z['time']
        info = json.loads((run/'summary.json').read_text())['worlds'][w]; WS.place_holders(info.get('spec')); rng = episode(w, a.levels)
        g = np.flatnonzero(ref[GATE[side]]); t_end = tr[g[-1]] if info['first_fail_s'] < 0 else min(tr[g[-1]], info['first_fail_s']+.6)
        cam = f'forearm_{side}'; est = {}; rows = []
        for k in range(0, len(Q), a.stride):
            if T[k] < tr[g[0]] or T[k] > t_end: continue
            kr = int(np.argmin(np.abs(tr-T[k])))
            if not ref[GATE[side]][kr]: continue
            setq(Qr[kr]); _, _, pr, Rr = inhand(side)
            setq(Q[k]); Rw, pw, pt, Rt = inhand(side)
            row = dict(t=float(T[k]), true_dev_mm=1e3*float(np.linalg.norm(pt-pr)), true_tilt_deg=tilt(Rr, Rt))
            for lev in a.levels:
                pe, Re = est.get(lev, (pr, Rr))                                  # tracking prior; the first frame: the intended grasp
                fit = estimate(cam, side, lev, Rw@Re, pw+Rw@pe, rng)
                if fit is not None:
                    Rf, tf = fit; pr_, Rr_ = Rw.T@(tf-pw), Rw.T@Rf
                    pe, Re = (pr_, Rr_) if lev not in est else ((1-a.alpha)*pe+a.alpha*pr_, blend(Re, Rr_, a.alpha))
                    row[f'{lev}_raw_err_mm'] = 1e3*float(np.linalg.norm(pr_-pt)); row[f'{lev}_raw_dev_mm'] = 1e3*float(np.linalg.norm(pr_-pr))
                est[lev] = (pe, Re)
                row[f'{lev}_dev_mm'] = 1e3*float(np.linalg.norm(pe-pr)); row[f'{lev}_tilt_deg'] = tilt(Rr, Re)
                row[f'{lev}_err_mm'] = 1e3*float(np.linalg.norm(pe-pt)); row[f'{lev}_err_tilt_deg'] = tilt(Rt, Re)
            rows.append(row)
        res.append(dict(label=lab, run=str(run), world=w, side=side, first_fail_s=info['first_fail_s'], rows=rows))
        print(lab, len(rows), 'frames', flush=True)
    (a.out/'track.json').write_text(json.dumps(res, indent=1))

    (a.out/'track.json').write_text(json.dumps(res, indent=1)); plot(res, a.out, a.levels)
