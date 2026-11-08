"""Layer: TASK EXPORT (no new physics). Real-robot line: a residual-RL task from a Vega whole-robot DAID physics stage (vega_mpc.py output).

The scene is the stage's simulated scene, verbatim (scene frame, DAID names: right_object / left_object / right_wrist / left_wrist,
plunger_carrier weld, object welds inactive). The reference is the EXECUTED physics: states every 10 ms (every 4th 2.5 ms state) and the
executed ctrl of every actuator at the sim rate (ctrl_sim_verbatim: the env applies it as is, the policy adds a residual).

  .venv-rl-head/bin/python -I export_task.py STAGE OUT
"""
import json, shutil, sys, time, hashlib
from pathlib import Path
import numpy as np, mujoco

import os as _os
ROOT = Path(_os.environ.get('G1PS_ROOT', str(Path(__file__).resolve().parents[1]/'deps_from_g1ps')))
sys.path.insert(0, str(ROOT/'tools'))
import importlib.util
sp = importlib.util.spec_from_file_location('xr', ROOT/'tools/export_residual_ppo_task.py'); XR = importlib.util.module_from_spec(sp); sp.loader.exec_module(XR)
PHASES = ('pre_lift', 'carry', 'press', 'post_press')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    stage, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]); smooth = '--smooth-fingers' in sys.argv
    if out.exists(): raise SystemExit(f'{out} exists; choose a new directory')
    ph = XR.load_physics_stage(stage); run = ph['run']; m = ph['model']; dt = ph['dt']
    if run.get('robot') != 'vega': raise SystemExit(f"not a Vega stage (robot={run.get('robot')})")
    xml = Path(ph['xml']); stride = int(round(.01/dt)); frames = np.arange(0, len(ph['q']), stride); n = len(frames)
    qpos, qvel, times = ph['q'][frames].astype(np.float64), ph['v'][frames].astype(np.float64), ph['t'][frames]
    ctrl_sim = ph['ctrl'].astype(np.float64); t_sim = ph['t'][:len(ctrl_sim)]
    controls = ctrl_sim[np.minimum(frames, len(ctrl_sim)-1)].copy()
    g = ph['gates']; rows = np.minimum(np.floor(frames*dt*ph['kin_fps']+1e-9).astype(int), len(g['times'])-1)
    gate = {k: v[rows].astype(bool) for k, v in g.items() if v.ndim == 1 and v.dtype == bool}
    lift = int(np.argmax(~gate['pipette_in_stand'])) if (~gate['pipette_in_stand']).any() else 0
    on = int(np.argmax(gate['press'])) if gate['press'].any() else n-1; off = int(len(gate['press'])-1-np.argmax(gate['press'][::-1])) if gate['press'].any() else n-1
    phase = np.zeros(n, np.int8); phase[lift:] = 1; phase[on:off+1] = 2; phase[off+1:] = 3
    sm_rep = None
    if smooth:   # (11-06) the policy must not learn the MPC's finger jitter: zero-phase smoothing of every finger actuator's executed ctrl (sim rate)
        # and finger joint's qpos_ref (100 Hz), cubic Savitzky-Golay: 0.2 s window on non-latched rows, 0.1 s on latched rows (the grip squeeze is kept), 0.3 s blends at
        # the latch edges (stage hold_latch_mask); finger qvel_ref re-derived from the smoothed qpos
        from scipy.ndimage import uniform_filter1d as uf
        from scipy.signal import savgol_filter
        L = g['hold_latch_mask'].astype(bool); sm_rep = {}
        sg = lambda x, w: savgol_filter(x, int(w) | 1, 3, mode='interp')   # zero-phase cubic Savitzky-Golay: removes the >~5-10 Hz jitter, keeps smooth closing / opening motion
        free_only = '--smooth-free-only' in sys.argv   # (11-06, data gate: grasp fingers within 1 deg of the demo) latched rows kept VERBATIM (the executed squeeze), only free rows smoothed
        def blended(X, rate, Lr):
            fr = (~Lr).astype(float); a = np.clip(uf(fr, max(1, int(round(.6*rate))), axis=0, mode='nearest'), 0, 1)
            if free_only: a = np.where(Lr, 0., a); return X+a*(sg(X, round(.2*rate))-X)
            return a*sg(X, round(.2*rate))+(1-a)*sg(X, round(.1*rate))
        rows_sim = np.minimum(np.floor(np.arange(len(ctrl_sim))*dt*ph['kin_fps']+1e-9).astype(int), len(L)-1)
        for i in range(m.nu):
            jn = m.joint(int(m.actuator_trnid[i, 0])).name
            if not jn.startswith(('r_', 'l_')): continue
            c0 = ctrl_sim[:, i].copy(); ctrl_sim[:, i] = blended(c0, 1/dt, L[rows_sim, i])
            a_ = int(m.jnt_qposadr[m.actuator_trnid[i, 0]]); v_ = int(m.jnt_dofadr[m.actuator_trnid[i, 0]]); q0 = qpos[:, a_].copy()
            qpos[:, a_] = blended(q0, 100., L[rows, i]); qvel[:, v_] = np.gradient(qpos[:, a_], .01)
            lt = L[rows, i]
            sm_rep[jn] = {'ctrl_max_change_deg_latched_free': [round(float(np.degrees(np.abs(ctrl_sim[:, i]-c0)[L[rows_sim, i]].max())) if L[rows_sim, i].any() else 0., 3),
                                                               round(float(np.degrees(np.abs(ctrl_sim[:, i]-c0)[~L[rows_sim, i]].max())) if (~L[rows_sim, i]).any() else 0., 3)],
                          'qpos_max_change_deg_latched': round(float(np.degrees(np.abs(qpos[:, a_]-q0)[lt].max())) if lt.any() else 0., 3)}
        sm_rep['_mode'] = 'free rows only (latched rows verbatim)' if free_only else 'free rows SG 0.2 s, latched rows SG 0.1 s, 0.3 s blends'
        from scipy.signal import butter, filtfilt
        def opt(name):
            return float(sys.argv[sys.argv.index(name)+1]) if name in sys.argv else None
        def lp_on(X, rate, sel, hz, keep_mean):
            # zero-phase 2nd-order Butterworth low-pass of X on the rows `sel` (0.15 s cosine blends at the window edges); keep_mean: the mean of
            # (new - old) over every contiguous window is put back to 0 (the low-frequency squeeze / preload is kept exactly)
            if not sel.any(): return X, []
            b_, a_ = butter(2, hz/(rate/2)); Y = filtfilt(b_, a_, X); w = np.clip(uf(sel.astype(float), max(1, int(round(.15*rate))), mode='nearest'), 0, 1)*sel
            idx = np.flatnonzero(sel); cuts = np.flatnonzero(np.diff(idx) > 1); segs = np.split(idx, cuts+1); new = X+w*(Y-X); means = []
            for sg_ in segs:
                if keep_mean: new[sg_] -= np.mean(new[sg_]-X[sg_])*w[sg_]/max(np.mean(w[sg_]), 1e-9)
                means.append(round(float(np.degrees(np.mean(new[sg_]-X[sg_]))), 4))
            return new, means
        hz_l = opt('--latched-lp')
        if hz_l:   # (11-06, coordinator) latched grasp fingers: low-pass the MPC ctrl / qpos jitter, keep the per-window mean (squeeze) exactly
            lrep = {}
            for i in range(m.nu):
                jn = m.joint(int(m.actuator_trnid[i, 0])).name
                if not jn.startswith(('r_', 'l_')) or jn.startswith('r_thumb'): continue   # the right thumb's press is handled by --press-thumb-lp
                ls_, lq_ = L[rows_sim, i], L[rows, i]
                if not ls_.any(): continue
                a_ = int(m.jnt_qposadr[m.actuator_trnid[i, 0]]); v_ = int(m.jnt_dofadr[m.actuator_trnid[i, 0]]); c0 = ctrl_sim[:, i].copy(); q0 = qpos[:, a_].copy()
                ctrl_sim[:, i], mc = lp_on(c0, 1/dt, ls_, hz_l, True); qpos[:, a_], mq = lp_on(q0, 100., lq_, hz_l, True); qvel[:, v_] = np.gradient(qpos[:, a_], .01)
                lrep[jn] = {'ctrl_max_change_deg': round(float(np.degrees(np.abs(ctrl_sim[:, i]-c0).max())), 3), 'qpos_max_change_deg': round(float(np.degrees(np.abs(qpos[:, a_]-q0).max())), 3),
                            'ctrl_window_mean_change_deg': mc, 'qpos_window_mean_change_deg': mq}
            sm_rep['_latched_lp'] = {'hz': hz_l, 'joints': lrep}
        hz_p = opt('--press-thumb-lp')
        if hz_p:   # (11-06) right thumb over the WHOLE press (descent, hold, ascent: press gate +- 0.3 s), zero-phase low-pass of ctrl and qpos
            pr = gate['press'].copy(); pad = 30; i0, i1 = max(0, on-pad), min(n-1, off+pad); pr[i0:i1+1] = True
            prs = pr[np.minimum((np.arange(len(ctrl_sim))*dt/.01).astype(int), n-1)]; prep = {}
            for jn in ('r_thumb_cmc_flex', 'r_thumb_cmc_abd', 'r_thumb_mcp', 'r_thumb_ip'):
                i = m.actuator(jn).id; a_ = int(m.jnt_qposadr[m.joint(jn).id]); v_ = int(m.jnt_dofadr[m.joint(jn).id]); c0 = ctrl_sim[:, i].copy(); q0 = qpos[:, a_].copy()
                ctrl_sim[:, i], _ = lp_on(c0, 1/dt, prs, hz_p, False); qpos[:, a_], _ = lp_on(q0, 100., pr, hz_p, False); qvel[:, v_] = np.gradient(qpos[:, a_], .01)
                prep[jn] = {'ctrl_max_change_deg': round(float(np.degrees(np.abs(ctrl_sim[:, i]-c0).max())), 3), 'qpos_max_change_deg': round(float(np.degrees(np.abs(qpos[:, a_]-q0).max())), 3)}
            sm_rep['_press_thumb_lp'] = {'hz': hz_p, 'window_s': [round(float(times[i0]), 3), round(float(times[i1]), 3)], 'joints': prep}
        if '--fingers-still' in sys.argv:
            # (11-06 round 3, user: "fingers move when they don't need to"; both hands, all 40 joints) every finger joint HOLDS a constant target between task events and makes ONE minimum-jerk transition at each
            # event: pre-shape before a closure, opening at a release. Protected (kept exactly as exported above): the joint's own latched rows,
            # the 0.6 s before every latch start of that joint AND of its hand (closure windows, MPC samples kept; T1_h lesson), the right press.
            # Pre-event transitions end 0.3 s before the closure window (late, the finger does not wander early) unless the change exceeds
            # BIG_DEG (the rack-clearance curl of the left ring / pinky, 70-100 deg): those move once, EARLY, starting 0.3 s after the segment
            # start (the hand starting toward the tube). Post-release transitions start at the release. Duration = max(0.3 s, 1.875 |dq| / VMAX)
            # (minimum-jerk peak speed <= VMAX). ctrl on the changed rows = q + kv/kp qdot (stage convention; the no-FF variant removes the FF later).
            VMAX, BIG_DEG, LEAD = 2.0, 30., .3   # VMAX: default; v3 sets it per joint below (drive capability)
            ORDER = '--right-open-order' in sys.argv   # (11-06) right release: pinky / ring first, index / middle +0.3 s, thumb +0.6 s; the right arm holds still WD s, then withdraws
            OPEN_DELAY = {'pinky': 0, 'ring': 0, 'index': 30, 'middle': 30, 'thumb': 60}; WD = float(sys.argv[sys.argv.index('--right-open-order')+1]) if ORDER else 0.
            mj = lambda x: np.clip(x, 0, 1)**3*(10-15*np.clip(x, 0, 1)+6*np.clip(x, 0, 1)**2)
            def runs(mask):
                idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0])); return list(zip(idx[::2], idx[1::2]))
            n100 = len(times); hand_latch = {}
            for i in range(m.nu):
                jn = m.joint(int(m.actuator_trnid[i, 0])).name
                if jn.startswith(('l_', 'r_')): hand_latch.setdefault(jn[0], np.zeros(n100, bool)); hand_latch[jn[0]] |= L[rows, i]
            def closure_mask(lm):
                cm = np.zeros(n100, bool)
                for a_, _ in runs(lm): cm[max(0, a_-60):a_] = True
                return cm
            hand_cl = {h: closure_mask(v) for h, v in hand_latch.items()}; hand_rel = {h: [b_ for _, b_ in runs(v)] for h, v in hand_latch.items()}
            prw = np.zeros(n100, bool); prw[max(0, on-30):min(n100, off+31)] = True; st_rep = {}
            for i in range(m.nu):
                jn = m.joint(int(m.actuator_trnid[i, 0])).name
                if not jn.startswith(('l_', 'r_')): continue
                a_ = int(m.jnt_qposadr[m.actuator_trnid[i, 0]]); v_ = int(m.jnt_dofadr[m.actuator_trnid[i, 0]]); h = jn[0]; lm = L[rows, i]
                prot = lm | closure_mask(lm) | hand_cl[h] | (prw if jn.startswith('r_thumb') else False)   # (user 11-06: both hands; right thumb press protected)
                if jn.startswith('r_thumb') and prw.any():   # the thumb's way from leaving the grip (its latch end) onto the cap is the press event itself: keep it
                    p0_ = int(np.flatnonzero(prw)[0]); ends_ = [b_ for _, b_ in runs(lm) if b_ <= p0_]
                    if ends_: prot[ends_[-1]:p0_] = True
                q0 = qpos[:, a_].copy(); qn = q0.copy(); segs_rep = []
                # (v3) per-joint speed the finger drive can follow: damping torque <= half its force limit at the RL nominal kd (x KD_SCALE: round 2/3 use x3)
                KD_SCALE = float(sys.argv[sys.argv.index('--still-kd-scale')+1]) if '--still-kd-scale' in sys.argv else 1.
                kd_ = -m.actuator_biasprm[i, 2]*KD_SCALE+m.dof_damping[v_]; VMAX = min(2.0, .5*float(m.actuator_forcerange[i, 1])/max(kd_, 1e-6))
                for a_r, b_r in runs(~prot):
                    cuts = sorted({a_r, b_r} | {r_ for r_ in hand_rel[h] if a_r < r_ < b_r})
                    for s0, s1 in zip(cuts[:-1], cuts[1:]):
                        v0 = qn[s0-1] if s0 > 0 else q0[0]; v1 = q0[s1] if s1 < n100 else q0[-1]   # continuity with the (possibly changed) previous segment
                        if s1 in hand_rel[h]: v1 = v0                                                 # up to a release: just hold (the opening happens after it)
                        post = s0 > 0 and (lm[s0-1] or s0 in hand_rel[h])                 # right after this joint's or its hand's release
                        T = max(.3, 1.875*abs(v1-v0)/VMAX); ns = max(1, int(round(T*100))); L_ = s1-s0
                        if post: t0_ = s0+(OPEN_DELAY.get(jn.split('_')[1], 0) if (h == 'r' and ORDER and s0 == (hand_rel['r'][-1] if hand_rel['r'] else -1)) else 0)
                        elif abs(np.degrees(v1-v0)) > BIG_DEG: t0_ = min(s0+30, max(s0, s1-ns))
                        else: t0_ = max(s0, s1-int(LEAD*100)-ns)
                        ns = min(ns, s1-t0_) if s1 > t0_ else 1
                        x = (np.arange(s0, s1)-t0_)/max(ns, 1); qn[s0:s1] = v0+(v1-v0)*mj(x)
                        segs_rep.append([round(float(times[s0]), 2), round(float(times[min(s1, n100-1)]), 2), round(float(np.degrees(v1-v0)), 1), 'post-release' if post else ('early (big)' if abs(np.degrees(v1-v0)) > BIG_DEG else 'late'), round(T, 2)])
                ch = ~prot
                qpos[:, a_] = qn; qvel[:, v_] = np.gradient(qn, .01)
                kk_ = -m.actuator_biasprm[i, 2]/m.actuator_gainprm[i, 0]; cs_t = np.interp(t_sim, times, qn+kk_*qvel[:, v_]); chs = np.interp(t_sim, times, ch.astype(float)) > .5
                ctrl_sim[chs, i] = cs_t[chs]
                st_rep[jn] = {'vmax_rad_s': round(VMAX, 2), 'changed_rows_100hz': int(ch.sum()), 'max_change_deg': round(float(np.degrees(np.abs(qn-q0).max())), 1), 'segments [t0, t1, dq_deg, kind, T_s]': segs_rep}
            if ORDER and hand_rel['r']:   # right arm: hold the pose at the release for WD s, then the original withdrawal, time-shifted (the last WD s of it are cut at the episode end)
                R_ = hand_rel['r'][-1]; dS = int(round(WD*100)); arm_rep = {}
                for i in range(m.nu):
                    an = m.actuator(i).name
                    if not an.startswith('R_arm'): continue
                    a_ = int(m.jnt_qposadr[m.actuator_trnid[i, 0]]); v_ = int(m.jnt_dofadr[m.actuator_trnid[i, 0]]); q0 = qpos[:, a_].copy(); qs_ = q0.copy()
                    qs_[R_:] = np.r_[np.full(min(dS, n100-R_), q0[R_]), q0[R_:max(R_, n100-dS)]][:n100-R_]
                    qpos[:, a_] = qs_; qvel[:, v_] = np.gradient(qs_, .01)
                    c0 = ctrl_sim[:, i].copy(); ts_R = float(times[R_]); sh = t_sim >= ts_R
                    src_t = np.where(t_sim < ts_R+WD, ts_R, t_sim-WD); ctrl_sim[sh, i] = np.interp(src_t[sh], t_sim, c0)
                    arm_rep[an] = round(float(np.degrees(np.abs(qs_-q0).max())), 2)
                sm_rep['_right_open_order'] = {'release_s': round(float(times[R_]), 3), 'finger_delays_s': {k_: v_/100 for k_, v_ in OPEN_DELAY.items()}, 'arm_hold_s': WD, 'arm_max_change_deg': arm_rep}
            sm_rep['_fingers_still'] = {'rule': 'hold between events; one minimum-jerk transition per event (late pre-shape, early for > %g deg, at release after); protected: latched rows, 0.6 s closure windows (joint and hand), right-thumb press window; all 40 finger joints' % BIG_DEG, 'vmax_rad_s': VMAX, 'joints': st_rep}
        controls = ctrl_sim[np.minimum(frames, len(ctrl_sim)-1)].copy()
    plq = int(m.jnt_qposadr[m.joint('pipette_plunger').id]); plunger = -qpos[:, plq]
    acts = [m.actuator(i).name for i in range(m.nu)]
    out.mkdir(parents=True); shutil.copy2(xml, out/'scene.xml')
    mc = mujoco.MjModel.from_xml_path(str(out/'scene.xml'))
    same = (mc.nq, mc.nv, mc.nu, mc.ngeom, mc.npair) == (m.nq, m.nv, m.nu, m.ngeom, m.npair)
    if not same: raise SystemExit('copied scene compiles differently from the stage model')
    np.savez_compressed(out/'reference.npz', timestamps=times, controls=controls, plunger=plunger, qpos_ref=qpos, qvel_ref=qvel,
                        ctrl_sim=ctrl_sim, ctrl_sim_verbatim=True, sim_times=t_sim, sim_dt=dt, phase=phase, phase_names=np.array(PHASES),
                        actuator_names=np.array(acts), qpos_joint_names=np.array([m.joint(j).name for j in range(m.njnt)]),
                        **{f'gate_{k}': v for k, v in gate.items()})
    task = {'mode': 'physics (Vega-U + Wuji beta1, whole robot simulated in the MPC); real-robot line', 'created': time.strftime('%Y-%m-%d %H:%M'),
            'command': ' '.join(sys.argv), 'stage': str(stage), 'scene_source': str(xml), 'sim_dt': dt, 'frames_100hz': n, 'ctrl_rows': len(ctrl_sim),
            'window_physics_s': [float(times[0]), float(times[-1])], 'phases': {'lift': float(times[lift]), 'press_on': float(times[on]), 'press_off': float(times[off])},
            'inputs': {'run_json': sha(stage/'run.json'), 'trajectory': sha(ph['root']/'trajectory_mjwp.npz'), 'scene': sha(xml)},
            'status_of_source': json.loads((stage.parents[2]/'status.json').read_text()).get('status') if (stage.parents[2]/'status.json').exists() else None,
            'layout': 'T1(b) (user-approved 11-05)' if '/rrline/t1/' in str(stage) else 'ORIGINAL table layout',
            'finger_smoothing': sm_rep,
            'tube_slip_window_s': (lambda j: [j['from_physics_s'], j['to_physics_s']])(json.loads((stage/'tube_grip_check.json').read_text())['columns']['physics']['slip']) if (stage/'tube_grip_check.json').exists() else None}
    (out/'task.json').write_text(json.dumps(task, indent=1, default=float))
    print(json.dumps({k: task[k] for k in ('frames_100hz', 'ctrl_rows', 'window_physics_s', 'phases', 'status_of_source')}, default=float))


if __name__ == '__main__': main()
