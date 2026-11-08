"""TRACKER Stage 0 (11-06): a residual-RL task whose BASE is the KINEMATIC reference (IK arms + finger reference + E1 squeeze prior),
not an MPC physics demo. Layer: TASK EXPORT (no physics run).

Built on the timeline of rrline/task_T1k_still3_noff (same 100 Hz frames, sim-rate rows, gates, phases, scene, contact labels, starts),
which itself is the T1_k physics window: frame f <-> stage row warm + 4 f of actual_reference.npz (2.5 ms rows), sim row j <-> warm + j.

  * all non-finger joints (arms, objects, plunger, carrier): the stage's kinematic reference (actual_reference.npz = DAID's reference after
    retime, incl. object poses that follow the hand while held) -- NOT the executed physics.
  * finger joints: latched rows (stage hold_latch_mask) and the 0.6 s closure window before every latch start -> kinematic reference
    (it carries the E1 left squeeze, --finger-offset applied to reference and ctrl together); all other rows -> still3's hold-still
    finger targets (one minimum-jerk move per event; derived from the kinematic reference, no MPC data in those rows); 0.1 s blends.
  * ctrl_sim: arms = kinematic arm q (the env adds kv/kp qdot + inverse-dynamics lead: ctrl_sim_verbatim False); fingers = the finger
    reference above (position targets, no velocity feed-forward, as task_T1k_still3_noff).
  * --no-prior: the same with the E1 left squeeze offsets removed (baseline (a) "IK base only").
Contact labels (contact_ref.*) and valid starts are still3's (labels come from the physics demo: noted deviation for Stage 0 only).

  python -I build_kin_task.py OUT [--no-prior]
"""
import json, shutil, sys, hashlib, time
from pathlib import Path
import numpy as np, mujoco

HERE = Path(__file__).resolve().parent; V = HERE.parent
SRC = V/'tasks/task_T1k_still3_noff'; STAGE = V/'rrline/t1/runs/T1_k/physics/stages/T1_k'; PKG = V/'rrline/t1/pkg_T1_k/pkg.json'
CLOSURE_S, BLEND_S = .6, .1


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def qmul(a, b):
    w1, x1, y1, z1 = a.T; w2, x2, y2, z2 = b.T
    return np.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], 1)


def qvel_from_qpos(m, Q, dt):
    """finite-difference qvel (nv) from qpos rows: hinge / slide d/dt; free joints: world linear velocity, LOCAL angular velocity."""
    n = len(Q); Vv = np.zeros((n, m.nv))
    for j in range(m.njnt):
        a, d = m.jnt_qposadr[j], m.jnt_dofadr[j]; t = m.jnt_type[j]
        if t == mujoco.mjtJoint.mjJNT_FREE:
            Vv[:, d:d+3] = np.gradient(Q[:, a:a+3], dt, axis=0)
            q = Q[:, a+3:a+7]/np.linalg.norm(Q[:, a+3:a+7], axis=1, keepdims=True)
            q = q*np.sign(np.sum(q*np.roll(q, 1, 0), 1, keepdims=True)+1e-12)   # hemisphere continuity
            dq = np.gradient(q, dt, axis=0); qc = q*np.array([1, -1, -1, -1]); w = 2*qmul(qc, dq)[:, 1:]; Vv[:, d+3:d+6] = w
        elif t == mujoco.mjtJoint.mjJNT_BALL:
            raise ValueError('ball joint not handled')
        else:
            Vv[:, d] = np.gradient(Q[:, a], dt)
    return Vv


def blend_mask(mask, rate):
    """0/1 mask -> soft weight with BLEND_S linear ramps on the outside of each True run"""
    from scipy.ndimage import distance_transform_edt
    if not mask.any(): return mask.astype(float)
    d = distance_transform_edt(~mask)/rate
    return np.clip(1-d/BLEND_S, 0, 1)*(d > 0)+mask


def main():
    out = Path(sys.argv[1]).resolve(); no_prior = '--no-prior' in sys.argv
    if out.exists(): raise SystemExit(f'{out} exists')
    r = dict(np.load(SRC/'reference.npz')); m = mujoco.MjModel.from_xml_path(str(SRC/'scene.xml'))
    ar = np.load(STAGE/'actual_reference.npz'); warm = int(ar['warmup_steps']); Qk = ar['qpos']
    g = np.load(STAGE/'gates.npz'); L = g['hold_latch_mask'].astype(bool); kin_fps = 30.
    nf, ns = len(r['timestamps']), len(r['sim_times']); dt = float(r['sim_dt'])
    rows_f = np.minimum(np.floor(np.arange(nf)*.01*kin_fps+1e-9).astype(int), len(L)-1)
    rows_s = np.minimum(np.floor(np.arange(ns)*dt*kin_fps+1e-9).astype(int), len(L)-1)
    Kf = Qk[np.minimum(warm+4*np.arange(nf), len(Qk)-1)].copy(); Ks = Qk[np.minimum(warm+np.arange(ns), len(Qk)-1)].copy()
    acts = [m.actuator(i).name for i in range(m.nu)]; jid = m.actuator_trnid[:, 0]; qa = m.jnt_qposadr[jid]
    finger = np.array([n.startswith(('r_', 'l_')) for n in acts]); arm = ~finger
    # E1 squeeze (pkg per_video.finger_offset, cosine ramp over finger_offset_ramp, stage physics time); in Kf/Ks already (reference and ctrl together)
    pv = json.loads(PKG.read_text())['per_video']; T0, T1 = pv['finger_offset_ramp']; off = {}
    for s_ in pv['finger_offset']:   # JOINT=RAD or JOINT=RAD@T0,T1 (stage physics time of its own cosine ramp)
        jn, v = s_.split('='); t0_, t1_ = (T0, T1)
        if '@' in v: v, tt = v.split('@'); t0_, t1_ = map(float, tt.split(','))
        off[jn] = (float(v), t0_, t1_)
    ramp = lambda t, t0_, t1_: .5-.5*np.cos(np.pi*np.clip((t-t0_)/(t1_-t0_), 0, 1))
    tf, ts = 3.0+np.arange(nf)*.01, 3.0+np.arange(ns)*dt
    if no_prior:   # baseline (a): every per-video finger offset removed (E1 left squeeze + the right ring/pinky flex)
        for jn, (v, t0_, t1_) in off.items():
            a_ = m.jnt_qposadr[m.joint(jn).id]; Kf[:, a_] -= ramp(tf, t0_, t1_)*v; Ks[:, a_] -= ramp(ts, t0_, t1_)*v
    # time base check: stage physics time of frame 0 = window start of the physics export (task.json window_physics_s)
    tj = json.loads((SRC/'task.json').read_text())
    qpos = Kf.copy(); ctrl = np.zeros((ns, m.nu)); rep = {}
    for i in range(m.nu):
        a_ = qa[i]
        if arm[i]:
            ctrl[:, i] = Ks[:, a_]; continue
        Lf, Ls = L[rows_f, i], L[rows_s, i]
        # a joint that is latched at any time while its hand holds its object follows the KINEMATIC reference over the whole held window
        # (e.g. the right thumb's way onto the cap and the press: still3 keeps that path verbatim from the physics demo = MPC data)
        held = r['gate_pipette_held' if acts[i].startswith('r_') else 'gate_tube_held'].astype(bool)
        if (Lf & held).any():
            Lf = Lf | held; Ls = Ls | held[np.minimum(np.arange(ns)//4, nf-1)]
        # closure windows: CLOSURE_S before every latch start
        cf, cs = Lf.copy(), Ls.copy()
        for Lm, cm, rate in ((Lf, cf, 100.), (Ls, cs, 1/dt)):
            st = np.flatnonzero(Lm[1:] & ~Lm[:-1])+1
            for s0 in st: cm[max(0, s0-int(round(CLOSURE_S*rate))):s0] = True
        wf, ws = blend_mask(cf, 100.), blend_mask(cs, 1/dt)
        qpos[:, a_] = wf*Kf[:, a_]+(1-wf)*r['qpos_ref'][:, a_]
        ctrl[:, i] = ws*Ks[:, a_]+(1-ws)*r['ctrl_sim'][:, i]
        rep[acts[i]] = {'kin_rows_frac': round(float(wf.mean()), 3), 'max_jump_deg_at_blend': round(float(np.degrees(np.abs(np.diff(qpos[:, a_])).max())), 2)}
    qvel = qvel_from_qpos(m, qpos, .01)
    pl = int(m.jnt_qposadr[m.joint('pipette_plunger').id])
    r2 = dict(r); r2['qpos_ref'] = qpos; r2['qvel_ref'] = qvel; r2['ctrl_sim'] = ctrl; r2['ctrl_sim_verbatim'] = np.array(False)
    r2['controls'] = ctrl[np.minimum(np.arange(nf)*4, ns-1)].copy(); r2['plunger'] = -qpos[:, pl]
    out.mkdir(parents=True)
    for f in ('scene.xml', 'contact_ref.json', 'contact_ref.npz', 'valid_starts.npy', 'valid_starts_audit.json'): shutil.copy2(SRC/f, out/f)
    np.savez_compressed(out/'reference.npz', **r2)
    tj2 = dict(tj); tj2.update(mode='TRACKER Stage 0 kinematic base (no MPC base)', created=time.strftime('%Y-%m-%d %H:%M:%S'), base_kind='kinematic',
                               prior='none (E1 squeeze removed)' if no_prior else 'E1 left squeeze (pkg_T1_k finger_offset, thumb ip +0.61 rad)',
                               built_from={'timeline_task': str(SRC), 'timeline_task_reference_sha': sha(SRC/'reference.npz'), 'stage': str(STAGE),
                                           'actual_reference_sha': sha(STAGE/'actual_reference.npz'), 'gates_sha': sha(STAGE/'gates.npz'), 'pkg': str(PKG)},
                               finger_composite={'closure_s': CLOSURE_S, 'blend_s': BLEND_S, 'per_actuator': rep},
                               noted_deviation='contact_ref labels and valid starts copied from task_T1k_still3_noff (physics-demo labels)',
                               plunger_ref_max_mm=round(float(r2['plunger'].max()*1e3), 2))
    (out/'task.json').write_text(json.dumps(tj2, indent=1)+'\n')
    print(json.dumps({'out': str(out), 'prior': tj2['prior'], 'plunger_ref_max_mm': tj2['plunger_ref_max_mm'],
                      'finger_kin_frac_mean': round(float(np.mean([v['kin_rows_frac'] for v in rep.values()])), 3),
                      'max_finger_jump_deg': max(v['max_jump_deg_at_blend'] for v in rep.values())}))


if __name__ == '__main__':
    main()
