"""Real-robot line: policy vs zero residual under the SAME perturbations (layer: RL ENV rollouts, MuJoCo Warp; deterministic actor mean).
Per world: reaches the end without a trainer failure; end pose of pipette / tube vs the reference end (seat rule 5 mm / 3 deg; tube axis
only, spin free); press (max plunger depth in the press window >= 50 % of the reference and seconds >= 90 %); tube in the left wrist while
held: position drift, axis tilt (judged), spin (logged); naturalness: free-finger high-pass RMS (deg; joint minus its 0.3 s moving mean,
fingers per the physics-demo free mask), action rate, elbow height minus shoulder (cm, max), arm joint margin (deg, min).

  .venv-rl-head/bin/python -I eval_rr.py CONFIG.json [--ckpt PT] [--repeat R] [--out OUT.json]
"""
import argparse, json, math, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
HERE = Path(__file__).resolve().parent
import os as _os
ROOT = Path(_os.environ.get('G1PS_ROOT', str(Path(__file__).resolve().parents[1]/'deps_from_g1ps')))   # node: export G1PS_ROOT=<g1-pipette-sim checkout>
sys.path[:0] = [str(HERE), str(ROOT/'tools/rl')]
import vega_res_env as E
import gpu_train as GT

p = argparse.ArgumentParser(); p.add_argument('config', type=Path); p.add_argument('--ckpt', type=Path); p.add_argument('--repeat', type=int, default=4)
p.add_argument('--every-s', type=float, default=1.0); p.add_argument('--out', type=Path); p.add_argument('--no-dr', action='store_true')
p.add_argument('--diag-pipette', action='store_true', help='(11-06) per world: pipette seat error at the right release, right-hand contacts with the pipette after it, wrist error and stand-contact force at the extraction')
p.add_argument('--from-start', type=int, default=0, help='(11-06) N worlds that all start at frame 0 (whole task; full slip window) instead of the every-s starts')
p.add_argument('--student', type=Path, help='distilled deployable student (distill.py checkpoint); the config must carry extras.student'); a0 = p.parse_args()
C = json.loads(a0.config.read_text()); rel = lambda x: (x if Path(x).is_absolute() else (a0.config.parent/x)).resolve()
task = rel(C['task']); dr = None if a0.no_dr else (C.get('dr') if isinstance(C.get('dr'), dict) or C.get('dr') is None else json.loads(rel(C['dr']).read_text()))
r = np.load(task/'reference.npz'); t = r['timestamps']; nfr = len(t)
starts = [k-(k % 2) for k in np.arange(0, nfr-200, int(round(a0.every_s/.01)))]*a0.repeat; N = len(starts)
if a0.from_start: starts = [0]*a0.from_start; N = len(starts)
kw = dict(dr=dr, extras=C.get('extras', {}), reward_scale=C.get('reward_scale', .01))
for k in ('arm_scale', 'finger_scale', 'group_scale', 'thumb_press_scale', 'plunger_weight'):   # (11-09) tracker configs: must match training
    if C.get(k) is not None: kw[k] = C[k]
if a0.student: kw['obs_mode'] = 'deployable'
env = E.BimanualGpuEnv(task, N, seed=123, nconmax=C.get('nconmax', 200), njmax=C.get('njmax', 1600), naconmax_per_world=C.get('naconmax_per_world', 400),
                       terminate=False, auto_reset=False, **kw)
policy = None
if a0.ckpt:
    a = SimpleNamespace(steps_per_env=C.get('steps_per_env', 24), lr=3e-4, init_std=C.get('init_std', .1), no_value_clip=True, deployable=False)
    venv, runner = GT.make_runner(env, a, None); runner.load(str(a0.ckpt), map_location=env.device); pol = runner.get_inference_policy(device=env.device)
    policy = lambda: pol(env.get_observations())
if a0.student:
    import distill as D
    ck = torch.load(a0.student, map_location=env.device); stu = D.Student(ck['nin'], ck['nout'], ck['hidden']).to(env.device); stu.load_state_dict(ck['student']); stu.eval()
    assert ck['nin'] == env.obs_actor.shape[1], (ck['nin'], env.obs_actor.shape)
    policy = lambda: stu(env.obs_actor)
fp = C.get('eval_fixed_placements')   # (11-06) held-out placements: list of [dx_mm, dy_mm, dyaw_deg] cycled over the worlds (rack; stand gets the mirrored offset)
if fp:
    P_ = torch.as_tensor([fp[i % len(fp)] for i in range(N)], dtype=torch.float32, device=env.device)
    d_ = P_[:, :2]*1e-3; th_ = torch.deg2rad(P_[:, 2]); env.x_hold_fixed = {'rack': (d_, th_), 'stand': (-d_, -th_)}
fr = torch.as_tensor(starts, device=env.device)
env.reset_envs(torch.ones(N, dtype=torch.bool, device=env.device), frames=fr); env._launch(env.g_kin); env.obs = env._observe()
dev = env.device; first_fail = torch.full((N,), -1., device=dev); steps = int(max((nfr-1-f)//env.frame_step for f in starts))
_eg = (C.get('extras') or {}).get('event_gate')
if _eg: steps += int(round(float(_eg.get('cap_s', 1.))/E.CONTROL_DT))*4   # (11-06) the event gate may hold the reference clock up to cap_s per event (4 events)
press = torch.as_tensor(r['phase'] == 2, device=dev); pl_ref = torch.as_tensor(r['plunger'], device=dev).float(); pl_max = torch.zeros(N, device=dev); pl_s = torch.zeros(N, device=dev)
held_t = torch.as_tensor(r['gate_tube_held'], device=dev)
# (11-06) tube slip = the physics-demo checker's definition (tools/check_physics_tube_grip.py): tube in the left wrist frame vs its pose at the
# end of the grasp dwell, over [dwell end, end of the held span]; window read from the source stage's tube_grip_check.json. Reference pose =
# the demo's own (holder-adjusted) in-hand pose at the dwell end, so mid-episode starts are measured against the same pose.
tj = json.loads((task/'task.json').read_text()); sw = tj.get('tube_slip_window_s')
if sw is None:
    tg = json.loads((Path(tj['stage'])/'tube_grip_check.json').read_text())['columns']['physics']['slip']; sw = [tg['from_physics_s'], tg['to_physics_s']]
k_sf = int(np.argmin(np.abs(t-sw[0]))); slip_win = torch.as_tensor((t >= sw[0]-1e-9) & (t <= sw[1]+1e-9), device=dev); slip = torch.zeros(N, device=dev); slip_ang = torch.zeros(N, device=dev); slip_ref = None
fa = env.x_cref['fing_acts']; free = env.x_cref['free']; qf_hist = []; free_hist = []; ar = torch.zeros(N, device=dev); elbow = torch.full((N,), -1., device=dev)
margin = torch.full((N,), 1e9, device=dev); tube0 = {}; drift = torch.zeros(N, device=dev); tilt = torch.zeros(N, device=dev); spin = torch.zeros(N, device=dev)
prev_a = torch.zeros(N, env.nu, device=dev); live_steps = torch.zeros(N, device=dev); k_hist = []; live_hist = []; wdev_hist = []; adev_hist = []; vref_hist = []
fail_reason = torch.zeros(N, 2, dtype=torch.long, device=dev); _mm = env.mjm; GB = np.array([int(_mm.geom_bodyid[g]) for g in range(_mm.ngeom)]); OBJB = [_mm.body('right_object').id, _mm.body('left_object').id]
CATS = ['rack', 'stand', 'table_or_other_furniture', 'left_hand', 'right_hand', 'left_arm', 'right_arm', 'pipette', 'tube', 'other']
def _cat(g):
    n_ = (_mm.geom(g).name or ''); b_ = _mm.body(int(_mm.geom_bodyid[g])).name
    if n_.startswith(('furn_rack',)): return 0
    if n_.startswith('furn_stand'): return 1
    if n_.startswith('furn_table') or n_.startswith('furn_'): return 2
    if b_ == 'right_object' or b_.startswith(('pipette', 'plunger')): return 7
    if b_ == 'left_object': return 8
    if b_.startswith('left_'): return 3
    if b_.startswith('right_'): return 4
    if b_.startswith('L_arm'): return 5
    if b_.startswith('R_arm'): return 6
    return 9
GCAT = np.array([_cat(g) for g in range(_mm.ngeom)]); first_touch = [[{} for _ in range(N)], [{} for _ in range(N)]]
_gs = np.flatnonzero(np.diff(np.r_[0, r['gate_pipette_in_stand'].astype(int)]) == 1); _kr = int(_gs[1]) if len(_gs) > 1 else int(np.flatnonzero(r['gate_pipette_held'])[-1])   # placed back in the stand (fingers start opening)
_ke = np.flatnonzero(~r['gate_pipette_in_stand'].astype(bool)); _te = float(t[_ke[0]]) if len(_ke) else 5.2
PD = {'k_rel': _kr, 't_rel': float(t[_kr]), 'ext': [_te-.4, _te+.3], 'ext_wrist_mm': torch.zeros(N, device=dev), 'ext_wrist_deg': torch.zeros(N, device=dev),
      'rel_pos_mm': torch.full((N,), float('nan'), device=dev), 'rel_dz_mm': torch.full((N,), float('nan'), device=dev), 'rel_tilt_deg': torch.full((N,), float('nan'), device=dev),
      'rel_wrist_mm': torch.full((N,), float('nan'), device=dev), 'ext_stand_N': np.zeros(N), 'post_rel_touch': [{} for _ in range(N)]}
fdof_ = torch.as_tensor([int(env.mjm.jnt_dofadr[env.mjm.actuator_trnid[int(i), 0]]) for i in fa.cpu().numpy()], device=dev)
margin_cmd = torch.full((N,), 1e9, device=dev); k_rel = int(np.flatnonzero(r['gate_tube_held'])[-1]); rel_lat = torch.full((N,), float('nan'), device=dev); rel_dz = torch.full((N,), float('nan'), device=dev)
slip_tilt = torch.zeros(N, device=dev); ph_t = torch.as_tensor(r['phase'].astype(np.int64), device=dev)   # (11-06) tube axis tilt in hand + per-phase slip (carry / press / place)
slip_ph = {n_: torch.zeros(N, device=dev) for n_ in ('carry', 'press', 'place')}; tilt_ph = {n_: torch.zeros(N, device=dev) for n_ in ('carry', 'press', 'place')}
for i in range(steps):
    with torch.inference_mode():
        act = policy().clamp(-1, 1) if policy is not None else torch.zeros(N, env.nu, device=dev)
        obs, rew, done, ex = env.step(act)
    k = env.frame.clamp(max=nfr-1); live = (env.frame <= nfr-1) & (first_fail < 0); f_any = (ex['reasons'] > 0).any(1)
    newly = f_any & (first_fail < 0); first_fail[newly] = env.frame[newly].float()*.01+float(t[0])
    if newly.any(): fail_reason[newly] = ex['reasons'][newly].long()
    if a0.diag_pipette:
        kp_ = env.frame.clamp(max=nfr-1); tnow = float(t[int(kp_[0])]); Wp, Op = env._poses('right'); Wpr = E.X.ref_w(env, 'right', kp_); Opr = E.X.ref_o(env, 'right', kp_)
        we = (Wp[:, :3, 3]-Wpr[:, :3, 3]).norm(dim=-1)*1e3; wa = torch.rad2deg(E.rot_angle(Wp[:, :3, :3], Wpr[:, :3, :3]))
        if PD['ext'][0] <= tnow <= PD['ext'][1]: PD['ext_wrist_mm'] = torch.maximum(PD['ext_wrist_mm'], we); PD['ext_wrist_deg'] = torch.maximum(PD['ext_wrist_deg'], wa)
        at_ = (kp_ >= PD['k_rel']) & torch.isnan(PD['rel_pos_mm'])
        if at_.any():
            PD['rel_pos_mm'][at_] = ((Op[at_, :3, 3]-Opr[at_, :3, 3]).norm(dim=-1)*1e3); PD['rel_dz_mm'][at_] = (Op[at_, 2, 3]-Opr[at_, 2, 3])*1e3
            PD['rel_tilt_deg'][at_] = torch.rad2deg(torch.arccos((Op[at_, :3, 2]*Opr[at_, :3, 2]).sum(-1).clamp(-1, 1))); PD['rel_wrist_mm'][at_] = we[at_]
        n2 = int(env.nacon.reshape(-1)[0])
        if n2:
            g2 = env.x_cgeom[:n2].long(); w2 = env.x_cworld[:n2].long(); f2 = env.x_force[w2, env.x_cefc[:n2, 0].long()].cpu().numpy(); g2n = g2.cpu().numpy(); w2n = w2.cpu().numpy()
            b1, b2 = GB[g2n[:, 0]], GB[g2n[:, 1]]; hitp = (b1 == OBJB[0]) | (b2 == OBJB[0]); oth = np.where(b1 == OBJB[0], g2n[:, 1], g2n[:, 0])
            if PD['ext'][0] <= tnow <= PD['ext'][1]:
                sel_ = hitp & (GCAT[oth] == 1)
                for w_, f_ in zip(w2n[sel_], f2[sel_]): PD['ext_stand_N'][w_] = max(PD['ext_stand_N'][w_], float(f_))
            if tnow >= PD['t_rel']:
                sel_ = hitp & (GCAT[oth] == 4)
                for w_, g_, f_ in zip(w2n[sel_], oth[sel_], f2[sel_]):
                    if f_ > .05: PD['post_rel_touch'][w_].setdefault(_mm.body(int(_mm.geom_bodyid[g_])).name, [round(tnow, 2), round(tnow, 2), 0.])
                    if f_ > .05: e_ = PD['post_rel_touch'][w_][_mm.body(int(_mm.geom_bodyid[g_])).name]; e_[1] = round(tnow, 2); e_[2] = round(max(e_[2], float(f_)), 2)          # (11-06) per object [pipette, tube]: 1 + index into E.FAILURES, 0 = ok
    # (11-06) first contact time of each partner category with the tube / the pipette (who knocks it)
    nc_ = int(env.nacon.reshape(-1)[0])
    if nc_:
        cg_ = env.wp.to_torch(env.d.contact.geom)[:nc_].cpu().numpy(); cw_ = env.wp.to_torch(env.d.contact.worldid)[:nc_].cpu().numpy(); ts_now = float(t[int(env.frame[0].clamp(max=nfr-1))])
        for oi_, ob_ in enumerate(OBJB):
            b1_, b2_ = GB[cg_[:, 0]], GB[cg_[:, 1]]; hit = (b1_ == ob_) | (b2_ == ob_); other_ = np.where(b1_ == ob_, cg_[:, 1], cg_[:, 0])[hit]; wv_ = cw_[hit]
            for c_, w_ in zip(GCAT[other_], wv_):
                if first_touch[oi_][w_].get(int(c_)) is None: first_touch[oi_][w_][int(c_)] = round(ts_now, 2)
    depth = -env.qpos[:, env.pl_q]; inp = press[k] & live; pl_max = torch.where(inp, torch.maximum(pl_max, depth), pl_max)
    pl_s += (inp & (depth >= .9*pl_ref.max())).float()*E.CONTROL_DT
    Wr_, Wl_ = env._poses('right')[0], env._poses('left')[0]; kk_ = env.frame.clamp(max=nfr-1)
    wdev_hist.append(torch.cat([Wr_[:, :3, 3]-E.X.ref_w(env, 'right', kk_)[:, :3, 3], Wl_[:, :3, 3]-E.X.ref_w(env, 'left', kk_)[:, :3, 3]], 1).cpu())
    adev_hist.append((env.qpos[:, env.aq][:, env.x_arm]-env.qpos_ref[kk_][:, env.aq][:, env.x_arm]).cpu())
    vref_hist.append(env.qvel_ref[env.frame.clamp(max=nfr-1)][:, fdof_].cpu())
    qf_hist.append(env.qpos[:, env.aq][:, fa].clone()); free_hist.append((free[k] > .5) & live[:, None]); k_hist.append(k.clone()); live_hist.append(live.clone())
    ar += torch.where(live, ((act-prev_a)**2).mean(1), torch.zeros_like(ar)); prev_a = act.clone(); live_steps += live.float()
    dz = (env.xpos[:, env.x_elbow, 2]-env.xpos[:, env.x_shoulder, 2]).max(1).values; elbow = torch.where(live, torch.maximum(elbow, dz), elbow)
    qa = env.qpos[:, env.aq][:, env.x_arm]; mg = torch.minimum(qa-(env.x_alo-.1), (env.x_ahi+.1)-qa).min(1).values; margin = torch.where(live, torch.minimum(margin, mg), margin)
    ca_ = env.ctrl[:, env.x_arm]; mgc = torch.minimum(ca_-(env.x_alo-.1), (env.x_ahi+.1)-ca_).min(1).values; margin_cmd = torch.where(live, torch.minimum(margin_cmd, mgc), margin_cmd)   # (11-06) commanded targets
    _, Ot_ = env._poses('left'); Or_ = E.X.ref_o(env, 'left', k); at_rel = live & (k >= k_rel) & torch.isnan(rel_lat)   # tube vs its (holder-adjusted) reference at the planned release frame
    if at_rel.any():
        rel_lat[at_rel] = (Ot_[at_rel, :2, 3]-Or_[at_rel, :2, 3]).norm(dim=-1); rel_dz[at_rel] = Ot_[at_rel, 2, 3]-Or_[at_rel, 2, 3]
    W, O = env._poses('left'); loc = E.rinv(W)@O; h = held_t[k] & live
    if 'p' not in tube0: tube0['p'] = loc[:, :3, 3].clone(); tube0['R'] = loc[:, :3, :3].clone(); tube0['set'] = torch.zeros(N, dtype=torch.bool, device=dev)
    setn = h & ~tube0['set']; tube0['p'][setn] = loc[setn, :3, 3]; tube0['R'][setn] = loc[setn, :3, :3]; tube0['set'] |= h
    dp = (loc[:, :3, 3]-tube0['p']).norm(dim=-1); ax = torch.arccos((loc[:, :3, 2]*tube0['R'][:, :3, 2]).sum(-1).clamp(-1, 1))
    full = E.rot_angle(loc[:, :3, :3], tube0['R'])
    drift = torch.where(h, torch.maximum(drift, dp), drift); tilt = torch.where(h, torch.maximum(tilt, ax), tilt); spin = torch.where(h, torch.maximum(spin, full), spin)
    if slip_ref is None:
        kk_ = torch.full((N,), k_sf, device=dev, dtype=torch.long); slip_ref = E.rinv(E.X.ref_w(env, 'left', kk_))@E.X.ref_o(env, 'left', kk_)
    sw_ = slip_win[k] & live; dps = (loc[:, :3, 3]-slip_ref[:, :3, 3]).norm(dim=-1); asl = E.rot_angle(loc[:, :3, :3], slip_ref[:, :3, :3])
    slip = torch.where(sw_, torch.maximum(slip, dps), slip); slip_ang = torch.where(sw_, torch.maximum(slip_ang, asl), slip_ang)
    tax = torch.arccos((loc[:, :3, 2]*slip_ref[:, :3, 2]).sum(-1).clamp(-1, 1)); slip_tilt = torch.where(sw_, torch.maximum(slip_tilt, tax), slip_tilt)
    phk = ph_t[k]
    for pi_, pn_ in ((1, 'carry'), (2, 'press'), (3, 'place')):
        sel_ = sw_ & (phk == pi_); slip_ph[pn_] = torch.where(sel_, torch.maximum(slip_ph[pn_], dps), slip_ph[pn_]); tilt_ph[pn_] = torch.where(sel_, torch.maximum(tilt_ph[pn_], tax), tilt_ph[pn_])
end_ok = first_fail < 0
# free-finger high-pass RMS (control rate; 0.3 s moving mean), over live free samples
Q = torch.stack(qf_hist).cpu().numpy(); M = torch.stack(free_hist).cpu().numpy(); win = max(1, int(round(.3/E.CONTROL_DT)))
ker = np.ones(win)/win; hp = np.zeros_like(Q)
for w in range(N):
    for j in range(Q.shape[2]): hp[:, w, j] = Q[:, w, j]-np.convolve(Q[:, w, j], ker, mode='same')
hp_rms = np.sqrt((np.where(M, hp, 0)**2).sum(0).sum(1)/np.maximum(M.sum(0).sum(1), 1))*180/math.pi
# (11-06) jitter v2 cells on the RL rollout: zero-phase 3-10 Hz band (4th-order Butterworth at the 50 Hz control rate) of every finger joint
# angle, RMS over the live rows of each window, median over worlds, worst joint per hand / finger / window. Windows from the reference gates:
# left hold = tube held; right hold = pipette held outside the press; right press = press phase +-0.3 s; post-release = after that hand's held span.
from scipy.signal import butter, filtfilt
BPj = butter(4, [3/(.5/E.CONTROL_DT), 10/(.5/E.CONTROL_DT)], 'band'); KH = torch.stack(k_hist).cpu().numpy(); LV = torch.stack(live_hist).cpu().numpy()
fnames = [env.action_names[int(i)] for i in fa.cpu().numpy()]
gt = {nm: r[nm].astype(bool) for nm in ('gate_tube_held', 'gate_pipette_held')}; ph_ = r['phase']; pidx = np.flatnonzero(ph_ == 2); prw = np.zeros(nfr, bool)
if len(pidx): prw[max(0, pidx[0]-30):pidx[-1]+31] = True
lh_last, rh_last = np.flatnonzero(gt['gate_tube_held'])[-1], np.flatnonzero(gt['gate_pipette_held'])[-1]
WIN = {'left': {'hold': gt['gate_tube_held'], 'post_release': np.arange(nfr) > lh_last},
       'right': {'hold': gt['gate_pipette_held'] & ~prw, 'press': prw, 'post_release': np.arange(nfr) > rh_last}}
BQ = np.zeros_like(Q)
for w in range(N):
    for j in range(Q.shape[2]): BQ[:, w, j] = filtfilt(*BPj, Q[:, w, j]) if Q.shape[0] > 30 else 0.
jit = {}
for side, pre in (('left', 'l_'), ('right', 'r_')):
    for f in ('thumb', 'index', 'middle', 'ring', 'pinky'):
        js = [j for j, nm in enumerate(fnames) if nm.startswith(pre+f)]
        for wn, gate_ in WIN[side].items():
            sel = gate_[KH] & LV                                                  # (T, N)
            vals = [np.sqrt((BQ[:, :, j]**2*sel).sum(0)/np.maximum(sel.sum(0), 1)) for j in js]   # per joint (N,)
            ok_w = sel.sum(0) >= 30
            if not ok_w.any() or not js: continue
            jit[f'{side} {f} {wn}'] = round(float(max(np.degrees(np.median(v[ok_w])) for v in vals)), 3)
thr_j = 0.25
# (11-06) whole-hand / arm shake: wrist position deviation from the (holder-adjusted) reference and arm joint deviation, zero-phase band RMS per
# window (0.5-3 Hz info; 3-8 Hz and 8-24 Hz judged, PROPOSED <= 0.3 mm / 0.1 mm), median over worlds, per hand / worst arm joint
# (11-06, user: fingers must not move when nothing calls for it) IDLE fingers: rows where the reference joint speed < 0.005 rad/s for >= 0.3 s;
# per hand / finger / phase: max range inside one idle window (deg) and RMS speed over idle rows (deg/s), worst joint, median over worlds (PROPOSED: range <= 2 deg, RMS <= 3 deg/s)
VR = np.abs(torch.stack(vref_hist).numpy()); Qd = np.degrees(Q); Vd = np.vstack([np.zeros((1,)+Qd.shape[1:]), np.diff(Qd, axis=0)/E.CONTROL_DT])
idle = {}
def _runs(b):
    ix = np.flatnonzero(np.diff(np.r_[0, b.astype(int), 0])); return list(zip(ix[::2], ix[1::2]))
held_first = {'left': np.flatnonzero(gt['gate_tube_held'])[0], 'right': np.flatnonzero(gt['gate_pipette_held'])[0]}
for side, pre in (('left', 'l_'), ('right', 'r_')):
    PH = {'pre_grasp': np.arange(nfr) < held_first[side], 'hold': WIN[side]['hold'] | (WIN[side].get('press', np.zeros(nfr, bool))), 'post_release': WIN[side]['post_release']}
    for f in ('thumb', 'index', 'middle', 'ring', 'pinky'):
        js = [j for j, nm in enumerate(fnames) if nm.startswith(pre+f)]
        for pn, pm in PH.items():
            rng_w, rms_w = [], []
            for w in range(N):
                rj, vj = 0., 0.
                for j in js:
                    m_ = (VR[:, w, j] < .005) & pm[KH[:, w]] & LV[:, w]; rr = [(a_, b_) for a_, b_ in _runs(m_) if b_-a_ >= 15]   # reference < 0.3 deg/s (excludes minimum-jerk tails)
                    if not rr: continue
                    rj = max(rj, max(float(Qd[a_:b_, w, j].max()-Qd[a_:b_, w, j].min()) for a_, b_ in rr))
                    sel_ = np.zeros(len(m_), bool)
                    for a_, b_ in rr: sel_[a_:b_] = True
                    vj = max(vj, float(np.sqrt(np.mean(Vd[sel_, w, j]**2))))
                rng_w.append(rj); rms_w.append(vj)
            if rng_w: idle[f'{side} {f} {pn}'] = [round(float(np.median(rng_w)), 2), round(float(np.median(rms_w)), 2)]
WD = torch.stack(wdev_hist).numpy()*1e3; AD = np.degrees(torch.stack(adev_hist).numpy()); BANDS = {'0.5-3Hz': (.5, 3.), '3-8Hz': (3., 8.), '8-24Hz': (8., 24.)}
WINS = {'carry': ph_[KH] == 1, 'press': ph_[KH] == 2, 'post_release': (np.arange(nfr) > max(lh_last, rh_last)+30)[KH]}
shake = {'wrist_dev_band_mm [R, L]': {}, 'arm_dev_band_deg (worst joint)': {}}
for bn_, (lo_, hi_) in BANDS.items():
    bb_ = butter(2, [lo_/(.5/E.CONTROL_DT), hi_/(.5/E.CONTROL_DT)], 'band'); YW = filtfilt(*bb_, WD, axis=0) if WD.shape[0] > 30 else WD*0; YA = filtfilt(*bb_, AD, axis=0) if AD.shape[0] > 30 else AD*0
    for wn_, sel_w in WINS.items():
        sel = sel_w & LV; ok_ = sel.sum(0) >= 30
        if not ok_.any(): continue
        r_ = [np.sqrt(((YW[:, :, 3*h:3*h+3]**2).sum(-1)*sel).sum(0)/np.maximum(sel.sum(0), 1)) for h in (0, 1)]
        a_ = np.sqrt(((YA**2)*sel[..., None]).sum(0)/np.maximum(sel.sum(0), 1)[:, None]).max(1)
        shake['wrist_dev_band_mm [R, L]'].setdefault(wn_, {})[bn_] = [round(float(np.median(x[ok_])), 3) for x in r_]
        shake['arm_dev_band_deg (worst joint)'].setdefault(wn_, {})[bn_] = round(float(np.median(a_[ok_])), 4)
ab = env.action_names.index('r_thumb_cmc_abd') if 'r_thumb_cmc_abd' in env.action_names else None
W_end, O_end = {}, {}
for s in E.SIDES:
    _, O = env._poses(s); O_end[s] = O
kend = torch.full((N,), nfr-1, device=dev, dtype=torch.long); Oref_end = {s: E.X.ref_o(env, s, kend) for s in E.SIDES}   # holder-adjusted end pose
seat = {}
for s, name in zip(E.SIDES, ('pipette', 'tube')):
    pe = (O_end[s][:, :3, 3]-Oref_end[s][:, :3, 3]).norm(dim=-1); ae = torch.arccos((O_end[s][:, :3, 2]*Oref_end[s][:, :3, 2]).sum(-1).clamp(-1, 1))
    seat[name] = (end_ok & (pe <= .005) & (ae <= math.radians(3))).cpu().numpy()
ref_s = float(((pl_ref >= .9*pl_ref.max()) & press).sum())*.01   # the demo's own seconds at >= 90 % of its max depth in the press window
pl_ok = ((pl_max >= .5*pl_ref.max()) & (pl_s >= 1.5)).cpu().numpy()   # DECIDES (coordinator 11-05: absolute rule until the user rules)
pl_ok_rel = ((pl_max >= .5*pl_ref.max()) & (pl_s >= .9*ref_s)).cpu().numpy();   # reported: relative to the demo (E1 1.79 s, T1b 1.25 s); user to decide
press_rows = np.array([bool(r['phase'][f:].tolist().count(2)) and f < int(np.argmax(r['phase'] == 2)) for f in starts])
f2 = lambda x: round(float(x), 3)
res = {'layer': 'RL env rollouts (MuJoCo Warp), deterministic actor mean' if policy else 'RL env rollouts, ZERO residual', 'ckpt': str(a0.ckpt or a0.student) if (a0.ckpt or a0.student) else None, 'actor': 'student (deployable obs)' if a0.student else ('teacher (full state)' if a0.ckpt else 'none'),
       'dr': None if dr is None else 'on', 'n': N, 'reach_end': int(end_ok.sum()), 'pipette_seated': int(seat['pipette'].sum()), 'tube_seated': int(seat['tube'].sum()),
       'press_ok_of_press_starts': f'{int((pl_ok & press_rows).sum())}/{int(press_rows.sum())}', 'press_rule (decides)': 'max depth >= 50 % of the demo max and >= 90 % depth for >= 1.5 s',
       'press_ok_relative (info, user to rule)': f'{int((pl_ok_rel & press_rows).sum())}/{int(press_rows.sum())} (>= 0.9 x the demo {ref_s:.2f} s)',
       'tube_slip_checker_def': {'window_s': [round(float(sw[0]), 3), round(float(sw[1]), 3)], 'slip_mm_p50_max': [f2(np.median(slip.cpu().numpy())*1e3), f2(slip.max()*1e3)],
                                 'turn_deg_p50_max': [f2(np.degrees(np.median(slip_ang.cpu().numpy()))), f2(np.degrees(float(slip_ang.max())))],
                                 'rule': 'tube in the left wrist frame vs the demo in-hand pose at the grasp-dwell end, over [dwell end, held end] (check_physics_tube_grip definition)'},
       'tube_in_hand (old, info: vs first held frame incl. the grasp closure)': {'drift_mm_p50_max': [f2(np.median(drift.cpu().numpy())*1e3), f2(drift.max()*1e3)], 'axis_tilt_deg_p50_max': [f2(np.degrees(np.median(tilt.cpu().numpy()))), f2(np.degrees(float(tilt.max())))],
                        'spin_deg_p50_max (logged, deferred)': [f2(np.degrees(np.median(spin.cpu().numpy()))), f2(np.degrees(float(spin.max())))]},
       'natural': {'free_finger_hp_rms_deg_p50_p95': [f2(np.median(hp_rms)), f2(np.percentile(hp_rms, 95))], 'action_rate_mean_sq': float('%.3g' % float((ar/live_steps.clamp_min(1)).mean())),
                   'elbow_minus_shoulder_cm_max': f2(elbow.max()*100), 'arm_joint_margin_deg_min': f2(np.degrees(float(margin.min()))),
                   'jitter_v2_band_3_10hz_rms_deg (median over worlds, worst joint)': jit,
                   'jitter_v2_over_0.25 (PROPOSED; hold / press / post-release judged)': {k_: v_ for k_, v_ in jit.items() if v_ > thr_j},
                   'shake (wrist / arm deviation from the reference, band RMS; PROPOSED 3-8 Hz <= 0.3 mm, 8-24 Hz <= 0.1 mm)': shake,
                   'idle_fingers [range deg, rms speed deg/s] (PROPOSED range <= 2, rms <= 3)': idle,
                   'idle_fingers_over (PROPOSED)': {k_: v_ for k_, v_ in idle.items() if v_[0] > 2 or v_[1] > 3}}}
# (11-06) per-world rows (same seed 123 -> world i has the same perturbation for the policy and for zero residual: matched comparisons)
Oe_, Oer_ = O_end['left'], Oref_end['left']; ax_e = torch.arccos((Oe_[:, :3, 2]*Oer_[:, :3, 2]).sum(-1).clamp(-1, 1))
hx = getattr(env, 'x_holes', None); hd = env.x_hold if getattr(env, 'x_hold', None) else None
res['natural']['arm_joint_margin_cmd_deg_min (commanded targets; rule >= 5.73)'] = f2(np.degrees(float(margin_cmd.min())))
res['per_world_detail'] = {'end_tube_lat_mm': ((Oe_[:, :2, 3]-Oer_[:, :2, 3]).norm(dim=-1).cpu().numpy()*1e3).round(2).tolist(), 'end_tube_dz_mm': ((Oe_[:, 2, 3]-Oer_[:, 2, 3]).cpu().numpy()*1e3).round(2).tolist(),
                           'end_tube_tilt_deg': np.degrees(ax_e.cpu().numpy()).round(2).tolist(), 'release_lat_mm': (rel_lat.cpu().numpy()*1e3).round(2).tolist(), 'release_dz_mm': (rel_dz.cpu().numpy()*1e3).round(2).tolist(),
                           'hole_pair': ['-'.join(hx['pairs'][int(i)]) for i in hx['idx'].cpu().numpy()] if hx is not None else None,
                           'rack_dxy_mm': (hd['rack']['d'].cpu().numpy()*1e3).round(1).tolist() if hd else None, 'rack_dyaw_deg': np.degrees(hd['rack']['th'].cpu().numpy()).round(2).tolist() if hd else None,
                           'stand_dxy_mm': (hd['stand']['d'].cpu().numpy()*1e3).round(1).tolist() if hd else None, 'stand_dyaw_deg': np.degrees(hd['stand']['th'].cpu().numpy()).round(2).tolist() if hd else None,
                           'first_fail_s': first_fail.cpu().numpy().round(2).tolist(),
                           'fail_reason [pipette, tube]': [[(E.FAILURES[int(x)-1] if int(x) > 0 else None) for x in row] for row in fail_reason.cpu().numpy()],
                           'first_touch_s pipette': [{CATS[c_]: v_ for c_, v_ in sorted(d_.items())} for d_ in first_touch[0]],
                           'first_touch_s tube': [{CATS[c_]: v_ for c_, v_ in sorted(d_.items())} for d_ in first_touch[1]], 'arm_margin_cmd_deg': np.degrees(margin_cmd.cpu().numpy()).round(2).tolist(), 'arm_margin_q_deg': np.degrees(margin.cpu().numpy()).round(2).tolist()}
if a0.diag_pipette:
    c_ = lambda x: [round(float(v), 2) for v in x.cpu().numpy()]
    res['pipette_diag'] = {'placed_in_stand_t_s (release window starts)': round(PD['t_rel'], 3), 'extraction_window_s': [round(x, 3) for x in PD['ext']], 'at_release_pipette_pos_err_mm': c_(PD['rel_pos_mm']), 'at_release_pipette_dz_mm': c_(PD['rel_dz_mm']),
                           'at_release_pipette_tilt_deg': c_(PD['rel_tilt_deg']), 'at_release_right_wrist_err_mm': c_(PD['rel_wrist_mm']), 'extraction_right_wrist_err_mm_max': c_(PD['ext_wrist_mm']),
                           'extraction_right_wrist_err_deg_max': c_(PD['ext_wrist_deg']), 'extraction_stand_contact_N_max': [round(float(v), 2) for v in PD['ext_stand_N']],
                           'after_placed_right_hand_touch {body: [t_first, t_last, max_N]}': PD['post_rel_touch'],
                           'holder_meas_err stand [dx_mm, dy_mm, dyaw_deg]': [[round(float(a_), 1) for a_ in row] for row in torch.cat([(env.x_hold['stand']['dm']-env.x_hold['stand']['d'])*1e3, torch.rad2deg(env.x_hold['stand']['thm']-env.x_hold['stand']['th'])[:, None]], 1).cpu().numpy()] if getattr(env, 'x_hold', None) else None}
if getattr(env, 'x_gate', None) is not None:
    Hg = env.x_gate['held'].cpu().numpy()*E.CONTROL_DT
    res['event_gate'] = {'events [side, frame, kind]': [[s_, int(g_), k_] for s_, g_, k_ in env.x_gate['events']], 'held_s_mean': Hg.mean(0).round(3).tolist(), 'held_s_max': Hg.max(0).round(3).tolist(),
                         'worlds_held_to_cap': (Hg >= float(_eg.get('cap_s', 1.))-1e-6).sum(0).tolist(), 'rule': 'reference clock held until the wrist (FK of measured joints) is within tol of its event pose (robot holder estimate); cap per event'}
# (11-06) cause table per world (default): which object fails, when, and the end state of the tube
def _segs(g):
    ix = np.flatnonzero(np.diff(np.r_[0, g.astype(int), 0])); return list(zip(ix[::2], ix[1::2]))
_th = _segs(r['gate_tube_held'].astype(bool)); _t_th0, _t_th1 = (float(t[_th[0][0]]), float(t[min(_th[0][1], nfr-1)])) if _th else (1e9, 1e9)
_ps = _segs(r['gate_pipette_in_stand'].astype(bool)); _t_place = float(t[_ps[1][0]]) if len(_ps) > 1 else 1e9; _t_ext = float(t[_ps[0][1]]) if _ps else 0.
_ff = first_fail.cpu().numpy(); _fr = fail_reason.cpu().numpy(); _ldz = np.array(res['per_world_detail']['end_tube_dz_mm']); _ltl = np.array(res['per_world_detail']['end_tube_tilt_deg'])
_rdz = np.array(res['per_world_detail']['release_dz_mm'], dtype=float); causes = []
for w in range(N):
    jam = (_ldz[w] > 3) and (_ltl[w] > 3) and (_ldz[w] < 10)
    if _ff[w] < 0:
        c_ = 'success' if (seat['tube'][w] and seat['pipette'][w]) else ('tube jammed tilted ~5 mm high (reached the end)' if jam else ('pipette not seated (reached the end)' if not seat['pipette'][w] else 'tube not seated, other (reached the end)'))
    elif _fr[w, 0] > 0:   # the pipette failed (first, or together with the tube)
        c_ = 'pipette extraction from the stand' if _ff[w] < _t_ext+1.5 else ('pipette after placement in the stand (fingers opening)' if _ff[w] >= _t_place-.5 else 'pipette, carry / press')
    else:
        if _ff[w] < _t_th0: c_ = 'tube disturbed before / at the grasp'
        elif _ff[w] < _t_th0+1.5: c_ = 'tube disturbed at the grasp / lost at the lift'
        elif _ff[w] >= _t_th1-1.: c_ = 'tube released high (place descent lagging)' if (np.isfinite(_rdz[w]) and _rdz[w] > 20) else ('tube jammed tilted ~5 mm high' if jam else 'tube fell / missed during the place')
        else: c_ = 'tube lost during the carry / press'
    causes.append(c_)
res['cause_table'] = dict(sorted(((c_, causes.count(c_)) for c_ in set(causes)), key=lambda x: -x[1])); res['per_world_detail']['cause'] = causes
res['per_world'] = {'start_frame': [int(x) for x in starts], 'end_ok': end_ok.cpu().numpy().astype(bool).tolist(), 'tube_seated': seat['tube'].astype(bool).tolist(),
                    'slip_mm': (slip.cpu().numpy()*1e3).round(2).tolist(), 'tilt_deg': np.degrees(slip_tilt.cpu().numpy()).round(2).tolist(),
                    'slip_phase_mm': {n_: (v_.cpu().numpy()*1e3).round(2).tolist() for n_, v_ in slip_ph.items()}, 'tilt_phase_deg': {n_: np.degrees(v_.cpu().numpy()).round(2).tolist() for n_, v_ in tilt_ph.items()}}
print(json.dumps({k_: v_ for k_, v_ in res.items() if k_ != 'per_world'}))
if a0.out: a0.out.write_text(json.dumps(res, indent=1))
