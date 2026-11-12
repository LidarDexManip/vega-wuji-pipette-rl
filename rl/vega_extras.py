"""Real-robot line: extras for vega_res_env (layer: RL ENV). Real-robot arm command path, residual smoothing, and the extra reward terms
(shared library h13_work/vega/common/rewards.py: Jiming T1-T3, naturalness, elbow rule). All weights / switches come from the config dict
(rrline/env_rr.yaml); weight 0 = term off (still logged if log=True).
"""
import json, sys
from pathlib import Path
import math
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'common'))
import rewards as RW

DEFAULT = {'real_cmd': True, 'margin_rad': .1, 'speed_cap': 2.5, 'ema_alpha': None, 'cmd_hz': None, 'vel_ff': True, 'clamp_after_ff': False, 'band_on_deviation': False, 'event_gate': None,
           'w': {'contact_set': 0., 'contact_graph': 0., 'wrench_support': 0., 'free_finger_acc': 0., 'action_rate': 0., 'elbow_height': 50., 'joint_margin': 0.,
                 'finger_jerk': 0., 'thumb_prior': 0., 'posture': 0., 'finger_band': 0., 'tube_tilt': 0., 'arm_band': 0., 'finger_still': 0., 'pipette_seat': 0.,
                 'contact_keep': 0., 'contact_drift': 0., 'contact_slip': 0., 'wrench_margin': 0., 'torque_rate_arm': 0., 'torque_rate_finger': 0., 'tip_align': 0.},
           # (11-12) pipette tip vs the tube (tip_init): 'tip': {'obs': privileged teacher observation, sigmas of the tip_align kernel}
           'tip': {'obs': False, 'sigma_coarse_mm': 20., 'sigma_fine_mm': 4., 'sigma_ang_deg': 8., 'obs_scale_mm': 20., 'obs_clip': 5.},
           # (11-09) torque rate (torque_rate_terms): change of the control-step mean actuator torque (averaged over the substeps, so nothing
           # above 25 Hz aliases in) between control steps, / (torque_rate_frac x the actuator's torque limit), squared, mean over the arm /
           # finger actuators, capped at torque_rate_cap. action_rate_norm (None = the old raw rad^2): action rate on the normalised actions,
           # 1 = a change of action_rate_norm per control step on every actuator; both 0 on the first step after a reset
           'torque_rate_frac': .01, 'torque_rate_cap': 25., 'action_rate_norm': None, 'action_rate_cap': None,
           'sigma_graph_mm': 3., 'force_alpha': .1, 'torque_length': .05, 'log': True,
           # (11-08) reference-free contact stability between consecutive control steps (contact_temporal): a link is in contact when its normal
           # force > force_alpha x object weight and stays in contact until it drops below keep_hysteresis x that; drift / slip normalised so
           # 1 = drift_mm / slip_mm per control step
           'keep_hysteresis': .5, 'drift_mm': 1., 'slip_mm': 1.,
           # reference.npz phase gates (task timeline, not contact labels) during which the three are off, e.g. deliberate in-hand regrasps
           'temporal_off_gates': [],
           # (11-08) task-wrench margin (margin_term): per-contact normal force <= margin_fmax (N), circular friction cones mu = margin_mu,
           # disturbance margin_rho (N; torques as tau / torque_length); solved every margin_every control steps (held in between), only while the
           # reference holds the object (gate_pipette_held / gate_tube_held) if margin_gate; log_margin computes it for logging at weight 0
           'margin_fmax': 3., 'margin_mu': 1.5, 'margin_rho': .3, 'margin_iters': 100, 'margin_every': 4, 'margin_gate': True, 'log_margin': False,
           # (11-09) weights of the base tracking terms per side (vega_res_env._reward_and_termination): object = object axis points in the wrist
           # frame (grasp), link / joint = finger links in the wrist frame / finger joints, wrist = wrist world pose, world = object world pose;
           # wrist / world carry their own x2 inside the term; the defaults are the original hard-coded weights (unchanged reward)
           'track_w': {'object': 8., 'link': 1., 'joint': 1., 'wrist': 1., 'world': 1.}}


def merge(a, b):
    out = dict(a)
    for k, v in (b or {}).items(): out[k] = merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return out


def init(env, cfg):
    m = env.mjm; dev = env.device; X = merge(DEFAULT, cfg); env.xcfg = X; env.cfg['extras'] = X; env.x_hold = None
    acts = env.action_names; arm = [i for i, n in enumerate(acts) if not n.startswith(('r_', 'l_'))]
    env.x_arm = torch.as_tensor(arm, device=dev); jid = m.actuator_trnid[arm, 0]
    jr = torch.as_tensor(m.jnt_range[jid], dtype=torch.float32, device=dev)
    env.x_alo, env.x_ahi = jr[:, 0]+X['margin_rad'], jr[:, 1]-X['margin_rad']
    env.x_prev_arm = env.ctrl[:, env.x_arm].clone(); env.x_af = torch.zeros(env.num_envs, env.nu, device=dev)
    # real-robot streaming (cmd_hz, 11-05): arm targets held at cmd_hz; residual upsampled linearly from the 50 Hz policy; velocity feedforward
    # through the position actuator (ctrl = q* + kv/kp * qd*, nominal gains: kp (q*-q) + kv (qd*-qd))
    env.x_rlast = torch.zeros(env.num_envs, env.nu, device=dev); env.x_rprev = torch.zeros_like(env.x_rlast); env.x_tick = env.ctrl[:, env.x_arm].clone()
    env.x_tickq = env.x_tick.clone(); env.x_sub = 0; env.x_base = torch.zeros(env.num_envs, dtype=torch.long, device=dev)
    env.x_ffr = torch.as_tensor(-m.actuator_biasprm[arm, 2]/np.maximum(m.actuator_gainprm[arm, 0], 1e-9), dtype=torch.float32, device=dev)
    env.x_elbow = torch.as_tensor([m.body('L_arm_l4').id, m.body('R_arm_l4').id], device=dev); env.x_shoulder = torch.as_tensor([m.body('L_arm_l1').id, m.body('R_arm_l1').id], device=dev)
    env.x_q1 = None; env.x_q2 = None; env.x_q3 = None
    torque_init(env)
    # contacts (Jiming terms) and the free-finger mask from the physics-demo labels
    cp = env.task/'contact_ref.npz'; env.x_cref = None
    if cp.exists():
        z = np.load(cp, allow_pickle=True); links = [str(x) for x in z['links']]; li = {n: i for i, n in enumerate(links)}
        T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)
        env.x_cref = {'mask': torch.as_tensor(z['mask'], device=dev), 'pts': T(z['pts']), 'u': T(z['u_req']), 'free': T(z['free_mask']), 'fing_acts': torch.as_tensor(z['fing_acts'], device=dev).long(),
                      'wn': T(z['w_req_norm']) if 'w_req_norm' in z.files else torch.ones(len(z['u_req']), 2, device=dev)}
        def subtree(root):
            s = {root}
            for b in range(m.nbody):
                p = b
                while p > 0:
                    if p in s: s.add(b); break
                    p = int(m.body_parentid[p])
            return s
        pip = subtree(m.body('right_object').id) | subtree(m.body('plunger_carrier').id); tube = subtree(m.body('left_object').id)
        gl = np.full(m.ngeom, -1); gs = np.full(m.ngeom, -1)
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g]); n = m.body(b).name
            if n in li: gl[g] = li[n]
            if b in pip: gs[g] = 0
            elif b in tube: gs[g] = 1
        env.x_gl, env.x_gs = torch.as_tensor(gl, device=dev), torch.as_tensor(gs, device=dev)
        env.x_lside = torch.as_tensor(z['side_of_link'], device=dev).long(); env.x_L = len(links)
        env.x_mass = torch.as_tensor([sum(m.body_mass[b] for b in pip), sum(m.body_mass[b] for b in tube)], dtype=torch.float32, device=dev)
        env.x_objb = torch.as_tensor([m.body('right_object').id, m.body('left_object').id], device=dev)
        tt = env.wp.to_torch; d = env.d
        env.x_cgeom, env.x_cworld = tt(d.contact.geom), tt(d.contact.worldid); env.x_cpos, env.x_cframe, env.x_cefc = tt(d.contact.pos), tt(d.contact.frame), tt(d.contact.efc_address)
        env.x_force = tt(d.efc.force); env.x_xipos = tt(d.xipos)
        temporal_init(env)


def torque_init(env):
    """(11-09) state of torque_rate_terms: a view of the actuator torques (nworld, nu), their per-control-step sum, the previous step's mean."""
    m = env.mjm; dev = env.device; N, nu = env.num_envs, env.nu
    env.x_trq = env.wp.to_torch(env.d.actuator_force); env.x_tsum = torch.zeros(N, nu, device=dev); env.x_tn = 0
    env.x_tprev = torch.zeros(N, nu, device=dev); env.x_tvalid = torch.zeros(N, dtype=torch.bool, device=dev)
    env.x_tmax = torch.as_tensor(np.abs(m.actuator_forcerange).max(1), dtype=torch.float32, device=dev).clamp_min(1e-6)
    env.x_fing = torch.as_tensor([i for i, n in enumerate(env.action_names) if n.startswith(('r_', 'l_'))], device=dev)


def substep_end(env):
    """Called after every physics substep (vega_res_env.step)."""
    if getattr(env, 'x_trq', None) is not None: env.x_tsum += env.x_trq; env.x_tn += 1


def torque_rate_terms(env):
    """(11-09, user: shivering of the wrists and of the fingers not in contact) Penalties (N,): torque_rate_arm / torque_rate_finger = mean over
    the arm / finger actuators of ((tau_t - tau_{t-1}) / (torque_rate_frac x torque limit))^2, tau_t the mean actuator torque over control step t
    (the substep average is an anti-alias low-pass: any chatter up to 25 Hz is seen, the PD ringing above it is not). Capped at torque_rate_cap.
    A steady grip or gravity load costs nothing (constant torque); the first step after a reset costs nothing."""
    X = env.xcfg; z = torch.zeros(env.num_envs, device=env.device)
    if getattr(env, 'x_trq', None) is None or env.x_tn == 0: return {'torque_rate_arm': z, 'torque_rate_finger': z}
    tq = env.x_tsum/env.x_tn; d = (tq-env.x_tprev)/(float(X['torque_rate_frac'])*env.x_tmax); cap = float(X['torque_rate_cap'])
    out = {'torque_rate_arm': (d[:, env.x_arm]**2).mean(1).clamp(max=cap), 'torque_rate_finger': (d[:, env.x_fing]**2).mean(1).clamp(max=cap)}
    out = {n_: torch.where(env.x_tvalid, torch.nan_to_num(t, 0., 0., 0.), z) for n_, t in out.items()}
    env.x_tprev = tq; env.x_tsum = torch.zeros_like(env.x_tsum); env.x_tn = 0; env.x_tvalid = torch.ones_like(env.x_tvalid)
    return out


def action_rate_term(env, r):
    """action_rate: old form mean((r - r_prev)^2) in rad^2 when action_rate_norm is None (configs before 11-09: numerically ~1e-5, i.e. off);
    else mean(((r - r_prev) / (scale x action_rate_norm))^2) on the normalised actions, capped at action_rate_cap, 0 on the first step after a reset."""
    an = env.xcfg.get('action_rate_norm')
    if not an: return RW.action_rate(r, env.prev)
    s = (env.scale*float(an)).clamp_min(1e-9)
    v = RW.action_rate(r/s, env.prev/s)
    if env.xcfg.get('action_rate_cap'): v = v.clamp(max=float(env.xcfg['action_rate_cap']))   # rare jumps (raw max ~1200) would dominate the variance
    return torch.where(env.episode_length_buf <= 1, torch.zeros_like(v), v)


def temporal_init(env):
    """(11-08) state for contact_temporal / margin_term, and the plunger slide (contacts(rest_plunger=True) undo its stroke)."""
    m = env.mjm; dev = env.device; N, L = env.num_envs, env.x_L
    assert 0. < env.xcfg['keep_hysteresis'] <= 1., 'keep_hysteresis is the release threshold as a fraction of the contact threshold'
    env.x_gb = torch.as_tensor(m.geom_bodyid, device=dev).long(); env.x_plb = -1
    try:
        b, j = m.body('pipette_plunger').id, m.joint('pipette_plunger').id
        env.x_plb, env.x_plq, env.x_plq0 = b, int(m.jnt_qposadr[j]), float(m.qpos0[m.jnt_qposadr[j]])
        env.x_plax = torch.as_tensor(m.jnt_axis[j], dtype=torch.float32, device=dev)
    except KeyError: pass
    env.x_kon = torch.zeros(N, L, dtype=torch.bool, device=dev); env.x_kpts = torch.zeros(N, L, 3, device=dev); env.x_kvalid = torch.zeros(N, dtype=torch.bool, device=dev)
    env.x_mdreq = torch.zeros(N, 2, device=dev); env.x_mdmean = torch.zeros(N, 2, device=dev); env.x_mstep = 0; env.x_mgate = None


def reset(env, ids):
    env.x_af[ids] = 0; env.x_prev_arm[ids] = env.ctrl[ids][:, env.x_arm]
    env.x_rlast[ids] = 0; env.x_rprev[ids] = 0; env.x_tick[ids] = env.ctrl[ids][:, env.x_arm]; env.x_tickq[ids] = env.x_tick[ids]
    if env.x_q1 is not None: env.x_q1[ids] = env.qpos[ids][:, env.aq]; env.x_q2[ids] = env.x_q1[ids]
    if getattr(env, 'x_q3', None) is not None: env.x_q3[ids] = env.x_q1[ids]
    if getattr(env, 'x_tvalid', None) is not None: env.x_tvalid[ids] = False   # (11-09) no previous torque right after a reset
    if getattr(env, 'x_kvalid', None) is not None:   # (11-08) no "previous step" right after a reset; stale margins cleared
        env.x_kvalid[ids] = False; env.x_kon[ids] = False; env.x_mdreq[ids] = 0; env.x_mdmean[ids] = 0


def smooth_residual(env, r):
    al = env.xcfg['ema_alpha']
    if al is None: return r
    env.x_af = al*r+(1-al)*env.x_af; return env.x_af


def begin_step(env, rapp, base):
    """Called once per control step before the physics substeps: the residual actually applied this step and the demo command row."""
    env.x_rprev = env.x_rlast; env.x_rlast = rapp; env.x_base = base


def arm_command(env, cmd):
    """Real-robot arm path (11-05): targets >= margin inside every arm joint limit, change <= speed_cap per second.
    cmd_hz None: updated every physics step. cmd_hz (e.g. 100): the real streamer -- target held for 1/cmd_hz s; the 50 Hz residual is
    upsampled linearly (previous -> current over the control step); the demo command is sampled at the tick (it already contains the demo's
    kv/kp qdot + tau_ff/kp); velocity feedforward added for the residual's slope only, capped at speed_cap."""
    X = env.xcfg
    if getattr(env, 'x_hold', None): cmd = cmd+env.x_hdelta
    if not X['real_cmd']: return cmd
    ia = env.x_arm; hz = X.get('cmd_hz')
    if not hz:
        ca = cmd[:, ia].clamp(env.x_alo, env.x_ahi); lim = X['speed_cap']*env.dt
        ca = env.x_prev_arm+(ca-env.x_prev_arm).clamp(-lim, lim); env.x_prev_arm = ca
        cmd = cmd.clone(); cmd[:, ia] = ca; return cmd
    hold = max(1, int(round(1./(hz*env.dt)))); s = env.x_sub
    if s % hold == 0:
        nt = max(1, env.substeps//hold); i = s//hold; dr = (env.x_rlast-env.x_rprev)[:, ia]
        q = cmd[:, ia]-dr*(1.-(i+1)/nt)
        # the demo command already carries its own feedforward (stage: ctrl = q + kv/kp qdot + tau_ff/kp), so sampling it at the tick IS the
        # streamed (q, qdot, tau) command; only the residual gets the velocity feedforward (its 50 Hz slope)
        v = dr/(env.substeps*env.dt)
        q = q.clamp(env.x_alo, env.x_ahi); lim = X['speed_cap']*hold*env.dt
        q = env.x_prev_arm+(q-env.x_prev_arm).clamp(-lim, lim); env.x_prev_arm = q; env.x_tickq = q
        v = v.clamp(-X['speed_cap'], X['speed_cap'])
        env.x_tick = q+env.x_ffr*v if X['vel_ff'] else q
        # (11-06 round 3) the sim actuator is position-only, so the velocity feed-forward lands in the position target: clamp AFTER adding it,
        # so the commanded target itself keeps >= margin_rad from every arm limit (round 1/2: clamp(q) + ff reached 5.24 deg < 5.73 deg)
        if X.get('clamp_after_ff'): env.x_tick = env.x_tick.clamp(env.x_alo, env.x_ahi)
    cmd = cmd.clone(); cmd[:, ia] = env.x_tick; return cmd


def contacts(env, alpha=None, rest_plunger=False):
    """Hand-link contacts on their own side's object with normal force > alpha (default force_alpha) x object weight: per link (N, L) mask and
    force-weighted mean contact point in the object frame, plus the raw contacts (normal, position, world, side).
    rest_plunger (11-08): points on the plunger are taken with the plunger stroke undone (on the plunger as if at rest), so pressing the plunger
    does not count as the thumb moving relative to the other fingers."""
    n = int(env.nacon.reshape(-1)[0]); N = env.num_envs; L = env.x_L; dev = env.device
    cur = torch.zeros(N, L, dtype=torch.bool, device=dev); pts = torch.zeros(N, L, 3, device=dev); wsum = torch.zeros(N, L, device=dev)
    empty = (cur, pts, (torch.zeros(0, 3, device=dev), torch.zeros(0, 3, device=dev), torch.zeros(0, dtype=torch.long, device=dev), torch.zeros(0, dtype=torch.long, device=dev)))
    if n == 0: return empty
    g = env.x_cgeom[:n].long(); w = env.x_cworld[:n].long(); f = env.x_force[w, env.x_cefc[:n, 0].long()]
    l0, l1 = env.x_gl[g[:, 0]], env.x_gl[g[:, 1]]; s0, s1 = env.x_gs[g[:, 0]], env.x_gs[g[:, 1]]
    link = torch.where(l0 >= 0, l0, l1); side = torch.where(l0 >= 0, s1, s0); sgn = torch.where(l0 >= 0, 1., -1.)
    ok = (link >= 0) & (side >= 0); ok = ok & (env.x_lside[link.clamp_min(0)] == side)
    thr = (env.xcfg['force_alpha'] if alpha is None else alpha)*env.x_mass*9.81; ok = ok & (f > thr[side.clamp_min(0)])
    if not ok.any(): return empty
    og = torch.where(l0 >= 0, g[:, 1], g[:, 0])[ok]                    # the object-side geom of each contact
    w, link, side, f, sgn = w[ok], link[ok], side[ok], f[ok], sgn[ok]
    pos = env.x_cpos[:n][ok]; nrm = env.x_cframe[:n][ok][:, 0]*sgn[:, None]
    cur.index_put_((w, link), torch.ones_like(w, dtype=torch.bool))
    ob = env.x_objb[side]; Rw = env.xmat[w, ob]; po = env.xpos[w, ob]; rel = pos-po
    if rest_plunger and getattr(env, 'x_plb', -1) >= 0:
        onp = (env.x_gb[og] == env.x_plb)[:, None]; ax = env.xmat[w, env.x_plb]@env.x_plax      # slide axis in the world (the slide does not rotate the body)
        rel = rel-torch.where(onp, ax*(env.qpos[w, env.x_plq]-env.x_plq0)[:, None], torch.zeros_like(rel))
    loc = (rel[:, None]@Rw)[:, 0]
    flat = w*L+link; pts.view(-1, 3).index_add_(0, flat, loc*f[:, None]); wsum.view(-1).index_add_(0, flat, f)
    return cur, pts/wsum.clamp_min(1e-9)[..., None], (nrm, pos, w, side)


def contact_temporal(env, cur, k=None):
    """(11-08, user) Reference-free contact stability between consecutive control steps, from the policy's own contacts (no demo labels).
    State per link: in contact when cur (force > force_alpha x weight), and stays in contact until the force drops below keep_hysteresis x that
    (so a light touch hovering at the threshold is not counted as on / off chatter). Penalties (N,), each the mean of the two hands, 0 on the first
    step after a reset:
      contact_keep  -- fraction of the previous step's contacts that are lost now (events: a deliberate release costs one step, chatter every flip);
      contact_drift -- mean |d_ij(t) - d_ij(t-1)| over pairs of links in contact at both steps (object-frame points, plunger stroke undone),
                       / drift_mm (1 = 1 mm per step), capped at 5: the grasp shape on the object should not change;
      contact_slip  -- mean |p_i(t) - p_i(t-1)| of each such link's point on the object / slip_mm, capped at 5: catches the whole grasp sliding
                       along the object (pairwise distances do not change then).
    A link that is not in contact costs nothing, so these never reward touching more; pair them with a term that does (tracking, wrench_margin).
    temporal_off_gates: names of reference.npz gates (frame k) during which all three are 0 (the state is still updated)."""
    X = env.xcfg; z = torch.zeros(env.num_envs, device=env.device)
    lo, pts, _ = contacts(env, alpha=X['force_alpha']*X['keep_hysteresis'], rest_plunger=True)
    prev, pprev, valid = env.x_kon, env.x_kpts, env.x_kvalid
    if X['temporal_off_gates'] and k is not None:
        if getattr(env, 'x_tgate', None) is None:
            ref = np.load(env.task/'reference.npz'); miss = [g_ for g_ in X['temporal_off_gates'] if g_ not in ref.files]
            assert not miss, f'temporal_off_gates not in reference.npz: {miss}'
            env.x_tgate = torch.as_tensor(np.any([ref[g_].astype(bool) for g_ in X['temporal_off_gates']], 0), device=env.device)
        valid = valid & ~env.x_tgate[k]
    on = cur | (prev & lo)
    lost, drift, slip = [], [], []
    for sd in (0, 1):
        m_ = env.x_lside == sd
        lost.append(RW.contact_lost(prev[:, m_], lo[:, m_]))
        drift.append(RW.contact_graph(pts[:, m_], on[:, m_], pprev[:, m_], prev[:, m_])[0])
        slip.append(RW.contact_slip(pts[:, m_], on[:, m_], pprev[:, m_], prev[:, m_]))
    out = {'contact_keep': (lost[0]+lost[1])/2, 'contact_drift': ((drift[0]+drift[1])/2*1e3/X['drift_mm']).clamp(max=5.),
           'contact_slip': ((slip[0]+slip[1])/2*1e3/X['slip_mm']).clamp(max=5.)}
    out = {n_: torch.where(valid, torch.nan_to_num(t, 0., 0., 0.), z) for n_, t in out.items()}
    env.x_kon, env.x_kpts, env.x_kvalid = on, pts, torch.ones_like(env.x_kvalid)
    return out


def margin_term(env, k, nrm, pos, cw, cside):
    """(11-08) Task-wrench margin penalty (N,) in [0, 1): mean over the hands of tanh(d_mean / margin_rho), d_mean = RW.wrench_margin's mean
    distance (N) of the required wrench +- margin_rho on each wrench axis from what the current contacts can supply (forces <= margin_fmax in
    friction cones). 0 = holds the required wrench (gravity + the reference object's inertia) with margin_rho to spare in every axis. Solved every
    margin_every control steps, held in between. Gated by the reference holding phase (object in hand) when margin_gate."""
    X = env.xcfg; N = env.num_envs; dev = env.device
    if env.x_mstep % max(1, int(X['margin_every'])) == 0:
        wq = env.x_cref['u'][k]*env.x_cref['wn'][k][..., None]           # (N, 2, 6) required wrench [m (a - g); I alpha / L] of the reference
        for sd in (0, 1):
            sel = cside == sd; ns = int(sel.sum())
            dr, dm = RW.wrench_margin(nrm[sel], pos[sel], env.x_xipos[:, env.x_objb[sd]], wq[:, sd], torch.full((ns,), float(X['margin_mu']), device=dev), cw[sel], N,
                                      X['torque_length'], float(X['margin_fmax']), float(X['margin_rho']), int(X['margin_iters']))
            env.x_mdreq[:, sd] = torch.nan_to_num(dr, 0., 0., 0.); env.x_mdmean[:, sd] = torch.nan_to_num(dm, 0., 0., 0.)
    env.x_mstep += 1
    if X['margin_gate']:
        if env.x_mgate is None:
            ref = np.load(env.task/'reference.npz'); F = len(env.x_cref['u'])
            env.x_mgate = torch.as_tensor(np.stack([ref['gate_pipette_held'] if 'gate_pipette_held' in ref.files else np.ones(F),
                                                    ref['gate_tube_held'] if 'gate_tube_held' in ref.files else np.ones(F)], 1).astype(np.float32), device=dev)
        gate = env.x_mgate[k]
    else: gate = torch.ones(N, 2, device=dev)
    pen = (gate*torch.tanh(env.x_mdmean/float(X['margin_rho']))).sum(1)/2
    return pen, {'_margin_req_pipette_N': env.x_mdreq[:, 0], '_margin_req_tube_N': env.x_mdreq[:, 1],
                 '_margin_mean_pipette_N': env.x_mdmean[:, 0], '_margin_mean_tube_N': env.x_mdmean[:, 1]}


def reward(env, r, k):
    """Extra reward (N,) and info. Positive kernels added, penalties subtracted (weights from the config)."""
    X = env.xcfg; Wt = X['w']; N = env.num_envs; dev = env.device; terms, pen = {}, {}
    temporal = any(Wt[n_] > 0 for n_ in ('contact_keep', 'contact_drift', 'contact_slip'))
    margin = Wt['wrench_margin'] > 0 or X['log_margin']
    if env.x_cref is not None and (Wt['contact_set'] > 0 or Wt['contact_graph'] > 0 or Wt['wrench_support'] > 0 or X['log'] or temporal or margin):
        cur, pts, (nrm, pos, cw, cside) = contacts(env); ref = env.x_cref['mask'][k]; pref = env.x_cref['pts'][k]
        if temporal or X['log']: pen.update(contact_temporal(env, cur, k))                     # (11-08) reference-free contact stability
        if margin: pen['wrench_margin'], mi_ = margin_term(env, k, nrm, pos, cw, cside); terms.update(mi_)
        miss, unint, err = [], [], []
        for sd in (0, 1):
            msk = env.x_lside == sd; mi, un = RW.contact_set(cur[:, msk], ref[:, msk]); miss.append(mi); unint.append(un)
            e, _ = RW.contact_graph(pts[:, msk], cur[:, msk], pref[:, msk], ref[:, msk]); err.append(e)
        pen['contact_set'] = (miss[0]+miss[1])/2+(unint[0]+unint[1]).clamp(max=6)/6
        terms['contact_graph'] = torch.exp(-((err[0]+err[1])/2*1e3)/X['sigma_graph_mm'])
        u = env.x_cref['u'][k]; sup = []
        for sd in (0, 1):
            sel = cside == sd; com = env.x_xipos[:, env.x_objb[sd]]
            sup.append(RW.cone_support(nrm[sel], pos[sel], com, u[:, sd], torch.full((int(sel.sum()),), 1.5, device=dev), cw[sel], N, X['torque_length']))
        terms['wrench_support'] = torch.tanh((sup[0]+sup[1])/2)
        terms['_missed_contact'] = (miss[0]+miss[1])/2
    qf = env.qpos[:, env.aq][:, env.x_cref['fing_acts']] if env.x_cref is not None else None
    if qf is not None:
        if env.x_q1 is None: env.x_q1 = env.qpos[:, env.aq].clone(); env.x_q2 = env.x_q1.clone()
        q1f, q2f = env.x_q1[:, env.x_cref['fing_acts']], env.x_q2[:, env.x_cref['fing_acts']]
        pf = RW.free_finger_acc(qf, q1f, q2f, env.x_cref['free'][k], env.frame_step*.01)
        pen['free_finger_acc'] = torch.where(env.episode_length_buf <= 2, torch.zeros_like(pf), pf)/1e4
        if getattr(env, 'x_q3', None) is None: env.x_q3 = env.x_q2.clone()
        q3f = env.x_q3[:, env.x_cref['fing_acts']]; jk = ((qf-3*q1f+3*q2f-q3f)**2).mean(1)/1e-4   # third difference per control step, normalised: 1 = 0.01 rad (0.57 deg) jitter
        pen['finger_jerk'] = torch.where(env.episode_length_buf <= 3, torch.zeros_like(jk), jk)
        if X['w'].get('finger_band', 0.) > 0:   # (11-06, round 2) causal ~3-10 Hz band (MA 2 steps - MA 8 steps at 50 Hz) of EVERY finger joint, latched ones included
            hb = getattr(env, 'x_fhist', None); qfb = qf
            if X.get('band_on_deviation'):   # (11-06 round 3) on q - q_ref: the plain-q band also charged the INTENDED motion (MA2 - MA8 ~ v x 0.06 s)
                qfb = qf-env.qpos_ref[k][:, env.aq][:, env.x_cref['fing_acts']]
            if hb is None or hb.shape[1] != qfb.shape[0]: hb = env.x_fhist = qfb[None].repeat(8, 1, 1)
            hb = env.x_fhist = torch.cat([hb[1:], qfb[None]], 0); bp = hb[-2:].mean(0)-hb.mean(0)
            pen['finger_band'] = torch.where(env.episode_length_buf <= 8, torch.zeros(qf.shape[0], device=qf.device), (bp**2).mean(1)/1e-4)   # 1 = 0.01 rad (0.57 deg) band amplitude
        env.x_q3 = env.x_q2.clone(); env.x_q2 = env.x_q1.clone(); env.x_q1 = env.qpos[:, env.aq].clone()
    if qf is not None:
        # (11-06) right thumb prior: squared deviation of the four right thumb joints from the reference at this frame (the reference carries the
        # natural start / relaxed end poses), and an arm posture prior (arm joints vs the reference), both mean squared rad
        qa_all = env.qpos[:, env.aq]; qr = env.q_ref_all[k] if hasattr(env, 'q_ref_all') else None
        if qr is None:
            env.q_ref_all = env.qpos_ref[:, env.aq]; qr = env.q_ref_all[k]
        if not hasattr(env, 'x_thumb_r'):
            env.x_thumb_r = torch.as_tensor([i for i, nm in enumerate(env.action_names) if nm.startswith('r_thumb')], device=dev)
        pen['thumb_prior'] = ((qa_all[:, env.x_thumb_r]-qr[:, env.x_thumb_r])**2).mean(1)
        pen['posture'] = ((qa_all[:, env.x_arm]-qr[:, env.x_arm])**2).mean(1)
    if Wt.get('finger_still', 0.) > 0:   # (11-06 round 3, user: fingers must not move when nothing calls for it; all 40 finger joints, both hands)
        if not hasattr(env, 'x_fdof'):
            mm = env.mjm; fa_ = [i for i in range(mm.nu) if mm.joint(int(mm.actuator_trnid[i, 0])).name.startswith(('l_', 'r_'))]
            env.x_fdof = torch.as_tensor([int(mm.jnt_dofadr[mm.actuator_trnid[i, 0]]) for i in fa_], device=dev)
        vr = env.qvel_ref[k][:, env.x_fdof]; v_ = env.qvel[:, env.x_fdof]; still = torch.exp(-(vr/.05)**2)   # 1 where the reference finger is still (|qdot_ref| << 0.05 rad/s)
        pen['finger_still'] = (still*(v_/.1)**2).sum(1)/still.sum(1).clamp_min(1.)                    # 1 = 0.1 rad/s (5.7 deg/s) RMS while the reference is still
    if Wt.get('pipette_seat', 0.) > 0:   # (11-06 round 3, teacher-only privileged) pipette pose vs its (holder-adjusted) reference from 1 s before it is back in the stand to 1.5 s after (finger opening)
        if not hasattr(env, 'x_pseat'):
            g_ = np.load(env.task/'reference.npz')['gate_pipette_in_stand'].astype(bool); st_ = np.flatnonzero(np.diff(np.r_[0, g_.astype(int)]) == 1)
            w_ = np.zeros(len(g_), bool)
            if len(st_) > 1: w_[max(0, st_[1]-100):min(len(g_), st_[1]+150)] = True
            env.x_pseat = torch.as_tensor(w_, device=dev)
        _, Op_ = env._poses('right'); Opr_ = ref_o(env, 'right', k)
        pe_ = (Op_[:, :3, 3]-Opr_[:, :3, 3]).norm(dim=-1); te_ = torch.arccos((Op_[:, :3, 2]*Opr_[:, :3, 2]).sum(-1).clamp(-1, 1))
        pen['pipette_seat'] = torch.where(env.x_pseat[k], ((pe_/.005)**2+(te_/.05)**2).clamp(max=50.), torch.zeros_like(pe_))   # 1 = 5 mm or 2.9 deg
    if Wt.get('arm_band', 0.) > 0:   # (11-06 round 3) causal ~3-10 Hz band (MA 2 - MA 8 control steps) of the 14 arm joints: damps DR-gain ringing (zeta down to 0.37 drawn)
        qa_b = env.qpos[:, env.aq][:, env.x_arm]-env.qpos_ref[k][:, env.aq][:, env.x_arm]; hb = getattr(env, 'x_ahist', None)   # deviation from the reference (smoke 11-06: on q it was ~53 from the intended arm speed alone)
        if hb is None or hb.shape[1] != qa_b.shape[0]: hb = qa_b[None].repeat(8, 1, 1)
        hb = env.x_ahist = torch.cat([hb[1:], qa_b[None]], 0); bpa = hb[-2:].mean(0)-hb.mean(0)
        pen['arm_band'] = torch.where(env.episode_length_buf <= 8, torch.zeros(qa_b.shape[0], device=dev), (bpa**2).mean(1)/4e-6)   # 1 = 0.002 rad (~1 mm at the wrist)
    if Wt.get('tube_tilt', 0.) > 0:   # (11-06 round 3) tube axis tilt in the LEFT hand vs the reference in-hand pose, while the reference holds it
        if not hasattr(env, 'x_tube_held'): env.x_tube_held = torch.as_tensor(np.load(env.task/'reference.npz')['gate_tube_held'].astype(bool), device=dev)
        Wl, Ol = env._poses('left'); loc = torch.linalg.inv(Wl)@Ol; rz = env.rel_ref['left'][k][:, :3, 2]
        tilt = torch.arccos((loc[:, :3, 2]*rz).sum(-1).clamp(-1, 1))   # rad; seat misses were ~8 deg in hand -> jam ~5 mm high
        pen['tube_tilt'] = torch.where(env.x_tube_held[k], (tilt/.05)**2, torch.zeros_like(tilt))   # 1 = 0.05 rad (2.9 deg)
        terms['_tube_tilt_deg'] = torch.rad2deg(tilt)
    if getattr(env, 'x_tip', None) is not None:   # (11-12) pipette tip / axis in the TUBE frame vs the reference's, gated approach .. extraction end
        rel, ax = tip_state(env); T = env.x_tip; c = T['cfg']; g = T['gate'][k]; e = rel-T['ref'][k]; lat = e[:, :2].norm(dim=-1)
        dd = torch.sqrt(lat**2+(.5*e[:, 2])**2); ang = torch.arccos((ax*T['ax'][k]).sum(-1).clamp(-1, 1))
        kern = (.5*torch.exp(-(dd/(c['sigma_coarse_mm']*1e-3))**2)+.5*torch.exp(-(dd/(c['sigma_fine_mm']*1e-3))**2))*torch.exp(-(ang/math.radians(c['sigma_ang_deg']))**2)
        terms['tip_align'] = torch.where(g, kern, torch.zeros_like(kern))
        terms['_tip_lat_mm_gated'] = torch.where(g, lat*1e3, torch.zeros_like(lat)); terms['_tip_gate'] = g.float()
        inb = (rel[:, :2].norm(dim=-1) < .0129-.0018) & (rel[:, 2] < .0638-.005); terms['_tip_in_bore_ref_in'] = (T['inref'][k] & inb).float()
    pen['action_rate'] = action_rate_term(env, r)
    pen.update(torque_rate_terms(env))                                                       # (11-09)
    pen['elbow_height'] = RW.elbow_above_shoulder(env.xpos[:, env.x_elbow, 2], env.xpos[:, env.x_shoulder, 2], 0.)
    qa = env.qpos[:, env.aq][:, env.x_arm]; pen['joint_margin'] = RW.joint_margin(qa, env.x_alo-X['margin_rad'], env.x_ahi+X['margin_rad'], X['margin_rad'])
    extra = sum(float(Wt[n])*t for n, t in terms.items() if n in Wt)-sum(float(Wt[n])*t for n, t in pen.items() if n in Wt)
    info = {**{f't_{n}': t.detach() for n, t in terms.items()}, **{f'p_{n}': t.detach() for n, t in pen.items()}}
    return torch.nan_to_num(extra, 0., 0., 0.), info


# ---------------------------------------------------------------- (11-12) pipette tip vs the tube
# The task reference (tracker/build_demo_task.py tip_insert) carries the tip's own path in the TUBE frame: tip_in_tube_ref, pip_axis_in_tube_ref,
# pip_tip_local (the tip in the pipette frame), gate_tip_align (approach .. extraction end), gate_tip_in_tube. Tracking each wrist / object in the
# world frame does not put the tip in the 25.8 mm bore: in physics the two hands are ~10 mm apart from their references relative to each other and
# the tube sits 3-18 mm off its in-hand template. The reward w.tip_align scores the tip / axis in the tube frame directly; extras.tip.obs gives
# the (privileged) teacher the tip / axis in the tube frame and the error to the reference at the OFFSETS frames.
def tip_init(env):
    X = env.xcfg; env.x_tip = None
    if not (X['tip'].get('obs') or float(X['w'].get('tip_align', 0.)) > 0): return
    r = np.load(env.task/'reference.npz'); assert 'tip_in_tube_ref' in r.files, f'{env.task}: no tip_in_tube_ref (task built before 11-12)'
    tt = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=env.device)
    env.x_tip = {'ref': tt(r['tip_in_tube_ref']), 'ax': tt(r['pip_axis_in_tube_ref']), 'pt': tt(r['pip_tip_local']), 'gate': tt(r['gate_tip_align'], torch.bool),
                 'inref': tt(r['gate_tip_in_tube'], torch.bool), 'cfg': X['tip']}


def tip_state(env, pt=None):
    """(N, 3) pipette tip and (N, 3) pipette axis, both in the tube frame"""
    _, Op = env._poses('right'); _, Ou = env._poses('left'); pt = env.x_tip['pt'] if pt is None else pt
    tip = Op[:, :3, :3]@pt+Op[:, :3, 3]
    return ((tip-Ou[:, :3, 3])[:, None, :]@Ou[:, :3, :3])[:, 0], (Op[:, :3, 2][:, None, :]@Ou[:, :3, :3])[:, 0]


def obs_extra(env, k, offsets):
    """(11-12) teacher observation (extras.tip.obs): tip / axis in the tube frame, then per offset frame the reference tip minus the tip (scaled,
    clipped) and the alignment gate; 6 + 4 x len(offsets) values"""
    T = getattr(env, 'x_tip', None)
    if T is None or not T['cfg'].get('obs'): return []
    rel, ax = tip_state(env); s = float(T['cfg']['obs_scale_mm'])*1e-3; cl = float(T['cfg']['obs_clip']); last = env.nframes-1; out = [(rel/s).clamp(-cl, cl), ax]
    for o in offsets:
        j = (k+o*env.frame_step).clamp(max=last); out += [((T['ref'][j]-rel)/s).clamp(-cl, cl), T['gate'][j, None].float()]
    return out


# ---------------------------------------------------------------- (11-12) external pushes on the hands (user: "external perturbation to the hand
# during the training"). extras 'perturb': per world and per body (default both wrists) a push starts with probability rate_hz x dt per control step
# (none active, start_delay_s after the reset); force magnitude U(force_n) in a uniformly random direction, torque U(torque_nm) about a random
# axis, duration U(dur_s), half-sine envelope; applied as d.xfrc_applied (world frame, at the body's COM), constant over the substeps of a control
# step. 'scale' multiplies force and torque (training scripts ramp it: tracker.perturb_ramp_steps / distill perturb_ramp_iters). Evaluation
# scripts drop extras.perturb unless asked (--perturb).
PERTURB = {'rate_hz': .2, 'force_n': [2., 15.], 'torque_nm': [0., .5], 'dur_s': [.1, .4], 'bodies': ['right_wrist', 'left_wrist'], 'start_delay_s': .5, 'scale': 1.}


def perturb_init(env):
    pc = env.xcfg.get('perturb'); env.x_pert = None
    if not pc: return
    pc = merge(PERTURB, pc); dev = env.device; N = env.num_envs; ids = torch.as_tensor([env.mjm.body(b).id for b in pc['bodies']], device=dev); B = len(ids)
    xf = env.wp.to_torch(env.d.xfrc_applied); assert tuple(xf.shape) == (N, env.mjm.nbody, 6), xf.shape
    z = lambda *s: torch.zeros(*s, device=dev)
    env.x_pert = {'cfg': pc, 'ids': ids, 'xfrc': xf, 'W': z(N, B, 6), 't': z(N, B), 'T': z(N, B), 'scale': float(pc['scale']), 'pushes': 0, 'steps': 0, 'active': 0.}


def perturb_reset(env, ids):
    P = getattr(env, 'x_pert', None)
    if P is None or not len(ids): return
    P['t'][ids] = 0.; P['T'][ids] = 0.; P['W'][ids] = 0.; P['xfrc'][ids[:, None], P['ids'][None, :]] = 0.


def perturb_step(env):
    P = getattr(env, 'x_pert', None)
    if P is None: return
    c = P['cfg']; g = env.gen; N, B = P['t'].shape; dt = env.substeps*env.dt; dev = env.device
    idle = P['t'] >= P['T']
    start = idle & (env.episode_length_buf[:, None].float()*dt >= float(c['start_delay_s'])) & (torch.rand(N, B, device=dev, generator=g) < float(c['rate_hz'])*dt)
    n = int(start.sum())
    if n:
        u = lambda lo_hi: float(lo_hi[0])+(float(lo_hi[1])-float(lo_hi[0]))*torch.rand(n, device=dev, generator=g)
        df = torch.randn(n, 3, device=dev, generator=g); df = df/df.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        dq = torch.randn(n, 3, device=dev, generator=g); dq = dq/dq.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        P['W'][start] = torch.cat([df*u(c['force_n'])[:, None], dq*u(c['torque_nm'])[:, None]], 1); P['T'][start] = u(c['dur_s']); P['t'][start] = 0.
        P['pushes'] += n
    act = P['t'] < P['T']; env_ = torch.where(act, torch.sin(math.pi*(P['t']/P['T'].clamp_min(1e-6)).clamp(0, 1)), torch.zeros_like(P['t']))
    P['xfrc'][:, P['ids']] = P['W']*(env_*P['scale'])[..., None]
    P['t'] = torch.where(act, P['t']+dt, P['t']); P['steps'] += 1; P['active'] += float(act.float().mean())


def perturb_stats(env):
    """mean fraction of (world, body) pairs being pushed, pushes started per world-second, since the last call"""
    P = getattr(env, 'x_pert', None)
    if P is None or not P['steps']: return {}
    out = {'push_active_frac': round(P['active']/P['steps'], 4), 'pushes_per_world_s': round(P['pushes']/(P['steps']*env.num_envs*env.substeps*env.dt), 4), 'push_scale': P['scale']}
    P['pushes'] = 0; P['steps'] = 0; P['active'] = 0.
    return out


# ---------------------------------------------------------------- holder offsets with object-relative re-anchoring (rack / stand moved per episode)
HOLDER = {'stand': {'side': 'right', 'gate': 'gate_pipette_in_stand', 'geoms': ('furn_stand_',), 'obj': 'right_object'},
          'rack': {'side': 'left', 'gate': 'gate_tube_in_rack', 'geoms': ('furn_rackv_', 'furn_rack_plate_'), 'obj': 'left_object'}}


class _IK:
    """ArmIK of the shared layer for a scene with the Lift baked in and DAID wrist names (no code change in arm_ik.py)."""
    def __new__(cls, m, device):
        import arm_ik as AK
        self = object.__new__(AK.ArmIK); self.device, self.dtype = device, torch.float32
        import mujoco
        d = mujoco.MjData(m); d.qpos[:] = m.qpos0; mujoco.mj_kinematics(m, d); ac = m.body(AK.ARM_ROOT).id
        T0 = np.eye(4); T0[:3, :3] = d.xmat[ac].reshape(3, 3); T0[:3, 3] = d.xpos[ac]; self.T_root = torch.as_tensor(T0, dtype=torch.float32, device=device)
        old = dict(AK.WRIST); AK.WRIST.clear(); AK.WRIST.update({'L': 'left_wrist', 'R': 'right_wrist'})
        try:
            self.chain, self.qadr, self.lo, self.hi, self.jnames = {}, {}, {}, {}, {}
            for s, wn in AK.WRIST.items():
                b = m.body(wn).id; bodies = []
                while b != ac: bodies.append(b); b = int(m.body_parentid[b])
                ch, qa, lo, hi, jn = [], [], [], [], []
                for b in bodies[::-1]:
                    pos = torch.as_tensor(m.body_pos[b], dtype=torch.float32, device=device); Rb = AK._qmat(torch.as_tensor(m.body_quat[b], dtype=torch.float32, device=device))
                    js = list(range(m.body_jntadr[b], m.body_jntadr[b]+m.body_jntnum[b])) if m.body_jntnum[b] > 0 else []
                    if js:
                        j = js[0]; ch.append((pos, Rb, torch.as_tensor(m.jnt_pos[j], dtype=torch.float32, device=device), torch.as_tensor(m.jnt_axis[j], dtype=torch.float32, device=device), len(qa)))
                        qa.append(int(m.jnt_qposadr[j])); lo.append(m.jnt_range[j, 0]); hi.append(m.jnt_range[j, 1]); jn.append(m.joint(j).name)
                    else: ch.append((pos, Rb, None, None, None))
                assert len(qa) == 7, (s, jn)
                self.chain[s] = ch; self.qadr[s] = qa; self.jnames[s] = jn
                self.lo[s] = torch.as_tensor(lo, dtype=torch.float32, device=device); self.hi[s] = torch.as_tensor(hi, dtype=torch.float32, device=device)
        finally:
            AK.WRIST.clear(); AK.WRIST.update(old)
        return self


def _rz(th):
    c, s = torch.cos(th), torch.sin(th); R = torch.zeros(th.shape[0], 3, 3, device=th.device); R[:, 0, 0] = c; R[:, 0, 1] = -s; R[:, 1, 0] = s; R[:, 1, 1] = c; R[:, 2, 2] = 1; return R


def _H(w, d, th, c):
    """Weighted holder transform (N, 4, 4): rotate by w*th about the vertical through c, then translate by w*d."""
    N = w.shape[0]; R = _rz(w*th); T = torch.zeros(N, 4, 4, device=w.device); T[:, :3, :3] = R
    T[:, :3, 3] = c[None]-(R@c[None, :, None])[..., 0]; T[:, :2, 3] += w[:, None]*d; T[:, 3, 3] = 1; return T


def _qmul(a, b):
    w1, x1, y1, z1 = a.unbind(-1); w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], -1)


def _qmat(q):   # wxyz (..., 4) -> rotation matrix (..., 3, 3)
    q = q/q.norm(dim=-1, keepdim=True); w, x, y, z = q.unbind(-1)
    return torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y), 2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x), 2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1).reshape(*q.shape[:-1], 3, 3)


def holder_init(env):
    hc = env.xcfg.get('holder'); env.x_hold = None
    if not hc: return
    m = env.mjm; dev = env.device; r = np.load(env.task/'reference.npz'); nf = env.nframes; bf = max(1, int(round(float(hc.get('blend_s', 1.5))/.01)))
    import mujoco
    d = mujoco.MjData(m); d.qpos[:] = r['qpos_ref'][0]; mujoco.mj_kinematics(m, d); H = {}
    for name, h in HOLDER.items():
        g = [i for i in range(m.ngeom) if m.geom_bodyid[i] == 0 and (m.geom(i).name or '').startswith(h['geoms'])]
        c = torch.as_tensor(d.geom_xpos[g].mean(0), dtype=torch.float32, device=dev); c[2] = 0.
        gate = r[h['gate']].astype(bool); fr = np.arange(nf)
        # (11-06 fix) the holder transform must be complete while the object is still ENGAGED with the holder, not only on the in-holder gate:
        # the old weight ramped during the extraction / the descent (rack w 0.65-0.91 while the tube went 143 -> 92 mm into the new hole), so a
        # moved hole (or holder) was reached with a sideways slide inside the hole. Engaged = gate, or the object within its insertion depth
        # (+ 15 mm) of its seated height of that segment; the 1.5 s ramp now happens above the holder.
        if hc.get('engage_fix', True):
            oq = int(m.jnt_qposadr[m.joint(h['obj']+'_joint').id]); z = r['qpos_ref'][:, oq+2]; xy = r['qpos_ref'][:, oq:oq+2]; gi = np.flatnonzero(gate); gap_ = np.flatnonzero(np.diff(gi) > 1)
            segs = [gi[:gap_[0]+1], gi[gap_[0]+1:]] if len(gap_) else [gi]; kmid_ = (gi[gap_[0]]+gi[gap_[0]+1])//2 if len(gap_) else nf
            depth = {'stand': .1061, 'rack': .0533}[name]+.015; eng = gate.copy()
            for si, sg_ in enumerate(segs):
                if not len(sg_): continue
                zr = float(np.median(z[sg_])); xyr = np.median(xy[sg_], 0); part = (fr < kmid_) if si == 0 else (fr >= kmid_)
                eng |= part & (z-zr < depth) & (np.linalg.norm(xy-xyr, axis=1) < .015)   # above its own hole (xy within 15 mm) and within the insertion depth
            gate = eng
        idx = np.flatnonzero(gate)
        dist = np.min(np.abs(fr[:, None]-idx[None]), 1) if len(idx) else np.full(nf, 10**6)
        w = np.clip(1-dist/bf, 0, 1)
        H[name] = {'g': torch.as_tensor(g, device=dev), 'c': c, 'w': torch.as_tensor(w, dtype=torch.float32, device=dev),
                   'xy': float(hc.get(name+'_xy_mm', 0.))*1e-3, 'yaw': np.radians(float(hc.get(name+'_yaw_deg', 0.))),
                   'd': torch.zeros(env.num_envs, 2, device=dev), 'th': torch.zeros(env.num_envs, device=dev), 'obj_q': int(m.jnt_qposadr[m.joint(h['obj']+'_joint').id]),
                   'dm': torch.zeros(env.num_envs, 2, device=dev), 'thm': torch.zeros(env.num_envs, device=dev),
                   'm_xy': float(hc.get('meas_xy_mm', 0.))*1e-3, 'm_yaw': np.radians(float(hc.get('meas_yaw_deg', 0.)))}
    # (11-06) hole pairs: per episode the tube is picked from / placed into another rack hole (holes_lib PAIRS); the rack-tied reference
    # (objects, left wrist) is translated by the hole offset (pick offset before the out-of-rack gap's midpoint, place offset after),
    # composed BEFORE the rack placement transform (offsets are in the rack's own, unrotated frame); the rack geometry does not move for it
    hp = hc.get('hole_pairs'); env.x_holes = None
    if hp:
        prof = json.loads((Path(__file__).resolve().parents[1]/'common/rack_visual/profile.json').read_text()); cen = {k: np.array(f['lower']['centre_xy_m']) for k, f in prof['fits'].items()}
        if hc.get('hole_centres', 'scene') == 'scene':
            # (11-06 fix) profile.json is in the rack_visual source frame, which is yawed ~190 deg vs this scene's rack (profile r0c2-r0c3 =
            # (+35.9, +6.4) mm, scene = (-36.4, 0)): every pick / place offset pointed at the mirrored neighbour. Use this scene's own bore
            # rings (mean of the furn_rackv_low_<hole>_s* geoms, world frame at the nominal rack pose = the rack's unrotated frame here).
            import re as _re; acc = {}
            for i in range(m.ngeom):
                mm = _re.match(r'furn_rackv_low_(r\d+c\d+)_s\d+$', m.geom(i).name or '')
                if mm: acc.setdefault(mm.group(1), []).append(d.geom_xpos[i, :2].copy())
            cen = {k: np.mean(v, 0) for k, v in acc.items()}
        pairs = [tuple(x) for x in hp]
        g = r['gate_tube_in_rack'].astype(bool); idx = np.flatnonzero(g); gap = np.flatnonzero(np.diff(idx) > 1)
        k_mid = int((idx[gap[0]]+idx[gap[0]+1])//2) if len(gap) else nf//2
        # (11-10) the reference's own pick / place holes: task.json 'ref_holes' (tracker/build_demo_task.py tasks: the same hole twice);
        # tasks without it are the T1 demo's r0c3 -> r1c3
        tjp = env.task/'task.json'; rh = (json.loads(tjp.read_text()).get('ref_holes') if tjp.exists() else None) or ['r0c3', 'r1c3']
        env.x_holes = {'pairs': pairs, 'pick': torch.as_tensor(np.array([cen[a]-cen[rh[0]] for a, b in pairs]), dtype=torch.float32, device=dev),
                       'place': torch.as_tensor(np.array([cen[b]-cen[rh[1]] for a, b in pairs]), dtype=torch.float32, device=dev), 'k_mid': k_mid, 'ref_holes': rh,
                       'idx': torch.zeros(env.num_envs, dtype=torch.long, device=dev)}
    env.x_hold = H; env.x_ik = _IK(m, dev)
    # (11-11) extras.holder.meas_model: the robot's holder / object estimate = the head-camera estimator's error model instead of independent
    # Gaussians: {"errors": "ego/errors.npz" (task-relative; perception/ego_holders.py: one JOINT row per characterised scene: stand dx dy dyaw,
    # rack dx dy dyaw, tube dx dy dz), "scale": 1 (multiplies the row), "extra_xy_mm" / "extra_yaw_deg" / "extra_tube_mm": independent Gaussian
    # added on top (randomisation margin), "pipette_xy_mm" / "pipette_yaw_deg": the pipette's slack in the stand (not observed)}. Per episode one
    # row j for the stand, the rack and the tube together; the pipette is the stand estimate applied to it; the tube's spin is not observed (0).
    mm = hc.get('meas_model'); env.x_ego = None
    if mm:
        ep = Path(mm['errors']); ep = ep if ep.is_absolute() else env.task/ep; z = np.load(ep)
        assert list(z['cols']) == ['stand_dx', 'stand_dy', 'stand_dyaw', 'rack_dx', 'rack_dy', 'rack_dyaw', 'tube_dx', 'tube_dy', 'tube_dz'], z['cols']
        env.x_ego = {'E': torch.as_tensor(z['errors'], dtype=torch.float32, device=dev)*float(mm.get('scale', 1.)), 'j': torch.zeros(env.num_envs, dtype=torch.long, device=dev),
                     'xy': float(mm.get('extra_xy_mm', 0.))*1e-3, 'yaw': np.radians(float(mm.get('extra_yaw_deg', 0.))), 'tube': float(mm.get('extra_tube_mm', 0.))*1e-3,
                     'pxy': float(mm.get('pipette_xy_mm', 0.))*1e-3, 'pyaw': np.radians(float(mm.get('pipette_yaw_deg', 0.)))}
    # (11-12) extras.student.scan = {"bank": policy_bank.npz (task-relative or absolute; perception/train_scan_net.py), "obs": "embedding" |
    # "estimate"}: the "initial head-depth image only" student. Per episode one bank row = one rendered start scene: its TRUE holder poses,
    # pick hole and object slack become the episode's scene; the scan network's PREDICTION (from that scene's noisy head-camera depth) is the
    # robot's holder estimate (re-anchors the arm command, as the hand-written estimate did). obs "embedding": the actor sees the scan
    # network's 64-d embedding INSTEAD of the measured poses / re-anchoring delta / re-anchored wrist targets; "estimate": the old inputs,
    # computed from the network's prediction. Rows are limited to pick holes of the config's hole_pairs.
    sc_ = (env.xcfg.get('student') or {}).get('scan'); env.x_scan = None
    if sc_:
        bp = Path(sc_['bank']); bp = bp if bp.is_absolute() else env.task/bp; zb = np.load(bp); hn = [str(x) for x in zb['holes']]
        pairs = env.x_holes['pairs'] if env.x_holes is not None else None
        pair_of = {a: i for i, (a, b) in reversed(list(enumerate(pairs)))} if pairs else {}
        ok = np.array([hn[h] in pair_of for h in zb['hole']]) if pairs else np.ones(len(zb['hole']), bool)
        Tb = lambda k: torch.as_tensor(np.asarray(zb[k])[ok], dtype=torch.float32, device=dev)
        env.x_scan = {'stand': Tb('stand'), 'rack': Tb('rack'), 'pstand': Tb('pred_stand'), 'prack': Tb('pred_rack'), 'ptube': Tb('pred_tube'),
                      'emb': Tb('emb'), 'pip_slack': Tb('pip_slack'), 'tube_slack': Tb('tube_slack'), 'n': int(ok.sum()), 'obs': sc_.get('obs', 'embedding'),
                      'pair': torch.as_tensor([pair_of.get(hn[h], 0) for h in zb['hole'][ok]], dtype=torch.long, device=dev),
                      'j': torch.zeros(env.num_envs, dtype=torch.long, device=dev)}
        assert env.x_scan['n'] > 0, 'scan bank has no rows for these hole pairs'
    env.x_arm_side = {'L': torch.as_tensor([i for i, n in enumerate(env.action_names) if n.startswith('L_arm')], device=dev),
                      'R': torch.as_tensor([i for i, n in enumerate(env.action_names) if n.startswith('R_arm')], device=dev)}
    env.x_carrier_q = torch.as_tensor([int(m.jnt_qposadr[m.joint(j).id]) for j in ('pc_x', 'pc_y', 'pc_z', 'pc_rx', 'pc_ry', 'pc_rz')], device=dev)
    env.x_carrier_b = m.body('plunger_carrier').id
    env.x_hdelta = torch.zeros(env.num_envs, env.nu, device=dev)
    env.g0 = {k: env.mf[k][0].clone() for k in ('geom_pos', 'geom_quat')}


def holder_T(env, name, k, ids=None, meas=False):
    """meas=True: the robot's estimate of the holder offset (true offset + per-episode measurement error; drives the command re-anchoring).
    Rack + hole pairs: H_rack(w) . Trans(w * hole offset) (pick offset before the out-of-rack midpoint, place offset after; known exactly)."""
    h = env.x_hold[name]; sl = slice(None) if ids is None else ids
    T = _H(h['w'][k], h['dm' if meas else 'd'][sl], h['thm' if meas else 'th'][sl], h['c'])
    X = getattr(env, 'x_holes', None)
    if name == 'rack' and X is not None:
        kk = k if torch.is_tensor(k) else torch.as_tensor(k, device=T.device)
        pi = X['idx'][sl]; off = torch.where((kk < X['k_mid'])[..., None], X['pick'][pi], X['place'][pi])
        S = torch.eye(4, device=T.device).repeat(T.shape[0], 1, 1); S[:, :2, 3] = h['w'][k][:, None]*off if h['w'][k].dim() else h['w'][k]*off
        T = T@S
    return T


def ref_w(env, s, k):
    T = env.W_ref[s][k]
    if not getattr(env, 'x_hold', None): return T
    return holder_T(env, 'stand' if s == 'right' else 'rack', k)@T


def ref_o(env, s, k):
    T = env.O_ref[s][k]
    if not getattr(env, 'x_hold', None): return T
    return holder_T(env, 'stand' if s == 'right' else 'rack', k)@T


def _arm_delta(env, q_seed, Hs):
    """IK correction of both arms: target = H . FK(q_seed); returns Δq (N, nu) on the arm columns."""
    out = torch.zeros(q_seed.shape[0], env.nu, device=env.device)
    for side, name in (('R', 'stand'), ('L', 'rack')):
        cols = env.x_arm_side[side]; qs = q_seed[:, cols]; T = env.x_ik.fk(side, qs); Tt = Hs[name]@T
        q, _, _ = env.x_ik.solve(side, Tt, qs, qs, iters=4, margin=0.); out[:, cols] = q-qs
    return out


def holder_step(env, k):
    """Per control step: the IK correction of the demo's arm command for the moved holders (applied in arm_command)."""
    if not getattr(env, 'x_hold', None): return
    row = (env.frame*env.per_frame).clamp(max=len(env.cmd_sim)-1); q_seed = env.cmd_sim[row]
    Hs = {n: holder_T(env, n, k, meas=True) for n in env.x_hold}
    active = (env.x_hold['stand']['w'][k] > 0) | (env.x_hold['rack']['w'][k] > 0)
    env.x_hdelta.zero_()
    if active.any():
        dq = _arm_delta(env, q_seed[active], {n: T[active] for n, T in Hs.items()}); env.x_hdelta[active] = dq


def holder_reset(env, ids, frames):
    """Sample per-episode holder offsets, move the holder geoms, and make the start state consistent (arms by IK, held / seated objects rigidly)."""
    if not getattr(env, 'x_hold', None) or not len(ids): return
    n = len(ids); dev = env.device; gpos, gquat = env.mf['geom_pos'], env.mf['geom_quat']; XE = getattr(env, 'x_ego', None)
    if XE is not None: XE['j'][ids] = torch.randint(len(XE['E']), (n,), device=dev, generator=env.gen)
    XS = getattr(env, 'x_scan', None)
    if XS is not None:
        XS['j'][ids] = torch.randint(XS['n'], (n,), device=dev, generator=env.gen); js = XS['j'][ids]
    for name, h in env.x_hold.items():
        h['d'][ids] = h['xy']*(2*torch.rand(n, 2, device=dev, generator=env.gen)-1); h['th'][ids] = h['yaw']*(2*torch.rand(n, device=dev, generator=env.gen)-1)
        if name == 'rack' and getattr(env, 'x_holes', None) is not None:
            X = env.x_holes; X['idx'][ids] = torch.randint(len(X['pairs']), (n,), device=dev, generator=env.gen)
            fp = getattr(env, 'x_hole_fixed', None)
            if fp is not None: X['idx'][ids] = fp[ids]
        if XS is not None:   # (11-12) the bank row's scene
            h['d'][ids] = XS[name][js, :2]; h['th'][ids] = XS[name][js, 2]
            if name == 'rack' and getattr(env, 'x_holes', None) is not None: env.x_holes['idx'][ids] = XS['pair'][js]
        fx = getattr(env, 'x_hold_fixed', None)   # diagnostics only: per-world fixed offsets {name: (d (N, 2) m, th (N,) rad)}
        if fx and name in fx: h['d'][ids] = fx[name][0][ids]; h['th'][ids] = fx[name][1][ids]
        if XS is not None:   # (11-12) the scan network's prediction for this row's scene
            h['dm'][ids] = XS['p'+name][js, :2]; h['thm'][ids] = XS['p'+name][js, 2]
        elif XE is not None:   # (11-11) head-camera estimate: a row of the estimator's error model (+ the extra randomisation)
            e = XE['E'][XE['j'][ids]][:, (0 if name == 'stand' else 3):][:, :3]
            h['dm'][ids] = h['d'][ids]+e[:, :2]+XE['xy']*torch.randn(n, 2, device=dev, generator=env.gen); h['thm'][ids] = h['th'][ids]+e[:, 2]+XE['yaw']*torch.randn(n, device=dev, generator=env.gen)
        else:
            h['dm'][ids] = h['d'][ids]+h['m_xy']*torch.randn(n, 2, device=dev, generator=env.gen); h['thm'][ids] = h['th'][ids]+h['m_yaw']*torch.randn(n, device=dev, generator=env.gen)
        Hfull = _H(torch.ones(n, device=dev), h['d'][ids], h['th'][ids], h['c'])
        g = h['g']; p0 = env.g0['geom_pos'][g]; q0 = env.g0['geom_quat'][g]
        gpos[ids[:, None], g[None]] = (Hfull[:, None, :3, :3]@p0[None, :, :, None])[..., 0]+Hfull[:, None, :3, 3]
        qz = torch.stack([torch.cos(h['th'][ids]/2), torch.zeros(n, device=dev), torch.zeros(n, device=dev), torch.sin(h['th'][ids]/2)], -1)
        gquat[ids[:, None], g[None]] = _qmul(qz[:, None].expand(-1, len(g), -1), q0[None].expand(n, -1, -1))
        # MJWarp never recomputes the world body's (static) geom poses in kinematics (checked 11-05: per-world geom_pos changed, geom_xpos
        # stayed): write the data-side poses too; they persist because nothing overwrites them
        gx, gm = env.wp.to_torch(env.d.geom_xpos), env.wp.to_torch(env.d.geom_xmat)
        gx[ids[:, None], g[None]] = gpos[ids[:, None], g[None]]
        gm[ids[:, None], g[None]] = _qmat(gquat[ids[:, None], g[None]]).reshape(gm[ids[:, None], g[None]].shape)
    if XS is not None:   # (11-12) objects still in their holder: the bank row's slack (the scan shows it) instead of the DR draw
        qr = env.qpos_ref[frames]
        for sd, key in (('right', 'pip_slack'), ('left', 'tube_slack')):
            sel = frames < env.f_pre[sd]
            if not bool(sel.any()): continue
            w = ids[sel]; a = env.obj_q[sd]; sl = XS[key][js[sel]]
            env.qpos[w, a:a+2] = qr[sel, a:a+2]+sl[:, :2]; env.qpos[w, a+2] = qr[sel, a+2]
            qz = torch.stack([torch.cos(sl[:, 2]/2), torch.zeros_like(sl[:, 2]), torch.zeros_like(sl[:, 2]), torch.sin(sl[:, 2]/2)], -1)
            env.qpos[w, a+3:a+7] = _qmul(qz, qr[sel, a+3:a+7])
    # consistent start: arms by IK toward the moved targets, objects rigidly with their holder / hand, carrier with the pipette
    Hs = {nm: holder_T(env, nm, frames, ids) for nm in env.x_hold}
    q = env.qpos[ids]; qa = env.qpos_ref[frames][:, env.aq]
    dq = _arm_delta(env, qa, Hs); q[:, env.aq[env.x_arm]] += dq[:, env.x_arm]
    # the command path starts from the re-anchored command too (else the rate limit drags the arm back toward the nominal command for ~0.1 s)
    Hm = {nm: holder_T(env, nm, frames, ids, meas=True) for nm in env.x_hold}
    row = (frames*env.per_frame).clamp(max=len(env.cmd_sim)-1); c = env.cmd_sim[row]; dqc = _arm_delta(env, c, Hm)
    env.x_hdelta[ids] = dqc; ca = (c+dqc)[:, env.x_arm].clamp(env.x_alo, env.x_ahi)
    env.x_prev_arm[ids] = ca; env.x_tick[ids] = ca; env.x_tickq[ids] = ca; cc = c+dqc; cc[:, env.x_arm] = ca; env.ctrl[ids] = torch.clamp(cc, env.lo, env.hi)
    for nm, h in env.x_hold.items():
        a = h['obj_q']; T = Hs[nm]; p = q[:, a:a+3]; q[:, a:a+3] = (T[:, :3, :3]@p[..., None])[..., 0]+T[:, :3, 3]
        qz = torch.stack([torch.cos(h['th'][ids]*h['w'][frames]/2), torch.zeros(n, device=dev), torch.zeros(n, device=dev), torch.sin(h['th'][ids]*h['w'][frames]/2)], -1)
        q[:, a+3:a+7] = _qmul(qz, q[:, a+3:a+7])
    # plunger carrier (slides x y z, hinges x y z in that order): follow the pipette's transform
    T = Hs['stand']; cq = env.x_carrier_q; c = q[:, cq]
    ca, sa, cb, sb, cc, sc = torch.cos(c[:, 3]), torch.sin(c[:, 3]), torch.cos(c[:, 4]), torch.sin(c[:, 4]), torch.cos(c[:, 5]), torch.sin(c[:, 5])
    Rx = torch.zeros(n, 3, 3, device=dev); Rx[:, 0, 0] = 1; Rx[:, 1, 1] = ca; Rx[:, 1, 2] = -sa; Rx[:, 2, 1] = sa; Rx[:, 2, 2] = ca
    Ry = torch.zeros(n, 3, 3, device=dev); Ry[:, 1, 1] = 1; Ry[:, 0, 0] = cb; Ry[:, 0, 2] = sb; Ry[:, 2, 0] = -sb; Ry[:, 2, 2] = cb
    Rz = torch.zeros(n, 3, 3, device=dev); Rz[:, 2, 2] = 1; Rz[:, 0, 0] = cc; Rz[:, 0, 1] = -sc; Rz[:, 1, 0] = sc; Rz[:, 1, 1] = cc
    Rn = T[:, :3, :3]@(Rx@Ry@Rz); pc = env.xpos[ids, env.x_carrier_b] if False else None
    c[:, 3] = torch.atan2(-Rn[:, 1, 2], Rn[:, 2, 2]); c[:, 4] = torch.asin(Rn[:, 0, 2].clamp(-1, 1)); c[:, 5] = torch.atan2(-Rn[:, 0, 1], Rn[:, 0, 0])
    p = c[:, :3]; c[:, :3] = (T[:, :3, :3]@p[..., None])[..., 0]+T[:, :3, 3]          # slides are world-aligned from the world origin
    q[:, cq] = c; env.qpos[ids] = q


# ---------------------------------------------------------------------------------------------- deployable student (11-05)
# extras 'student': {'meas_pos_mm': 5, 'meas_yaw_deg': 2}: the actor (obs_mode deployable) additionally sees, measured ONCE at the episode
# start with noise: each object's pose vs its nominal reference (dx dy dz [m], dyaw [rad]); each holder's offset as the robot estimates it
# (the same estimate that re-anchors the arm command; zeros + noise when holders are off); and the re-anchoring arm command delta itself.
STUDENT_PARTS = [('obj_meas_right_dpos_dyaw', 4), ('obj_meas_left_dpos_dyaw', 4), ('holder_meas_stand_dxy_dyaw', 3), ('holder_meas_rack_dxy_dyaw', 3)]


def student_init(env):
    sc = env.xcfg.get('student'); env.x_meas = None
    if not sc: return
    m = env.mjm; dev = env.device
    env.x_objq = {s: int(m.jnt_qposadr[m.joint(n).id]) for s, n in (('right', 'right_object_joint'), ('left', 'left_object_joint'))}
    env.x_meas = torch.zeros(env.num_envs, 14, device=dev); env.x_hd_seen = torch.zeros(env.num_envs, len(env.x_arm), device=dev)
    if getattr(env, 'actor_layout', None) is not None:
        o = env.actor_dim
        extra = [(f'wrist_ref_rel_{s}[t+{c}] (pos 3, rot cols 6)', 9) for c in WRIST_REF_OFFSETS for s in ('right', 'left')] if sc.get('wrist_ref') else []
        parts = STUDENT_PARTS+[('holder_reanchor_arm_cmd_delta', len(env.x_arm))]+extra
        XS = getattr(env, 'x_scan', None)
        if XS is not None and XS['obs'] == 'embedding': parts = [('scan_embedding (head depth at the start; perception/train_scan_net.py)', int(XS['emb'].shape[1]))]
        for name, n in parts:
            env.actor_layout.append({'name': name, 'start': o, 'size': n}); o += n
        env.actor_dim = o
    depth_init(env)


def _yaw_quat(q):   # wxyz -> yaw of the body z-rotation part (world frame)
    w, x, y, z = q.unbind(-1); return torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def measure(env, ids, frames):
    if getattr(env, 'x_meas', None) is None or not len(ids): return
    depth_reset(env, ids)
    sc = env.xcfg['student']; n = len(ids); dev = env.device; g = env.gen
    sp, sy = float(sc.get('meas_pos_mm', 5.))*1e-3, np.radians(float(sc.get('meas_yaw_deg', 2.)))
    q = env.qpos[ids]; qr = env.qpos_ref[frames]; out = []; XE = getattr(env, 'x_ego', None) if getattr(env, 'x_hold', None) else None
    for s in ('right', 'left'):
        a = env.x_objq[s]; dp = q[:, a:a+3]-qr[:, a:a+3]; dy = _yaw_quat(q[:, a+3:a+7])-_yaw_quat(qr[:, a+3:a+7]); dy = torch.atan2(torch.sin(dy), torch.cos(dy))
        XS = getattr(env, 'x_scan', None)
        if XS is not None and s == 'right':   # (11-12) the stand prediction applied to the pipette (its slack in the stand is not observed)
            h = env.x_hold['stand']; one = torch.ones(n, device=dev)
            Tm = _H(one, h['dm'][ids], h['thm'][ids], h['c'])@torch.linalg.inv(_H(one, h['d'][ids], h['th'][ids], h['c']))
            pm = (Tm[:, :3, :3]@q[:, a:a+3, None])[..., 0]+Tm[:, :3, 3]; out += [pm-qr[:, a:a+3], (dy+h['thm'][ids]-h['th'][ids])[:, None]]
        elif XS is not None:                  # (11-12) the predicted tube position; spin not observed
            out += [XS['ptube'][XS['j'][ids]]-qr[:, a:a+3], torch.zeros(n, 1, device=dev)]
        elif XE is None: out += [dp+sp*torch.randn(n, 3, device=dev, generator=g), (dy+sy*torch.randn(n, device=dev, generator=g))[:, None]]
        elif s == 'right':   # (11-11) pipette = the stand estimate applied to it (+ its unobserved slack in the stand)
            h = env.x_hold['stand']; one = torch.ones(n, device=dev)
            Tm = _H(one, h['dm'][ids], h['thm'][ids], h['c'])@torch.linalg.inv(_H(one, h['d'][ids], h['th'][ids], h['c']))
            pm = (Tm[:, :3, :3]@q[:, a:a+3, None])[..., 0]+Tm[:, :3, 3]; pm[:, :2] += XE['pxy']*torch.randn(n, 2, device=dev, generator=g)
            out += [pm-qr[:, a:a+3], (dy+h['thm'][ids]-h['th'][ids]+XE['pyaw']*torch.randn(n, device=dev, generator=g))[:, None]]
        else:                # (11-11) tube = the estimator's tube fit (row j) + extra; its spin is not observed
            out += [dp+XE['E'][XE['j'][ids]][:, 6:9]+XE['tube']*torch.randn(n, 3, device=dev, generator=g), torch.zeros(n, 1, device=dev)]
    for nm in ('stand', 'rack'):
        if getattr(env, 'x_hold', None): h = env.x_hold[nm]; out += [h['dm'][ids], h['thm'][ids][:, None]]
        else: out += [sp*torch.randn(n, 2, device=dev, generator=g), sy*torch.randn(n, 1, device=dev, generator=g)]
    env.x_meas[ids] = torch.cat(out, 1)


WRIST_REF_OFFSETS = (0, 2, 5, 12)   # = vega_res_env.OFFSETS (control steps)


def student_parts(env):
    if getattr(env, 'x_meas', None) is None: return []
    depth_update(env)   # (11-10) renders every cfg.every calls; the image lives in env.x_depth['obs'], not in the vector
    XS = getattr(env, 'x_scan', None)
    if XS is not None and XS['obs'] == 'embedding': return [XS['emb'][XS['j']]]
    hd = env.x_hdelta[:, env.x_arm] if getattr(env, 'x_hold', None) else torch.zeros(env.num_envs, len(env.x_arm), device=env.device)
    out = [env.x_meas, hd]
    if env.xcfg['student'].get('wrist_ref'):
        # (11-10) end-effector reference for the student: the reference wrist pose re-anchored with the robot's OWN holder estimate
        # (holder_T meas=True), relative to the wrist pose from forward kinematics of the MEASURED arm joints (encoder offset + noise:
        # env.h_q[0]); position + first two rotation columns, like the teacher's wrist term (which uses the true wrist and holders)
        k = env.frame.clamp(max=env.nframes-1); last = env.nframes-1; qm = env.h_q[0]; Wm = {}
        for s, L in (('right', 'R'), ('left', 'L')):
            T = env.x_ik.fk(L, qm[:, env.x_arm_side[L]]); Ti = torch.zeros_like(T); Rt = T[:, :3, :3].transpose(1, 2)
            Ti[:, :3, :3] = Rt; Ti[:, :3, 3] = -(Rt@T[:, :3, 3, None])[..., 0]; Ti[:, 3, 3] = 1.; Wm[s] = Ti
        for c in WRIST_REF_OFFSETS:
            j = (k+c*env.frame_step).clamp(max=last)
            for s in ('right', 'left'):
                rel = Wm[s]@holder_T(env, 'stand' if s == 'right' else 'rack', j, meas=True)@env.W_ref[s][j]
                out += [rel[:, :3, 3], rel[:, :3, :2].flatten(1)]
    return out


# ---------------------------------------------------------------------------------------------- event gate (11-06 round 3, default off)
# extras 'event_gate': {'tol_mm': 3, 'tol_deg': 3, 'cap_s': 1.0, 'events': ['closure', 'release']}. At each hand event the REFERENCE CLOCK is held
# (the arm keeps its command for the frame just before the event) until the wrist is within tol of its pose at the event frame, at most cap_s per
# event; then the fingers close / open. Deployable: wrist pose = forward kinematics of the measured arm joints; the target uses the robot's own
# holder estimate (holder_T meas=True), never the object state. Events from the task gates: grasp closures = 0.6 s before tube_held / pipette_held
# start (the closure windows), releases = end of tube_held (tube place) and the return of the pipette to the stand (fingers start opening).
def _gate_init(env):
    G = env.xcfg.get('event_gate'); env.x_gate = None
    if not G: return
    r = np.load(env.task/'reference.npz'); dev = env.device; ev = []
    def first(g): i = np.flatnonzero(g.astype(bool)); return int(i[0]) if len(i) else None
    def last(g): i = np.flatnonzero(g.astype(bool)); return int(i[-1]) if len(i) else None
    if 'closure' in G.get('events', ['closure', 'release']):
        for side, gn in (('right', 'gate_pipette_held'), ('left', 'gate_tube_held')):
            f = first(r[gn]);
            if f is not None: ev.append((side, max(0, f-60), 'closure'))
    if 'release' in G.get('events', ['closure', 'release']):
        f = last(r['gate_tube_held'])
        if f is not None: ev.append(('left', f, 'release'))
        st = np.flatnonzero(np.diff(np.r_[0, r['gate_pipette_in_stand'].astype(int)]) == 1)
        if len(st) > 1: ev.append(('right', int(st[1]), 'release'))
    env.x_gate = {'events': ev, 'held': torch.zeros(env.num_envs, len(ev), dtype=torch.long, device=dev), 'tol_m': float(G.get('tol_mm', 3.))*1e-3,
                  'tol_rad': math.radians(float(G.get('tol_deg', 3.))), 'cap': int(round(float(G.get('cap_s', 1.))/CONTROL_DT_GATE))}


CONTROL_DT_GATE = .02
import math


def frame_advance(env):
    """per-world frame increment: frame_step, or up to the frame before a gated event while the wrist is not yet at its event pose (<= cap)"""
    step = env.frame_step
    if not hasattr(env, 'x_gate'): _gate_init(env)
    G = env.x_gate
    if G is None: return step
    inc = torch.full_like(env.frame, step)
    for e, (side, g, kind) in enumerate(G['events']):
        cross = (env.frame < g) & (env.frame+step >= g)
        if not bool(cross.any()): continue
        gk = torch.full_like(env.frame, g); Wt = env.W_ref[side][gk]
        if getattr(env, 'x_hold', None): Wt = holder_T(env, 'stand' if side == 'right' else 'rack', gk, meas=True)@Wt   # the robot's own holder estimate
        W, _ = env._poses(side); pe = (W[:, :3, 3]-Wt[:, :3, 3]).norm(dim=-1); ae = rot_angle_(W[:, :3, :3], Wt[:, :3, :3])
        hold = cross & ((pe > G['tol_m']) | (ae > G['tol_rad'])) & (G['held'][:, e] < G['cap'])
        G['held'][:, e] += hold.long(); inc = torch.where(hold, (g-1-env.frame).clamp(min=0), inc)
    return inc


def rot_angle_(A, B):
    c = ((A.transpose(1, 2)@B).diagonal(dim1=1, dim2=2).sum(-1)-1)/2; return torch.arccos(c.clamp(-1, 1))


def gate_reset(env, ids):
    if getattr(env, 'x_gate', None) is not None and len(ids): env.x_gate['held'][ids] = 0


# ---------------------------------------------------------------------------------------------- (11-10) wrist depth cameras for the student
# extras.student.depth = {...} (DEPTH_DEFAULT; true = defaults). One stereo-depth camera per hand on a bracket on the wrist body (perception/
# wristcam_sim.py 'forearm' mount: 6 cm back, 4.8 cm off the palm, looking at the grasp; right-hand wrist frame, left mirrored), rendered
# by MuJoCo Warp's batched ray-cast renderer at the policy resolution every `every` control steps (render geom groups 1 = robot + object
# visuals, 3 = furniture; collision copies moved to group 4 at load), with the stereo noise model of wristcam_sim.NOISE interpolated per
# episode between `noise_lo` (FFS-like) and `noise_hi` (D405 on-chip-like); disparity noise rescaled from 256 px to the render width
# (correlated part x r, white part x r^2 for the r x r averaging, r = width / 256). Per-episode depth scale error and camera mount error
# (applied to the rendered camera pose). The student sees the image of the PREVIOUS render (latency `every` steps); nothing for the first
# render interval of an episode. Stored as int16 mm (0 = invalid) in env.x_depth['obs'] (N, 2 cameras [right, left], H, W).
DEPTH_DEFAULT = dict(res=[64, 48], every=2, f256=156., baseline=.018, zmin=.044, mount=[[-.005, .065, .06], [-.01, .045, -.08]],
                     noise_lo=dict(white=.1, corr=.15, edge_fat=.5, edge_drop=.5, holes=.10, obj_holes=.10, speckle=.01, scale=.005, extr=[1., .5]),
                     noise_hi=dict(white=.25, corr=.4, edge_fat=.5, edge_drop=.6, holes=.35, obj_holes=.25, speckle=.03, scale=.01, extr=[2., 1.]),
                     aux_inhand=1., mount_rand=None)
# (11-12) mount_rand = {'xyz_mm': [x, y, z], 'rpy_deg': [r, p, y]} (None = off): per episode and per camera, an installation offset of the
# camera on top of the small extr error above: position uniform in +-xyz_mm along the WRIST axes (x thumb / little finger, y palm, z forearm),
# rotation uniform in +-rpy_deg about the camera's own axes (roll = about the viewing axis, pitch = about the image horizontal (tilt),
# yaw = about the image vertical (pan)), composed Rz(roll) Rx(pitch) Ry(yaw) in the MuJoCo camera frame. The student is not told the offset.
DEPTH_CAMS = ('wristcam_right', 'wristcam_left')


def depth_cfg(extras):
    dc = ((extras or {}).get('student') or {}).get('depth')
    if not dc: return None
    return merge(DEPTH_DEFAULT, dc if isinstance(dc, dict) else {})


def load_model(path, extras):
    """The task scene; with the student's depth cameras added (no change to the physics) when extras.student.depth is set."""
    import mujoco, math
    dc = depth_cfg(extras)
    if dc is None: return mujoco.MjModel.from_xml_path(str(path))
    sp = mujoco.MjSpec.from_file(str(path)); (pos, tgt) = dc['mount']; W, H = dc['res']; f = dc['f256']*W/256.
    for side, s in (('right', 1.), ('left', -1.)):
        p = np.array(pos)*[1, s, 1]; fw = np.array(tgt)*[1, s, 1]-p; fw /= np.linalg.norm(fw)
        zc = -fw; xc = np.cross([0, s, 0], zc); xc /= np.linalg.norm(xc); yc = np.cross(zc, xc); q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.stack([xc, yc, zc], 1).ravel())
        sp.body(f'{side}_wrist').add_camera(name=f'wristcam_{side}', pos=p, quat=q, fovy=math.degrees(2*math.atan(H/2/f)))
    m = sp.compile()
    for i in range(m.ngeom):   # group 3 on moving bodies = collision proxies overlapping the visuals: out of the rendered groups
        if m.geom_group[i] == 3 and int(m.geom_bodyid[i]) != 0: m.geom_group[i] = 4
    return m


def _gauss_kernel(sigma, dev):
    r = max(1, int(math.ceil(2.5*sigma))); x = torch.arange(-r, r+1, dtype=torch.float32, device=dev); k = torch.exp(-.5*(x/sigma)**2)
    k2 = k[:, None]*k[None]; k2 = k2/k2.sum(); return k2[None, None], float(torch.sqrt((k2**2).sum()))


def depth_init(env):
    dc = depth_cfg(env.xcfg); env.x_depth = None
    if dc is None: return
    import warp as wp, mujoco_warp as mjw
    m = env.mjm; W, H = dc['res']; N = env.num_envs; dev = env.device
    with wp.ScopedDevice(env.wpdev):
        rc = mjw.create_render_context(m, nworld=N, cam_res=(W, H), render_rgb=False, render_depth=True, render_seg=True, enabled_geom_groups=[1, 3],
                                       cam_active=list(DEPTH_CAMS), use_textures=False, use_shadows=False)
        zb = [wp.zeros((N, H, W), dtype=float) for _ in DEPTH_CAMS]; sb = [wp.zeros((N, H, W), dtype=wp.vec2i) for _ in DEPTH_CAMS]
    objb = {m.body(n).id for n in ('right_object', 'left_object', 'pipette_plunger')}
    objg = torch.tensor([int(m.geom_bodyid[i]) in objb for i in range(m.ngeom)]+[False], device=dev)   # last = background (-1)
    r = W/256.; K = {k: _gauss_kernel(s*r, dev) for k, s in (('corr', 3.), ('holes', 4.), ('obj', 3.))}
    lo, hi = dc['noise_lo'], dc['noise_hi']
    env.x_depth = dict(cfg=dc, rc=rc, cams=[m.camera(n).id for n in DEPTH_CAMS], zb=zb, sb=sb, objg=objg, W=W, H=H, f=dc['f256']*r, r=r, K=K, tick=0,
                       lo={k: torch.tensor(v, dtype=torch.float32, device=dev) for k, v in lo.items()},
                       hi={k: torch.tensor(v, dtype=torch.float32, device=dev) for k, v in hi.items()},
                       obs=torch.zeros(N, 2, H, W, dtype=torch.int16, device=dev), pend=torch.zeros(N, 2, H, W, dtype=torch.int16, device=dev),
                       Rwc=torch.tensor(np.stack([_quat2mat(m.cam_quat[m.camera(n).id]) for n in DEPTH_CAMS]), dtype=torch.float32, device=dev),
                       alpha=torch.zeros(N, device=dev), scale=torch.zeros(N, device=dev),
                       Re=torch.eye(3, device=dev).repeat(N, 2, 1, 1), te=torch.zeros(N, 2, 3, device=dev))


def _p(X, k):   # per-world noise parameter, interpolated between the two levels by the episode's alpha
    return X['lo'][k]+X['alpha'][:, None]*(X['hi'][k]-X['lo'][k]) if X['lo'][k].dim() else X['lo'][k]+X['alpha']*(X['hi'][k]-X['lo'][k])


def depth_reset(env, ids):
    X = getattr(env, 'x_depth', None)
    if X is None or not len(ids): return
    n = len(ids); dev = env.device; g = env.gen
    X['alpha'][ids] = torch.rand(n, device=dev, generator=g)
    X['scale'][ids] = _p(X, 'scale')[ids]*torch.randn(n, device=dev, generator=g)
    ex = _p(X, 'extr')[ids]                                                     # (n, 2): mm, deg
    for j in range(2):
        u = torch.randn(n, 3, device=dev, generator=g); u = u/u.norm(dim=1, keepdim=True); X['te'][ids, j] = u*ex[:, :1]*1e-3
        a = torch.randn(n, 3, device=dev, generator=g); a = a/a.norm(dim=1, keepdim=True); th = torch.deg2rad(ex[:, 1])
        Kx = torch.zeros(n, 3, 3, device=dev); Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 2] = -a[:, 2], a[:, 1], -a[:, 0]; Kx = Kx-Kx.transpose(1, 2)
        X['Re'][ids, j] = torch.eye(3, device=dev)+torch.sin(th)[:, None, None]*Kx+(1-torch.cos(th))[:, None, None]*(Kx@Kx)
    mr = X['cfg'].get('mount_rand')
    if mr:   # (11-12) installation offset (see DEPTH_DEFAULT): te / Re are in the nominal camera frame (applied in depth_update)
        xyz = torch.tensor(mr.get('xyz_mm', [0, 0, 0]), dtype=torch.float32, device=dev)*1e-3
        rpy = torch.deg2rad(torch.tensor(mr.get('rpy_deg', [0, 0, 0]), dtype=torch.float32, device=dev))
        for j in range(2):
            dp = (2*torch.rand(n, 3, device=dev, generator=g)-1)*xyz                       # wrist frame
            X['te'][ids, j] += dp@X['Rwc'][j]                                               # -> camera frame (Rwc^T dp)
            r_, p_, y_ = ((2*torch.rand(n, 3, device=dev, generator=g)-1)*rpy).unbind(1)
            X['Re'][ids, j] = _rot(2, r_)@_rot(0, p_)@_rot(1, y_)@X['Re'][ids, j]
    X['obs'][ids] = 0; X['pend'][ids] = 0


def _rot(ax, th):   # (n, 3, 3) rotations about axis ax (0 x, 1 y, 2 z) by th (n,)
    c, s_ = torch.cos(th), torch.sin(th); R = torch.zeros(len(th), 3, 3, device=th.device); i, k = [a for a in range(3) if a != ax]
    R[:, ax, ax] = 1; R[:, i, i] = c; R[:, k, k] = c; R[:, i, k] = -s_; R[:, k, i] = s_; return R


def _quat2mat(q):
    import mujoco
    R = np.zeros(9); mujoco.mju_quat2Mat(R, np.asarray(q, float)); return R.reshape(3, 3)


def _blur(x, KS):
    k, nrm = KS; return torch.nn.functional.conv2d(x, k, padding=k.shape[-1]//2)/nrm


def depth_noise(env, z, obj):
    """Simulated stereo depth (m, 0 = invalid) from true depth z (N, 2, H, W) and the held-object mask."""
    X = env.x_depth; dc = X['cfg']; g = env.gen; dev = env.device; N, C, H, W = z.shape; r = X['r']; fB = X['f']*dc['baseline']
    P = lambda k: _p(X, k).view(N, 1, 1, 1)
    bg = z <= 0; zt = torch.where(bg, torch.full_like(z, 2.), z).clamp(min=1e-3)
    rn = lambda: torch.randn(N*C, 1, H, W, device=dev, generator=g)
    n = P('white')*r*r*rn().view(N, C, H, W)+P('corr')*r*_blur(rn(), X['K']['corr']).view(N, C, H, W)
    zn = fB/(fB/zt+n).clamp(min=1e-3)*(1+X['scale'].view(N, 1, 1, 1))
    jump = torch.zeros_like(z, dtype=torch.bool); dv = (zt[:, :, 1:]-zt[:, :, :-1]).abs() > .005; dh = (zt[..., 1:]-zt[..., :-1]).abs() > .005
    jump[:, :, 1:] |= dv; jump[:, :, :-1] |= dv; jump[..., 1:] |= dh; jump[..., :-1] |= dh
    u = torch.rand(z.shape, device=dev, generator=g); fat = jump & (u < P('edge_fat'))
    zfg = -torch.nn.functional.max_pool2d(-zt.view(N*C, 1, H, W), 3, 1, 1).view(N, C, H, W)
    zn = torch.where(fat, zfg*(1+X['scale'].view(N, 1, 1, 1)), zn)
    bad = jump & ~fat & (torch.rand(z.shape, device=dev, generator=g) < P('edge_drop'))
    bad |= _blur(rn(), X['K']['holes']).view(N, C, H, W) > torch.special.ndtri(1-P('holes').clamp(1e-4, .9999))
    bad |= obj & (_blur(rn(), X['K']['obj']).view(N, C, H, W) > torch.special.ndtri(1-P('obj_holes').clamp(1e-4, .9999)))
    sp = torch.rand(z.shape, device=dev, generator=g) < P('speckle'); zn = zn+sp*(torch.rand(z.shape, device=dev, generator=g)*.04-.02)
    bad |= bg | (zn < dc['zmin']) | (zn > 2.)
    return torch.where(bad, torch.zeros_like(zn), zn)


def depth_update(env):
    X = getattr(env, 'x_depth', None)
    if X is None: return
    X['tick'] += 1
    if X['tick'] % int(X['cfg']['every']): return
    import warp as wp, mujoco_warp as mjw
    with wp.ScopedDevice(env.wpdev):
        mjw.camlight(env.m, env.d)
        cx, cm = wp.to_torch(env.d.cam_xpos), wp.to_torch(env.d.cam_xmat)        # camera mount error of the episode (camera frame)
        for j, c in enumerate(X['cams']):
            R = cm[:, c].clone(); cx[:, c] += (R@X['te'][:, j, :, None])[..., 0]; cm[:, c] = R@X['Re'][:, j]
        mjw.refit_bvh(env.m, env.d, X['rc']); mjw.render(env.m, env.d, X['rc'])
        for j in range(2): mjw.get_depth(X['rc'], j, 1., X['zb'][j]); mjw.get_segmentation(X['rc'], j, X['sb'][j])
    z = torch.stack([wp.to_torch(b) for b in X['zb']], 1)
    seg = torch.stack([wp.to_torch(b)[..., 0] for b in X['sb']], 1).long(); seg = torch.where(seg < 0, torch.full_like(seg, len(X['objg'])-1), seg)
    zn = depth_noise(env, z, X['objg'][seg])
    X['obs'] = X['pend']; X['pend'] = (zn*1000.).round().clamp(0, 32767).to(torch.int16)


def inhand_targets(env):
    """(N, 36) privileged auxiliary target: each held object's six axis points (+-5 cm) in its wrist frame (the teacher's object term)."""
    out = []
    for s in ('right', 'left'):
        W, O = env._poses(s); R = W[:, :3, :3].transpose(1, 2); lo = R@O[:, :3, :3]; lp = (R@(O[:, :3, 3]-W[:, :3, 3])[..., None])[..., 0]
        out.append((env.axis@lo.transpose(1, 2)+lp[:, None]).flatten(1))
    return torch.cat(out, 1)
