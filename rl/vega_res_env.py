"""REAL-ROBOT LINE copy of tools/rl/gpu_env.py adapted to Vega-U + Wuji beta1 (rrline, 11-05): DAID scene names, 50 Hz control,
real-robot arm command limits, extra reward terms (rewards.py of the shared layer). Original header follows.

Batched MuJoCo Warp environment for the bimanual residual-RL task (tools/export_residual_ppo_task.py --physics-stage).

YS's residual tracking task (g1-wuji-retargeting baselines/residual_ppo/env.py, TrackingEnv in g1-full mode) on the GPU, both sides:

* Action: a residual on the reference command of every actuator (waist + both arms: scale 0.08 rad, both hands: 0.25 rad; YS's scales),
  held for one 20 Hz control step (10 physics steps of 5 ms, the physics' sim_dt). Under it the reference runs at the sim rate:
  fingers = the executed physics ctrl (reference.npz ctrl_sim), waist / arms = the IK, interpolated, plus the servo velocity lead
  kv / kp * qdot_ref (tools/rl/train.py ServoTrackingEnv: kp (q_ref - q) + kv (qdot_ref - qdot)).
* Reward (YS's terms per side, summed over the pipette side and the tube side): 8 object axis points in the wrist frame +
  links (wrist + finger bodies in the wrist frame) + finger joints + 2 wrist world pose + 2 object world pose, minus 0.01 x residual
  smoothness over all actuators; plus a plunger depth term (weight 2, exp(-(err / 3 mm)^2)) in the press window. Link and joint
  targets are the executed physics state (qpos_ref), not the commands (which carry the grip squeeze).
* Termination per object, YS's order and thresholds: relative position > 5 cm, relative orientation > 45 deg, axis points > 5 cm,
  world position > 5 cm, world orientation > 45 deg, height < 6 cm, linear speed > 10 m/s, angular speed > 500 rad/s; non-finite.
  Success = the end of the reference reached; no time limit inside the window.
* Reset: reference state + reference velocity (RSI) at a start frame drawn uniformly from the valid starts (check_task.py reset).
* MuJoCo Warp guards: patch_mjw_broadphase_cache(); after every physics step the broadphase pair count d.ncollision and the contact
  count d.nacon (both against naconmax) and the per-world constraint rows d.nefc (against njmax) are folded into a running max on
  the GPU, read once per control step; any overflow raises (no training on dropped contacts).
All observations, rewards and resets are torch on the GPU, no per-env Python loops.
"""
from __future__ import annotations

import json, math, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vega_extras as X

FAILURES = ('object_relative_position', 'object_relative_orientation', 'object_axis_points', 'object_world_position',
            'object_world_orientation', 'object_height', 'object_linear_velocity', 'object_angular_velocity', 'nonfinite')
SIDES = ('right', 'left')
DR_FIELDS = ('geom_friction', 'pair_friction', 'body_mass', 'body_subtreemass', 'body_inertia', 'body_invweight0', 'dof_invweight0', 'actuator_gainprm', 'actuator_biasprm', 'actuator_forcerange', 'jnt_actfrcrange', 'body_pos', 'geom_pos', 'geom_quat')
PREFIX = {'right': 'r', 'left': 'l'}
OBJECT_BODY = {'right': 'right_object', 'left': 'left_object'}
WRIST_BODY = {'right': 'right_wrist', 'left': 'left_wrist'}
OBJECT_JOINT = {'right': 'right_object_joint', 'left': 'left_object_joint'}
AXIS = np.concatenate([np.eye(3), -np.eye(3)])*.05      # YS: axis half-length 0.05 m
OFFSETS = (0, 2, 5, 12)                                 # future reference offsets in control steps (50 Hz: 0 / 40 / 100 / 240 ms ~ G1's 0 / 50 / 150 / 250 ms)
CONTROL_DT = .02                                        # Vega real-robot line: policy at 50 Hz


def refuse_backport_on_head():
    """(11-05) W27 --gjk-backport patches the pinned mujoco_warp 5d5a645 only; the head stack (retargeting_head, .venv-rl-head) has the fix natively."""
    from importlib.metadata import version
    if version('warp-lang') != '1.10.1': raise SystemExit(f"--gjk-backport is for the pinned stack (warp 1.10.1), this env has warp {version('warp-lang')}: the head stack has the fix natively")


def quat_to_mat(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), 2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1).reshape(q.shape[:-1]+(3, 3))


def rinv(T):
    R = T[..., :3, :3].transpose(-1, -2); out = torch.zeros_like(T); out[..., :3, :3] = R
    out[..., :3, 3] = -(R@T[..., :3, 3, None])[..., 0]; out[..., 3, 3] = 1; return out


def rot_angle(Ra, Rb):
    c = ((Ra.transpose(-1, -2)@Rb).diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2
    return torch.arccos(c.clamp(-1, 1))


class BimanualGpuEnv:
    def __init__(self, task, num_envs, device='cuda:0', arm_scale=.08, finger_scale=.25, plunger_weight=2., plunger_sigma=.003,
                 nconmax=None, njmax=None, naconmax_per_world=None, starts=None, seed=0, guard=True, record=False, terminate=True, auto_reset=True, feedforward=True, gjk_backport=False, thumb_press_scale=None, reward_scale=1., group_scale=None, obs_mode='full', dr=None, extras=None):
        import sys
        import os as _os; sys.path.insert(0, _os.environ.get('G1PS_ROOT', str(Path(__file__).resolve().parents[1]/'deps_from_g1ps')))
        if gjk_backport:   # same patched collision_gjk as the physics worker's --gjk-backport (default off); before mujoco_warp is first imported
            refuse_backport_on_head()
            from g1_pipette_sim.retarget.gpu.mjw_backport import install_gjk_backport
            install_gjk_backport()
        import mujoco, warp as wp, mujoco_warp as mjw
        from g1_pipette_sim.retarget.gpu.core import patch_mjw_broadphase_cache
        patch_mjw_broadphase_cache()
        self.wp, self.mjw, self.mujoco = wp, mjw, mujoco
        self.task = Path(task); self.num_envs = N = int(num_envs); self.device = device; self.guard = guard
        self.cfg = dict(task=str(self.task), num_envs=N, arm_scale=arm_scale, finger_scale=finger_scale, plunger_weight=plunger_weight,
                        plunger_sigma=plunger_sigma, control_dt=CONTROL_DT)
        sizes = json.loads((self.task/'gpu_sizes.json').read_text()) if (self.task/'gpu_sizes.json').exists() else {}
        self.nconmax = int(nconmax or sizes.get('nconmax', 600)); self.njmax = int(njmax or sizes.get('njmax', 2400))
        self.naconmax = int(N*(naconmax_per_world or sizes.get('naconmax_per_world', 1200)))
        self.cfg.update(nconmax=self.nconmax, njmax=self.njmax, naconmax=self.naconmax)
        # (11-05, default off) obs_mode 'deployable': the actor sees only joint encoders, previous action and the reference tables; object and plunger state go to the critic. dr: domain randomisation, see set_dr
        self.obs_mode = obs_mode; self.dr = None; self.dr_on = dr is not None
        if obs_mode != 'full': self.cfg['obs_mode'] = obs_mode
        if dr is not None: self.cfg['dr'] = {k: v for k, v in dr.items() if k != 'arm_gains'}
        # (11-05, both default off = unchanged) thumb_press_scale: factor on the residual of the four right thumb actuators while the step's first frame is in the press gate (phase 2);
        # reward_scale: factor on the returned reward (the critic's target scale; PPO's value clip is in reward units)
        self.thumb_press_scale = thumb_press_scale; self.reward_scale = float(reward_scale)
        if thumb_press_scale is not None: self.cfg['thumb_press_scale'] = float(thumb_press_scale)
        if self.reward_scale != 1.: self.cfg['reward_scale'] = self.reward_scale
        m = mujoco.MjModel.from_xml_path(str(self.task/'scene.xml')); self.mjm = m
        r = np.load(self.task/'reference.npz'); self.dt = float(m.opt.timestep)
        if not math.isclose(float(r['sim_dt']), self.dt): raise ValueError('scene timestep != physics sim_dt')
        self.substeps = int(round(CONTROL_DT/self.dt)); self.per_frame = int(round(.01/self.dt)); self.frame_step = int(round(CONTROL_DT/.01))
        dev = device; f32 = dict(dtype=torch.float32, device=dev)
        T = lambda x: torch.as_tensor(np.asarray(x), **f32)
        # ---- layout
        self.nu = m.nu; acts = [m.actuator(i).name for i in range(m.nu)]; self.action_names = acts
        jid = m.actuator_trnid[:, 0]; self.aq = torch.as_tensor(m.jnt_qposadr[jid], device=dev); self.av = torch.as_tensor(m.jnt_dofadr[jid], device=dev)
        finger = np.array([n.startswith(('r_', 'l_')) for n in acts])
        self.scale = T(np.where(finger, finger_scale, arm_scale))
        kp, kv = m.actuator_gainprm[:, 0], -m.actuator_biasprm[:, 2]
        self.lo, self.hi = T(m.actuator_ctrlrange[:, 0]), T(m.actuator_ctrlrange[:, 1])
        self.fing = {s: np.flatnonzero(np.array([n.startswith(PREFIX[s]+'_') for n in acts])) for s in SIDES}
        self.fing_t = {s: torch.as_tensor(v, device=dev) for s, v in self.fing.items()}
        self.rthumb = torch.as_tensor([i for i, n in enumerate(acts) if n.startswith('r_thumb_')], device=dev)
        if group_scale:   # (11-05, default off) residual scale per group, rad per action unit: FL / FR fingers per side, AL / AR arm per side, W waist; exploration noise in rad = action std x this
            sel = {'FL': lambda n: n.startswith('l_'), 'FR': lambda n: n.startswith('r_'), 'AL': lambda n: n.startswith('L_arm'), 'AR': lambda n: n.startswith('R_arm'), 'W': lambda n: False}
            for k, v in group_scale.items(): self.scale[torch.as_tensor([i for i, n in enumerate(acts) if sel[k](n)], device=dev)] = float(v)
            self.cfg['group_scale'] = {k: float(v) for k, v in group_scale.items()}
        jr = m.jnt_range[jid]; self.jrange = {s: T(jr[v, 1]-jr[v, 0]) for s, v in self.fing.items()}
        self.wrist_id = {s: m.body(WRIST_BODY[s]).id for s in SIDES}; self.obj_id = {s: m.body(OBJECT_BODY[s]).id for s in SIDES}
        self.link_ids = {s: [self.wrist_id[s]]+[int(m.jnt_bodyid[jid[i]]) for i in self.fing[s]] for s in SIDES}
        self.obj_q = {s: int(m.jnt_qposadr[m.joint(OBJECT_JOINT[s]).id]) for s in SIDES}; self.obj_v = {s: int(m.jnt_dofadr[m.joint(OBJECT_JOINT[s]).id]) for s in SIDES}
        self.pl_q = int(m.jnt_qposadr[m.joint('pipette_plunger').id]); self.pl_v = int(m.jnt_dofadr[m.joint('pipette_plunger').id])
        # ---- reference (100 Hz frames) and the sim-rate command
        self.nframes = len(r['timestamps']); self.times = r['timestamps']
        self.qpos_ref, self.qvel_ref = T(r['qpos_ref']), T(r['qvel_ref'])
        arm = ~finger
        lead = np.zeros_like(r['ctrl_sim'])
        self.verbatim = bool(r['ctrl_sim_verbatim']) if 'ctrl_sim_verbatim' in r.files else False
        if self.verbatim:   # (10-30) whole-robot physics export: ctrl_sim is the executed ctrl of every actuator; applied as is
            self.cfg['command'] = 'verbatim executed physics ctrl (no lead, no feed-forward)'; self.cfg['feedforward'] = False
        else:
            qd = r['qvel_ref'][:, m.jnt_dofadr[jid]]                      # reference joint velocities at the frames
            for i in np.flatnonzero(arm): lead[:, i] = kv[i]/kp[i]*np.interp(r['sim_times'], r['timestamps'], qd[:, i])
            self.cfg['feedforward'] = bool(feedforward)
            if feedforward:   # inverse dynamics of the reference (inertia + Coriolis; gravity is compensated by gravcomp) through the servos
                tau = self.reference_torques(m, r)
                for i in np.flatnonzero(arm): lead[:, i] += np.interp(r['sim_times'], r['timestamps'], tau[:, i])/kp[i]
                self.tau_ff = tau
        self.cmd_sim = T(r['ctrl_sim']+lead)
        self.ctrl_frame = T(r['controls'])                            # observation: the reference command at a frame
        self.phase = torch.as_tensor(r['phase'], device=dev).long(); self.press = self.phase == 2
        self.plunger_ref = T(r['plunger'])
        d = mujoco.MjData(m); W = {s: [] for s in SIDES}; O = {s: [] for s in SIDES}; L = {s: [] for s in SIDES}
        pose = lambda b: np.r_[np.c_[d.xmat[b].reshape(3, 3), d.xpos[b]], [[0, 0, 0, 1]]]
        for k in range(self.nframes):
            d.qpos[:] = r['qpos_ref'][k]; mujoco.mj_kinematics(m, d)
            for s in SIDES:
                Wk = pose(self.wrist_id[s]); W[s].append(Wk); O[s].append(pose(self.obj_id[s]))
                L[s].append((d.xpos[self.link_ids[s]]-Wk[:3, 3])@Wk[:3, :3])
        self.W_ref = {s: T(np.array(W[s])) for s in SIDES}; self.O_ref = {s: T(np.array(O[s])) for s in SIDES}
        self.L_ref = {s: T(np.array(L[s])) for s in SIDES}
        self.axis = T(AXIS)
        rel = {s: rinv(self.W_ref[s])@self.O_ref[s] for s in SIDES}
        self.rel_ref = rel; self.rel_axis_ref = {s: self.axis@rel[s][:, :3, :3].transpose(1, 2)+rel[s][:, None, :3, 3] for s in SIDES}
        self.qfing_ref = {s: self.qpos_ref[:, self.aq[self.fing_t[s]]] for s in SIDES}
        # ---- starts: 20 Hz frames from the reset audit (or given), each with at least one control step left
        vs = np.load(self.task/'valid_starts.npy') if starts is None else np.asarray(starts)
        vs = vs[vs+self.frame_step <= self.nframes-1]; self.valid_starts = torch.as_tensor(vs, device=dev).long()
        self.max_episode_length = (self.nframes-1)//self.frame_step
        # ---- MuJoCo Warp
        wp.init(); self.wpdev = wp.get_device(dev)
        with wp.ScopedDevice(self.wpdev):
            self.m = mjw.put_model(m, batch_sizes={k: N for k in DR_FIELDS}) if self.dr_on else mjw.put_model(m)
            self.d = mjw.make_data(m, nworld=N, nconmax=self.nconmax, njmax=self.njmax, naconmax=self.naconmax)
        tt = wp.to_torch
        self.qpos, self.qvel, self.ctrl = tt(self.d.qpos), tt(self.d.qvel), tt(self.d.ctrl)
        self.xpos, self.xmat = tt(self.d.xpos), tt(self.d.xmat)
        self.qacc_ws = tt(self.d.qacc_warmstart); self.ncol, self.nacon, self.nefc = tt(self.d.ncollision), tt(self.d.nacon), tt(self.d.nefc)
        self.gen = torch.Generator(device=dev); self.gen.manual_seed(seed)
        self.stats = torch.zeros(3, dtype=torch.int64, device=dev); self.stats_peak = torch.zeros(3, dtype=torch.int64, device=dev)
        self.limits = torch.tensor([self.naconmax, self.naconmax, self.njmax], device=dev)
        self.ovf = tt(self.d.overflow) if hasattr(self.d, 'overflow') else None   # (11-04, W28) upstream 4911167: sticky per-world bitmask, also covers naccdmax / njmax_nnz; None on the pinned stack = unchanged
        if self.ovf is not None:
            from mujoco_warp._src.types import OverflowType as OT_
            self.ovf_hard = int(OT_.NEFC | OT_.NJMAX_NNZ | OT_.BROADPHASE | OT_.NARROWPHASE | OT_.CCD | OT_.HFIELD | OT_.NVMAX)
        # ---- buffers
        self.frame = torch.zeros(N, dtype=torch.long, device=dev); self.start = torch.zeros_like(self.frame)
        self.episode_length_buf = torch.zeros(N, dtype=torch.long, device=dev)
        self.prev = torch.zeros(N, self.nu, **f32); self.prev2 = torch.zeros_like(self.prev); self.res = torch.zeros_like(self.prev)
        self.record = record; self.num_actions = self.nu; self.terminate = terminate; self.auto_reset = auto_reset; self.substep_hook = None
        self.prev_a = torch.zeros_like(self.prev); self.finger_mask = T(finger).bool(); self.t_nu = torch.arange(self.nu, device=dev); self.world = torch.arange(N, device=dev)
        self.joff = torch.zeros_like(self.prev); self.skew = torch.zeros(N, dtype=torch.long, device=dev); self.adelay = torch.zeros(N, dtype=torch.long, device=dev); self.a_buf = None; self.base_off = None
        self.obs_noise = {}; self.offset_ramp = .5
        if self.dr_on: self._dr_init(); self.set_dr(dr)
        if obs_mode == 'deployable': self._actor_init()
        X.init(self, extras or {})
        if self.dr_on: X.holder_init(self)
        X.student_init(self)
        self.ep_stats = self._new_stats()
        self._capture()
        self.reset_envs(torch.ones(N, dtype=torch.bool, device=dev))
        self.obs = self._observe()
        if obs_mode == 'deployable': self._observe_deployable()

    # ------------------------------------------------------------------ (11-05) domain randomisation, batched; per-world fields written in place, sampled at each reset
    def _dr_init(self):
        import mujoco
        wp, m = self.wp, self.mjm; tt = wp.to_torch; dev = self.device
        self.mf = {k: tt(getattr(self.m, k)) for k in DR_FIELDS}; self.m0 = {k: v[0].clone() for k, v in self.mf.items()}; self.gen_dr = torch.Generator(device=dev); self.gen_dr.manual_seed(12345)
        hand = set(); pip = {m.body(n).id for n in ('right_object', 'plunger_carrier', 'pipette_plunger')}; tube = {m.body('left_object').id}
        for s, root in (('r', m.body('right_wrist').id), ('l', m.body('left_wrist').id)):
            for b in range(m.nbody):
                x = b
                while x > 0 and x != root: x = m.body_parentid[x]
                if x == root: hand.add(b)
        ho, pt = [], []
        for k in range(m.npair):
            a, b = int(m.geom_bodyid[m.pair_geom1[k]]), int(m.geom_bodyid[m.pair_geom2[k]])
            if (a in hand and b in pip | tube) or (b in hand and a in pip | tube): ho.append(k)
            if (a in pip and b in tube) or (b in pip and a in tube): pt.append(k)
        self.pair_ho = torch.as_tensor(ho, device=dev); self.pair_pt = torch.as_tensor(pt, device=dev)
        self.geom_furn = torch.as_tensor([g for g in range(m.ngeom) if m.geom_bodyid[g] == 0 and (m.geom_contype[g] or m.geom_conaffinity[g])], device=dev)   # Vega DAID scene: furniture proxies on the world body
        self.pip_bodies = torch.as_tensor(sorted(pip), device=dev); self.tube_bodies = torch.as_tensor(sorted(tube), device=dev)
        self.obj_dofs = {s: torch.arange(self.obj_v[s], self.obj_v[s]+6, device=dev) for s in SIDES}; self.pelvis = m.body('base_link').id
        self.act_jid = torch.as_tensor(m.actuator_trnid[:, 0], device=dev); self.arm_ids = (~self.finger_mask).nonzero().squeeze(-1); self.fing_ids = self.finger_mask.nonzero().squeeze(-1)
        r = np.load(self.task/'reference.npz'); self.f_pre = {'right': int(np.argmax(r['gate_pipette_held'])), 'left': int(np.argmax(r['gate_tube_held']))}
        self.dr_vec = torch.zeros(self.num_envs, 7+4*self.nu, device=dev)

    def set_dr(self, dr, seed=12345):
        """dr keys, all optional, ranges [lo, hi] are factors unless stated: friction_hand_object, friction_furniture, friction_pipette_tube, object_mass, kp_g1, kv_g1, kp_hand, kv_hand (per joint), torque_limit (per joint),
        joint_offset_g1 / joint_offset_hand (rad, +- per joint per episode: encoder zero error, ramped in over offset_ramp_s from a reset), cmd_skew_s (hand command vs arm command, +- s per episode), action_delay_steps [lo, hi] (control steps on the residual only),
        obs_q_std_g1, obs_q_std_hand, obs_v_std_g1 (actor observation noise), object_start_xy_mm, object_start_yaw_deg (object in its rack / stand before its first grasp), base_xy_mm (robot base vs world, +- per axis, ramped), arm_gains {kp: [nu], kv: [nu]} (absolute gains on the waist and arm actuators, before the factors)."""
        if not self.dr_on: raise RuntimeError('env built without dr')
        for k, v in self.mf.items(): v[:] = self.m0[k]
        self.dr = dict(dr); self.gen_dr.manual_seed(seed); self.joff.zero_(); self.skew.zero_(); self.adelay.zero_(); self.dr_vec.zero_(); self.base_off = None
        self.offset_ramp = float(dr.get('offset_ramp_s', .5)); self.obs_noise = {k: dr[k] for k in ('obs_q_std_g1', 'obs_q_std_hand', 'obs_v_std_g1') if k in dr}
        if 'action_delay_steps' in dr: self.a_buf = torch.zeros(int(dr['action_delay_steps'][1])+1, self.num_envs, self.nu, device=self.device)
        else: self.a_buf = None
        if 'base_xy_mm' in dr: self.base_off = torch.zeros(self.num_envs, 2, device=self.device)
        if 'arm_gains' in dr:
            kp, kv = torch.as_tensor(dr['arm_gains']['kp'], dtype=torch.float32, device=self.device), torch.as_tensor(dr['arm_gains']['kv'], dtype=torch.float32, device=self.device); g0, b0 = self.m0['actuator_gainprm'], self.m0['actuator_biasprm']
            g0[self.arm_ids, 0] = kp; b0[self.arm_ids, 1] = -kp; b0[self.arm_ids, 2] = -kv; self.mf['actuator_gainprm'][:] = g0; self.mf['actuator_biasprm'][:] = b0

    def _uni(self, shape, rng):
        lo, hi = rng; return lo+(hi-lo)*torch.rand(shape, device=self.device, generator=self.gen_dr)

    def _sample_dr(self, ids, frames):
        d = self.dr; n = len(ids); nu = self.nu; mf, m0 = self.mf, self.m0; vec = self.dr_vec
        if 'friction_hand_object' in d or 'friction_pipette_tube' in d:
            pf = m0['pair_friction'][None].repeat(n, 1, 1)
            if 'friction_hand_object' in d: x = self._uni((n,), d['friction_hand_object']); pf[:, self.pair_ho] *= x[:, None, None]; vec[ids, 0] = x
            if 'friction_pipette_tube' in d: x = self._uni((n,), d['friction_pipette_tube']); pf[:, self.pair_pt] *= x[:, None, None]; vec[ids, 2] = x
            mf['pair_friction'][ids] = pf
        if 'friction_furniture' in d:
            gf = m0['geom_friction'][None].repeat(n, 1, 1); x = self._uni((n,), d['friction_furniture']); gf[:, self.geom_furn] *= x[:, None, None]; mf['geom_friction'][ids] = gf; vec[ids, 1] = x
        if 'object_mass' in d:
            for j, (bd, s) in enumerate(((self.pip_bodies, 'right'), (self.tube_bodies, 'left'))):
                x = self._uni((n,), d['object_mass']); vec[ids, 3+j] = x; I = ids[:, None]
                mf['body_mass'][I, bd] = m0['body_mass'][bd][None]*x[:, None]; mf['body_subtreemass'][I, bd] = m0['body_subtreemass'][bd][None]*x[:, None]; mf['body_inertia'][I, bd] = m0['body_inertia'][bd][None]*x[:, None, None]; mf['body_invweight0'][I, bd] = m0['body_invweight0'][bd][None]/x[:, None, None]
                od = self.obj_dofs[s]; mf['dof_invweight0'][I, od] = m0['dof_invweight0'][od][None]/x[:, None]
        if any(k in d for k in ('kp_g1', 'kv_g1', 'kp_hand', 'kv_hand')):
            kp = torch.ones(n, nu, device=self.device); kv = torch.ones(n, nu, device=self.device)
            for key, ix, tgt in (('kp_g1', self.arm_ids, kp), ('kv_g1', self.arm_ids, kv), ('kp_hand', self.fing_ids, kp), ('kv_hand', self.fing_ids, kv)):
                if key in d: tgt[:, ix] = self._uni((n, len(ix)), d[key])
            if 'kv_g1_x_sqrt_kp' in d:   # (11-06 round 3, default absent = unchanged) arm kv factor = sqrt(kp factor) x U(lo, hi): the damping ratio scales
                # by U only (zeta' = zeta0 kv_f / sqrt(kp_f)); lo 0.95 keeps zeta >= 0.70 on the weakest joint (R_arm_j1 nominal 0.74) while kp keeps its full range
                kv[:, self.arm_ids] = torch.sqrt(kp[:, self.arm_ids])*self._uni((n, len(self.arm_ids)), d['kv_g1_x_sqrt_kp'])
            g = m0['actuator_gainprm'][None].repeat(n, 1, 1); b = m0['actuator_biasprm'][None].repeat(n, 1, 1); g[:, :, 0] *= kp; b[:, :, 1] *= kp; b[:, :, 2] *= kv; mf['actuator_gainprm'][ids] = g; mf['actuator_biasprm'][ids] = b; vec[ids, 7:7+nu] = kp; vec[ids, 7+nu:7+2*nu] = kv
        if 'torque_limit' in d:
            x = self._uni((n, nu), d['torque_limit']); mf['actuator_forcerange'][ids] = m0['actuator_forcerange'][None]*x[:, :, None]; jf = m0['jnt_actfrcrange'][None].repeat(n, 1, 1); jf[:, self.act_jid] *= x[:, :, None]; mf['jnt_actfrcrange'][ids] = jf; vec[ids, 7+2*nu:7+3*nu] = x
        if 'joint_offset_g1' in d or 'joint_offset_hand' in d:
            lim = torch.where(self.finger_mask, torch.as_tensor(float(d.get('joint_offset_hand', 0.)), device=self.device), torch.as_tensor(float(d.get('joint_offset_g1', 0.)), device=self.device)); x = lim*(2*torch.rand(n, nu, device=self.device, generator=self.gen_dr)-1); self.joff[ids] = x; vec[ids, 7+3*nu:7+4*nu] = x
        if 'cmd_skew_s' in d:
            x = float(d['cmd_skew_s'])*(2*torch.rand(n, device=self.device, generator=self.gen_dr)-1); self.skew[ids] = torch.round(x/self.dt).long(); vec[ids, 5] = x
        if 'action_delay_steps' in d:
            lo, hi = d['action_delay_steps']; x = torch.randint(int(lo), int(hi)+1, (n,), device=self.device, generator=self.gen_dr); self.adelay[ids] = x; vec[ids, 6] = x.float(); self.a_buf[:, ids] = 0
        if 'base_xy_mm' in d:
            self.base_off[ids] = float(d['base_xy_mm'])*1e-3*(2*torch.rand(n, 2, device=self.device, generator=self.gen_dr)-1); mf['body_pos'][ids, self.pelvis] = m0['body_pos'][self.pelvis]
        if 'object_start_xy_mm' in d:
            for s in SIDES:
                sel = frames < self.f_pre[s]
                if not sel.any(): continue
                w = ids[sel]; k = len(w); a = self.obj_q[s]; r_, th = float(d['object_start_xy_mm'])*1e-3*torch.sqrt(torch.rand(k, device=self.device, generator=self.gen_dr)), 2*math.pi*torch.rand(k, device=self.device, generator=self.gen_dr)
                self.qpos[w, a] += r_*torch.cos(th); self.qpos[w, a+1] += r_*torch.sin(th); yaw = math.radians(float(d.get('object_start_yaw_deg', 0.)))*(2*torch.rand(k, device=self.device, generator=self.gen_dr)-1)
                c, sn = torch.cos(yaw/2), torch.sin(yaw/2); q = self.qpos[w, a+3:a+7].clone(); self.qpos[w, a+3] = c*q[:, 0]-sn*q[:, 3]; self.qpos[w, a+4] = c*q[:, 1]-sn*q[:, 2]; self.qpos[w, a+5] = c*q[:, 2]+sn*q[:, 1]; self.qpos[w, a+6] = c*q[:, 3]+sn*q[:, 0]

    def _offset_weight(self, extra_s=0.):
        return (((self.frame-self.start).float()*.01+extra_s)/max(self.offset_ramp, 1e-6)).clamp(0, 1)

    # ------------------------------------------------------------------ (11-05) the deployable actor's observation
    def _actor_init(self):
        H = 3; dev = self.device; N, nu = self.num_envs, self.nu; self.H = H
        self.h_q, self.h_v, self.h_a = (torch.zeros(H, N, nu, device=dev) for _ in range(3)); self.qm_prev = torch.zeros(N, nu, device=dev)
        self.q_ref_act = self.qpos_ref[:, self.aq]; self.phase_1h = torch.nn.functional.one_hot(self.phase.clamp(min=0), 4).float()
        lay = []; o = 0
        for name, n in [(f'{x}[t-{i}]', nu) for i in range(H) for x in ('q_measured', 'qdot_measured', 'prev_action')]+[(f'{x}[t+{c}]', nu) for c in OFFSETS for x in ('q_reference', 'base_command')]+[('plunger_reference_m[t+%d]' % c, 1) for c in OFFSETS]+[('phase_onehot', 4), ('clip_progress', 1)]:
            lay.append({'name': name, 'start': o, 'size': n}); o += n
        self.actor_layout = lay; self.actor_dim = o

    def _observe_deployable(self):
        H, N, nu = self.H, self.num_envs, self.nu; k = self.frame.clamp(max=self.nframes-1); last = self.nframes-1; fresh = self.episode_length_buf == 0
        q, v = self.qpos[:, self.aq], self.qvel[:, self.av]; qm = q+(self.joff*self._offset_weight()[:, None] if self.dr_on else 0.)
        if 'obs_q_std_g1' in self.obs_noise or 'obs_q_std_hand' in self.obs_noise:
            sq = torch.where(self.finger_mask, torch.as_tensor(float(self.obs_noise.get('obs_q_std_hand', 0.)), device=self.device), torch.as_tensor(float(self.obs_noise.get('obs_q_std_g1', 0.)), device=self.device)); qm = qm+sq*torch.randn(N, nu, device=self.device, generator=self.gen_dr)
        vm = v.clone()
        if 'obs_v_std_g1' in self.obs_noise: vm = vm+float(self.obs_noise['obs_v_std_g1'])*torch.randn(N, nu, device=self.device, generator=self.gen_dr)
        fd = (qm-self.qm_prev)/CONTROL_DT; fd[fresh] = 0.; vm[:, self.finger_mask] = fd[:, self.finger_mask]; self.qm_prev = qm.clone() if isinstance(qm, torch.Tensor) else qm   # Wuji drivers read position only: first difference at the control rate
        self.h_q = torch.roll(self.h_q, 1, 0); self.h_v = torch.roll(self.h_v, 1, 0); self.h_a = torch.roll(self.h_a, 1, 0); self.h_q[0] = qm; self.h_v[0] = vm; self.h_a[0] = self.prev_a
        self.h_q[:, fresh] = qm[fresh]; self.h_v[:, fresh] = vm[fresh]; self.h_a[:, fresh] = 0.
        parts = [x[i] for i in range(H) for x in (self.h_q, self.h_v, self.h_a)]
        for o in OFFSETS:
            j = (k+o*self.frame_step).clamp(max=last); parts += [self.q_ref_act[j], self.ctrl_frame[j]]
        parts += [self.plunger_ref[(k+o*self.frame_step).clamp(max=last), None] for o in OFFSETS]; parts += [self.phase_1h[k], (k.float()/last)[:, None]]
        parts += X.student_parts(self)
        self.obs_actor = torch.nan_to_num(torch.cat(parts, 1), 0., 0., 0.).clamp(-1e3, 1e3)
        self.obs_critic = torch.cat([self.obs, self.dr_vec], 1) if self.dr_on else self.obs

    @staticmethod
    def reference_torques(m, r):
        """Per frame and actuator: M(q_ref) qdd_ref + c(q_ref, qd_ref) - g(q_ref) at the actuated dofs (mj_rne; qdd by central differences)."""
        import mujoco
        d = mujoco.MjData(m); qacc = np.gradient(r['qvel_ref'], r['timestamps'], axis=0); dof = m.jnt_dofadr[m.actuator_trnid[:, 0]]
        out = np.zeros((len(r['timestamps']), m.nu)); full, grav = np.zeros(m.nv), np.zeros(m.nv)
        for k in range(len(out)):
            d.qpos[:] = r['qpos_ref'][k]; d.qvel[:] = r['qvel_ref'][k]; mujoco.mj_forward(m, d); d.qacc[:] = qacc[k]; mujoco.mj_rne(m, d, 1, full)
            d.qvel[:] = 0; mujoco.mj_forward(m, d); d.qacc[:] = 0; mujoco.mj_rne(m, d, 0, grav)
            out[k] = (full-grav)[dof]
        return out

    # ------------------------------------------------------------------ simulation
    def _capture(self):
        wp, mjw = self.wp, self.mjw
        with wp.ScopedDevice(self.wpdev):
            # warm-up from states spread over the window (all worlds at one state can need more broadphase room than the
            # training distribution: ~361 pairs / world with the pipette seated); the state is reset afterwards
            fr = self.valid_starts[torch.arange(self.num_envs, device=self.device) % len(self.valid_starts)]
            fr = fr[torch.randperm(len(fr), device=self.device, generator=self.gen)] if hasattr(self, 'gen') else fr
            self.qpos[:] = self.qpos_ref[fr]; self.qvel[:] = self.qvel_ref[fr]; self.ctrl[:] = self.cmd_sim[fr*self.per_frame]
            for _ in range(2): mjw.step(self.m, self.d); mjw.kinematics(self.m, self.d)
            wp.synchronize()
            with wp.ScopedCapture() as c: mjw.step(self.m, self.d)
            self.g_step = c.graph
            with wp.ScopedCapture() as c: mjw.kinematics(self.m, self.d)
            self.g_kin = c.graph

    def _launch(self, g):
        self.wp.capture_launch(g)

    def _fold_stats(self):
        s = torch.stack([self.ncol[0].long(), self.nacon[0].long(), self.nefc.max().long()])
        torch.maximum(self.stats, s, out=self.stats)

    def check_overflow(self):
        st = self.stats.cpu(); lim = self.limits.cpu()
        torch.maximum(self.stats_peak, self.stats, out=self.stats_peak); self.stats.zero_()
        if self.guard and (st > lim).any():
            raise RuntimeError(f'MuJoCo Warp buffer overflow: ncollision {int(st[0])} / nacon {int(st[1])} (naconmax {self.naconmax}), '
                               f'nefc per world {int(st[2])} (njmax {self.njmax}); contacts were dropped - resize gpu_sizes.json')
        if self.ovf is not None and self.guard and bool((self.ovf & self.ovf_hard).any()):
            raise RuntimeError(f'MuJoCo Warp Data.overflow hard bits in {int(((self.ovf & self.ovf_hard) != 0).sum())} worlds (largest mask {hex(int(self.ovf.max()))}); contacts or rows were dropped - resize gpu_sizes.json')
        return st

    def overflow_report(self):
        """(11-05) Worlds with each Data.overflow bit set so far (sticky per world; head stack only, None on the pin), hard bits as g1_pipette_sim/physics/mjw_head.py."""
        if self.ovf is None: return None
        from g1_pipette_sim.physics.mjw_head import overflow_report
        return overflow_report(self.d, self.wp)

    def reset_envs(self, mask, frames=None):
        ids = torch.nonzero(mask).squeeze(-1)
        if not len(ids): return ids
        if frames is None:
            frames = self.valid_starts[torch.randint(len(self.valid_starts), (len(ids),), device=self.device, generator=self.gen)]
        self.frame[ids] = frames; self.start[ids] = frames; self.episode_length_buf[ids] = 0
        self.qpos[ids] = self.qpos_ref[frames]; self.qvel[ids] = self.qvel_ref[frames]; self.qacc_ws[ids] = 0
        self.ctrl[ids] = self.cmd_sim[(frames*self.per_frame).clamp(max=len(self.cmd_sim)-1)]
        self.prev[ids] = 0; self.prev2[ids] = 0; self.res[ids] = 0
        self.prev_a[ids] = 0
        if self.dr is not None: self._sample_dr(ids, frames)
        X.reset(self, ids)
        X.holder_reset(self, ids, frames); X.gate_reset(self, ids)
        X.measure(self, ids, frames)
        return ids

    def step(self, actions):
        a = torch.nan_to_num(actions.to(self.device), 0.).clamp(-1, 1)
        r = a*self.scale
        if self.thumb_press_scale is not None:
            inp = self.press[self.frame.clamp(max=self.nframes-1)]; r[:, self.rthumb] = torch.where(inp[:, None], r[:, self.rthumb]*self.thumb_press_scale, r[:, self.rthumb])
        r = X.smooth_residual(self, r)
        self.res = r; self.prev_a = a
        rapp = r
        if self.dr is not None and self.a_buf is not None:   # (11-05) residual latency in control steps
            self.a_buf = torch.roll(self.a_buf, 1, 0); self.a_buf[0] = r; rapp = self.a_buf[self.adelay, self.world]
        off = self.joff*self._offset_weight(self.substeps*self.dt/2)[:, None] if self.dr is not None else None
        if self.base_off is not None: self.mf['body_pos'][:, self.pelvis, :2] = self.m0['body_pos'][self.pelvis, :2]+self.base_off*self._offset_weight(self.substeps*self.dt)[:, None]
        skew = self.dr is not None and 'cmd_skew_s' in self.dr
        X.holder_step(self, self.frame.clamp(max=self.nframes-1))
        base = self.frame*self.per_frame; X.begin_step(self, rapp, base)
        for s in range(self.substeps):
            self.x_sub = s
            if skew: cmd = self.cmd_sim[(base[:, None]+s-self.skew[:, None]*self.finger_mask).clamp(0, len(self.cmd_sim)-1), self.t_nu]+rapp
            else: cmd = self.cmd_sim[(base+s).clamp(max=len(self.cmd_sim)-1)]+rapp
            if off is not None: cmd = cmd-off
            torch.clamp(X.arm_command(self, cmd), self.lo, self.hi, out=self.ctrl)
            self._launch(self.g_step); self._fold_stats()
            if self.substep_hook is not None: self.substep_hook(s)
        self._launch(self.g_kin)
        self.frame += X.frame_advance(self); self.episode_length_buf += 1   # (11-06) frame_step, or less while a release / closure gate holds the reference clock
        st = self.check_overflow()
        rew, fail, reasons, info = self._reward_and_termination(r)
        end = self.frame >= self.nframes-1
        if not self.terminate: fail = fail & end
        success = end & ~fail
        done = fail | success
        self._accumulate(done, success, reasons)
        self.prev2 = self.prev.clone(); self.prev = r.clone()
        ids = self.reset_envs(done) if self.auto_reset else done.nonzero().squeeze(-1)[:0]
        if len(ids): self._launch(self.g_kin)
        self.obs = self._observe()
        if self.obs_mode == 'deployable': self._observe_deployable()
        extras = {'time_outs': torch.zeros_like(done), 'log': {}, 'success': success, 'fail': fail, 'reasons': reasons, 'info': info, 'stats': st}
        return self.obs, rew, done, extras

    # ------------------------------------------------------------------ task terms
    def _poses(self, s):
        W = torch.eye(4, device=self.device).repeat(self.num_envs, 1, 1); O = W.clone()
        W[:, :3, :3] = self.xmat[:, self.wrist_id[s]]; W[:, :3, 3] = self.xpos[:, self.wrist_id[s]]
        O[:, :3, :3] = self.xmat[:, self.obj_id[s]]; O[:, :3, 3] = self.xpos[:, self.obj_id[s]]
        return W, O

    def _reward_and_termination(self, r):
        k = self.frame.clamp(max=self.nframes-1); reward = torch.zeros(self.num_envs, device=self.device)
        reasons = torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)   # per object: 0 = ok, 1 + index into FAILURES
        q = self.qpos[:, self.aq]; info = {}
        finite = torch.isfinite(self.qpos).all(1) & torch.isfinite(self.qvel).all(1)
        for j, s in enumerate(SIDES):
            W, O = self._poses(s); Wi = rinv(W)
            local = Wi@O
            act_axis = self.axis@local[:, :3, :3].transpose(1, 2)+local[:, None, :3, 3]
            axis_err = (act_axis-self.rel_axis_ref[s][k]).norm(dim=-1)
            rel = self.rel_ref[s][k]
            pos = (local[:, :3, 3]-rel[:, :3, 3]).norm(dim=-1); ang = rot_angle(local[:, :3, :3], rel[:, :3, :3])
            Or, Wr = X.ref_o(self, s, k), X.ref_w(self, s, k)
            wpos = (O[:, :3, 3]-Or[:, :3, 3]).norm(dim=-1); wang = rot_angle(O[:, :3, :3], Or[:, :3, :3])
            wp_, wr_ = (W[:, :3, 3]-Wr[:, :3, 3]).norm(dim=-1), rot_angle(W[:, :3, :3], Wr[:, :3, :3])
            links = (self.xpos[:, self.link_ids[s]]-W[:, None, :3, 3])@W[:, :3, :3]
            le = (links-self.L_ref[s][k]).norm(dim=-1)
            je = (q[:, self.fing_t[s]]-self.qfing_ref[s][k])/self.jrange[s]
            obj_t = torch.exp(-(axis_err.mean(1)/.04)**2); link_t = torch.exp(-(le/.025)**2).mean(1); joint_t = torch.exp(-(je/.1)**2).mean(1)
            wrist_t = 2*torch.exp(-(wp_/.03)**2-(wr_/.15)**2); world_t = 2*torch.exp(-(wpos/.04)**2-(wang/.3)**2)
            reward += 8*obj_t+link_t+joint_t+wrist_t+world_t
            v = self.qvel[:, self.obj_v[s]:self.obj_v[s]+6]
            conds = torch.stack([pos > .05, ang > math.pi/4, axis_err.max(1).values > .05, wpos > .05, wang > math.pi/4,
                                 O[:, 2, 3] < .06, v[:, :3].norm(dim=-1) > 10, v[:, 3:].norm(dim=-1) > 500, ~finite], 1)
            any_ = conds.any(1); first = torch.argmax(conds.int(), 1)
            reasons[:, j] = torch.where(any_, first+1, 0)
            info[s] = dict(object_position_m=wpos, object_orientation_rad=wang, relative_position_m=pos, axis_mean_m=axis_err.mean(1),
                           axis_max_m=axis_err.max(1).values, wrist_position_m=wp_, wrist_orientation_rad=wr_, link_rmse_m=le.pow(2).mean(1).sqrt(),
                           terms=dict(object=obj_t, link=link_t, joint=joint_t, wrist=wrist_t, world=world_t))
        smooth = (r-self.prev).pow(2).sum(1)+(r-2*self.prev+self.prev2).pow(2).sum(1)
        depth = -self.qpos[:, self.pl_q]; dref = self.plunger_ref[k]
        pl_t = torch.where(self.press[k], self.cfg['plunger_weight']*torch.exp(-((depth-dref)/self.cfg['plunger_sigma'])**2), torch.zeros_like(depth))
        reward = reward-.01*smooth+pl_t
        ex_, xi_ = X.reward(self, r, k); reward = reward+ex_; info['extras'] = xi_
        info['reward_unscaled'] = reward
        info['plunger_depth_m'] = depth; info['plunger_ref_m'] = dref; info['plunger_term'] = pl_t; info['smooth'] = smooth
        fail = (reasons > 0).any(1)
        reward = torch.nan_to_num(reward, 0., 0., 0.)
        if self.reward_scale != 1.: reward = reward*self.reward_scale
        return reward, fail, reasons, info

    def _observe(self):
        k = self.frame.clamp(max=self.nframes-1); last = self.nframes-1
        q = self.qpos[:, self.aq]; v = self.qvel[:, self.av]
        parts = [q, v, self.prev]
        poses = {s: self._poses(s) for s in SIDES}
        for s in SIDES:
            W, O = poses[s]; local = rinv(W)@O
            parts.append((self.axis@local[:, :3, :3].transpose(1, 2)+local[:, None, :3, 3]).flatten(1))
        parts += [-self.qpos[:, self.pl_q, None], -self.qvel[:, self.pl_v, None]]
        for o in OFFSETS:
            j = (k+o*self.frame_step).clamp(max=last)
            parts.append(self.ctrl_frame[j])
            for s in SIDES:
                W = poses[s][0]
                parts += [self.rel_axis_ref[s][j].flatten(1), self.L_ref[s][j].flatten(1)]
                rel = rinv(W)@X.ref_w(self, s, j)
                parts += [rel[:, :3, 3], rel[:, :3, :2].flatten(1)]
            parts.append(self.plunger_ref[j, None])
        return torch.nan_to_num(torch.cat(parts, 1), 0., 0., 0.).clamp(-1e3, 1e3)

    # ------------------------------------------------------------------ episode statistics
    def _new_stats(self):
        return dict(episodes=0, success=0, length_steps=0, reasons=torch.zeros(2, len(FAILURES)+1, dtype=torch.long, device=self.device))

    def _accumulate(self, done, success, reasons):
        st = self.ep_stats
        st['episodes'] = st['episodes']+done.sum(); st['success'] = st['success']+success.sum()
        st['length_steps'] = st['length_steps']+(self.episode_length_buf*done).sum()
        for j in range(2): st['reasons'][j] += torch.bincount(reasons[:, j][done], minlength=len(FAILURES)+1)

    def pop_stats(self):
        st = self.ep_stats; self.ep_stats = self._new_stats(); n = int(st['episodes'])
        out = {'episodes': n, 'success_fraction': float(st['success'])/max(n, 1), 'mean_episode_s': float(st['length_steps'])*CONTROL_DT/max(n, 1)}
        R = st['reasons'].cpu().numpy()
        for j, s in enumerate(('pipette', 'tube')):
            out[f'fail_{s}'] = {FAILURES[i-1]: round(float(R[j, i])/max(n, 1), 4) for i in range(1, R.shape[1]) if R[j, i]}
        return out

    # ------------------------------------------------------------------ rsl_rl VecEnv interface
    def get_observations(self):
        from tensordict import TensorDict
        if self.obs_mode == 'deployable': return TensorDict({'policy': self.obs_actor, 'critic': self.obs_critic}, batch_size=[self.num_envs])
        return TensorDict({'policy': self.obs}, batch_size=[self.num_envs])
