"""TRACKER Stage 0 env (11-06): rrline's BimanualGpuEnv (imported, NOT edited) + two per-world additions. Layer: RL ENV (MuJoCo Warp).

  * VOC (virtual object controller, as h13_work/vega/common/vega_env.py _voc): a PD wrench on each object toward its holder-adjusted
    reference pose (X.ref_o), times a per-world scale; a share of the worlds (voc_free_frac) is kept at 0 for a clean readout. The
    curriculum (train_tracker.py) lowers the scale on success.
  * time-warp: per-world reference playback speed s ~ U(warp_lo, warp_hi) drawn at every reset. The 100 Hz reference clock advances by
    frame_step * s per control step (fractional accumulator, integer frames); within a step the sim-rate command row advances by s per
    physics step; the arm lead (kv/kp qdot_ref + inverse dynamics / kp) is scaled by s; the RSI start velocity by s.
Defaults (voc_scale 0, warp 1..1) reproduce the parent env exactly, so the copied eval script judges at VOC 0, real time.

Source of the parent (sha256[:16] at copy time 11-06): vega_res_env.py cfcd18203c42e7c7, vega_extras.py 6acb0be996af279d.
"""
import math, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; RR = HERE.parent/'rl'
sys.path.insert(0, str(RR))
import vega_res_env as VR
import vega_extras as X

SIDES = VR.SIDES


def _rotvec(R):   # (n, 3, 3) -> rotation vector (n, 3)
    c = ((R.diagonal(dim1=1, dim2=2).sum(-1)-1)/2).clamp(-1, 1); th = torch.arccos(c)
    v = torch.stack([R[:, 2, 1]-R[:, 1, 2], R[:, 0, 2]-R[:, 2, 0], R[:, 1, 0]-R[:, 0, 1]], -1)
    s = torch.sin(th); k = torch.where(s.abs() > 1e-6, th/(2*s), torch.full_like(th, .5))
    return v*k[:, None]


class BimanualGpuEnv(VR.BimanualGpuEnv):
    def __init__(self, task, num_envs, *args, voc_scale=0., voc_kp=400., voc_kd=40., voc_free_frac=0., warp=(1., 1.), idle_finger_mask=False, idle_speed_deg_s=.3, idle_min_s=.3, press_progress_w=0., **kw):
        self._pp_w = float(press_progress_w)
        self._idle_on = bool(idle_finger_mask); self._idle_speed = float(idle_speed_deg_s); self._idle_min = float(idle_min_s)
        self._warp_lo, self._warp_hi = float(warp[0]), float(warp[1]); self._voc_init = float(voc_scale)
        self._voc_kp, self._voc_kd, self._voc_free_frac = float(voc_kp), float(voc_kd), float(voc_free_frac)
        self._tw_ready = False
        super().__init__(task, num_envs, *args, **kw)
        dev = self.device; N = self.num_envs; m = self.mjm
        r = np.load(Path(task)/'reference.npz')
        self.ctrl_sim_t = torch.as_tensor(r['ctrl_sim'], dtype=torch.float32, device=dev); self.lead_t = self.cmd_sim-self.ctrl_sim_t
        self.tw = torch.ones(N, device=dev); self.tacc = torch.zeros(N, device=dev); self._tw_ready = True
        self.xfrc = self.wp.to_torch(self.d.xfrc_applied)
        self.voc_scale = torch.full((N,), self._voc_init, device=dev); self.voc_free = torch.zeros(N, dtype=torch.bool, device=dev)
        nfree = int(round(self._voc_free_frac*N)); self.voc_free[:nfree] = True; self.voc_scale[self.voc_free] = 0.
        self.obj_mass = {s: float(m.body_subtreemass[self.obj_id[s]]) for s in SIDES}
        self.obj_inert = {s: float(np.mean(m.body_inertia[self.obj_id[s]])) for s in SIDES}
        self.max_episode_length = int(math.ceil((self.nframes-1)/(self.frame_step*min(self._warp_lo, 1.))))+2
        self.cfg['tracker'] = dict(voc_scale=self._voc_init, voc_kp=self._voc_kp, voc_kd=self._voc_kd, voc_free_frac=self._voc_free_frac, warp=[self._warp_lo, self._warp_hi],
                                   idle_finger_mask=self._idle_on, idle_speed_deg_s=self._idle_speed, idle_min_s=self._idle_min)
        # (11-06, default OFF) idle-finger residual mask: zero the finger residual wherever the reference finger joint is still
        # (|qdot_ref| < idle_speed for >= idle_min s): the structural fix for idle-finger motion (coordinator's proposal for real-robot round 4)
        self.idle_tab = None
        if self._idle_on:
            from scipy.ndimage import binary_opening
            jid = m.actuator_trnid[:, 0]; qd = np.degrees(np.abs(r['qvel_ref'][:, m.jnt_dofadr[jid]]))
            fing = np.array([m.actuator(i).name.startswith(('r_', 'l_')) for i in range(m.nu)]); w = max(1, int(round(self._idle_min/.01)))
            tab = np.zeros_like(qd, dtype=bool)
            for i in np.flatnonzero(fing): tab[:, i] = binary_opening(qd[:, i] < self._idle_speed, structure=np.ones(w, bool))
            if 'grasp_latch_mask' in r.files:   # (11-06) never mask a GRASPING finger: its residual carries the squeeze while the object is held
                tab &= ~r['grasp_latch_mask'].astype(bool)
            self.idle_tab = torch.as_tensor(tab, device=dev); self.cfg['tracker']['idle_frac_of_finger_rows'] = round(float(tab[:, fing].mean()), 3)
        if self._warp_lo != 1. or self._warp_hi != 1.:   # draw the first episode's speeds (the parent's __init__ reset ran before tw existed)
            self.tw[:] = self._draw_tw(N); self.qvel[:] = self.qvel*self.tw[:, None]

    # ------------------------------------------------------------------ (11-06, default OFF) dense press-progress reward
    def _reward_and_termination(self, r):
        rew, fail, reasons, info = super()._reward_and_termination(r)
        if self._pp_w > 0:
            # the parent's plunger term exp(-((d - d_ref)/3 mm)^2) gives ~0 reward (no gradient) while the plunger is far from its
            # reference depth; this adds a monotone term in the press window: w * clamp(depth / max(ref, 1 mm), 0, 1.2)
            k = self.frame.clamp(max=self.nframes-1); depth = -self.qpos[:, self.pl_q]; dref = self.plunger_ref[k].clamp(min=1e-3)
            pp = torch.where(self.press[k], self._pp_w*(depth/dref).clamp(0, 1.2), torch.zeros_like(depth))
            info['press_progress'] = pp; rew = rew+(pp*self.reward_scale if self.reward_scale != 1. else pp)
        return rew, fail, reasons, info

    # ------------------------------------------------------------------ curriculum hook
    def set_voc(self, scale):
        self.voc_scale[~self.voc_free] = float(scale)

    def _draw_tw(self, n):
        if self._warp_lo == self._warp_hi: return torch.full((n,), self._warp_lo, device=self.device)
        return self._warp_lo+(self._warp_hi-self._warp_lo)*torch.rand(n, device=self.device, generator=self.gen)

    def reset_envs(self, mask, frames=None):
        ids = super().reset_envs(mask, frames)
        if not getattr(self, '_tw_ready', False) or not len(ids): return ids
        self.tw[ids] = self._draw_tw(len(ids)); self.tacc[ids] = 0.
        self.qvel[ids] = self.qvel[ids]*self.tw[ids, None]
        self.xfrc[ids] = 0.
        return ids

    def _voc_apply(self):
        if not bool((self.voc_scale > 0).any()): return
        k = self.frame.clamp(max=self.nframes-1)
        for s in SIDES:
            O = X.ref_o(self, s, k); b, a, v = self.obj_id[s], self.obj_q[s], self.obj_v[s]
            p = self.qpos[:, a:a+3]; R = self.xmat[:, b]
            lin = self.qvel[:, v:v+3]; w = (R@self.qvel[:, v+3:v+6, None])[..., 0]   # free joint: world linear, local angular velocity
            rv = _rotvec(O[:, :3, :3]@R.transpose(1, 2))
            g = torch.tensor([0., 0., 9.81], device=self.device)
            F = self.obj_mass[s]*(self._voc_kp*(O[:, :3, 3]-p)-self._voc_kd*lin+g)
            Tq = self.obj_inert[s]*(self._voc_kp*rv-self._voc_kd*w)
            self.xfrc[:, b, :3] = self.voc_scale[:, None]*F; self.xfrc[:, b, 3:] = self.voc_scale[:, None]*Tq

    # ------------------------------------------------------------------ step (parent's step with the time-warp and the VOC)
    def step(self, actions):
        if self._warp_lo == 1. and self._warp_hi == 1. and not bool((self.voc_scale > 0).any()) and self.idle_tab is None:
            return super().step(actions)
        a = torch.nan_to_num(actions.to(self.device), 0.).clamp(-1, 1)
        r = a*self.scale
        if self.thumb_press_scale is not None:
            inp = self.press[self.frame.clamp(max=self.nframes-1)]; r[:, self.rthumb] = torch.where(inp[:, None], r[:, self.rthumb]*self.thumb_press_scale, r[:, self.rthumb])
        r = X.smooth_residual(self, r)
        if self.idle_tab is not None: r = torch.where(self.idle_tab[self.frame.clamp(max=self.nframes-1)], torch.zeros_like(r), r)
        self.res = r; self.prev_a = a
        rapp = r
        if self.dr is not None and self.a_buf is not None:
            self.a_buf = torch.roll(self.a_buf, 1, 0); self.a_buf[0] = r; rapp = self.a_buf[self.adelay, self.world]
        off = self.joff*self._offset_weight(self.substeps*self.dt/2)[:, None] if self.dr is not None else None
        if self.base_off is not None: self.mf['body_pos'][:, self.pelvis, :2] = self.m0['body_pos'][self.pelvis, :2]+self.base_off*self._offset_weight(self.substeps*self.dt)[:, None]
        X.holder_step(self, self.frame.clamp(max=self.nframes-1))
        base = self.frame*self.per_frame; X.begin_step(self, rapp, base); X.perturb_step(self)   # (11-12) hand pushes (wrists; VOC forces act on the objects)
        last = len(self.cmd_sim)-1
        for s in range(self.substeps):
            self.x_sub = s
            idx = (base+torch.floor(self.tw*s).long()).clamp(max=last)
            cmd = self.ctrl_sim_t[idx]+self.tw[:, None]*self.lead_t[idx]+rapp
            if off is not None: cmd = cmd-off
            torch.clamp(X.arm_command(self, cmd), self.lo, self.hi, out=self.ctrl)
            self._voc_apply()
            self._launch(self.g_step); self._fold_stats()
            if self.substep_hook is not None: self.substep_hook(s)
        self._launch(self.g_kin)
        adv = self.tacc+self.frame_step*self.tw; inc = torch.floor(adv); self.tacc = adv-inc
        gate_inc = X.frame_advance(self)   # frame_step, or less while a release / closure gate holds the clock (gate off: frame_step)
        inc = inc.long()
        if torch.is_tensor(gate_inc): inc = torch.where(gate_inc < self.frame_step, torch.minimum(inc, gate_inc), inc)
        self.frame += inc; self.episode_length_buf += 1
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


# re-export the parent module's names (CONTROL_DT, OFFSETS, FAILURES, ...) so scripts written for vega_res_env work unchanged
for _n in dir(VR):
    if not _n.startswith('__') and _n not in globals(): globals()[_n] = getattr(VR, _n)


# ---- per-group episode stats for the VOC-free readout worlds (train_tracker.py logs them next to the all-world success)
def _acc(self, done, success, reasons):
    VR.BimanualGpuEnv._accumulate(self, done, success, reasons)
    if not hasattr(self, '_free_done'): self._free_done = torch.zeros((), device=self.device); self._free_succ = torch.zeros((), device=self.device)
    if hasattr(self, 'voc_free'):
        self._free_done = self._free_done+(done & self.voc_free).sum(); self._free_succ = self._free_succ+(success & self.voc_free).sum()   # out-of-place (inference-mode tensors)


def _pop_free(self):
    d = float(getattr(self, '_free_done', torch.zeros(())).item()); s_ = float(getattr(self, '_free_succ', torch.zeros(())).item())
    self._free_done = torch.zeros((), device=self.device); self._free_succ = torch.zeros((), device=self.device)
    return {'episodes': int(d), 'success_fraction': s_/max(d, 1.)}


BimanualGpuEnv._accumulate = _acc; BimanualGpuEnv.pop_free_stats = _pop_free
