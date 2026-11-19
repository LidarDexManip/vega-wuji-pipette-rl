"""(11-17) Replace a task's reference by its nominal PHYSICAL rollout: zero residual plus the regrasp knots of opt_regrasp.py, nominal conditions
(no DR, the pipette at its nominal place, no pushes), from frame 0 to the end.
  qpos_ref      the rollout (control steps interpolated to the 100 Hz frames; free-joint quaternions normalised), qvel_ref by finite differences
  wrist_targets the right wrist / pipette poses of the new qpos_ref
  ctrl_sim      corrected so that the env's command (ctrl_sim + its arm lead / feed-forward, which depend on qvel_ref) equals the command the
                rollout executed (the old command + the applied residual); controls (100 Hz, observation) = old + the applied residual
Zero residual on the new task replays the rollout (checked at the end).
  cd rl; python -I ref_from_rollout.py CONFIG.json OPT.npz DST_TASK"""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np, torch, mujoco
HERE = Path(__file__).resolve().parent; import os as _os
ROOT = Path(_os.environ.get('G1PS_ROOT', str(HERE.parent/'deps_from_g1ps'))); sys.path[:0] = [str(HERE), str(HERE.parent/'tracker'), str(ROOT/'tools/rl')]
import vega_res_env as E, build_kin_task as BK

p = argparse.ArgumentParser(); p.add_argument('config', type=Path); p.add_argument('opt', type=Path); p.add_argument('dst', type=Path); p.add_argument('--dr-center', action='store_true'); a0 = p.parse_args()
C = json.loads(a0.config.read_text()); rel = lambda x: (Path(x) if Path(x).is_absolute() else (a0.config.parent/x)).resolve(); src = rel(C['task'])
ex = {k: v for k, v in C.get('extras', {}).items() if k != 'perturb'}
if ex.get('holder'): ex['holder'] = {**ex['holder'], 'stand_xy_mm': 0, 'stand_yaw_deg': 0, 'meas_xy_mm': 0, 'meas_yaw_deg': 0}
dr = None
if a0.dr_center:
    d0 = json.loads(rel(C['dr']).read_text()) if isinstance(C.get('dr'), str) else C.get('dr')
    dr = {k: ([.5*(v[0]+v[1])]*2 if isinstance(v, (list, tuple)) and len(v) == 2 else 0.) for k, v in d0.items() if not k.endswith('_frac') and not isinstance(v, dict)}   # (11-19) dict entries (obs_stale) only touch observations: off
kw = dict(dr=dr, extras=ex, reward_scale=C.get('reward_scale', .01))
for k in ('group_scale', 'thumb_press_scale', 'plunger_weight'):
    if C.get(k) is not None: kw[k] = C[k]
mk = lambda task, n=2: E.BimanualGpuEnv(task, n, seed=123, nconmax=200, njmax=1600, naconmax_per_world=400, terminate=False, auto_reset=False, starts=np.array([0]), **kw)
Z = np.load(a0.opt); Kn = torch.as_tensor(Z['knots']).float(); sel = Z['sel']; k_a, k_h, k_r = int(Z['k_a']), int(Z['k_h']), int(Z['k_r']); K = Kn.shape[0]

def residual(k, dev):
    kt = [k_a+(j+1)*(k_h-k_a)/K for j in range(K)]
    if k <= k_a: return torch.zeros(len(sel), device=dev)
    if k <= k_h:
        x = [float(k_a)]+kt; v = torch.cat([torch.zeros(1, len(sel)), Kn], 0).to(dev); j = max(1, min(K, int(np.searchsorted(x, k))))
        w = (k-x[j-1])/(x[j]-x[j-1]); return (1-w)*v[j-1]+w*v[j]
    w = max(0., 1.-(k-k_h)/max(1, k_r)); return (w*Kn[-1]).to(dev)

@torch.inference_mode()
def run(env, with_knots):
    """all worlds; returns qpos (steps+1, N, nq), frames, step base frames, applied residual (world 0), end in-hand error vs the env's reference end"""
    N = env.num_envs; dev = env.device
    env.reset_envs(torch.ones(N, dtype=torch.bool, device=dev), frames=torch.zeros(N, dtype=torch.long, device=dev)); env._launch(env.g_kin); env.obs = env._observe()
    Q = [env.qpos.cpu().numpy().copy()]; F = [0]; B = []; RES = []
    while int(env.frame[0]) < env.nframes-1:
        k = int(env.frame[0]); act = torch.zeros(N, env.nu, device=dev)
        if with_knots: act[:, torch.as_tensor(sel, device=dev)] = residual(k, dev).clamp(-1, 1)[None]
        env.step(act); B.append(k); RES.append(env.res[0].cpu().numpy().copy()); Q.append(env.qpos.cpu().numpy().copy()); F.append(int(env.frame[0]))
    W, O = env._poses('right'); L = torch.linalg.inv(W)@O; R = env.rel_ref['right'][env.nframes-1]
    ep = ((L[:, :3, 3]-R[:3, 3]).norm(dim=-1)*1e3).cpu().numpy()
    ea = torch.rad2deg(torch.arccos((((L[:, :3, :3].transpose(1, 2)@R[None, :3, :3]).diagonal(dim1=1, dim2=2).sum(-1)-1)/2).clamp(-1, 1))).cpu().numpy()
    return np.array(Q), np.array(F), np.array(B), np.array(RES), ep, ea

envA = mk(src, 64); QA, F, Bk, RES, epA, eaA = run(envA, True); wbest = int(np.argmin(epA/3+eaA/3)); Q = QA[:, wbest]
print(f'64 nominal copies: final grip (<5 mm, <5 deg) in {100*np.mean((epA < 5) & (eaA < 5)):.0f}%; reference from world {wbest}: {epA[wbest]:.1f} mm {eaA[wbest]:.1f} deg', flush=True); cmd_old = envA.cmd_sim.cpu().numpy(); per, sub = envA.per_frame, envA.substeps
print(f'rollout: {len(Bk)} control steps, frames 0..{F[-1]}; residual max {np.abs(RES).max():.3f} rad', flush=True)
cmd_des = cmd_old.copy()
for b, rr in zip(Bk, RES):
    for s in range(sub):
        i = b*per+s
        if i < len(cmd_des): cmd_des[i] += rr
r = dict(np.load(src/'reference.npz')); nfr = len(r['timestamps']); m = mujoco.MjModel.from_xml_path(str(src/'scene.xml'))
# qpos at every frame: linear between the control-step states, free / ball joint quaternions normalised
Qn = np.stack([np.interp(np.arange(nfr), F, Q[:, j]) for j in range(Q.shape[1])], 1)
for j in range(m.njnt):
    t = m.jnt_type[j]; a_ = int(m.jnt_qposadr[j])
    if t in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
        o = a_+3 if t == mujoco.mjtJoint.mjJNT_FREE else a_; Qn[:, o:o+4] /= np.linalg.norm(Qn[:, o:o+4], axis=1, keepdims=True)
r['qpos_ref'] = Qn.astype(np.float32); r['qvel_ref'] = BK.qvel_from_qpos(m, Qn, .01)
ctrl_frame_res = np.zeros_like(r['controls'])
for b, rr in zip(Bk, RES): ctrl_frame_res[b:b+envA.frame_step] = rr
r['controls'] = r['controls']+ctrl_frame_res
if a0.dst.exists(): shutil.rmtree(a0.dst)
shutil.copytree(src, a0.dst, ignore=shutil.ignore_patterns('contact_ref*', 'valid_starts*', 'audit*'))
np.savez_compressed(a0.dst/'reference.npz', **r)
envB = mk(a0.dst); lead_new = envB.cmd_sim.cpu().numpy()-r['ctrl_sim']; r['ctrl_sim'] = (cmd_des-lead_new).astype(r['ctrl_sim'].dtype)
np.savez_compressed(a0.dst/'reference.npz', **r); del envB
d = mujoco.MjData(m); wt = dict(np.load(src/'wrist_targets.npz')); Wr = np.zeros((nfr, 4, 4)); Pr = np.zeros((nfr, 4, 4)); wb, ob = m.body('right_wrist').id, m.body('right_object').id
for k in range(nfr):
    d.qpos[:] = Qn[k]; mujoco.mj_kinematics(m, d)
    for T_, b in ((Wr, wb), (Pr, ob)): T_[k] = np.eye(4); T_[k][:3, :3] = d.xmat[b].reshape(3, 3); T_[k][:3, 3] = d.xpos[b]
wt['right'] = Wr; wt['pipette'] = Pr; np.savez_compressed(a0.dst/'wrist_targets.npz', **wt)
tj = json.loads((a0.dst/'task.json').read_text()); tj['physical_reference'] = {'from': str(src), 'opt': str(a0.opt), 'regrasp_frames': [k_a, k_h], 'note': '(11-17) qpos_ref = nominal physical rollout (rl/ref_from_rollout.py)'}
(a0.dst/'task.json').write_text(json.dumps(tj, indent=1, default=float)+'\n')
# check: zero residual on the new task
envC = mk(a0.dst, 64); Q2, F2, _, _, ep2, ea2 = run(envC, False); ob_q = int(m.jnt_qposadr[m.joint('right_object_joint').id])
n_ = min(len(Q2), len(Q)); dp = np.linalg.norm(Q2[:n_, :, ob_q:ob_q+3]-Q[:n_, None, ob_q:ob_q+3], axis=2)*1e3
print(f'zero-residual replay of {a0.dst.name}, 64 nominal copies: end within 5 mm / 5 deg of the reference end in {100*np.mean((ep2 < 5) & (ea2 < 5)):.0f}% '
      f'(ep p50 {np.median(ep2):.1f} mm, ea p50 {np.median(ea2):.1f} deg); pipette path deviation p50 of the max {np.median(dp.max(0)):.1f} mm', flush=True)
