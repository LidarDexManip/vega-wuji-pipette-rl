"""(11-17) A task whose reference is ONE recorded rollout (record_pickup.py npz, one world): qpos_ref = that world's states (control steps
interpolated to the 100 Hz frames; quaternions normalised), qvel_ref by finite differences, wrist_targets and pipette_rest from it (the
world's own placement becomes the nominal one). The rollout ran with zero residual, i.e. the source task's commands: ctrl_sim is corrected
(two-pass, as rl/ref_from_rollout.py) so that the env sends exactly those commands again although its arm lead / feed-forward now come from
the new qvel_ref. Check at the end: zero residual on the new task, no DR, nominal placement, 64 copies.
  cd rl; python -I ref_from_record.py CONFIG.json REC.npz WORLD DST_TASK"""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np, torch, mujoco
HERE = Path(__file__).resolve().parent; import os as _os
ROOT = Path(_os.environ.get('G1PS_ROOT', str(HERE.parent/'deps_from_g1ps'))); sys.path[:0] = [str(HERE), str(HERE.parent/'tracker'), str(ROOT/'tools/rl')]
import vega_res_env as E, build_kin_task as BK

p = argparse.ArgumentParser(); p.add_argument('config', type=Path); p.add_argument('rec', type=Path); p.add_argument('world', type=int); p.add_argument('dst', type=Path); a0 = p.parse_args()
C = json.loads(a0.config.read_text()); rel = lambda x: (Path(x) if Path(x).is_absolute() else (a0.config.parent/x)).resolve(); src = rel(C['task'])
ex = {k: v for k, v in C.get('extras', {}).items() if k != 'perturb'}
if ex.get('holder'): ex['holder'] = {**ex['holder'], 'stand_xy_mm': 0, 'stand_yaw_deg': 0, 'meas_xy_mm': 0, 'meas_yaw_deg': 0}
kw = dict(dr=None, extras=ex, reward_scale=C.get('reward_scale', .01))
for k in ('group_scale', 'thumb_press_scale', 'plunger_weight'):
    if C.get(k) is not None: kw[k] = C[k]
mk = lambda task, n=2: E.BimanualGpuEnv(task, n, seed=123, nconmax=200, njmax=1600, naconmax_per_world=400, terminate=False, auto_reset=False, starts=np.array([0]), **kw)
Z = np.load(a0.rec); w = a0.world; Q = Z['qpos'][:, w].astype(float); F = Z['frame'][:, w].astype(int)
assert int(Z['first_fail'][w]) < 0, 'that world failed'
r = dict(np.load(src/'reference.npz')); nfr = len(r['timestamps']); m = mujoco.MjModel.from_xml_path(str(src/'scene.xml'))
keep = np.r_[True, np.diff(F) > 0]; Q, F = Q[keep], F[keep]
Qn = np.stack([np.interp(np.arange(nfr), F, Q[:, j]) for j in range(Q.shape[1])], 1)
for j in range(m.njnt):
    t = m.jnt_type[j]; a_ = int(m.jnt_qposadr[j])
    if t in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
        o = a_+3 if t == mujoco.mjtJoint.mjJNT_FREE else a_; Qn[:, o:o+4] /= np.linalg.norm(Qn[:, o:o+4], axis=1, keepdims=True)
envA = mk(src); cmd_old = envA.cmd_sim.cpu().numpy(); del envA
d = mujoco.MjData(m); wb, ob = m.body('right_wrist').id, m.body('right_object').id; Wr = np.zeros((nfr, 4, 4)); Pr = np.zeros((nfr, 4, 4))
for k in range(nfr):
    d.qpos[:] = Qn[k]; mujoco.mj_kinematics(m, d)
    for T_, b in ((Wr, wb), (Pr, ob)): T_[k] = np.eye(4); T_[k][:3, :3] = d.xmat[b].reshape(3, 3); T_[k][:3, 3] = d.xpos[b]
r['qpos_ref'] = Qn.astype(np.float32); r['qvel_ref'] = BK.qvel_from_qpos(m, Qn, .01); r['pipette_rest'] = Pr[0]
if a0.dst.exists(): shutil.rmtree(a0.dst)
shutil.copytree(src, a0.dst, ignore=shutil.ignore_patterns('contact_ref*', 'valid_starts*', 'audit*'))
np.savez_compressed(a0.dst/'reference.npz', **r)
envB = mk(a0.dst); lead_new = envB.cmd_sim.cpu().numpy()-r['ctrl_sim']; r['ctrl_sim'] = (cmd_old-lead_new).astype(r['ctrl_sim'].dtype); del envB
np.savez_compressed(a0.dst/'reference.npz', **r)
wt = dict(np.load(src/'wrist_targets.npz')); wt['right'] = Wr; wt['pipette'] = Pr; np.savez_compressed(a0.dst/'wrist_targets.npz', **wt)
tj = json.loads((a0.dst/'task.json').read_text()); tj['physical_reference'] = {'from': str(src), 'record': str(a0.rec), 'world': w, 'note': '(11-17) qpos_ref = one recorded zero-residual rollout (rl/ref_from_record.py)'}
(a0.dst/'task.json').write_text(json.dumps(tj, indent=1, default=float)+'\n')

@torch.inference_mode()
def check(env):
    N = env.num_envs; dev = env.device
    env.reset_envs(torch.ones(N, dtype=torch.bool, device=dev), frames=torch.zeros(N, dtype=torch.long, device=dev)); env._launch(env.g_kin); env.obs = env._observe()
    while int(env.frame[0]) < env.nframes-1: env.step(torch.zeros(N, env.nu, device=dev))
    W, O = env._poses('right'); L = torch.linalg.inv(W)@O; R = env.rel_ref['right'][env.nframes-1]
    ep = ((L[:, :3, 3]-R[:3, 3]).norm(dim=-1)*1e3).cpu().numpy()
    ea = torch.rad2deg(torch.arccos((((L[:, :3, :3].transpose(1, 2)@R[None, :3, :3]).diagonal(dim1=1, dim2=2).sum(-1)-1)/2).clamp(-1, 1))).cpu().numpy()
    return ep, ea
ep, ea = check(mk(a0.dst, 64))
print(f'{a0.dst.name} = world {w} of {a0.rec.name}; zero-residual replay, no DR, 64 copies: end within 5 mm / 5 deg of the reference end in {100*np.mean((ep < 5) & (ea < 5)):.0f}% (ep p50 {np.median(ep):.1f} mm, ea p50 {np.median(ea):.1f} deg)', flush=True)
