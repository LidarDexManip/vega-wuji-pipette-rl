"""Pickup task (tracker/build_pickup_task.py: pick the pipette up from the flat table and hold it): teacher / student / zero residual, from the
start of the reference (11-14). Same env, DR, seed (123) and deterministic actor mean as eval_rr.py; pushes off unless --perturb.

Per world:
  end_ok    no trainer failure until the end (pipette > 5 cm / 45 deg off its reference, in the hand or in the world, etc.)
  lifted    the pipette rose >= 8 cm above its rest height at some point
  hold      at the end: pipette vs its reference <= 20 mm and its axis <= 15 deg; pipette in the hand (wrist frame) <= 15 mm / 15 deg
  thumb     at the end: the right thumb tip in the pipette frame <= 15 mm from the reference's (the thumb at the plunger)
  success   end_ok & lifted & hold & thumb
Also: first failure time / reason, the in-hand drift over the final hold, the largest in-hand error over the whole episode.

  cd rl; python -I eval_pickup.py CONFIG.json (--ckpt TEACHER.pt | --student STUDENT.pt | neither = zero residual) [--voc 1] [--n 64] [--out OUT.json]
"""
import argparse, json, math, sys, time
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import numpy as np, torch
HERE = Path(__file__).resolve().parent
import os as _os
ROOT = Path(_os.environ.get('G1PS_ROOT', str(HERE.parent/'deps_from_g1ps')))
sys.path[:0] = [str(HERE), str(HERE.parent/'tracker'), str(ROOT/'tools/rl')]
import vega_res_env as E
import gpu_train as GT

p = argparse.ArgumentParser(); p.add_argument('config', type=Path); p.add_argument('--ckpt', type=Path); p.add_argument('--student', type=Path)
p.add_argument('--task', type=Path, help='default: the config\'s task'); p.add_argument('--n', type=int, default=64); p.add_argument('--out', type=Path)
p.add_argument('--no-dr', action='store_true'); p.add_argument('--perturb', action='store_true'); p.add_argument('--no-holder', action='store_true', help='pipette at its nominal place')
p.add_argument('--voc', type=float, help='virtual object controller scale (tracker env; zero-residual checks)')
p.add_argument('--start-frame', type=int, default=0); p.add_argument('--dump-worlds', type=Path, help='per-world outcome + DR draws (npz)'); p.add_argument('--mount-rand', type=float, nargs=2, metavar=('XYZ_MM', 'RPY_DEG'))
a0 = p.parse_args(); t_start = time.time()
C = json.loads(a0.config.read_text()); rel = lambda x: (Path(x) if Path(x).is_absolute() else (a0.config.parent/x)).resolve()
task = (a0.task or rel(C['task'])).resolve()
if a0.mount_rand is not None and (C.get('extras') or {}).get('student', {}).get('depth'):
    C['extras']['student']['depth']['mount_rand'] = ({'xyz_mm': [a0.mount_rand[0]]*3, 'rpy_deg': [a0.mount_rand[1]]*3} if any(a0.mount_rand) else None)
dr = None if a0.no_dr else (C.get('dr') if isinstance(C.get('dr'), dict) or C.get('dr') is None else json.loads(rel(C['dr']).read_text()))
ex = {k_: v_ for k_, v_ in C.get('extras', {}).items() if a0.perturb or k_ != 'perturb'}
if a0.no_holder and ex.get('holder'): ex['holder'] = {**ex['holder'], 'stand_xy_mm': 0, 'stand_yaw_deg': 0, 'meas_xy_mm': 0, 'meas_yaw_deg': 0}
r = np.load(task/'reference.npz'); t = r['timestamps']; nfr = len(t); N = a0.n
kw = dict(dr=dr, extras=ex, reward_scale=C.get('reward_scale', .01))
for k in ('arm_scale', 'finger_scale', 'group_scale', 'thumb_press_scale', 'plunger_weight'):
    if C.get(k) is not None: kw[k] = C[k]
if a0.student: kw['obs_mode'] = 'deployable'
if a0.voc is not None:
    import tracker_env as TE; TR = C.get('tracker', {})
    env = TE.BimanualGpuEnv(task, N, seed=123, nconmax=200, njmax=1600, naconmax_per_world=400, terminate=False, auto_reset=False,
                            voc_scale=a0.voc, voc_kp=TR.get('voc_kp', 2000.), voc_kd=TR.get('voc_kd', 90.), voc_free_frac=0., **kw)
else:
    env = E.BimanualGpuEnv(task, N, seed=123, nconmax=200, njmax=1600, naconmax_per_world=400, terminate=False, auto_reset=False, **kw)
policy = None
if a0.ckpt:
    a = SimpleNamespace(steps_per_env=C.get('steps_per_env', 24), lr=3e-4, init_std=C.get('init_std', .1), no_value_clip=True, deployable=False)
    venv, runner = GT.make_runner(env, a, None); runner.load(str(a0.ckpt), map_location=env.device); pol = runner.get_inference_policy(device=env.device)
    policy = lambda: pol(env.get_observations())
if a0.student:
    import distill as D
    ck = torch.load(a0.student, map_location=env.device)
    if ck.get('depth'):
        DX = env.x_depth; assert DX is not None, 'student uses depth images: the config needs extras.student.depth'
        stu = D.StudentCNN(ck['nin'], ck['nout'], ck['hidden'], (DX['H'], DX['W']), aux=ck.get('aux', 0)).to(env.device)
    else: stu = D.Student(ck['nin'], ck['nout'], ck['hidden']).to(env.device)
    stu.load_state_dict(ck['student']); stu.eval()
    assert ck['nin'] == env.obs_actor.shape[1], (ck['nin'], env.obs_actor.shape)
    policy = (lambda: stu(env.obs_actor, env.x_depth['obs'])) if ck.get('depth') else (lambda: stu(env.obs_actor))
dev = env.device; m = env.mjm
env.reset_envs(torch.ones(N, dtype=torch.bool, device=dev), frames=torch.full((N,), a0.start_frame, dtype=torch.long, device=dev)); env._launch(env.g_kin); env.obs = env._observe()
if env.obs_mode == 'deployable': env._observe_deployable()
steps = (nfr-1-a0.start_frame)//env.frame_step
tj = json.loads((task/'task.json').read_text()); hold_from = float(tj['pickup']['hold_from_s']); k_hold = int(np.searchsorted(t, hold_from))
z_rest = float(np.asarray(r['pipette_rest'])[2, 3]); th_site = m.site('right_thumb_tip').id
import mujoco
dk = mujoco.MjData(m); dk.qpos[:] = r['qpos_ref'][nfr-1]; mujoco.mj_kinematics(m, dk); ob = m.body('right_object').id
th_ref = dk.xmat[ob].reshape(3, 3).T@(dk.site_xpos[th_site]-dk.xpos[ob])
sx = env.wp.to_torch(env.d.site_xpos)
first_fail = torch.full((N,), -1, dtype=torch.long, device=dev); fail_reason = torch.zeros(N, 2, dtype=torch.long, device=dev)
zmax = torch.zeros(N, device=dev); inhand_max = torch.zeros(N, device=dev); hold_drift = torch.zeros(N, device=dev); rel_hold0 = None
for i in range(steps):
    with torch.inference_mode():
        act = policy().clamp(-1, 1) if policy is not None else torch.zeros(N, env.nu, device=dev)
        obs, rew, done, ex_ = env.step(act)
    k = env.frame.clamp(max=nfr-1); live = first_fail < 0; f_any = (ex_['reasons'] > 0).any(1)
    newly = f_any & live; first_fail[newly] = k[newly]; fail_reason[newly] = ex_['reasons'][newly].long()
    W, O = env._poses('right'); zmax = torch.maximum(zmax, O[:, 2, 3]-z_rest)
    loc = E.rinv(W)@O; rr = env.rel_ref['right'][k]; ie = (loc[:, :3, 3]-rr[:, :3, 3]).norm(dim=-1)
    inhand_max = torch.where(first_fail < 0, torch.maximum(inhand_max, ie), inhand_max)
    if int(k[0]) >= k_hold:
        if rel_hold0 is None: rel_hold0 = loc.clone()
        hold_drift = torch.maximum(hold_drift, (loc[:, :3, 3]-rel_hold0[:, :3, 3]).norm(dim=-1))
kend = torch.full((N,), nfr-1, dtype=torch.long, device=dev); W, O = env._poses('right'); Or = E.X.ref_o(env, 'right', kend)
pe = (O[:, :3, 3]-Or[:, :3, 3]).norm(dim=-1); ae = torch.arccos((O[:, :3, 2]*Or[:, :3, 2]).sum(-1).clamp(-1, 1))
loc = E.rinv(W)@O; rr = env.rel_ref['right'][nfr-1]; ipe = (loc[:, :3, 3]-rr[None, :3, 3]).norm(dim=-1); iae = E.rot_angle(loc[:, :3, :3], rr[None, :3, :3].expand(N, 3, 3))
thumb = ((sx[:, th_site]-O[:, :3, 3])[:, None, :]@O[:, :3, :3])[:, 0]; the = (thumb-torch.as_tensor(th_ref, dtype=torch.float32, device=dev)).norm(dim=-1)
tilt = torch.rad2deg(torch.arccos(O[:, 2, 2].clamp(-1, 1)))
c = lambda x: x.cpu().numpy()
end_ok = c(first_fail < 0); lifted = c(zmax >= .08); hold = c((pe <= .02) & (ae <= math.radians(15)) & (ipe <= .015) & (iae <= math.radians(15))) & end_ok
thumb_ok = c(the <= .015) & end_ok; succ = end_ok & lifted & hold & thumb_ok
ff = c(first_fail); fr = c(fail_reason)
reasons = Counter(','.join(('pipette:' if j == 0 else 'tube:')+E.FAILURES[fr[w, j]-1] for j in (0, 1) if fr[w, j] > 0) for w in range(N) if ff[w] >= 0)
q = lambda x: [round(float(np.percentile(x, 50)), 1), round(float(np.percentile(x, 90)), 1), round(float(x.max()), 1)] if len(x) else None
res = {'config': str(a0.config), 'task': str(task), 'policy': str(a0.ckpt or a0.student or 'zero residual'), 'voc': a0.voc, 'n': N, 'dr': not a0.no_dr, 'perturb': a0.perturb,
       'holder_random': not a0.no_holder, 'start_frame': a0.start_frame,
       'success': int(succ.sum()), 'end_ok': int(end_ok.sum()), 'lifted': int(lifted.sum()), 'hold_ok': int(hold.sum()), 'thumb_ok': int(thumb_ok.sum()),
       'first_fail_s_p10_50_90': [round(float(t[int(np.percentile(ff[ff >= 0], x))]), 2) for x in (10, 50, 90)] if (ff >= 0).any() else None,
       'fail_reasons': reasons.most_common(6),
       'end_mm_p50_p90_max': {'pipette_pos': q(c(pe)[end_ok]*1e3), 'pipette_axis_deg': q(np.degrees(c(ae)[end_ok])), 'inhand_pos': q(c(ipe)[end_ok]*1e3),
                              'inhand_deg': q(np.degrees(c(iae)[end_ok])), 'thumb_vs_ref': q(c(the)[end_ok]*1e3), 'tilt_from_vertical_deg': q(c(tilt)[end_ok])},
       'hold_drift_mm_p50_p90_max': q(c(hold_drift)[end_ok]*1e3), 'inhand_err_max_mm_p50_p90_max': q(c(inhand_max)*1e3), 'lift_max_mm_p50': round(float(np.median(c(zmax)))*1e3, 1),
       'pushes': (int(env.x_pert['pushes']) if getattr(env, 'x_pert', None) else 0), 'push_active_frac': (round(env.x_pert['active']/max(1, env.x_pert['steps']), 3) if getattr(env, 'x_pert', None) else 0), 'wall_s': round(time.time()-t_start, 1)}
print(json.dumps(res, indent=1))
if a0.dump_worlds:   # (11-15) per-world outcome and randomisation draws (dr_vec: 0 hand-object friction, 1 furniture friction, 3 pipette mass factor, 7.. kp / kv factors)
    hold_ = c((pe <= .02) & (ae <= math.radians(15)) & (ipe <= .015) & (iae <= math.radians(15)))
    h_ = env.x_hold['stand'] if getattr(env, 'x_hold', None) else None
    np.savez(a0.dump_worlds, end_ok=end_ok, hold=hold_ & end_ok, first_fail=ff, reason=fr, dr_vec=c(env.dr_vec) if env.dr_on else np.zeros((N, 1)),
             place_d=c(h_['d']) if h_ else np.zeros((N, 2)), place_th=c(h_['th']) if h_ else np.zeros(N), meas_d=c(h_['dm']) if h_ else np.zeros((N, 2)),
             inhand_mm=c(ipe)*1e3, inhand_deg=np.degrees(c(iae)), tilt_deg=c(tilt), nu=env.nu, action_names=np.array(env.action_names))
if a0.out: a0.out.parent.mkdir(parents=True, exist_ok=True); a0.out.write_text(json.dumps(res, indent=1)+'\n')
