"""record qpos of a few worlds of a pickup rollout (teacher / student / zero residual) for offline rendering (11-14)
  cd rl; python -I record_pickup.py CONFIG --ckpt CKPT [--n 4] --out OUT.npz [--perturb] [--voc V]"""
import argparse, json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
HERE = Path(__file__).resolve().parent; import os as _os
ROOT = Path(_os.environ.get('G1PS_ROOT', str(HERE.parent/'deps_from_g1ps'))); sys.path[:0] = [str(HERE), str(HERE.parent/'tracker'), str(ROOT/'tools/rl')]
import vega_res_env as E, gpu_train as GT
p = argparse.ArgumentParser(); p.add_argument('config', type=Path); p.add_argument('--ckpt', type=Path); p.add_argument('--student', type=Path); p.add_argument('--n', type=int, default=4)
p.add_argument('--out', type=Path, required=True); p.add_argument('--perturb', action='store_true'); p.add_argument('--no-dr', action='store_true'); p.add_argument("--voc", type=float); p.add_argument("--start-frame", type=int, default=0)
p.add_argument('--full', action='store_true', help='(11-16) also save qvel and ctrl per step (CPU replay of contact forces: mj_forward at the recorded state)')
a0 = p.parse_args(); C = json.loads(a0.config.read_text()); rel = lambda x: (Path(x) if Path(x).is_absolute() else (a0.config.parent/x)).resolve(); task = rel(C['task']) if not C['task'].startswith('../../tasks') or (a0.config.parent/C['task']).exists() else (HERE.parent/C['task'][6:]).resolve()
dr = None if a0.no_dr else json.loads(rel(C['dr']).read_text()) if isinstance(C.get('dr'), str) else C.get('dr')
kw = dict(dr=dr, extras={k: v for k, v in C['extras'].items() if a0.perturb or k != 'perturb'}, reward_scale=C.get('reward_scale', .01))
for k in ('group_scale', 'thumb_press_scale', 'plunger_weight'):
    if C.get(k) is not None: kw[k] = C[k]
if a0.student: kw['obs_mode'] = 'deployable'
N = a0.n
if a0.voc is not None:
    import tracker_env as TE; env = TE.BimanualGpuEnv(task, N, seed=123, nconmax=200, njmax=1600, naconmax_per_world=400, terminate=False, auto_reset=False, voc_scale=a0.voc, voc_kp=2000., voc_kd=90., **kw)
else: env = E.BimanualGpuEnv(task, N, seed=123, nconmax=200, njmax=1600, naconmax_per_world=400, terminate=False, auto_reset=False, **kw)
pol = None
if a0.ckpt:
    a = SimpleNamespace(steps_per_env=24, lr=3e-4, init_std=.1, no_value_clip=True, deployable=False); venv, runner = GT.make_runner(env, a, None)
    runner.load(str(a0.ckpt), map_location=env.device); pp = runner.get_inference_policy(device=env.device); pol = lambda: pp(env.get_observations())
if a0.student:
    import distill as D
    ck = torch.load(a0.student, map_location=env.device); DX = env.x_depth
    stu = D.StudentCNN(ck['nin'], ck['nout'], ck['hidden'], (DX['H'], DX['W']), aux=ck.get('aux', 0)).to(env.device) if ck.get('depth') else D.Student(ck['nin'], ck['nout'], ck['hidden']).to(env.device)
    stu.load_state_dict(ck['student']); stu.eval(); pol = (lambda: stu(env.obs_actor, env.x_depth['obs'])) if ck.get('depth') else (lambda: stu(env.obs_actor))
dev = env.device; env.reset_envs(torch.ones(N, dtype=torch.bool, device=dev), frames=torch.full((N,), a0.start_frame, dtype=torch.long, device=dev)); env._launch(env.g_kin); env.obs = env._observe()
if env.obs_mode == 'deployable': env._observe_deployable()
nfr = env.nframes; qs = [env.qpos.cpu().numpy().copy()]; fr = [env.frame.cpu().numpy().copy()]; ff = np.full(N, -1)
vs = [env.qvel.cpu().numpy().copy()]; cs = [env.ctrl.cpu().numpy().copy()]
for i in range((nfr-1)//env.frame_step):
    with torch.inference_mode():
        act = pol().clamp(-1, 1) if pol else torch.zeros(N, env.nu, device=dev); _, _, _, ex = env.step(act)
    f = (ex['reasons'] > 0).any(1).cpu().numpy(); k = env.frame.cpu().numpy(); ff = np.where((ff < 0) & f, k, ff)
    qs.append(env.qpos.cpu().numpy().copy()); fr.append(k.copy())
    if a0.full: vs.append(env.qvel.cpu().numpy().copy()); cs.append(env.ctrl.cpu().numpy().copy())
ext = dict(qvel=np.array(vs), ctrl=np.array(cs)) if a0.full else {}
# (11-17) per-world conditions of this recording (one reset at the start): DR vector, robot base height, arm lag, holder placement and its measured value
if getattr(env, 'dr_on', False): ext['dr_vec'] = env.dr_vec.cpu().numpy()
if getattr(env, 'base_offz', None) is not None: ext['base_z'] = env.base_offz.cpu().numpy()
if getattr(env, 'x_lag', None) is not None: ext.update(lag_d=env.x_lag['d'].cpu().numpy(), lag_tau=env.x_lag['tau'].cpu().numpy(), lag_alpha=env.x_lag['alpha'].cpu().numpy())
for nm_, h_ in (getattr(env, 'x_hold', None) or {}).items():
    for k_ in ('d', 'th', 'dm', 'thm'):
        if k_ in h_: ext[f'hold_{nm_}_{k_}'] = h_[k_].cpu().numpy()
np.savez_compressed(a0.out, qpos=np.array(qs), frame=np.array(fr), first_fail=ff, task=str(task), **ext); print('wrote', a0.out, 'first fail frames', ff)
