"""Layer: ZERO-RESIDUAL PHYSICS REPLAY on the RL env (MuJoCo Warp head stack), real-robot line. Does the exported task reproduce the Vega MPC
physics demo, and how often does the demo alone survive the training perturbations?

  .venv-rl-head/bin/python -I replay_check.py TASK [--dr DR.json] [--n-per-start K] [--real-cmd 0|1] [--out OUT.json]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vega_res_env as E

p = argparse.ArgumentParser(); p.add_argument('task', type=Path); p.add_argument('--dr', type=Path); p.add_argument('--n-per-start', type=int, default=1)
p.add_argument('--starts-s', type=float, nargs='*', default=[3.04, 4.0, 5.5, 7.0, 9.0, 11.0, 12.5, 14.5, 17.0, 19.0, 20.5, 22.5])
p.add_argument('--real-cmd', type=int, default=None); p.add_argument('--out', type=Path); p.add_argument('--seed', type=int, default=0)
a = p.parse_args()
r = np.load(a.task/'reference.npz'); t = r['timestamps']
fr = []
for s in a.starts_s:
    k = int(np.argmin(np.abs(t-s))); k -= k % 2; fr.append(min(k, len(t)-3))
fr = fr*a.n_per_start; N = len(fr)
dr = json.loads(a.dr.read_text()) if a.dr else None
kw = {} if a.real_cmd is None else {'extras': {'real_cmd': bool(a.real_cmd)}}
env = E.BimanualGpuEnv(a.task, N, nconmax=200, njmax=1600, naconmax_per_world=600, seed=a.seed, terminate=False, auto_reset=False, dr=dr, **kw)
frames = torch.as_tensor(fr, device=env.device)
env.reset_envs(torch.ones(N, dtype=torch.bool, device=env.device), frames=frames); env._launch(env.g_kin); env.obs = env._observe()
alive = torch.ones(N, dtype=torch.bool, device=env.device); first_fail = torch.full((N,), -1., device=env.device)
emax = {s: torch.zeros(N, device=env.device) for s in E.SIDES}; t0 = time.time()
steps = int(max((env.nframes-1-f)//env.frame_step for f in fr))
for i in range(steps):
    with torch.inference_mode(): obs, rew, done, ex = env.step(torch.zeros(N, env.nu, device=env.device))
    info = ex['info']; f_any = (ex['reasons'] > 0).any(1)
    newly = f_any & (first_fail < 0); first_fail[newly] = (env.frame[newly].float()*.01+float(t[0]))
    live = env.frame <= env.nframes-1
    for s in E.SIDES: emax[s] = torch.where(live & (first_fail < 0), torch.maximum(emax[s], info[s]['object_position_m']), emax[s])
end_ok = first_fail < 0
res = {'layer': 'zero-residual physics replay, RL env (Warp head)', 'task': str(a.task), 'dr': a.dr.name if a.dr else None, 'cfg_command': env.cfg.get('command'),
       'real_cmd': env.xcfg['real_cmd'], 'n': N, 'reach_end_without_failure': int(end_ok.sum()), 'wall_s': round(time.time()-t0, 1),
       'per_start': [{'start_s': round(float(t[f]), 2), 'ok': bool(end_ok[i]), 'first_fail_s': (round(float(first_fail[i]), 2) if first_fail[i] >= 0 else None),
                      'max_obj_err_mm': {s: round(float(emax[s][i])*1e3, 1) for s in E.SIDES}} for i, f in enumerate(fr)]}
print(json.dumps({k: v for k, v in res.items() if k != 'per_start'}))
for row in res['per_start'][:min(N, 24)]: print(' ', row)
if a.out: a.out.write_text(json.dumps(res, indent=1))
