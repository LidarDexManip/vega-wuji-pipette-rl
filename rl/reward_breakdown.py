"""Weighted reward breakdown of a policy under training-like rollouts (11-09): random valid starts, DR, termination + auto reset, deterministic
actor mean (zero residual without --ckpt). Per term: weight x value per control step (before reward_scale), mean and std over worlds x steps, and
the check that the terms add up to the env's reward.
  python -I reward_breakdown.py CONFIG.json [--ckpt CKPT.pt] [--envs 512] [--steps 300] [--noise-std S] [--out out.json]
--noise-std S adds N(0, S^2) to the normalised action every step, like PPO's sampling during training (11-09: the rate penalties also see the
exploration noise; deterministic rollouts alone under-state what the trainer optimises)."""
import argparse, json, os, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
HERE = Path(__file__).resolve().parent; ROOT = Path(os.environ.get('G1PS_ROOT', str(HERE.parent/'deps_from_g1ps')))
sys.path[:0] = [str(HERE), str(ROOT/'tools/rl')]
import vega_res_env as E
import gpu_train as GT

p = argparse.ArgumentParser(); p.add_argument('config', type=Path); p.add_argument('--ckpt', type=Path); p.add_argument('--envs', type=int, default=512)
p.add_argument('--steps', type=int, default=300); p.add_argument('--noise-std', type=float, default=0.); p.add_argument('--out', type=Path); a0 = p.parse_args()
C = json.loads(a0.config.read_text()); rel = lambda x: (Path(x) if Path(x).is_absolute() else (a0.config.parent/x)).resolve()
dr = C.get('dr') if isinstance(C.get('dr'), dict) or C.get('dr') is None else json.loads(rel(C['dr']).read_text())
kw = dict(dr=dr, extras={k_: v_ for k_, v_ in C.get('extras', {}).items() if k_ != 'perturb'}, reward_scale=C.get('reward_scale', .01))   # (11-12) no hand pushes in evals
for k in ('arm_scale', 'finger_scale', 'group_scale', 'thumb_press_scale', 'plunger_weight'):   # (11-09) tracker configs: must match training
    if C.get(k) is not None: kw[k] = C[k]
N = a0.envs
env = E.BimanualGpuEnv(rel(C['task']), N, seed=7, nconmax=C.get('nconmax', 200), njmax=C.get('njmax', 1600), naconmax_per_world=C.get('naconmax_per_world', 400),
                       terminate=True, auto_reset=True, starts=C.get('starts'), **kw)
policy = None
if a0.ckpt:
    a = SimpleNamespace(steps_per_env=C.get('steps_per_env', 24), lr=3e-4, init_std=C.get('init_std', .1), no_value_clip=True, deployable=False)
    venv, runner = GT.make_runner(env, a, None); runner.load(str(a0.ckpt), map_location=env.device); pol = runner.get_inference_policy(device=env.device)
    policy = lambda: pol(env.get_observations())
Wt = env.xcfg['w']; acc = {}; resid = []; tot = []; raw = {}

def add(name, v):
    acc.setdefault(name, []).append(v.float().cpu().numpy())

for i in range(a0.steps):
    with torch.inference_mode():
        act = policy() if policy is not None else torch.zeros(N, env.nu, device=env.device)
        if a0.noise_std > 0: act = act + a0.noise_std*torch.randn_like(act)
        obs, rew, done, ex = env.step(act.clamp(-1, 1))
    if i < 5: continue
    info = ex['info']; parts = []
    for s in ('right', 'left'):
        T = info[s]['terms']
        for n_, w_ in env.xcfg['track_w'].items():   # (11-09) the env's weights (default 8 / 1 / 1 / 1 / 1; wrist / world carry their own x2)
            v = w_*T[n_]; add(f'track_{s}_{n_}', v); parts.append(v)
    v = -.01*info['smooth']; add('env_smooth (-0.01 x)', v); parts.append(v)
    v = info['plunger_term']; add('env_plunger', v); parts.append(v)
    for key, val in info['extras'].items():
        nm = key[2:]; raw.setdefault(key, []).append(val.float().cpu().numpy())
        if nm in Wt and float(Wt[nm]) != 0:
            v = (1. if key.startswith('t_') else -1.)*float(Wt[nm])*val; add(f'x_{nm}', v); parts.append(v)
    s_ = torch.stack(parts).sum(0); ru = info['reward_unscaled']; ok = torch.isfinite(ru)
    resid.append(float((s_-ru)[ok].abs().max()) if ok.any() else 0.); tot.append(ru[ok].float().cpu().numpy())
rows = []
for n_, L in acc.items():
    v = np.concatenate(L); rows.append((n_, float(v.mean()), float(v.std())))
T_ = np.concatenate(tot); pos = sum(m for _, m, _ in rows if m > 0); neg = sum(m for _, m, _ in rows if m < 0)
rows.sort(key=lambda r: -abs(r[1]))
print(f'{"term":34s} {"mean":>9s} {"std":>8s}  share')
for n_, m, s in rows: print(f'{n_:34s} {m:9.4f} {s:8.4f}  {m/(pos if m > 0 else -neg):6.1%} of {"+" if m > 0 else "-"}')
print(f'total per step: mean {T_.mean():.3f}, std {T_.std():.3f} | positives {pos:.3f}, penalties {neg:.3f} | max |sum(terms) - env reward| {max(resid):.2e}')
print('\nraw (unweighted) extras, per control step:')
for key in sorted(raw):
    v = np.concatenate(raw[key]); print(f'  {key:30s} mean {v.mean():9.4f}  p50 {np.median(v):9.4f}  p95 {np.percentile(v, 95):9.4f}  max {v.max():9.4f}  (w {Wt.get(key[2:], "-")})')
if a0.out:
    a0.out.parent.mkdir(parents=True, exist_ok=True)
    a0.out.write_text(json.dumps({'ckpt': str(a0.ckpt) if a0.ckpt else 'ZERO residual', 'config': str(a0.config), 'envs': N, 'steps': a0.steps, 'noise_std': a0.noise_std,
                                  'terms': {n_: {'mean': round(m, 5), 'std': round(s, 5)} for n_, m, s in rows},
                                  'raw': {k_: {'mean': float(np.concatenate(v_).mean()), 'p95': float(np.percentile(np.concatenate(v_), 95))} for k_, v_ in raw.items()}, 'total_mean': float(T_.mean()), 'total_std': float(T_.std()), 'sum_check_max_abs': max(resid)}, indent=1)+'\n')
