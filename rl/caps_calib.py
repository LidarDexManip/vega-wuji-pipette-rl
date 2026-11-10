"""(11-09) Calibration of the caps coefficient on the real env: with a trained checkpoint, one training-like rollout (PPO sampling), then
  - the smoothness loss L of the policy mean (caps.py units: 1 = a change of 0.01 on every actuator per control step), p50 / p95 per pair;
  - the actor-gradient norms of the PPO surrogate (one mini-batch, as in rsl_rl's update) and of L, so that coeff x |grad L| can be set
    against |grad surrogate| (rsl_rl clips the actor gradient at max_grad_norm 1 jointly).
  python -I caps_calib.py CONFIG.json --ckpt CKPT.pt [--envs 1024] [--reps 3]"""
import argparse, json, os, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
HERE = Path(__file__).resolve().parent; ROOT = Path(os.environ.get('G1PS_ROOT', str(HERE.parent/'deps_from_g1ps')))
sys.path[:0] = [str(HERE), str(ROOT/'tools/rl')]
import vega_res_env as E
import gpu_train as GT
import caps

p = argparse.ArgumentParser(); p.add_argument('config', type=Path); p.add_argument('--ckpt', type=Path); p.add_argument('--envs', type=int, default=1024)
p.add_argument('--reps', type=int, default=3); a0 = p.parse_args()
C = json.loads(a0.config.read_text()); rel = lambda x: (x if Path(x).is_absolute() else (a0.config.parent/x)).resolve()
dr = C.get('dr') if isinstance(C.get('dr'), dict) or C.get('dr') is None else json.loads(rel(C['dr']).read_text())
kw = dict(dr=dr, extras=C.get('extras', {}), reward_scale=C.get('reward_scale', .01))
for k in ('arm_scale', 'finger_scale', 'group_scale', 'thumb_press_scale', 'plunger_weight'):   # (11-09) tracker configs: must match training
    if C.get(k) is not None: kw[k] = C[k]
env = E.BimanualGpuEnv(rel(C['task']), a0.envs, seed=11, nconmax=C.get('nconmax', 200), njmax=C.get('njmax', 1600), naconmax_per_world=C.get('naconmax_per_world', 400),
                       starts=C.get('starts'), **kw)
a = SimpleNamespace(steps_per_env=C.get('steps_per_env', 24), lr=3e-4, init_std=C.get('init_std', .1), no_value_clip=True, deployable=False)
venv, runner = GT.make_runner(env, a, None)
if a0.ckpt: runner.load(str(a0.ckpt), map_location=env.device)
alg = runner.alg; h = caps.ActionSmoothness(alg, 1., pairs=16384); alg.train_mode(); obs = venv.get_observations()
params = [q for q in alg.actor.parameters() if q.requires_grad]


def gnorm(loss):
    g = torch.autograd.grad(loss, params, allow_unused=True); return float(torch.sqrt(sum((x.float()**2).sum() for x in g if x is not None)))


res = {'L': [], 'p50': [], 'p95': [], 'g_caps': [], 'g_surr': []}
for rep in range(a0.reps + 1):
    with torch.inference_mode():
        for _ in range(a.steps_per_env):
            obs, rew, d, ex = venv.step(alg.act(obs)); alg.process_env_step(obs, rew, d, ex)
        alg.compute_returns(obs)
    if rep == 0: alg.storage.clear(); continue   # first rollout: let the episodes spread (auto reset), discard
    o0, o1, ok = h.sample_pairs()
    with torch.no_grad():
        K = ok.shape[0]; mu = alg.actor(torch.cat([o0, o1], 0)); per = ((((mu[K:]-mu[:K]).float()/h.norm)**2).mean(1))[ok].cpu().numpy()
    res['p50'].append(float(np.median(per))); res['p95'].append(float(np.percentile(per, 95)))
    L = h.loss_on(alg.actor, o0, o1, ok); res['L'].append(float(L)); res['g_caps'].append(gnorm(L))
    gen = alg.storage.mini_batch_generator(alg.num_mini_batches, 1); batch = next(gen)
    batch.advantages = (batch.advantages-batch.advantages.mean())/(batch.advantages.std()+1e-8)
    alg.actor(batch.observations, stochastic_output=True); lp = alg.actor.get_output_log_prob(batch.actions)
    ratio = torch.exp(lp-batch.old_actions_log_prob.squeeze(-1)); A = batch.advantages.squeeze(-1)
    surr = torch.max(-A*ratio, -A*torch.clamp(ratio, 1-alg.clip_param, 1+alg.clip_param)).mean(); res['g_surr'].append(gnorm(surr))
    alg.storage.clear()
out = {k: round(float(np.mean(v)), 4) for k, v in res.items()}; out['ckpt'] = str(a0.ckpt); out['ratio g_surr / g_caps'] = round(out['g_surr']/max(out['g_caps'], 1e-12), 5)
print(json.dumps(out))
