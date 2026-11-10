"""Contact-stability metrics of a policy (11-08): from-start rollouts under the eval_rr.py setup (seed 123, DR as in the config, deterministic actor
mean, no termination), logging the reference-free contact terms of vega_extras on every control step, whatever the config's reward weights:
  lost      -- fraction of the previous step's link contacts lost (hysteresis 0.1 / 0.05 x object weight), per step and as loss events per second
  drift     -- change of the pairwise contact distances on the object, mm per control step
  slip      -- motion of each contact point on the object, mm per control step
  holds     -- share of reference-held steps where the contacts can supply the required wrench (distance < 0.05 N, <= margin_fmax per contact)
  margin    -- mean distance (N) of required wrench +- margin_rho per axis (0 = full margin), reference-held steps
Steps of a world count until its first failure (eval_rr's failure reasons); temporal terms skip the temporal_off_gates frames (deliberate regrasps).
  python -I eval_contact_stability.py CONFIG.json [--ckpt CKPT.pt] [--from-start 64] [--out out.json] [--metrics-config configs/rr_T1k_tc_s5.json]
The metric parameters (hysteresis, fmax, rho, off gates) come from --metrics-config so that policies trained with different configs are measured alike."""
import argparse, json, os, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
HERE = Path(__file__).resolve().parent; ROOT = Path(os.environ.get('G1PS_ROOT', str(HERE.parent/'deps_from_g1ps')))
sys.path[:0] = [str(HERE), str(ROOT/'tools/rl')]
import vega_res_env as E
import gpu_train as GT

p = argparse.ArgumentParser(); p.add_argument('config', type=Path); p.add_argument('--ckpt', type=Path); p.add_argument('--from-start', type=int, default=64)
p.add_argument('--out', type=Path); p.add_argument('--metrics-config', type=Path, default=HERE/'configs/rr_T1k_tc_s5.json'); a0 = p.parse_args()
C = json.loads(a0.config.read_text()); rel = lambda x: (x if Path(x).is_absolute() else (a0.config.parent/x)).resolve()
task = rel(C['task']); dr = C.get('dr') if isinstance(C.get('dr'), dict) or C.get('dr') is None else json.loads(rel(C['dr']).read_text())
MX = json.loads(a0.metrics_config.read_text())['extras']
ex_cfg = json.loads(json.dumps(C.get('extras', {})))
for k in ('keep_hysteresis', 'drift_mm', 'slip_mm', 'temporal_off_gates', 'margin_fmax', 'margin_mu', 'margin_rho', 'margin_iters', 'margin_gate'):
    if k in MX: ex_cfg[k] = MX[k]
ex_cfg.update({'log': True, 'log_margin': True, 'margin_every': 1})        # metrics only: reward weights untouched (they do not change the rollout)
r = np.load(task/'reference.npz'); nfr = len(r['timestamps']); N = a0.from_start
kw = dict(dr=dr, extras=ex_cfg, reward_scale=C.get('reward_scale', .01))
for k in ('arm_scale', 'finger_scale', 'group_scale', 'thumb_press_scale', 'plunger_weight'):   # (11-09) tracker configs: must match training
    if C.get(k) is not None: kw[k] = C[k]
env = E.BimanualGpuEnv(task, N, seed=123, nconmax=C.get('nconmax', 200), njmax=C.get('njmax', 1600), naconmax_per_world=C.get('naconmax_per_world', 400),
                       terminate=False, auto_reset=False, **kw)
policy = None
if a0.ckpt:
    a = SimpleNamespace(steps_per_env=C.get('steps_per_env', 24), lr=3e-4, init_std=C.get('init_std', .1), no_value_clip=True, deployable=False)
    venv, runner = GT.make_runner(env, a, None); runner.load(str(a0.ckpt), map_location=env.device); pol = runner.get_inference_policy(device=env.device)
    policy = lambda: pol(env.get_observations())
dev = env.device
env.reset_envs(torch.ones(N, dtype=torch.bool, device=dev), frames=torch.zeros(N, dtype=torch.long, device=dev)); env._launch(env.g_kin); env.obs = env._observe()
steps = int((nfr-1)//env.frame_step)
eg = ex_cfg.get('event_gate')
if eg: steps += int(round(float(eg.get('cap_s', 1.))/E.CONTROL_DT))*4
off = np.zeros(nfr, bool)
for g_ in ex_cfg.get('temporal_off_gates', []): off |= r[g_].astype(bool)
off = torch.as_tensor(off, device=dev)
held = torch.as_tensor(np.stack([r['gate_pipette_held'], r['gate_tube_held']], 1).astype(bool), device=dev)
alive = torch.ones(N, dtype=torch.bool, device=dev); reached_end = torch.zeros(N, dtype=torch.bool, device=dev)
acc = {k: [] for k in ('lost', 'drift', 'slip')}; nlinks = []; hold_ok = [[], []]; marg = [[], []]; t_steps = 0
for i in range(steps):
    with torch.inference_mode():
        act = policy().clamp(-1, 1) if policy is not None else torch.zeros(N, env.nu, device=dev)
        obs, rew, done, ex = env.step(act)
    k = env.frame.clamp(max=nfr-1); alive &= ~(ex['reasons'] > 0).any(1) & (env.frame <= nfr-1); reached_end |= alive & (env.frame >= nfr-1-env.frame_step)
    xi = ex['info']['extras']; m_t = alive & ~off[k]
    if i >= 2:
        for key, name in (('lost', 'p_contact_keep'), ('drift', 'p_contact_drift'), ('slip', 'p_contact_slip')):
            acc[key].append(xi[name][m_t].float().cpu().numpy())
        t_steps += int(m_t.sum()); nlinks.append(env.x_kon[alive].sum(1).float().cpu().numpy())
        for sd, nm in ((0, 'pipette'), (1, 'tube')):
            m_h = alive & held[k, sd]
            hold_ok[sd].append((xi[f't__margin_req_{nm}_N'][m_h] < .05).float().cpu().numpy()); marg[sd].append(xi[f't__margin_mean_{nm}_N'][m_h].float().cpu().numpy())
    if not alive.any(): break
cat = lambda L: np.concatenate(L) if L else np.zeros(0)
lost, drift, slip = (cat(acc[k]) for k in ('lost', 'drift', 'slip'))
s = lambda x, f=3: {'mean': round(float(x.mean()), f), 'p95': round(float(np.percentile(x, 95)), f)} if x.size else None
# lost / drift / slip are side averages (vega_extras): x2 gives the per-hand value when one hand is active; reported as logged (comparable across runs)
res = {'ckpt': str(a0.ckpt) if a0.ckpt else 'ZERO residual', 'config': str(a0.config), 'metrics_config': str(a0.metrics_config), 'n_worlds': N,
       'survived_to_end': int(reached_end.sum()), 'counted_steps (alive, outside regrasp gates)': t_steps,
       'lost_fraction_per_step': s(lost, 4), 'lost_fraction_per_s (side-averaged, x 50 Hz)': round(float(lost.mean()*50), 3),
       'drift_mm_per_step (penalty units, side-averaged, capped 5)': s(drift), 'slip_mm_per_step (same)': s(slip),
       'links_in_contact_per_world': round(float(cat(nlinks).mean()), 2),
       'holds_required_wrench_share (ref-held steps)': {nm: round(float(cat(hold_ok[sd]).mean()), 3) for sd, nm in ((0, 'pipette'), (1, 'tube'))},
       'margin_mean_N (ref-held; 0 = full rho margin)': {nm: round(float(cat(marg[sd]).mean()), 3) for sd, nm in ((0, 'pipette'), (1, 'tube'))},
       'params': {k: env.xcfg.get(k) for k in ('force_alpha', 'keep_hysteresis', 'margin_fmax', 'margin_mu', 'margin_rho', 'temporal_off_gates')}}
print(json.dumps(res, indent=1))
if a0.out: a0.out.parent.mkdir(parents=True, exist_ok=True); a0.out.write_text(json.dumps(res, indent=1)+'\n')
