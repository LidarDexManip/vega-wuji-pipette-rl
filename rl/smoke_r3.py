"""Round-3 pre-launch smoke (local GPU, short): build the RL env from a config with a small env count, check (1) the coupled arm DR draws keep
zeta >= 0.70 on every arm joint and world, kp range intact; (2) 40 control steps with random residuals: rewards / penalties finite, the new terms
(arm_band, tube_tilt, finger_band) present and finite, commanded arm targets >= 0.1 rad inside the limits (clamp_after_ff).
  python smoke_r3.py CONFIG.json [--envs 256]"""
import argparse, json, math, os, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; ROOT = Path(os.environ.get('G1PS_ROOT', str(Path(__file__).resolve().parents[1]/'deps_from_g1ps')))
sys.path[:0] = [str(HERE), str(ROOT/'tools/rl')]
import vega_res_env as E, mujoco
p = argparse.ArgumentParser(); p.add_argument('config', type=Path); p.add_argument('--envs', type=int, default=256); p.add_argument('--act-std', type=float, default=.3); a0 = p.parse_args()
C = json.loads(a0.config.read_text()); rel = lambda x: (x if Path(x).is_absolute() else (a0.config.parent/x)).resolve()
task = rel(C['task']); dr = json.loads(rel(C['dr']).read_text()); N = a0.envs
env = E.BimanualGpuEnv(task, N, seed=11, nconmax=C.get('nconmax', 200), njmax=C.get('njmax', 1600), naconmax_per_world=C.get('naconmax_per_world', 400), terminate=True, auto_reset=True,
                       dr=dr, extras=C['extras'], reward_scale=C.get('reward_scale', .01), group_scale=C.get('group_scale'))
dev = env.device; m = env.mjm; out = {'config': str(a0.config)}
# (1) damping ratio per world / arm joint from the drawn gains (diagonal inertia at a carry pose, armature included)
d = mujoco.MjData(m); r = np.load(task/'reference.npz'); d.qpos[:] = r['qpos_ref'][int(np.argmin(np.abs(r['timestamps']-9)))]; mujoco.mj_forward(m, d)
M = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, M, d.qM) if hasattr(d, 'qM') else mujoco.mj_fullM(m, d, M)
g = env.mf['actuator_gainprm'].cpu().numpy(); b = env.mf['actuator_biasprm'].cpu().numpy(); arm = [i for i in range(m.nu) if 'arm' in m.actuator(i).name]
Z, KP = [], []
for i in arm:
    dv = int(m.jnt_dofadr[m.actuator_trnid[i, 0]]); kp = g[:, i, 0]; kv = -b[:, i, 2]; Z.append((kv+m.dof_damping[dv])/(2*np.sqrt(kp*M[dv, dv]))); KP.append(kp/m.actuator_gainprm[i, 0])
Z, KP = np.array(Z), np.array(KP)
out['arm_zeta_min_over_worlds_joints'] = round(float(Z.min()), 3); out['arm_zeta_p5'] = round(float(np.percentile(Z, 5)), 3); out['kp_factor_range'] = [round(float(KP.min()), 3), round(float(KP.max()), 3)]
# (2) 40 control steps with random residual actions
pen_keys = ('arm_band', 'tube_tilt', 'finger_band', 'finger_still', 'pipette_seat'); seen = {k: [] for k in pen_keys}; rews = []; mg = []
for _ in range(40):
    with torch.inference_mode(): obs, rew, done, ex = env.step(torch.randn(N, env.nu, device=dev)*a0.act_std)
    rews.append(rew.cpu().numpy()); xi = ex['info'].get('extras', {})
    for k in pen_keys:
        v = xi.get('p_'+k) if isinstance(xi, dict) else None
        if v is not None: seen[k].append(float(v.mean()) if torch.is_tensor(v) else float(np.mean(v)))
    ca = env.ctrl[:, env.x_arm]; mg.append(float(torch.minimum(ca-(env.x_alo-.1), (env.x_ahi+.1)-ca).min()))
R = np.concatenate(rews); out['reward_finite'] = bool(np.isfinite(R).all()); out['reward_mean'] = round(float(R.mean()), 5)
out['new_terms_mean'] = {k: (round(float(np.mean(v)), 4) if v else 'not logged') for k, v in seen.items()}
out['new_terms_finite'] = all(np.isfinite(v).all() for v in seen.values() if v)
out['cmd_arm_margin_deg_min (rule >= 5.73)'] = round(math.degrees(min(mg)), 3)
out['pass'] = bool(out['reward_finite'] and out['new_terms_finite'] and out['arm_zeta_min_over_worlds_joints'] >= .699 and min(mg) >= .1-1e-6 and all(v for v in seen.values()))
print(json.dumps(out, indent=1))
