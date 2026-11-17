"""(11-17) Transitions for a learned world model of the right arm + hand + pipette (SuperMimic-style, hybridrobotics/supermimic):
the env runs as in training (DR, random placement, random start frames, auto reset), actions = teacher mean + per-episode Gaussian noise.
Per control step and world:
  state  (74)  right arm q/qd (7+7), right hand q/qd (20+20), pipette in the wrist frame: position (3), rotation 6D (6), linear / angular
               velocity (3+3), plunger q/qd (2), gravity direction in the wrist frame (3)
  cmd    (54)  right-side actuator commands at the first and the last physics substep of the step (27+27, rad)
  ctx    (108) DR context: dr_vec[:7] (frictions, masses, skew) + right-side kp, kv, joint offsets (27 each) + right finger frictionloss (20; 11-17)
  frame, done (episode ended after this step), fail
Writes OUT.npz with arrays (T, N, ...).
  cd rl; python -I wm_collect.py CONFIG.json --ckpt TEACHER.pt --n 512 --steps 1500 --out OUT.npz [--noise 0,0.1,0.3]"""
import argparse, json, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
HERE = Path(__file__).resolve().parent; import os as _os
ROOT = Path(_os.environ.get('G1PS_ROOT', str(HERE.parent/'deps_from_g1ps'))); sys.path[:0] = [str(HERE), str(HERE.parent/'tracker'), str(ROOT/'tools/rl')]
import vega_res_env as E, gpu_train as GT

p = argparse.ArgumentParser(); p.add_argument('config', type=Path); p.add_argument('--ckpt', type=Path, required=True); p.add_argument('--n', type=int, default=512)
p.add_argument('--steps', type=int, default=1500); p.add_argument('--noise', default='0,0.1,0.3'); p.add_argument('--out', type=Path, required=True); p.add_argument('--seed', type=int, default=0)
a0 = p.parse_args(); C = json.loads(a0.config.read_text()); rel = lambda x: (Path(x) if Path(x).is_absolute() else (a0.config.parent/x)).resolve(); task = rel(C['task'])
dr = json.loads(rel(C['dr']).read_text()) if isinstance(C.get('dr'), str) else C.get('dr')
ex = {k: v for k, v in C.get('extras', {}).items() if k != 'perturb'}
kw = dict(dr=dr, extras=ex, reward_scale=C.get('reward_scale', .01))
for k in ('group_scale', 'thumb_press_scale', 'plunger_weight'):
    if C.get(k) is not None: kw[k] = C[k]
N = a0.n; env = E.BimanualGpuEnv(task, N, seed=123+a0.seed, nconmax=200, njmax=1600, naconmax_per_world=400, terminate=True, auto_reset=True, **kw); dev = env.device
a = SimpleNamespace(steps_per_env=24, lr=3e-4, init_std=.1, no_value_clip=True, deployable=False); venv, runner = GT.make_runner(env, a, None)
runner.load(str(a0.ckpt), map_location=dev); pol = runner.get_inference_policy(device=dev)
names = env.action_names; R = torch.as_tensor([i for i, n in enumerate(names) if n.startswith('R_arm')]+[i for i, n in enumerate(names) if n.startswith('r_')], device=dev)
nu = env.nu; ctx_idx = torch.cat([torch.arange(7, device=dev), 7+R, 7+nu+R, 7+3*nu+R])
fl_dof = env.av[R[7:]]   # (11-17) + the right finger joints' stiction (frictionloss_hand DR; not in dr_vec, which the critic observes)
levels = torch.tensor([float(x) for x in a0.noise.split(',')], device=dev); g = torch.Generator(device=dev); g.manual_seed(a0.seed)
sig = levels[torch.randint(len(levels), (N,), device=dev, generator=g)]
ow = env.obj_id['right']; ov = env.obj_v['right']

def rot6(M): return torch.cat([M[..., :, 0], M[..., :, 1]], -1)
def state():
    W, O = env._poses('right'); Rw = W[:, :3, :3]; L = torch.linalg.inv(W)@O
    q, v = env.qpos[:, env.aq[R]], env.qvel[:, env.av[R]]
    vo = env.qvel[:, ov:ov+3]; wo = env.qvel[:, ov+3:ov+6]                          # free joint: world linear, body-local angular
    vo_w = (Rw.transpose(1, 2)@vo[:, :, None])[:, :, 0]; wo_w = (Rw.transpose(1, 2)@(O[:, :3, :3]@wo[:, :, None]))[:, :, 0]
    gw = (Rw.transpose(1, 2)@torch.tensor([0., 0., -1.], device=dev)[None, :, None].expand(N, 3, 1))[:, :, 0]
    pl = torch.stack([env.qpos[:, env.pl_q], env.qvel[:, env.pl_v]], 1)
    return torch.cat([q[:, :7], v[:, :7], q[:, 7:], v[:, 7:], L[:, :3, 3], rot6(L[:, :3, :3]), vo_w, wo_w, pl, gw], 1)

ctrl_first = torch.zeros(N, nu, device=dev)
def hook(s):
    if s == 0: ctrl_first.copy_(env.ctrl)
env.substep_hook = hook
env.reset_envs(torch.ones(N, dtype=torch.bool, device=dev)); env._launch(env.g_kin); env.obs = env._observe()
S, CMD, CTX, FR, DONE, FAIL, ACT = [], [], [], [], [], [], []; t0 = time.time()
with torch.inference_mode():
    for t in range(a0.steps):
        s = state(); ctx = torch.cat([env.dr_vec[:, ctx_idx], env.mf['dof_frictionloss'][:, fl_dof]], 1).clone(); k = env.frame.clone()
        act = (pol(env.get_observations())+sig[:, None]*torch.randn(N, nu, device=dev, generator=g)).clamp(-1, 1)
        _, _, done, x = env.step(act)
        S.append(s.cpu()); CMD.append(torch.cat([ctrl_first[:, R], env.ctrl[:, R]], 1).cpu()); CTX.append(ctx.cpu()); FR.append(k.cpu()); ACT.append(act[:, R].cpu())
        DONE.append(done.cpu()); FAIL.append(x['fail'].cpu())
        if done.any(): sig[done] = levels[torch.randint(len(levels), (int(done.sum()),), device=dev, generator=g)]
        if t % 200 == 0: print(f'step {t}/{a0.steps}  {time.time()-t0:.0f} s  episodes ended so far {int(torch.stack(DONE).sum())}', flush=True)
# the state AFTER each step = the next row's state unless the episode ended (then the row starts a new episode)
S.append(state().cpu())
np.savez(a0.out, state=torch.stack(S).numpy().astype(np.float32), cmd=torch.stack(CMD).numpy().astype(np.float32), ctx=torch.stack(CTX).numpy().astype(np.float32),
         act=torch.stack(ACT).numpy().astype(np.float32), frame=torch.stack(FR).numpy(), done=torch.stack(DONE).numpy(), fail=torch.stack(FAIL).numpy(),
         names=np.array([names[i] for i in R.tolist()]), task=str(task), noise=a0.noise)
print('wrote', a0.out, f'{a0.steps} steps x {N} worlds in {time.time()-t0:.0f} s')
