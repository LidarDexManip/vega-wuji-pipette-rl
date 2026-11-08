"""Unit tests for tracker_env (11-06). Layer: RL ENV (MuJoCo Warp), zero action, short rollouts.
  1. identity: warp 1, VOC 0 -> same trajectory as the parent env (same starts, same seed).
  2. time-warp: frames advance at frame_step * s per control step; slow worlds take longer; commands follow the warped clock.
  3. VOC: at VOC 1 the kinematic base keeps both objects near their reference; at VOC 0 it does not (the reason for the curriculum).
  4. per-world holders / hole pairs still drawn (reference bank = per-world holder offsets, hole pairs, speed).
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import tracker_env as TE
import vega_res_env as VR

TASK = HERE.parent/'tasks/task_kin_prior'; out = {}
EXTRAS = json.loads((HERE/'configs/stage0.json').read_text())['extras'] if (HERE/'configs/stage0.json').exists() else {}
kw = dict(nconmax=200, njmax=1600, naconmax_per_world=400, terminate=False, auto_reset=False, seed=7)

# 1. identity
N = 16; st = np.arange(N)*100
e0 = VR.BimanualGpuEnv(TASK, N, starts=st, **kw); e1 = TE.BimanualGpuEnv(TASK, N, starts=st, **kw)
for e in (e0, e1): e.reset_envs(torch.ones(N, dtype=torch.bool, device=e.device), frames=torch.as_tensor(st, device=e.device))
z = torch.zeros(N, e0.nu, device=e0.device)
for _ in range(25): e0.step(z); e1.step(z)
out['identity_max_abs_qpos_diff'] = float((e0.qpos-e1.qpos).abs().max()); out['identity_frames_equal'] = bool((e0.frame == e1.frame).all())
del e0, e1; torch.cuda.empty_cache()

# 2. time-warp: speeds fixed per world by drawing then overriding
e = TE.BimanualGpuEnv(TASK, 4, starts=[200]*4, warp=(.8, 1.2), **kw)
e.reset_envs(torch.ones(4, dtype=torch.bool, device=e.device), frames=torch.full((4,), 200, device=e.device, dtype=torch.long))
e.tw[:] = torch.tensor([.8, 1., 1.2, 1.], device=e.device); e.tacc[:] = 0
z = torch.zeros(4, e.nu, device=e.device)
for _ in range(50): e.step(z)
adv = (e.frame-200).float().tolist(); out['warp_frames_after_50_steps'] = adv; out['warp_expected'] = [80., 100., 120., 100.]
out['warp_ok'] = all(abs(a-b) <= 1 for a, b in zip(adv, out['warp_expected']))
out['max_episode_length'] = int(e.max_episode_length); del e; torch.cuda.empty_cache()

# 3. VOC on the kinematic base, zero action from frame 0 for 6 s
res = {}
for scale in (1., 0.):
    e = TE.BimanualGpuEnv(TASK, 8, starts=[0]*8, voc_scale=scale, **kw)
    e.reset_envs(torch.ones(8, dtype=torch.bool, device=e.device), frames=torch.zeros(8, dtype=torch.long, device=e.device))
    z = torch.zeros(8, e.nu, device=e.device); errs = []
    for i in range(300):
        _, _, _, ex = e.step(z)
        if i % 25 == 24: errs.append([round(float(ex['info'][s]['object_position_m'].median())*1e3, 1) for s in VR.SIDES])
    res[f'voc_{scale}'] = {'object_err_mm_median_every_0.5s[right,left]': errs, 'reached_frame': int(e.frame.median())}
    del e; torch.cuda.empty_cache()
out['voc'] = res
print(json.dumps(out, indent=1)); (HERE/'test_tracker_env.json').write_text(json.dumps(out, indent=1)+'\n')
