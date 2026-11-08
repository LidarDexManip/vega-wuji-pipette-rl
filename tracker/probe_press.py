"""Probe (11-06): plunger depth reached by a policy in the press window, from the start, VOC 0. Layer: RL env."""
import json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path[:0] = [str(HERE), str(Path(__file__).resolve().parents[1]/'deps_from_g1ps/tools/rl')]
import tracker_env as E, gpu_train as GT
cfg, ck, N = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8; VOC = float(sys.argv[4]) if len(sys.argv) > 4 else 0.; TASK = sys.argv[5] if len(sys.argv) > 5 else None
C = json.loads(cfg.read_text()); rel = lambda x: (cfg.parent/x).resolve(); task = rel(C['task']) if TASK is None else Path(TASK).resolve(); dr = json.loads(rel(C['dr']).read_text())
env = E.BimanualGpuEnv(task, N, seed=5, starts=[0]*N, terminate=False, auto_reset=False, dr=dr, extras=C['extras'], reward_scale=C['reward_scale'], group_scale=C['group_scale'],
                       voc_scale=VOC, voc_kp=C.get('tracker', {}).get('voc_kp', 400.), voc_kd=C.get('tracker', {}).get('voc_kd', 40.), **({'thumb_press_scale': C['thumb_press_scale']} if C.get('thumb_press_scale') else {}))
a = SimpleNamespace(steps_per_env=24, lr=3e-4, init_std=.1, no_value_clip=True, deployable=False)
if ck == 'zero': pol = lambda o: torch.zeros(N, env.nu, device=env.device)
else: venv, runner = GT.make_runner(env, a, None); runner.load(str(ck), map_location=env.device); pol = runner.get_inference_policy(device=env.device)
env.reset_envs(torch.ones(N, dtype=torch.bool, device=env.device), frames=torch.zeros(N, dtype=torch.long, device=env.device))
press = env.press; pref = env.plunger_ref; dmax = torch.zeros(N, device=env.device); refmax = float(pref.max()); held90 = torch.zeros(N, device=env.device)
thumb_err = []; th_q = []; th_r = []
with torch.inference_mode():
    for i in range(int((env.nframes-1)/env.frame_step)+2):
        obs, _, _, ex = env.step(pol(env.get_observations()))
        k = env.frame.clamp(max=env.nframes-1); inp = press[k]; depth = -env.qpos[:, env.pl_q]
        dmax = torch.where(inp, torch.maximum(dmax, depth), dmax); held90 += (inp & (depth >= .9*refmax)).float()*E.CONTROL_DT
        if bool(inp.any()):
            rq = env.qpos_ref[k][:, env.aq[env.rthumb]]; q = env.qpos[:, env.aq[env.rthumb]]; thumb_err.append(float((q-rq).abs().max(1).values[inp].mean()))
            th_q.append(torch.where(inp[:, None], q, torch.full_like(q, float('nan'))).cpu().numpy()); th_r.append(torch.where(inp[:, None], rq, torch.full_like(rq, float('nan'))).cpu().numpy())
print(json.dumps({'ref_max_depth_mm': round(refmax*1e3, 2), 'policy_max_depth_mm': [round(float(x)*1e3, 2) for x in dmax], 'held_ge90_s': [round(float(x), 2) for x in held90],
                  'right_thumb_abs_err_rad_mean_in_press': round(float(np.mean(thumb_err)), 3) if thumb_err else None,
                  'right_thumb_deg_mean_in_press_policy_per_world [cmc_flex, cmc_abd, mcp, ip]': np.round(np.degrees(np.nanmean(np.array(th_q), 0)), 1).tolist() if th_q else None,
                  'right_thumb_deg_mean_in_press_reference [cmc_flex, cmc_abd, mcp, ip]': np.round(np.degrees(np.nanmean(np.array(th_r), (0, 1))), 1).tolist() if th_r else None}))
