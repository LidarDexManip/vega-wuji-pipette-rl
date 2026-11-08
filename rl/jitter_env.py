"""Finger jitter of the RL BASE (layer: RL env, MuJoCo Warp physics, ZERO residual, no DR, start frame 0, N worlds): the 3-10 Hz band RMS
(zero-phase 4th-order Butterworth, same measure as t1/jitter2.py) of every finger joint angle, worst joint per hand / finger / window.
Windows from the task reference gates: left hold = gate_tube_held, right hold = gate_pipette_held, right press = gate_press (+-0.3 s),
post-release = after the held gate of that hand. Also the base command itself (ctrl_sim) on the same windows.
  python jitter_env.py TASK [TASK ...] [--worlds 4] [--out J]"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch
from scipy.signal import butter, filtfilt
HERE = Path(__file__).resolve().parent; ROOT = Path(os.environ.get('G1PS_ROOT', str(Path(__file__).resolve().parents[1]/'deps_from_g1ps')))
sys.path[:0] = [str(HERE), str(ROOT/'tools/rl')]
import vega_res_env as E
p = argparse.ArgumentParser(); p.add_argument('tasks', nargs='+', type=Path); p.add_argument('--worlds', type=int, default=4); p.add_argument('--out', type=Path); a = p.parse_args()
BP = butter(4, [3/50, 10/50], 'band'); TH = 0.25; res = {}
for task in a.tasks:
    r = np.load(task/'reference.npz'); t = r['timestamps']; nfr = len(t); N = a.worlds
    env = E.BimanualGpuEnv(task, N, seed=3, terminate=False, auto_reset=False, nconmax=200, njmax=1600, naconmax_per_world=600)
    dev = env.device; env.reset_envs(torch.ones(N, dtype=torch.bool, device=dev), frames=torch.zeros(N, dtype=torch.long, device=dev)); env._launch(env.g_kin); env.obs = env._observe()
    m = env.mjm; fj = [(j, m.joint(j).name) for j in range(m.njnt) if m.joint(j).name.startswith(('r_', 'l_'))]; qa = [int(m.jnt_qposadr[j]) for j, _ in fj]
    Q, K = [], []
    for i in range((nfr-1)//env.frame_step):
        with torch.inference_mode(): env.step(torch.zeros(N, env.nu, device=dev))
        Q.append(env.qpos[:, qa].cpu().numpy()); K.append(int(env.frame[0].clamp(max=nfr-1)))
    Q = np.degrees(np.stack(Q, 1)); tq = t[np.array(K)]                       # (N, T, J) at the policy rate
    t100 = np.arange(tq[0], tq[-1], .01); P = np.stack([np.stack([np.interp(t100, tq, Q[w, :, k]) for k in range(len(qa))], 1) for w in range(N)])
    g = lambda name: np.interp(t100, t, r[name].astype(float)) > .5
    pr = g('gate_press'); idx = np.flatnonzero(pr); pr2 = pr.copy()
    if len(idx): pr2[max(0, idx[0]-30):idx[-1]+31] = True
    lh, rh = g('gate_tube_held'), g('gate_pipette_held')
    W = {'left': {'hold': lh, 'post_release': t100 > t100[np.flatnonzero(lh)[-1]]}, 'right': {'hold': rh & ~pr2, 'press': pr2, 'post_release': t100 > t100[np.flatnonzero(rh)[-1]]}}
    out = {}
    for side, pre in (('left', 'l_'), ('right', 'r_')):
        for f in ('thumb', 'index', 'middle', 'ring', 'pinky'):
            ks = [k for k, (_, n_) in enumerate(fj) if n_.startswith(pre+f)]
            for wn, sel in W[side].items():
                if sel.sum() < 30: continue
                v = max(float(np.median([np.sqrt(np.mean(filtfilt(*BP, P[w, :, k])[sel]**2)) for w in range(N)])) for k in ks)
                out[f'{side} {f} {wn}'] = round(v, 3)
    res[str(task.name)] = {'band_3_10hz_rms_deg_median_over_worlds': out, 'over_0.25': {k: v for k, v in out.items() if v > TH}}
    print(task.name, 'over 0.25 deg:', res[str(task.name)]['over_0.25'])
    del env; torch.cuda.empty_cache()
if a.out: a.out.write_text(json.dumps({'layer': 'RL env zero-residual replay, no DR, frame 0', 'threshold_PROPOSED': TH, 'tasks': res}, indent=1))
