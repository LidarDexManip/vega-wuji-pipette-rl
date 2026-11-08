"""Layer: start audit on the RL env (one zero-residual control step from each candidate start, no DR). Keeps starts with no trainer failure."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vega_res_env as E
task = Path(sys.argv[1]); r = np.load(task/'reference.npz'); n = len(r['timestamps']); cand = np.arange(0, n-3, 2)
env = E.BimanualGpuEnv(task, len(cand), nconmax=200, njmax=1600, naconmax_per_world=400, seed=0, terminate=False, auto_reset=False, starts=cand)
fr = torch.as_tensor(cand, device=env.device); env.reset_envs(torch.ones(len(cand), dtype=torch.bool, device=env.device), frames=fr); env._launch(env.g_kin)
with torch.inference_mode(): _, _, _, ex = env.step(torch.zeros(len(cand), env.nu, device=env.device))
bad = (ex['reasons'] > 0).any(1).cpu().numpy(); keep = cand[~bad]
np.save(task/'valid_starts.npy', keep); json.dump({'candidates': int(len(cand)), 'kept': int(len(keep)), 'dropped_s': [round(float(r['timestamps'][k]), 2) for k in cand[bad]][:40]}, open(task/'valid_starts_audit.json', 'w'), indent=1)
print(len(cand), 'kept', len(keep), 'dropped at', [round(float(r['timestamps'][k]), 2) for k in cand[bad]][:20])
