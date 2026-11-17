"""(11-17) Bake the arm-response pre-compensation into a task: ctrl_sim := the env's command with the nominal lead / feed-forward AND
extras.arm_lag_comp applied (vega_extras.arm_comp_init: inverse lag filter + time advance on the arm columns), marked ctrl_sim_verbatim so the
env applies it as is (no lead, no compensation on top). The real runtime streams the same command (plus the policy residual).
  cd rl; python -I bake_arm_comp.py SRC_TASK DST_TASK --comp '{"delay_s": .04, "tau_j34": .0375, "tau_other": .25, "ff_j34": .925, "ff_other": .665}'"""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path[:0] = [str(HERE), str(HERE.parent/'tracker')]
import vega_res_env as E
p = argparse.ArgumentParser(); p.add_argument('src', type=Path); p.add_argument('dst', type=Path); p.add_argument('--comp', required=True); a = p.parse_args()
comp = json.loads(a.comp); env = E.BimanualGpuEnv(a.src, 2, seed=1, nconmax=200, njmax=1600, naconmax_per_world=400, terminate=False, auto_reset=False, starts=np.array([0]), extras={'arm_lag_comp': comp})
cs = env.cmd_sim.cpu().numpy(); r = dict(np.load(a.src/'reference.npz')); assert cs.shape == r['ctrl_sim'].shape
if a.dst.exists(): shutil.rmtree(a.dst)
shutil.copytree(a.src, a.dst)
r['ctrl_sim'] = cs.astype(r['ctrl_sim'].dtype); r['ctrl_sim_verbatim'] = np.array(True); r['arm_lag_comp'] = np.array(json.dumps(comp))
np.savez_compressed(a.dst/'reference.npz', **r)
tj = json.loads((a.dst/'task.json').read_text()); tj['arm_lag_comp'] = dict(comp, note='11-17 ctrl_sim = compensated command (verbatim): stream it as is on the robot'); (a.dst/'task.json').write_text(json.dumps(tj, indent=1))
print('wrote', a.dst, 'arm command change max %.3f rad' % float(np.abs(cs-(r['ctrl_sim'])).max()))
