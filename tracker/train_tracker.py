"""Real-robot line: resumable PPO on vega_res_env (layer: TRAINING on the RL env, MuJoCo Warp; not a physics demo). Node-ready: every
setting comes from the JSON config (env count, device, steps, wall hours, paths relative to the config unless absolute); checkpoints
latest.pt / model_<steps>.pt / final.pt; a restart with the same out dir resumes. Writes <out>/done.marker (status) at the end.

  .venv-rl-head/bin/python -I train.py CONFIG.json [--out DIR] [--minutes M]   (--minutes caps this invocation's wall time)
"""
import argparse, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
import torch
HERE = Path(__file__).resolve().parent
import os as _os
ROOT = Path(_os.environ.get('G1PS_ROOT', str(Path(__file__).resolve().parents[1]/'deps_from_g1ps')))   # node: export G1PS_ROOT=<g1-pipette-sim checkout>
sys.path[:0] = [str(HERE), str(ROOT/'tools/rl')]
import tracker_env as E   # TRACKER copy of rrline/train.py (sha256[:16] 418473dc27bab2ba, 11-06): env = tracker_env (VOC curriculum, time-warp)
import gpu_train as GT

DEF = dict(num_envs=1024, device='cuda:0', seed=0, max_env_steps=2e7, wall_hours=.33, ckpt_minutes=20., latest_minutes=5., steps_per_env=24,
           lr=3e-4, init_std=.1, arm_scale=None, finger_scale=None, group_scale=None, reward_scale=.01, no_value_clip=True, deployable=False,
           dr=None, extras={}, nconmax=200, njmax=1600, naconmax_per_world=400, starts=None)

p = argparse.ArgumentParser(); p.add_argument('config', type=Path); p.add_argument('--out', type=Path); p.add_argument('--minutes', type=float); a0 = p.parse_args()
C = {**DEF, **json.loads(a0.config.read_text())}
rel = lambda x: (Path(x) if Path(x).is_absolute() else (a0.config.parent/x)).resolve()   # (11-09 fix: an absolute path was returned as str)
task = rel(C['task']); out = a0.out or rel(C.get('out', 'runs/'+a0.config.stem)); out.mkdir(parents=True, exist_ok=True)
dr = C['dr'] if isinstance(C['dr'], dict) or C['dr'] is None else json.loads(rel(C['dr']).read_text())
a = SimpleNamespace(**{k: C[k] for k in ('steps_per_env', 'lr', 'init_std', 'no_value_clip', 'deployable')})
fstate = out/'state.json'; latest = out/'latest.pt'
state = json.loads(fstate.read_text()) if fstate.exists() else {'env_steps': 0, 'wall_s': 0., 'it': 0, 'status': 'new', 'starts': 0, 'ckpts': []}
if state['status'] == 'complete': print('already complete', state); sys.exit(0)
(out/'done.marker').unlink(missing_ok=True)


def save_state(**kw):
    state.update(kw); tmp = fstate.with_suffix('.tmp'); tmp.write_text(json.dumps(state, indent=1)+'\n'); os.replace(tmp, fstate)


kw = dict(dr=dr, extras=C['extras'], reward_scale=C['reward_scale'])
for k in ('arm_scale', 'finger_scale'):
    if C[k] is not None: kw[k] = C[k]
if C['group_scale']: kw['group_scale'] = C['group_scale']
if C['deployable']: kw['obs_mode'] = 'deployable'
TR = C.get('tracker', {}); CUR = TR.get('curriculum', {})
kw.update(voc_scale=float(state.get('voc_scale', TR.get('voc_start', 0.))), voc_kp=TR.get('voc_kp', 400.), voc_kd=TR.get('voc_kd', 40.), voc_free_frac=TR.get('voc_free_frac', 0.), warp=tuple(TR.get('warp', (1., 1.))), idle_finger_mask=bool(TR.get('idle_finger_mask', False)), press_progress_w=float(TR.get('press_progress_w', 0.)))
for _k in ('plunger_weight', 'thumb_press_scale'):
    if C.get(_k) is not None: kw[_k] = C[_k]
env = E.BimanualGpuEnv(task, C['num_envs'], device=C['device'], seed=C['seed']+state['starts'], nconmax=C['nconmax'], njmax=C['njmax'],
                       naconmax_per_world=C['naconmax_per_world'], starts=C['starts'], **kw)
venv, runner = GT.make_runner(env, a, None)
if latest.exists(): runner.load(str(latest), map_location=env.device); print('resumed from', latest, 'at', state['env_steps'], flush=True)
import caps   # (11-09) optional CAPS action smoothness on the policy mean (config "caps"; logged as loss 'symmetry'), as rl/train.py; absent -> nothing changes
if caps.attach(runner.alg, C.get('caps')): print('caps action smoothness attached:', C['caps'], flush=True)
save_state(starts=state['starts']+1, status='running', config=dict(file=str(a0.config), C={k: (str(v) if isinstance(v, Path) else v) for k, v in C.items()}, env=env.cfg,
                                                                    ppo=GT.train_cfg(a), obs_dim=int(env.obs.shape[1]), valid_starts=len(env.valid_starts)))
nf = torch.zeros(1, dtype=torch.long, device=env.device); orig = env._reward_and_termination
def wrapped(res_):
    nf[0] += (~(torch.isfinite(env.qpos).all(1) & torch.isfinite(env.qvel).all(1))).sum(); return orig(res_)
env._reward_and_termination = wrapped
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(str(out/'tb')); flog = open(out/'progress.jsonl', 'a')
alg = runner.alg; obs = venv.get_observations(); alg.train_mode(); nsteps = C['steps_per_env']; t_all = time.time(); wall0 = state['wall_s']
t_latest = t_ckpt = time.time(); rc = 0; reason = None; cap = (a0.minutes*60) if a0.minutes else float('inf')
voc = float(state.get('voc_scale', TR.get('voc_start', 0.))); succ_win = []; its_at_level = int(state.get('its_at_level', 0))


def save_ckpt(path):
    runner.current_learning_iteration = state['it']; tmp = path.with_suffix('.tmp'); runner.save(str(tmp)); os.replace(tmp, path)


try:
    while state['env_steps'] < C['max_env_steps'] and wall0+(time.time()-t_all) < C['wall_hours']*3600 and time.time()-t_all < cap:
        t0 = time.time(); xs = {}
        with torch.inference_mode():
            for _ in range(nsteps):
                obs, rew, dones, extras = venv.step(alg.act(obs)); alg.process_env_step(obs, rew, dones, extras)
            alg.compute_returns(obs)
        t1 = time.time(); loss = alg.update(); t2 = time.time(); st = env.pop_stats()
        wall = wall0+(time.time()-t_all); state['env_steps'] += nsteps*env.num_envs; state['it'] += 1; state['wall_s'] = wall
        row = {'it': state['it'], 'env_steps': state['env_steps'], 'wall_s': round(wall, 1), 'steps_per_s': round(nsteps*env.num_envs/(t2-t0)),
               'action_std': float(alg.get_policy().output_std.mean()), **{k: float(v) for k, v in loss.items()}, **st,
               'contact_peak': env.stats_peak.tolist(), 'nonfinite_worlds': int(nf[0]), 'vram_gb': round(torch.cuda.max_memory_allocated()/2**30, 2)}
        fs = env.pop_free_stats(); row['voc_scale'] = voc; row['free_success_fraction'] = fs['success_fraction']; row['free_episodes'] = fs['episodes']
        succ_win = (succ_win+[row['success_fraction']])[-int(CUR.get('window_its', 5)):]; its_at_level += 1
        if voc > 0 and its_at_level >= int(CUR.get('min_its_per_level', 20)) and len(succ_win) == int(CUR.get('window_its', 5)) and min(succ_win) >= float(CUR.get('success_threshold', .6)):
            voc = voc*float(CUR.get('factor', .8)); voc = 0. if voc < float(CUR.get('floor', .05)) else voc; env.set_voc(voc); its_at_level = 0; succ_win = []
            row['voc_change'] = voc
        vz = TR.get('voc_zero_by_steps')   # (11-09) optional [s0, s1]: VOC <= a linear 1 -> 0 cap between env steps s0 and s1 (the success curriculum can go lower sooner)
        if vz and voc > 0:
            capv = min(1., max(0., (float(vz[1])-state['env_steps'])/(float(vz[1])-float(vz[0]))))
            if voc > capv: voc = capv if capv >= float(CUR.get('floor', .05)) else 0.; env.set_voc(voc); row['voc_change'] = voc; row['voc_capped'] = True
        state['voc_scale'] = voc; state['its_at_level'] = its_at_level
        flog.write(json.dumps(row)+'\n'); flog.flush()
        print(json.dumps({k: row[k] for k in ('it', 'env_steps', 'wall_s', 'steps_per_s', 'action_std', 'success_fraction', 'mean_episode_s', 'nonfinite_worlds', 'voc_scale', 'free_success_fraction')}), flush=True)
        for k in ('mean_episode_s', 'success_fraction', 'steps_per_s', 'action_std', 'voc_scale', 'free_success_fraction'): writer.add_scalar(f'train/{k}', row[k], state['env_steps'])
        if int(nf[0]) > 0: rc, reason = 3, f"{int(nf[0])} non-finite world steps by it {state['it']}"; break
        if time.time()-t_latest >= C['latest_minutes']*60: save_ckpt(latest); save_state(); t_latest = time.time()
        ck_s = C.get('ckpt_steps')   # (11-06) also a checkpoint every ckpt_steps env steps (crossing a multiple)
        if time.time()-t_ckpt >= C['ckpt_minutes']*60 or (ck_s and (state['env_steps']//int(ck_s)) > ((state['env_steps']-nsteps*env.num_envs)//int(ck_s))):
            pth = out/f"model_{state['env_steps']}.pt"; save_ckpt(pth); state['ckpts'].append(pth.name); save_state(); t_ckpt = time.time()
except RuntimeError as e:
    rc, reason = 4, f'RuntimeError: {e}'[:500]
save_ckpt(latest)
done = state['env_steps'] >= C['max_env_steps'] or wall0+(time.time()-t_all) >= C['wall_hours']*3600
if rc == 0 and done: save_ckpt(out/'final.pt')
save_state(status='complete' if (rc == 0 and done) else ('aborted' if rc else 'paused'), reason=reason)
(out/'done.marker').write_text(json.dumps({'status': state['status'], 'reason': reason, 'env_steps': state['env_steps'], 'wall_s': state['wall_s']})+'\n')
print('END', state['status'], reason, flush=True); sys.exit(rc)
