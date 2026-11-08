"""rsl_rl PPO on the batched MuJoCo Warp bimanual env (tools/rl/gpu_env.py). Run with the repo's .venv-rl (conda retargeting env +
rsl-rl-lib), under the GPU flock.

  sizes  measure the contact / constraint peaks (zero and random residuals, auto-reset over the valid starts) with large buffers and
         write TASK/gpu_sizes.json (naconmax per world = peak broadphase pairs per world x 1.5, njmax = peak rows x 1.5)
  sweep  training throughput (collection + PPO update, env steps/s) and peak VRAM at --num-envs (a few iterations each, own process each)
  train  one run, time-boxed by --minutes (wall clock); per iteration: mean episode length (s), success fraction, failure reasons
         per object, losses, steps/s, VRAM -> RUN/progress.jsonl + tensorboard; checkpoints RUN/model_<it>.pt
  eval   a checkpoint (deterministic, the actor's mean) from the first frame and the spread starts of tools/rl/gpu_replay.py,
         next to the zero-residual baseline

  .venv-rl/bin/python tools/rl/gpu_train.py sizes TASK
  .venv-rl/bin/python tools/rl/gpu_train.py sweep TASK --num-envs 1024 --iterations 3 --out RUN_DIR
  .venv-rl/bin/python tools/rl/gpu_train.py train TASK --num-envs 4096 --minutes 5 --out RUN_DIR
  .venv-rl/bin/python tools/rl/gpu_train.py eval TASK --checkpoint RUN_DIR/model_N.pt --out RUN_DIR/eval
  (11-05: the head stack is .venv-rl-head, conda retargeting_head + rsl-rl-lib; no --gjk-backport there)

PPO (rsl_rl defaults unless noted): 24 steps per env per iteration, 5 epochs x 4 mini-batches, clip 0.2, gamma 0.99, lambda 0.95, adaptive
lr from 3e-4 (desired KL 0.01), entropy 0.001 (YS), value loss 1.0, grad norm 1.0; actor MLP 512-256-128 (YS's), critic 512-512-256-128 (YS's),
ELU, empirical observation normalisation, Gaussian std 0.3 (YS's init), the actor's last layer zero-initialised (first policy mean =
the zero-residual reference)."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_env import BimanualGpuEnv, CONTROL_DT, refuse_backport_on_head
from rsl_rl.env import VecEnv


class RslEnv(VecEnv):
    """Thin rsl_rl VecEnv wrapper."""
    def __init__(self, env):
        self.env = env; self.num_envs = env.num_envs; self.num_actions = env.nu; self.device = env.device
        self.max_episode_length = env.max_episode_length; self.cfg = env.cfg

    @property
    def episode_length_buf(self): return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, v): self.env.episode_length_buf = v

    def get_observations(self): return self.env.get_observations()

    def step(self, actions):
        obs, rew, done, ex = self.env.step(actions)
        return self.env.get_observations(), rew, done, {'time_outs': ex['time_outs'], 'log': {}}


def env_kwargs(args):
    """(11-05) default-off env options from the command line; none given = {} = unchanged."""
    kw = {k: getattr(args, k) for k in ('arm_scale', 'finger_scale', 'thumb_press_scale') if getattr(args, k, None) is not None}
    if getattr(args, 'reward_scale', None) is not None: kw['reward_scale'] = args.reward_scale
    if getattr(args, 'group_scale', None): kw['group_scale'] = {x.split('=')[0]: float(x.split('=')[1]) for x in args.group_scale.split(',')}
    if getattr(args, 'deployable', False): kw['obs_mode'] = 'deployable'
    if getattr(args, 'dr_config', None): kw['dr'] = json.loads(Path(args.dr_config).read_text())
    return kw


def train_cfg(args):
    cfg = _train_cfg(args)
    if getattr(args, 'no_value_clip', False): cfg['algorithm']['use_clipped_value_loss'] = False   # (11-05, default off)
    if getattr(args, 'deployable', False): cfg['obs_groups'] = {'actor': ['policy'], 'critic': ['critic']}   # (11-05, default off) asymmetric: gpu_env obs_mode deployable
    return cfg


def _train_cfg(args):
    return {'num_steps_per_env': args.steps_per_env, 'save_interval': 10**9, 'obs_groups': {'actor': ['policy'], 'critic': ['policy']},
            'logger': 'tensorboard', 'check_for_nan': True,
            'algorithm': {'class_name': 'PPO', 'num_learning_epochs': 5, 'num_mini_batches': 4, 'clip_param': .2, 'gamma': .99, 'lam': .95,
                          'value_loss_coef': 1., 'entropy_coef': .001, 'learning_rate': args.lr, 'max_grad_norm': 1., 'schedule': 'adaptive',
                          'desired_kl': .01},
            'actor': {'class_name': 'MLPModel', 'hidden_dims': [512, 256, 128], 'activation': 'elu', 'obs_normalization': True,
                      'distribution_cfg': {'class_name': 'GaussianDistribution', 'init_std': args.init_std, 'std_type': 'scalar'}},
            'critic': {'class_name': 'MLPModel', 'hidden_dims': [512, 512, 256, 128], 'activation': 'elu', 'obs_normalization': True}}


def make_runner(env, args, log_dir):
    from rsl_rl.runners import OnPolicyRunner
    venv = RslEnv(env); runner = OnPolicyRunner(venv, train_cfg(args), log_dir=log_dir, device=env.device)
    last = [mod for mod in runner.alg.actor.mlp.modules() if isinstance(mod, torch.nn.Linear)][-1]
    with torch.no_grad(): last.weight.zero_(); last.bias.zero_()
    return venv, runner


def run_iterations(env, venv, runner, iterations=None, minutes=None, log=None, writer=None, tag=''):
    alg = runner.alg; obs = venv.get_observations(); alg.train_mode(); rows = []
    t_start = time.time(); it = 0; nsteps = runner.cfg['num_steps_per_env']; total = 0
    torch.cuda.reset_peak_memory_stats()
    while True:
        t0 = time.time()
        with torch.inference_mode():
            for _ in range(nsteps):
                actions = alg.act(obs)
                obs, rew, dones, extras = venv.step(actions)
                alg.process_env_step(obs, rew, dones, extras)
            alg.compute_returns(obs)
        torch.cuda.synchronize(); t1 = time.time()
        loss = alg.update(); torch.cuda.synchronize(); t2 = time.time()
        total += nsteps*env.num_envs
        st = env.pop_stats()
        row = {'it': it, 'env_steps': total, 'wall_s': round(t2-t_start, 2), 'collect_s': round(t1-t0, 3), 'update_s': round(t2-t1, 3),
               'steps_per_s': round(nsteps*env.num_envs/(t2-t0), 1), 'collect_steps_per_s': round(nsteps*env.num_envs/(t1-t0), 1),
               'vram_peak_gb': round(torch.cuda.max_memory_allocated()/1e9, 3), 'vram_reserved_gb': round(torch.cuda.max_memory_reserved()/1e9, 3),
               'lr': alg.learning_rate, 'action_std': float(alg.get_policy().output_std.mean()), **{k: float(v) for k, v in loss.items()}, **st,
               'contact_peak': env.stats_peak.tolist()}
        rows.append(row)
        if log: log(json.dumps(row))
        if writer is not None:
            for k in ('mean_episode_s', 'success_fraction', 'steps_per_s', 'action_std', 'lr', 'vram_peak_gb'): writer.add_scalar(f'train/{k}', row[k], it)
            for k, v in loss.items(): writer.add_scalar(f'loss/{k}', float(v), it)
            for obj in ('pipette', 'tube'):
                for reason, frac in row[f'fail_{obj}'].items(): writer.add_scalar(f'fail_{obj}/{reason}', frac, it)
        it += 1
        if iterations is not None and it >= iterations: break
        if minutes is not None and (time.time()-t_start) >= minutes*60: break
    return rows


def cmd_sizes(a):
    task = Path(a.task); N = a.num_envs
    env = BimanualGpuEnv(task, N, nconmax=500, njmax=2400, naconmax_per_world=500, seed=1)
    peak = np.zeros(3); g = torch.Generator(device=env.device); g.manual_seed(2)
    for i in range(a.steps):
        std = 0. if i < a.steps//3 else a.init_std*(2 if i > 2*a.steps//3 else 1)
        act = torch.randn(N, env.nu, device=env.device, generator=g)*std
        env.step(act); st = env.stats_peak.cpu().numpy(); peak = np.maximum(peak, st)
    sizes = {'measured': {'num_envs': N, 'steps': a.steps, 'actions': f'zero, then N(0, {a.init_std}), then N(0, {2*a.init_std}); auto-reset over the valid starts',
                          'ncollision_total_peak': int(peak[0]), 'nacon_total_peak': int(peak[1]), 'nefc_per_world_peak': int(peak[2]),
                          'ncollision_per_world': round(peak[0]/N, 1), 'nacon_per_world': round(peak[1]/N, 1)},
             **({'overflow': env.overflow_report()} if env.ovf is not None else {}),
             'margin': 1.5, 'naconmax_per_world': int(np.ceil(max(peak[0], peak[1])/N*1.5)), 'njmax': int(np.ceil(peak[2]*1.5)), 'nconmax': int(np.ceil(peak[1]/N*1.5))}
    (task/'gpu_sizes.json').write_text(json.dumps(sizes, indent=1)+'\n'); print(json.dumps(sizes, indent=1))


def cmd_sweep(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for n in a.num_envs:
        t0 = time.time(); env = BimanualGpuEnv(a.task, n, seed=a.seed); init = time.time()-t0
        venv, runner = make_runner(env, a, None)
        rows = run_iterations(env, venv, runner, iterations=a.iterations, log=print)
        res = {'num_envs': n, 'init_s': round(init, 1), 'iterations': rows, 'steps_per_s_last': rows[-1]['steps_per_s'],
               'steps_per_s_median': float(np.median([r['steps_per_s'] for r in rows[1:]] or [rows[-1]['steps_per_s']])),
               'collect_steps_per_s_median': float(np.median([r['collect_steps_per_s'] for r in rows[1:]] or [rows[-1]['collect_steps_per_s']])),
               'vram_peak_gb': max(r['vram_peak_gb'] for r in rows), 'vram_reserved_gb': max(r['vram_reserved_gb'] for r in rows),
               'nvidia_smi_mib': nvsmi(), 'sizes': env.cfg}
        (out/f'sweep_{n}.json').write_text(json.dumps(res, indent=1)+'\n')
        print('SWEEP', n, res['steps_per_s_median'], 'steps/s', res['vram_peak_gb'], 'GB', flush=True)


def nvsmi():
    import subprocess
    try: return int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']).decode().split()[0])
    except Exception: return None


def cmd_train(a):
    from torch.utils.tensorboard import SummaryWriter
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    env = BimanualGpuEnv(a.task, a.num_envs, seed=a.seed, **env_kwargs(a)); venv, runner = make_runner(env, a, str(out))
    (out/'config.json').write_text(json.dumps({'args': {k: str(v) for k, v in vars(a).items()}, 'env': env.cfg, 'ppo': train_cfg(a),
                                               'valid_starts': len(env.valid_starts), 'obs_dim': int(env.obs.shape[1])}, indent=1)+'\n')
    writer = SummaryWriter(str(out/'tb')); f = open(out/'progress.jsonl', 'w')
    def log(s): f.write(s+'\n'); f.flush(); print(s, flush=True)
    rows = run_iterations(env, venv, runner, minutes=a.minutes, log=log, writer=writer)
    path = out/f'model_{len(rows)-1}.pt'; runner.save(str(path)); writer.close()
    (out/'summary.json').write_text(json.dumps({'iterations': len(rows), 'env_steps': rows[-1]['env_steps'], 'wall_s': rows[-1]['wall_s'],
                                                'checkpoint': str(path), 'steps_per_s_median': float(np.median([r['steps_per_s'] for r in rows])),
                                                'vram_peak_gb': max(r['vram_peak_gb'] for r in rows), **({'overflow': env.overflow_report()} if env.ovf is not None else {})}, indent=1)+'\n')


def cmd_eval(a):
    from gpu_replay import rollout, SPREAD_S
    starts = list(a.starts or SPREAD_S); env = BimanualGpuEnv(a.task, len(starts), seed=a.seed, naconmax_per_world=600, njmax=1200, **env_kwargs(a))
    venv, runner = make_runner(env, a, None); runner.load(str(a.checkpoint), map_location=env.device)
    policy = runner.get_inference_policy(device=env.device)
    res, R, q = rollout(env, starts, policy=lambda obs: policy(obs), terminate=not a.no_terminate, record=a.record)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); tag = 'policy_noterm' if a.no_terminate else 'policy'
    (out/f'{tag}.json').write_text(json.dumps({'checkpoint': str(a.checkpoint), 'starts': starts, 'results': res}, indent=1)+'\n')
    np.savez_compressed(out/f'{tag}_series.npz', **R)
    if q is not None: np.savez_compressed(out/f'{tag}_world{a.record}_qpos.npz', qpos=q, dt=env.dt)
    for r in res: print(r['start_s'], 'survived', r['survived_s'], 's success', r['success'], 'pipette', r['pipette_first_fail_s'], r['pipette_first_fail_reason'],
                        '| tube', r['tube_first_fail_s'], r['tube_first_fail_reason'], '| wristR', r['wrist_right_err_mm'], '| press', r.get('press'), '| end', r.get('end_pose'))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd', choices=('sizes', 'sweep', 'train', 'eval')); p.add_argument('task', type=Path)
    p.add_argument('--num-envs', type=int, nargs='*', default=[4096]); p.add_argument('--iterations', type=int, default=3)
    p.add_argument('--minutes', type=float, default=5.); p.add_argument('--out', type=Path); p.add_argument('--steps', type=int, default=150)
    p.add_argument('--steps-per-env', type=int, default=24); p.add_argument('--lr', type=float, default=3e-4); p.add_argument('--init-std', type=float, default=.3)
    p.add_argument('--seed', type=int, default=0); p.add_argument('--checkpoint', type=Path); p.add_argument('--starts', type=float, nargs='*')
    p.add_argument('--no-terminate', action='store_true'); p.add_argument('--record', type=int)
    p.add_argument('--arm-scale', type=float, help='residual scale, waist + arms (rad per action unit); default the env default .08 (11-05)')
    p.add_argument('--finger-scale', type=float, help='residual scale, fingers (rad per action unit); default the env default .25 (11-05)')
    p.add_argument('--thumb-press-scale', type=float, help='factor on the right thumb residual in the press gate; default off (11-05)')
    p.add_argument('--reward-scale', type=float, help='factor on the reward; default off (11-05)')
    p.add_argument('--group-scale', help='residual scale per group, rad per action unit, e.g. FL=.02,FR=.08,AL=.005,AR=.03,W=.005 (groups FL FR AL AR W); default off (11-05)')
    p.add_argument('--no-value-clip', action='store_true', help='rsl_rl use_clipped_value_loss False; default off (11-05)')
    p.add_argument('--deployable', action='store_true', help='asymmetric observations: actor = joint encoders, previous action, reference tables; critic = full state; default off (11-05)')
    p.add_argument('--dr-config', type=Path, help='domain randomisation json (gpu_env.set_dr); default off (11-05)')
    p.add_argument('--gjk-backport', action='store_true', help='patched mujoco_warp collision_gjk, as the physics worker; default off')
    a = p.parse_args()
    if a.gjk_backport:   # same patched collision_gjk as the physics worker's --gjk-backport; before the env imports mujoco_warp
        refuse_backport_on_head()
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from g1_pipette_sim.retarget.gpu.mjw_backport import install_gjk_backport
        install_gjk_backport()
    if a.cmd != 'sweep': a.num_envs = a.num_envs[0]
    {'sizes': cmd_sizes, 'sweep': cmd_sweep, 'train': cmd_train, 'eval': cmd_eval}[a.cmd](a)


if __name__ == '__main__':
    main()
