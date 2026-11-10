"""Test (CPU, no MuJoCo) of caps.py, the action-smoothness loss on the policy mean, inside the real rsl_rl PPO (gpu_train's config):
  1. the loss by hand: equal means -> 0; a change of 0.01 on every actuator -> 1; a pair across an episode end is skipped; the cap;
  2. attached to PPO it is added to every mini-batch loss (logged as 'symmetry') and, with zero reward, the updates make the mean smoother
     (the loss falls) while the std is left to PPO (no pull from the smoothness term: it does not see the noise);
  3. coeff 0 / no config -> nothing attached (existing configs unchanged).
  python -I test_caps.py"""
import os, sys
from pathlib import Path
from types import SimpleNamespace
import torch
from tensordict import TensorDict
HERE = Path(__file__).resolve().parent; ROOT = Path(os.environ.get('G1PS_ROOT', str(HERE.parent/'deps_from_g1ps')))
sys.path[:0] = [str(HERE), str(ROOT/'tools/rl')]
import caps
import gpu_train as GT

N, NO, NU, T = 64, 12, 5, 24


class FakeEnv:
    """obs jump randomly every step (a policy that follows them is jerky); reward 0; episodes of 10 steps."""
    def __init__(self):
        self.num_envs, self.nu, self.device, self.max_episode_length, self.cfg = N, NU, 'cpu', 10, {}
        self.episode_length_buf = torch.zeros(N, dtype=torch.long); self.g = torch.Generator().manual_seed(1); self.o = self._new()

    def _new(self): return torch.randn(N, NO, generator=self.g)
    def get_observations(self): return TensorDict({'policy': self.o}, batch_size=[N])

    def step(self, a):
        self.episode_length_buf += 1; done = self.episode_length_buf >= 10; self.episode_length_buf[done] = 0; self.o = self._new()
        return self.get_observations(), torch.zeros(N), done.long(), {'time_outs': torch.zeros(N)}


def make():
    env = FakeEnv(); venv = GT.RslEnv(env)
    from rsl_rl.runners import OnPolicyRunner
    a = SimpleNamespace(steps_per_env=T, lr=3e-4, init_std=.1, no_value_clip=True, deployable=False)
    return env, venv, OnPolicyRunner(venv, GT.train_cfg(a), log_dir=None, device='cpu')   # actor NOT zeroed here: it starts jerky


def collect(runner, venv):
    alg = runner.alg; obs = venv.get_observations()
    with torch.inference_mode():
        for _ in range(T):
            obs, rew, d, ex = venv.step(alg.act(obs)); alg.process_env_step(obs, rew, d, ex)
        alg.compute_returns(obs)


def test_by_hand():
    class Lin(torch.nn.Module):
        def forward(self, td): return td['policy'][:, :NU]
    h = caps.ActionSmoothness(SimpleNamespace(device='cpu'), 1.)
    o0 = TensorDict({'policy': torch.zeros(4, NO)}, batch_size=[4]); ok = torch.ones(4, dtype=torch.bool)
    assert float(h.loss_on(Lin(), o0, o0.clone(), ok)) == 0, 'equal means must cost nothing'
    o1 = TensorDict({'policy': torch.full((4, NO), .01)}, batch_size=[4])
    v = float(h.loss_on(Lin(), o0, o1, ok)); print(f'change of 0.01 on every actuator: {v:.4f} (expected 1)'); assert abs(v-1) < 1e-4
    o2 = o1.clone(); o2['policy'][0] = 5.; ok2 = ok.clone(); ok2[0] = False
    assert abs(float(h.loss_on(Lin(), o0, o2, ok2))-1) < 1e-4, 'a pair across an episode end must be skipped'
    assert abs(float(h.loss_on(Lin(), o0, o2, ok))-(25+3)/4) < 1e-3, 'cap 25'


def test_ppo():
    torch.manual_seed(0); env, venv, runner = make(); alg = runner.alg
    h = caps.attach(alg, {'coeff': .05, 'pairs': 1024}); assert h is alg.symmetry
    collect(runner, venv)
    with torch.no_grad(): L0 = float(h.compute_loss(alg.actor, None, 0))
    std0 = float(alg.get_policy().output_std.mean()); losses = []
    for it in range(30):
        if it: collect(runner, venv)
        out = alg.update(); losses.append(out['symmetry'])
    collect(runner, venv)
    with torch.no_grad(): L1 = float(h.compute_loss(alg.actor, None, 0))
    std1 = float(alg.get_policy().output_std.mean())
    print(f'smoothness loss on the mean: {L0:.2f} -> {L1:.3f} after 30 PPO updates (zero reward); action std {std0:.4f} -> {std1:.4f}')
    assert 'symmetry' in out and L1 < .2*L0, 'the updates did not smooth the mean'
    assert std1 > .9*std0, 'the std collapsed: the term must not see the exploration noise'


def test_off():
    env, venv, runner = make()
    assert caps.attach(runner.alg, None) is None and caps.attach(runner.alg, {'coeff': 0}) is None and runner.alg.symmetry is None


if __name__ == '__main__':
    test_by_hand(); test_ppo(); test_off(); print('PASS')
