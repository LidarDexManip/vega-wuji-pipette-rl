"""Unit test (CPU, no MuJoCo): vega_extras.arm_command keeps the COMMANDED arm target >= margin_rad inside every arm limit when
clamp_after_ff is on, and reproduces the round-1/2 violation when it is off (the residual's velocity feed-forward added after the clamp).
  python test_arm_clamp.py"""
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vega_extras as X

def run(clamp_after_ff):
    N, nu = 64, 7; lo, hi = -1.0, 1.0; margin = .1
    cfg = X.merge(X.DEFAULT, {'cmd_hz': 100, 'vel_ff': True, 'clamp_after_ff': clamp_after_ff})
    env = SimpleNamespace(xcfg=cfg, x_hold=None, x_arm=torch.arange(nu), x_alo=torch.full((nu,), lo+margin), x_ahi=torch.full((nu,), hi-margin),
                          dt=.0025, substeps=8, x_ffr=torch.full((nu,), .07), x_prev_arm=torch.zeros(N, nu), x_tick=torch.zeros(N, nu), x_tickq=torch.zeros(N, nu))
    g = torch.Generator().manual_seed(0); worst = 1e9
    sign = torch.where(torch.rand(N, nu, generator=g) > .5, 1., -1.); base = sign*1.05                       # demo command held beyond the bounds
    for step in range(400):
        env.x_rprev = torch.zeros(N, nu); env.x_rlast = sign*torch.rand(N, nu, generator=g)*.06             # residual pushing outward (rad per control step)
        for s in range(env.substeps):
            env.x_sub = s; cmd = base+env.x_rlast; out = X.arm_command(env, cmd)[:, env.x_arm]
            worst = min(worst, float(torch.minimum(out-lo, hi-out).min()))
    return worst

if __name__ == '__main__':
    off, on = run(False), run(True)
    print(f'commanded-target margin to the limit (rad): clamp_after_ff off {off:.4f}, on {on:.4f} (rule >= 0.1)')
    assert off < .1-1e-6, 'expected the old path to violate the 0.1 rad margin (reproduces the bug)'
    assert on >= .1-1e-6, 'commanded target closer than 0.1 rad to a limit with clamp_after_ff on'
    print('PASS')
