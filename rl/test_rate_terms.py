"""Unit test (CPU, no MuJoCo): the 11-09 torque-rate and normalised action-rate terms of vega_extras.
  1. a constant torque (steady grip, gravity load) costs nothing; the first step after a reset costs nothing;
  2. arm torque alternating by 2 % of the limit between control steps -> torque_rate_arm = (2 x 2 % / 1 %)^2 = 16 on those joints (mean over arm);
  3. chatter WITHIN a control step averages out (anti-alias: the substep mean is what is differenced); the cap applies;
  4. action_rate with action_rate_norm 0.01: a change of 0.01 normalised action on every actuator = 1; old raw form unchanged without it.
  python test_rate_terms.py"""
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vega_extras as X

N, NU = 3, 6                      # actuators 0-1 arm, 2-5 fingers
TMAX = torch.tensor([100., 25., 2., 2., .3, .3])


def env_(**cfg):
    c = X.merge(X.DEFAULT, cfg)
    e = SimpleNamespace(xcfg=c, num_envs=N, nu=NU, device='cpu', x_trq=torch.zeros(N, NU), x_tsum=torch.zeros(N, NU), x_tn=0, x_tprev=torch.zeros(N, NU),
                        x_tvalid=torch.zeros(N, dtype=torch.bool), x_tmax=TMAX, x_arm=torch.tensor([0, 1]), x_fing=torch.tensor([2, 3, 4, 5]),
                        scale=torch.tensor([.04, .04, .1, .1, .05, .05]), prev=torch.zeros(N, NU), episode_length_buf=torch.full((N,), 5),
                        x_af=torch.zeros(N, NU), x_prev_arm=torch.zeros(N, 2), ctrl=torch.zeros(N, NU), x_rlast=torch.zeros(N, NU), x_rprev=torch.zeros(N, NU),
                        x_tick=torch.zeros(N, 2), x_tickq=torch.zeros(N, 2), x_q1=None)
    return e


def control_step(e, taus):
    """taus: list over the substeps of (N, NU) torques."""
    for t in taus: e.x_trq.copy_(t); X.substep_end(e)
    return X.torque_rate_terms(e)


def test_torque_rate():
    e = env_(); base = torch.tensor([30., -5., .5, .5, .1, .1]).expand(N, NU)
    o = control_step(e, [base]*8); assert float(o['torque_rate_arm'].abs().max()) == 0 and float(o['torque_rate_finger'].abs().max()) == 0, 'first step must cost nothing'
    o = control_step(e, [base]*8); assert float(o['torque_rate_arm'].abs().max()) < 1e-9 and float(o['torque_rate_finger'].abs().max()) < 1e-9, 'constant torque was penalised'
    sgn = 1.; vals = []
    for _ in range(4):        # arm joint 0 alternates +-2 % of its limit between control steps (25 Hz chatter): change 4 % = 4 units of 1 % -> 16 on that joint
        t = base.clone(); t[:, 0] += sgn*.02*TMAX[0]; sgn = -sgn; vals.append(control_step(e, [t]*8))
    v = float(vals[-1]['torque_rate_arm'][0]); print(f'25 Hz arm chatter of +-2 % of the limit: torque_rate_arm {v:.2f} (expected 16 / 2 arm joints = 8)')
    assert abs(v-8.) < 1e-4 and float(vals[-1]['torque_rate_finger'].abs().max()) < 1e-9
    t1, t2 = base.clone(), base.clone(); t1[:, 2] += .2; t2[:, 2] -= .2                  # finger chatter within the control step (alternating substeps): averages out
    o = control_step(e, [t1, t2]*4); assert float(o['torque_rate_finger'].abs().max()) < 1e-6, 'intra-step chatter aliased in'
    t = base.clone(); t[:, 0] += 50.; o = control_step(e, [t]*8); assert abs(float(o['torque_rate_arm'][0])-25.) < 1e-6, 'cap'
    X.reset(e, torch.tensor([1])); o = control_step(e, [base]*8)
    assert float(o['torque_rate_arm'][1]) == 0 and float(o['torque_rate_arm'][0]) > 0, 'reset must clear the previous torque for that world only'


def test_action_rate():
    e = env_(action_rate_norm=.01); r = e.scale*.01*torch.ones(N, NU)      # a change of 0.01 normalised action on every actuator
    v = X.action_rate_term(e, r); print(f'action_rate (norm 0.01) for a 0.01 change on every actuator: {float(v[0]):.3f} (expected 1)')
    assert torch.allclose(v, torch.ones(N), atol=1e-5)
    e.episode_length_buf = torch.tensor([1, 5, 5]); v = X.action_rate_term(e, r); assert float(v[0]) == 0 and abs(float(v[1])-1) < 1e-5, 'first step after reset'
    e3 = env_(action_rate_norm=.01, action_rate_cap=25.); assert float(X.action_rate_term(e3, r*100).max()) == 25., 'cap'
    e2 = env_(); v2 = X.action_rate_term(e2, r); assert torch.allclose(v2, ((e2.scale*.01)**2).mean().expand(N)), 'old raw form changed'
    print(f'  old raw form for the same change: {float(v2[0]):.2e} (why the old action_rate was effectively off)')


if __name__ == '__main__':
    test_torque_rate(); test_action_rate(); print('PASS')
