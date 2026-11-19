"""(11-18) Warm start across a residual-scale change: when a group's scale goes from s_old to s_new (e.g. group_scale AR 0.06 -> 0.10), the
same residual in rad needs actions x s_old / s_new. Scales the output rows (weight, bias) of those actions and, for a PPO teacher, its action
std. The optimizer state is kept (the resume paths load it; Adam moments of the scaled rows are only slightly off). Works on train_tracker checkpoints (actor_state_dict) and distill.py student checkpoints (student).
   rescale_action_rows.py SRC.pt DST.pt --prefix R_arm --factor 0.6"""
import argparse, torch
p = argparse.ArgumentParser(); p.add_argument('src'); p.add_argument('dst'); p.add_argument('--prefix', default='R_arm'); p.add_argument('--factor', type=float, required=True)
a = p.parse_args(); z = torch.load(a.src, map_location='cpu', weights_only=False)
# action order of vega_res_env (L arm 7, left hand 20, R arm 7, right hand 20)
NAMES = [f'L_arm_j{i}' for i in range(1, 8)]+[f'l_{j}' for j in range(20)]+[f'R_arm_j{i}' for i in range(1, 8)]+[f'r_{j}' for j in range(20)]
names = list(z.get('action_names') or NAMES); rows = [i for i, n in enumerate(names) if n.startswith(a.prefix)]
assert len(rows) in (7, 20), rows   # (11-20) 20: a hand (prefix r_ / l_)
sd = z['actor_state_dict'] if 'actor_state_dict' in z else z['student']
outs = [k for k, v in sd.items() if k.endswith('.weight') and v.ndim == 2 and v.shape[0] == len(names)]
assert len(outs) == 1, outs
k = outs[0]; sd[k][rows] *= a.factor; sd[k.replace('.weight', '.bias')][rows] *= a.factor
if 'distribution.std_param' in sd: sd['distribution.std_param'][rows] *= a.factor
z['rescaled_action_rows'] = {'prefix': a.prefix, 'factor': a.factor, 'rows': rows, 'layer': k, 'from': a.src}
torch.save(z, a.dst); print(f'{a.dst}: {k} rows {rows[0]}-{rows[-1]} ({a.prefix}) x {a.factor}' + (' + action std' if 'distribution.std_param' in sd else ''))
