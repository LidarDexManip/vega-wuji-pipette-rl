"""(11-12) Warm-start a tracker / teacher checkpoint (rsl_rl OnPolicyRunner: actor + critic MLPs with empirical obs normalisers, Adam) on an
observation that has ADD more values appended at the end (e.g. extras.tip.obs: 6 + 4 x 4 = 22). New first-layer weight columns = 0 (the
expanded policy starts as exactly the old one), normaliser mean 0 / var 1 / std 1 for the new entries (the new values are already O(1)), Adam
moments of the first-layer weights padded with 0.
  python expand_obs_ckpt.py IN.pt OUT.pt --add 22
"""
import argparse
from pathlib import Path
import torch

p = argparse.ArgumentParser(); p.add_argument('src', type=Path); p.add_argument('dst', type=Path); p.add_argument('--add', type=int, required=True); a = p.parse_args()
ck = torch.load(a.src, map_location='cpu', weights_only=False); n_old = None; changed = []
for net in ('actor_state_dict', 'critic_state_dict'):
    sd = ck[net]; w = sd['mlp.0.weight']; n_old = w.shape[1] if n_old is None else n_old; assert w.shape[1] == n_old, (net, w.shape)
    sd['mlp.0.weight'] = torch.cat([w, torch.zeros(w.shape[0], a.add, dtype=w.dtype)], 1); changed.append(f'{net}.mlp.0.weight')
    for k, fill in (('obs_normalizer._mean', 0.), ('obs_normalizer._var', 1.), ('obs_normalizer._std', 1.)):
        v = sd[k]; assert v.shape[-1] == n_old, (net, k, v.shape); sd[k] = torch.cat([v, torch.full((*v.shape[:-1], a.add), fill, dtype=v.dtype)], -1); changed.append(f'{net}.{k}')
npad = 0
for st in ck['optimizer_state_dict']['state'].values():
    for k in ('exp_avg', 'exp_avg_sq'):
        v = st.get(k)
        if v is not None and v.dim() == 2 and v.shape[1] == n_old: st[k] = torch.cat([v, torch.zeros(v.shape[0], a.add, dtype=v.dtype)], 1); npad += 1
assert npad == 4, f'expected the Adam moments of the two first layers (4 tensors), padded {npad}'
torch.save(ck, a.dst); print(f'{a.src} -> {a.dst}: obs {n_old} -> {n_old+a.add}; padded {changed} and {npad} Adam moments')
