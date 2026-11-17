"""(11-17) Learned world model of the right arm + hand + pipette, after SuperMimic (hybridrobotics/supermimic, Apache-2.0: models.Dynamics,
learners.wm_learner): an MLP (Linear-LayerNorm-GELU) maps (normalised state, normalised commands, DR context) to the normalised state change;
trained on multi-step open-loop rollouts (horizon H) with per-group weights 1 / nominal_error^2 (Bryson's rule).
Data: rl/wm_collect.py npz files. Evaluation on held-out worlds: open-loop rollouts of 1 / 8 / 32 control steps (0.02 s each) from real
states with the recorded commands; the pipette's position (mm) and orientation (deg) error in the wrist frame, per task phase, against the
rigid-grasp baseline (the pipette keeps its pose in the hand).
  cd rl; python -I wm_train.py DATA.npz [DATA2.npz ...] --out runs/wm/wm_a.pt [--iters 20000] [--horizon 8] [--hidden 1024,1024,1024]"""
import argparse, json, math, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn

# state layout (rl/wm_collect.py)
G = {'arm_q': slice(0, 7), 'arm_qd': slice(7, 14), 'hand_q': slice(14, 34), 'hand_qd': slice(34, 54), 'pip_pos': slice(54, 57), 'pip_rot': slice(57, 63),
     'pip_v': slice(63, 66), 'pip_w': slice(66, 69), 'plunger': slice(69, 71), 'grav': slice(71, 74)}
NOMINAL = {'arm_q': .01, 'arm_qd': .3, 'hand_q': .02, 'hand_qd': 1., 'pip_pos': .002, 'pip_rot': .02, 'pip_v': .05, 'pip_w': 1., 'plunger': .002, 'grav': .01}
PHASES = {'pickup (<2.6 s)': (0, 260), 'upright (2.6-4.2 s)': (260, 420), 'regrasp (4.2-6.7 s)': (420, 672), 'hold (>6.7 s)': (672, 10**6)}


def build_mlp(i, hidden, o):
    L, c = [], i
    for h in hidden: L += [nn.Linear(c, h), nn.LayerNorm(h), nn.GELU()]; c = h
    L.append(nn.Linear(c, o)); return nn.Sequential(*L)


def gram6(r):
    a, b = r[..., :3], r[..., 3:]; a = a/a.norm(dim=-1, keepdim=True).clamp(min=1e-6); b = b-(a*b).sum(-1, keepdim=True)*a; b = b/b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return torch.cat([a, b], -1)


def rot_from6(r):
    a, b = r[..., :3], r[..., 3:]; return torch.stack([a, b, torch.cross(a, b, dim=-1)], -1)


class Dynamics(nn.Module):
    def __init__(s, ns, nc, nx, hidden, stats):
        super().__init__(); s.mlp = build_mlp(ns+nc+nx, hidden, ns)
        for k, v in stats.items(): s.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))
    def forward(s, x, c, ctx):
        h = torch.cat([(x-s.s_mu)/s.s_sd, (c-s.c_mu)/s.c_sd, (ctx-s.x_mu)/s.x_sd], -1)
        y = x+s.mlp(h)*s.d_sd+s.d_mu
        y = torch.cat([y[..., :57], gram6(y[..., 57:63]), y[..., 63:71], y[..., 71:74]/y[..., 71:74].norm(dim=-1, keepdim=True).clamp(min=1e-6)], -1)
        return y


def load(paths):
    S, C, X, F, D = [], [], [], [], []
    for p in paths:
        z = np.load(p); S.append(z['state']); C.append(z['cmd']); X.append(z['ctx']); F.append(z['frame']); D.append(z['done'])
    # concatenate along worlds (each file: T+1 states, T steps)
    T = min(c.shape[0] for c in C)
    return (np.concatenate([s[:T+1] for s in S], 1), np.concatenate([c[:T] for c in C], 1), np.concatenate([x[:T] for x in X], 1),
            np.concatenate([f[:T] for f in F], 1), np.concatenate([d[:T] for d in D], 1))


def windows(done, H):
    """(t, w) starts whose H transitions stay inside one episode"""
    T, N = done.shape; ok = np.ones((T-H+1, N), bool)
    for k in range(H): ok &= ~done[k:T-H+1+k]
    return np.argwhere(ok)


def rollout(model, s0, cmd, ctx):
    xs, x = [], s0
    for k in range(cmd.shape[1]): x = model(x, cmd[:, k], ctx[:, k]); xs.append(x)
    return torch.stack(xs, 1)


def group_loss(pred, gt):
    out = {}
    for k, sl in G.items(): out[k] = ((pred[..., sl]-gt[..., sl])**2).mean()
    return out


def pip_err(pred, gt):
    ep = (pred[..., 54:57]-gt[..., 54:57]).norm(dim=-1)*1e3
    Rp, Rg = rot_from6(pred[..., 57:63]), rot_from6(gt[..., 57:63]); tr = (Rp.transpose(-1, -2)@Rg).diagonal(dim1=-2, dim2=-1).sum(-1)
    return ep, torch.rad2deg(torch.arccos(((tr-1)/2).clamp(-1, 1)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('data', nargs='+'); ap.add_argument('--out', type=Path, required=True); ap.add_argument('--iters', type=int, default=20000)
    ap.add_argument('--horizon', type=int, default=8); ap.add_argument('--batch', type=int, default=2048); ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--hidden', default='1024,1024,1024'); ap.add_argument('--holdout', type=float, default=.1); ap.add_argument('--eval-every', type=int, default=2000); a = ap.parse_args()
    dev = 'cuda'; torch.manual_seed(0)
    S, C, X, F, D = load(a.data); T, N = C.shape[:2]; nv = max(1, int(N*a.holdout)); tr_w, te_w = np.arange(nv, N), np.arange(nv)
    print(f'data: {T} steps x {N} worlds ({len(tr_w)} train / {len(te_w)} held-out worlds), episodes ended {int(D.sum())}', flush=True)
    St, Ct, Xt = (torch.as_tensor(v, device=dev) for v in (S, C, X)); Dn = D
    valid = ~D; dS = (S[1:]-S[:-1])[valid]; s_all = S[:-1][valid]
    stats = dict(s_mu=s_all.mean(0), s_sd=s_all.std(0)+1e-4, c_mu=C[valid].mean(0), c_sd=C[valid].std(0)+1e-4, x_mu=X[valid].mean(0), x_sd=X[valid].std(0)+1e-3,
                 d_mu=dS.mean(0), d_sd=dS.std(0)+1e-6)
    hidden = tuple(int(h) for h in a.hidden.split(',')); model = Dynamics(S.shape[-1], C.shape[-1], X.shape[-1], hidden, stats).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5); W = {k: 1/NOMINAL[k]**2 for k in G}
    win = windows(Dn, a.horizon); tr = win[np.isin(win[:, 1], tr_w)]; te_w_set = te_w
    print(f'{len(tr)} training windows of {a.horizon} steps; model {sum(p.numel() for p in model.parameters())/1e6:.1f} M params', flush=True)
    trw = torch.as_tensor(tr, device=dev); H = a.horizon; ar = torch.arange(H, device=dev); t0 = time.time(); log = []
    for it in range(a.iters+1):
        if it % a.eval_every == 0 or it == a.iters:
            r = evaluate(model, S, C, X, F, Dn, te_w_set, dev); r['it'] = it; r['wall_s'] = round(time.time()-t0); log.append(r); print(json.dumps(r), flush=True)
        if it == a.iters: break
        b = trw[torch.randint(len(trw), (a.batch,), device=dev)]; t, w = b[:, 0], b[:, 1]
        idx = t[:, None]+ar[None]
        s0 = St[t, w]; cmd = Ct[idx, w[:, None]]; ctx = Xt[idx, w[:, None]]; gt = St[idx+1, w[:, None]]
        pred = rollout(model, s0, cmd, ctx); gl = group_loss(pred, gt)
        loss = sum(W[k]*v*(G[k].stop-G[k].start) for k, v in gl.items())/S.shape[-1]
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
        if it % 500 == 0: print(f'it {it} loss {loss.item():.3f} | ' + ' '.join(f'{k} {math.sqrt(v.item())/NOMINAL[k]:.2f}' for k, v in gl.items()), flush=True)
    torch.save({'model': model.state_dict(), 'hidden': hidden, 'dims': (S.shape[-1], C.shape[-1], X.shape[-1]), 'stats': {k: v for k, v in stats.items()}, 'log': log,
                'data': a.data, 'horizon': a.horizon}, a.out); print('wrote', a.out)


@torch.no_grad()
def evaluate(model, S, C, X, F, D, worlds, dev, horizons=(1, 8, 32), n_max=4000):
    model.eval(); out = {}
    for H in horizons:
        win = windows(D, H); win = win[np.isin(win[:, 1], worlds)]
        if len(win) > n_max: win = win[np.random.default_rng(H).choice(len(win), n_max, replace=False)]
        t, w = win[:, 0], win[:, 1]; idx = t[:, None]+np.arange(H)[None]
        s0 = torch.as_tensor(S[t, w], device=dev); cmd = torch.as_tensor(C[idx, w[:, None]], device=dev); ctx = torch.as_tensor(X[idx, w[:, None]], device=dev)
        gt = torch.as_tensor(S[idx[:, -1]+1, w], device=dev); pred = rollout(model, s0, cmd, ctx)[:, -1]
        ep, ea = pip_err(pred, gt); bp, ba = pip_err(s0, gt)                         # baseline: the pipette keeps its pose in the hand
        fr = F[t, w]; row = {}
        for name, (lo, hi) in PHASES.items():
            m_ = torch.as_tensor((fr >= lo) & (fr < hi), device=dev)
            if m_.sum() < 20: continue
            row[name] = [round(float(ep[m_].median()), 2), round(float(ea[m_].median()), 2), round(float(bp[m_].median()), 2), round(float(ba[m_].median()), 2),
                         round(float(ep[m_].quantile(.9)), 2), round(float(ea[m_].quantile(.9)), 2)]
        hq = (pred[:, 14:34]-gt[:, 14:34]).abs().mean()
        out[f'H{H}'] = {'pipette mm / deg p50 (model), rigid-grasp baseline mm / deg p50, model p90 mm / deg': row, 'hand joint err deg': round(float(torch.rad2deg(hq)), 2)}
    model.train(); return out


if __name__ == '__main__':
    main()
