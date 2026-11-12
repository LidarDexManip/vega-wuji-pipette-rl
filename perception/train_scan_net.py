"""(11-12) Learned holder estimate from the start-of-episode head-camera depth crop (perception/ego_scan_bank.py), the drop-in for
ego_holders.estimate in the "initial depth image only" student.

ScanNet: depth crop (int16 mm, 0 invalid) -> [depth 0.4-1.0 m -> 0..1, validity, pixel x, pixel y] -> conv encoder -> 64-d EMBEDDING ->
  regression: stand (dx, dy, dyaw), rack (dx, dy, dyaw) in the env's holder parametrisation, tube position offset from the reference's start
  tube position (dx, dy, dz); classification: the pick hole (10). The embedding is what the student sees; the regression output re-anchors
  the arm command (as the hand-written estimate did).
Writes OUT/scan_net.pt (weights + crop / normalisation meta), OUT/scan_eval.json (val errors in the units of ego/summary.json, next to the
hand-written estimator's), OUT/policy_bank.npz (truth, prediction and embedding of every scene of the unseen 'policy' split: the student's
per-episode measurement bank).
  .venv-rl-head/bin/python -I perception/train_scan_net.py rl/runs/ego_scan [--epochs 40]
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn

SC = dict(xy=.05, yaw=math.radians(10.), tube=.1, zlo=.4, zhi=1.0)   # target / input normalisation


class ScanNet(nn.Module):
    def __init__(s, H, W, nhole=10, emb=64):
        super().__init__(); s.H, s.W = H, W
        C = lambda a, b, k, st: [nn.Conv2d(a, b, k, st, k//2), nn.ELU()]
        s.enc = nn.Sequential(*C(4, 32, 5, 2), *C(32, 64, 3, 2), *C(64, 64, 3, 1), *C(64, 128, 3, 2), *C(128, 128, 3, 1), *C(128, 128, 3, 2), nn.Flatten())
        with torch.no_grad(): n = s.enc(torch.zeros(1, 4, H, W)).shape[1]
        s.fc = nn.Sequential(nn.Linear(n, 256), nn.ELU(), nn.Linear(256, emb), nn.ELU())
        s.reg = nn.Linear(emb, 9); s.cls = nn.Linear(emb, nhole)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij'); s.register_buffer('xy', torch.stack([xx, yy])[None])

    def inputs(s, d_mm):
        z = d_mm.float()*1e-3; v = (z > 0).float(); d = ((z-SC['zlo'])/(SC['zhi']-SC['zlo'])).clamp(0, 1)*v
        return torch.cat([d[:, None], v[:, None], s.xy.expand(len(z), -1, -1, -1)], 1)

    def forward(s, d_mm):
        e = s.fc(s.enc(s.inputs(d_mm))); return e, s.reg(e), s.cls(e)


def targets(Z):
    t = np.concatenate([Z['stand'][:, :2]/SC['xy'], Z['stand'][:, 2:]/SC['yaw'], Z['rack'][:, :2]/SC['xy'], Z['rack'][:, 2:]/SC['yaw'], (Z['tube']-Z['tube0'])/SC['tube']], 1)
    return torch.tensor(t, dtype=torch.float32), torch.tensor(Z['hole'])


def decode(y):   # normalised regression -> physical (m, rad)
    k = torch.tensor([SC['xy'], SC['xy'], SC['yaw'], SC['xy'], SC['xy'], SC['yaw'], SC['tube'], SC['tube'], SC['tube']], device=y.device); return y*k


def augment(d, g):
    """extra robustness on top of the bank's stereo noise: per-image depth scale +-0.4 %, 1 % pixel dropout"""
    s = 1+.004*torch.randn(len(d), 1, 1, device=d.device, generator=g); x = (d.float()*s).round()
    drop = torch.rand(d.shape, device=d.device, generator=g) < .01; return torch.where(drop | (d <= 0), torch.zeros_like(x), x)


COLS = ['stand_dx mm', 'stand_dy mm', 'stand_dyaw deg', 'rack_dx mm', 'rack_dy mm', 'rack_dyaw deg', 'tube_dx mm', 'tube_dy mm', 'tube_dz mm']


def err_stats(E):   # E (n, 9) physical errors -> per column stats in mm / deg
    out = {}
    for j, c in enumerate(COLS):
        x = E[:, j]*(180/math.pi if 'deg' in c else 1e3)
        out[c] = {'mean': round(float(x.mean()), 2), 'std': round(float(x.std()), 2), 'p95_abs': round(float(np.percentile(np.abs(x), 95)), 2), 'max_abs': round(float(np.abs(x).max()), 2)}
    return out


@torch.no_grad()
def predict(net, D, dev, bs=1024):
    E, Y, C = [], [], []
    for a in range(0, len(D), bs):
        e, y, c = net(D[a:a+bs].to(dev)); E.append(e.cpu()); Y.append(decode(y).cpu()); C.append(c.argmax(1).cpu())
    return torch.cat(E), torch.cat(Y), torch.cat(C)


def export_bank(net, Z, path, dev):
    """a split's scenes -> the student's measurement bank (truth, the network's prediction and embedding)"""
    Emb, Y, C = predict(net, torch.tensor(Z['depth'][:, 0]), dev); Y = Y.numpy()
    np.savez(path, stand=Z['stand'], rack=Z['rack'], hole=Z['hole'], tube=Z['tube'], pipette=Z['pipette'], tube0=Z['tube0'], pipette0=Z['pipette0'],
             pip_slack=Z['pip_slack'], tube_slack=Z['tube_slack'], holes=Z['holes'],
             pred_stand=Y[:, 0:3], pred_rack=Y[:, 3:6], pred_tube=Y[:, 6:9]+Z['tube0'], pred_hole=C.numpy(), emb=Emb.numpy().astype(np.float32))
    return Y, C


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('bank', type=Path); ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--export-only', nargs=2, metavar=('SPLIT', 'OUT_NAME'), help='load OUT/scan_net.pt and write the bank of SPLIT.npz to OUT/OUT_NAME.npz')
    a = ap.parse_args(); dev = 'cuda'; torch.manual_seed(0); g = torch.Generator(device=dev).manual_seed(0)
    if a.export_only:
        ck = torch.load(a.bank/'scan_net.pt', map_location=dev); net = ScanNet(ck['H'], ck['W'], ck['nhole']).to(dev); net.load_state_dict(ck['state_dict']); net.eval()
        export_bank(net, np.load(a.bank/f'{a.export_only[0]}.npz'), a.bank/f'{a.export_only[1]}.npz', dev); print('wrote', a.bank/f'{a.export_only[1]}.npz'); return
    Tr, Va, Po = (np.load(a.bank/f'{n}.npz') for n in ('train', 'val', 'policy'))
    Dtr = torch.tensor(Tr['depth']).to(dev); N, K, H, W = Dtr.shape; ytr, htr = (x.to(dev) for x in targets(Tr))
    Dva = torch.tensor(Va['depth'][:, 0]); yva, hva = targets(Va)
    net = ScanNet(H, W, len(Tr['holes'])).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs*(N//a.bs); sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=.05)
    t0 = time.time(); log = []
    for ep in range(a.epochs):
        net.train(); perm = torch.randperm(N, device=dev, generator=g); k = torch.randint(K, (N,), device=dev, generator=g); tl = 0.
        for b in range(N//a.bs):
            i = perm[b*a.bs:(b+1)*a.bs]; d = augment(Dtr[i, k[i]], g); e, y, c = net(d)
            loss = Fn.smooth_l1_loss(y, ytr[i], beta=.02)+.1*Fn.cross_entropy(c, htr[i])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step(); tl += float(loss)
        net.eval(); _, Y, C = predict(net, Dva, dev); E = (Y-decode(yva)).numpy(); acc = float((C == hva).float().mean())
        row = {'ep': ep, 'loss': round(tl/(N//a.bs), 5), 'val_hole_acc': round(acc, 4), 's': round(time.time()-t0, 1),
               **{c: v['std'] for c, v in err_stats(E).items()}}
        log.append(row); print(row, flush=True)
    _, Y, C = predict(net, Dva, dev); E = (Y-decode(yva)).numpy()
    ref = a.bank.parents[2]/'tasks/task_d02_straight_v3/ego/summary.json'
    hand = json.loads(ref.read_text())['errors'] if ref.exists() else None
    ev = {'val_n': len(Dva), 'val_hole_acc': float((C == hva).float().mean()), 'scan_net_errors': err_stats(E), 'hand_written_estimator_errors (ego/summary.json, 400 scenes)': hand,
          'epochs': a.epochs, 'train_scenes': N, 'noise_draws': K, 'train_s': round(time.time()-t0, 1), 'log': log}
    (a.bank/'scan_eval.json').write_text(json.dumps(ev, indent=1)+'\n')
    torch.save({'state_dict': net.state_dict(), 'H': H, 'W': W, 'nhole': len(Tr['holes']), 'holes': Tr['holes'].tolist(), 'SC': SC, 'crop': json.loads(str(Tr['crop'])),
                'cam': json.loads(str(Tr['cam'])), 'neck': Tr['neck'].tolist(), 'tube0': Tr['tube0'].tolist()}, a.bank/'scan_net.pt')
    # the student's measurement bank: unseen scenes -> truth, prediction, embedding
    Y, C = export_bank(net, Po, a.bank/'policy_bank.npz', dev); export_bank(net, Va, a.bank/'eval_bank.npz', dev)   # eval: scenes unseen by both networks
    Ep = np.concatenate([Y[:, :6]-np.concatenate([Po['stand'], Po['rack']], 1), Y[:, 6:9]+Po['tube0']-Po['tube']], 1)
    ev['policy_split_errors'] = err_stats(Ep); ev['policy_hole_acc'] = float((C.numpy() == Po['hole']).mean())
    (a.bank/'scan_eval.json').write_text(json.dumps(ev, indent=1)+'\n')
    print(json.dumps({k: v for k, v in ev.items() if k != 'log'}, indent=1))


if __name__ == '__main__':
    main()
