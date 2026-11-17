"""(11-17) Open-loop regrasp optimisation in physics: CEM on the GPU env, nominal conditions (no DR, the pipette at its nominal place, no pushes).
Every world replays the reference with zero residual up to the regrasp start k_a (the same physical state in all worlds); from k_a to k_hold each
world adds its own residual on the right hand (optionally also the right arm), piecewise linear over K knots (0 at k_a), ramped out over
--ramp-s after k_hold so the hold runs on the reference's own final-grip commands; then the hold until k_hold + --hold-s.
Cost per world: a drop (pipette > 5 cm off its place in the hand or > 8 cm in the world, below 6 cm, non-finite; the in-hand ROTATION is free), the pipette-in-wrist pose vs the reference's at the end, the in-hand path vs the
reference during the window (weak), knot size. The best world's rollout (qpos, ctrl per control step) is saved for the reference rebuild.
  cd rl; python -I opt_regrasp.py CONFIG.json --n 1024 --iters 30 --out OUT.npz [--arm] [--ka-s 4.2 --khold-s 6.72] [--init PREV.npz]
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; import os as _os
ROOT = Path(_os.environ.get('G1PS_ROOT', str(HERE.parent/'deps_from_g1ps'))); sys.path[:0] = [str(HERE), str(HERE.parent/'tracker'), str(ROOT/'tools/rl')]
import vega_res_env as E

p = argparse.ArgumentParser(); p.add_argument('config', type=Path); p.add_argument('--n', type=int, default=1024); p.add_argument('--iters', type=int, default=30)
p.add_argument('--knots', type=int, default=6); p.add_argument('--hold-s', type=float, default=1.); p.add_argument('--ramp-s', type=float, default=.5)
p.add_argument('--std', type=float, default=.5); p.add_argument('--elite', type=int, default=64); p.add_argument('--arm', action='store_true')
p.add_argument('--ka-s', type=float); p.add_argument('--khold-s', type=float); p.add_argument('--init', type=Path); p.add_argument('--seed', type=int, default=0)
p.add_argument('--w-path', type=float, default=.05); p.add_argument('--init-key', default='mean'); p.add_argument('--dr-only', default='', help='with --dr-center: these DR keys (comma list) keep their full range'); p.add_argument('--dr-center', action='store_true', help='physics fixed at the centre of every DR range (the training distribution\'s middle; e.g. kv_hand x3.15), no offsets, nominal placement'); p.add_argument('--dr', action='store_true', help='physics DR and placement as in the config (copies then = different DR draws)'); p.add_argument('--w-tilt', type=float, default=0., help='cost per 5 deg of the largest pipette tilt above --tilt-free during the regrasp'); p.add_argument('--tilt-free', type=float, default=15.)
p.add_argument('--w-contact', type=float, default=0., help='(11-17) cost per control step and missing link while fewer than --min-links right-hand links press on the pipette (regrasp window + ramp): no free flight / single-point pivot'); p.add_argument('--min-links', type=int, default=2); p.add_argument('--drop-dist', type=float, default=.16)
p.add_argument('--copies', type=int, default=1, help='worlds per candidate (the GPU contact solve is not bit-reproducible: cost = mean over copies)'); p.add_argument('--out', type=Path, required=True)
a0 = p.parse_args(); C = json.loads(a0.config.read_text()); rel = lambda x: (Path(x) if Path(x).is_absolute() else (a0.config.parent/x)).resolve()
task = rel(C['task']); tj = json.loads((task/'task.json').read_text()); ramp = tj['regrasp']['final_grip']['ramp_s']
ka_s = a0.ka_s if a0.ka_s is not None else ramp[0]; kh_s = a0.khold_s if a0.khold_s is not None else ramp[1]
ex = {k: v for k, v in C.get('extras', {}).items() if k != 'perturb'}
if ex.get('holder') and not a0.dr: ex['holder'] = {**ex['holder'], 'stand_xy_mm': 0, 'stand_yaw_deg': 0, 'meas_xy_mm': 0, 'meas_yaw_deg': 0}
dr = (json.loads(rel(C['dr']).read_text()) if isinstance(C.get('dr'), str) else C.get('dr')) if a0.dr else None
if a0.dr_center:
    d0 = json.loads(rel(C['dr']).read_text()) if isinstance(C.get('dr'), str) else C.get('dr')
    keep = [k for k in a0.dr_only.split(',') if k]
    dr = {k: (v if k in keep else [.5*(v[0]+v[1])]*2 if isinstance(v, (list, tuple)) and len(v) == 2 else 0.) for k, v in d0.items() if not k.endswith('_frac')}
    print('DR centre:', dr, flush=True)
kw = dict(dr=dr, extras=ex, reward_scale=C.get('reward_scale', .01))
for k in ('group_scale', 'thumb_press_scale', 'plunger_weight'):
    if C.get(k) is not None: kw[k] = C[k]
N = a0.n; env = E.BimanualGpuEnv(task, N, seed=123, nconmax=200, njmax=1600, naconmax_per_world=400, terminate=False, auto_reset=False, starts=np.array([0]), **kw); dev = env.device
names = env.action_names; sel = [i for i, n in enumerate(names) if n.startswith('r_')]+([i for i, n in enumerate(names) if n.startswith('R_arm')] if a0.arm else [])
D = len(sel); K = a0.knots; k_a = int(round(ka_s*100)); k_h = int(round(kh_s*100)); fs = env.frame_step
k_end = min(env.nframes-1, k_h+int(round(a0.hold_s*100))); k_r = int(round(a0.ramp_s*100))
kt = torch.tensor([k_a+(j+1)*(k_h-k_a)/K for j in range(K)], device=dev)     # knot frames (value 0 at k_a)
relR = env.rel_ref['right']
print(f'task {task.name}: regrasp {ka_s:.2f}-{kh_s:.2f} s, hold to {k_end/100:.2f} s; {D} residual dims x {K} knots; {N} worlds', flush=True)

def residual(Kn, k):
    """Kn: N x K x D knot values; k: the current frame (int) -> N x D"""
    if k <= k_a: return torch.zeros(N, D, device=dev)
    if k <= k_h:
        x = torch.cat([torch.tensor([float(k_a)], device=dev), kt]); v = torch.cat([torch.zeros(N, 1, D, device=dev), Kn], 1)
        j = int(torch.searchsorted(x, torch.tensor(float(k), device=dev)).clamp(1, K)); w = (k-x[j-1])/(x[j]-x[j-1]); return (1-w)*v[:, j-1]+w*v[:, j]
    w = max(0., 1.-(k-k_h)/max(1, k_r)); return w*Kn[:, -1]

def rot_deg(A, B): return torch.rad2deg(torch.arccos((((A.transpose(1, 2)@B).diagonal(dim1=1, dim2=2).sum(-1)-1)/2).clamp(-1, 1)))

@torch.inference_mode()
def rollout(Kn, keep=False):
    env.reset_envs(torch.ones(N, dtype=torch.bool, device=dev), frames=torch.zeros(N, dtype=torch.long, device=dev)); env._launch(env.g_kin); env.obs = env._observe()
    fail_k = torch.full((N,), -1, device=dev, dtype=torch.long); cdef = torch.zeros(N, device=dev); clow = torch.zeros(N, device=dev); path = torch.zeros(N, 2, device=dev); tmax = torch.zeros(N, device=dev); npath = 0; qs, cs, frs = [], [], []
    while True:
        k = int(env.frame[0]);
        if k >= k_end: break
        act = torch.zeros(N, env.nu, device=dev); act[:, sel] = residual(Kn, k).clamp(-1, 1)
        _, _, _, x = env.step(act)
        k2 = int(env.frame[0]); ir = x['info']['right']; Oz = env.xpos[:, env.obj_id['right'], 2]
        # (11-17) a real drop = the pipette > --drop-dist from the wrist (reference 7.7-11.2 cm after the lift); the old test (> 5 cm / 8 cm off the
        # KINEMATIC reference) fired on 96% of good physical regrasps, whose path differs from the kinematic one by up to ~5 cm (it sat in a comment until 11-17)
        f = (((env.xpos[:, env.obj_id['right']]-env.xpos[:, env.wrist_id['right']]).norm(dim=-1) > a0.drop_dist) & (k2 >= k_a)) | ~torch.isfinite(env.qpos).all(1)   # from the regrasp start (before: still on the table)
        fail_k = torch.where((fail_k < 0) & f, torch.full_like(fail_k, k2), fail_k)
        if a0.w_contact > 0 and k_a <= k2 <= k_h+k_r:   # (11-17) links pressing on the pipette (force > force_alpha x its weight)
            cur = E.X.contacts(env)[0]; nl = (cur & (env.x_lside == 0)[None]).sum(1); dfc = (a0.min_links-nl).clamp(min=0).float(); cdef += dfc; clow += (dfc > 0).float()
        if k_a <= k2 <= k_h:
            W, O = env._poses('right'); L = torch.linalg.inv(W)@O; R = relR[min(k2, env.nframes-1)]
            tmax = torch.maximum(tmax, torch.rad2deg(torch.arccos(O[:, 2, 2].abs().clamp(max=1.))))
            path[:, 0] += (L[:, :3, 3]-R[:3, 3]).norm(dim=-1)*1e3; path[:, 1] += rot_deg(L[:, :3, :3], R[None, :3, :3].expand(N, 3, 3)); npath += 1
        if keep: qs.append(env.qpos.clone()); cs.append(env.ctrl.clone()); frs.append(k2)
    W, O = env._poses('right'); L = torch.linalg.inv(W)@O; R = relR[k_end]
    ep = (L[:, :3, 3]-R[:3, 3]).norm(dim=-1)*1e3; ea = rot_deg(L[:, :3, :3], R[None, :3, :3].expand(N, 3, 3)); path = path/max(1, npath)
    failed = fail_k >= 0
    cost = torch.where(failed, 100.+(k_end-fail_k).float()/10, torch.zeros_like(ep))+ep/3+ea/3+a0.w_path*(path[:, 0]+path[:, 1])+.1*Kn.pow(2).mean((1, 2))+a0.w_tilt*(tmax-a0.tilt_free).clamp(min=0)/5+a0.w_contact*cdef
    out = dict(cost=cost, failed=failed, fail_k=fail_k, ep=ep, ea=ea, path=path, tmax=tmax, cdef=cdef, clow_s=clow*.02)
    if keep: out.update(qpos=torch.stack(qs), ctrl=torch.stack(cs), frames=frs)
    return out

g = torch.Generator(device=dev); g.manual_seed(a0.seed)
mean = torch.zeros(K, D, device=dev); std = torch.full((K, D), a0.std, device=dev)
if a0.init:
    z = np.load(a0.init); m0_ = torch.as_tensor(z[a0.init_key], device=dev).float(); std = torch.full((K, D), a0.std, device=dev)
    mean = torch.zeros(K, D, device=dev); mean[:, :m0_.shape[1]] = m0_   # (11-17) a hand-only init (20 dims) pads the arm dims (--arm) with 0
best = None; hist = []; t0 = time.time()
M = a0.copies; Nc = N//M; assert Nc*M == N
for it in range(a0.iters+1):
    Kc = (mean[None]+std[None]*torch.randn(Nc, K, D, device=dev, generator=g)).clamp(-1, 1)
    Kc[0] = mean.clamp(-1, 1)                        # candidate 0 = the current mean
    if it == 0 and not a0.init: Kc[1] = 0.             # candidate 1 = zero residual (baseline)
    Kn = Kc.repeat_interleave(M, 0); R_ = rollout(Kn); cw = R_['cost']
    good_w = (R_['ep'] < 5) & (R_['ea'] < 5) & ~R_['failed']; good = good_w.view(Nc, M).float().mean(1)
    fail_all = float(R_['failed'].float().mean()); c = cw.view(Nc, M).mean(1); order = torch.argsort(c); e = order[:max(1, a0.elite//M)]
    R_ = {k_: (v_.view(Nc, M, *v_.shape[1:])[:, 0] if torch.is_tensor(v_) and v_.shape[:1] == (N,) else v_) for k_, v_ in R_.items()}
    j = int(order[0]); rec = dict(it=it, best=float(c[j]), best_good=float(good[j]), mean_good=float(good[0]), all_good=float(good_w.float().mean()), mean_world=float(c[0]), failed_frac=float(R_['failed'].float().mean()), best_ep_mm=float(R_['ep'][j]), best_ea_deg=float(R_['ea'][j]),
                                  best_failed=bool(R_['failed'][j]), elite_cost=float(c[e].mean()), zero=(float(c[1]) if it == 0 and not a0.init else None), zero_fail_s=(float(R_['fail_k'][1])/100 if it == 0 and not a0.init else None),
                                  fail_s_p50=(float(R_['fail_k'][R_['failed']].float().median())/100 if bool(R_['failed'].any()) else None), best_fail_s=float(R_['fail_k'][j])/100, best_tmax=float(R_['tmax'][j]), drop_all=fail_all, best_clow_s=float(R_['clow_s'][j]), best_cdef=float(R_['cdef'][j]), clow_s_p50=float(R_['clow_s'].median()),
                                  ep_p10_50_90=[round(float(v), 1) for v in torch.quantile(R_['ep'], torch.tensor([.1, .5, .9], device=dev))], ea_p10_50_90=[round(float(v), 1) for v in torch.quantile(R_['ea'], torch.tensor([.1, .5, .9], device=dev))], wall_s=round(time.time()-t0, 1))
    hist.append(rec); print(json.dumps(rec), flush=True)
    if best is None or float(c[j]) < best[0]: best = (float(c[j]), Kc[j].clone())
    if it == a0.iters: break
    mean = .8*Kc[e].mean(0)+.2*mean; std = (.8*Kc[e].std(0)+.2*std).clamp(min=.05)
# final: replay the best knots in every world (identical) and keep the rollout
Kb = best[1][None].expand(N, K, D).contiguous(); R_ = rollout(Kb, keep=True)
gw = ((R_['ep'] < 5) & (R_['ea'] < 5) & ~R_['failed']).float().mean(); print(f'replay of the best in all {N} worlds: final grip reached (<5 mm, <5 deg) in {100*float(gw):.0f}%, ep p50 {float(R_["ep"].median()):.1f} mm, ea p50 {float(R_["ea"].median()):.1f} deg', flush=True)
print(f'contact: time with < {a0.min_links} links on the pipette p50 {float(R_["clow_s"].median()):.2f} s, max {float(R_["clow_s"].max()):.2f} s; worlds never below: {100*float((R_["clow_s"] == 0).float().mean()):.0f}%; drops {100*float(R_["failed"].float().mean()):.0f}%', flush=True)
print(f'best cost {best[0]:.2f}: replay failed {bool(R_["failed"][0])}, end in-hand {float(R_["ep"][0]):.1f} mm {float(R_["ea"][0]):.1f} deg, path {R_["path"][0].tolist()}', flush=True)
np.savez_compressed(a0.out, knots=best[1].cpu().numpy(), mean=mean.cpu().numpy(), sel=np.array(sel), names=np.array(names), k_a=k_a, k_h=k_h, k_end=k_end, k_r=k_r,
                    qpos=R_['qpos'][:, 0].cpu().numpy(), ctrl=R_['ctrl'][:, 0].cpu().numpy(), frames=np.array(R_['frames']), hist=json.dumps(hist), task=str(task))
print('wrote', a0.out)
