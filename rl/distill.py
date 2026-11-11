"""Real-robot line, step 4: DAgger distillation of the privileged teacher into a deployable student (layer: TRAINING on the RL env, MuJoCo Warp;
not a physics demo). The student's actor input is obs_mode 'deployable' + extras.student: joint encoders (+noise; finger velocity by first
difference), previous actions (3-step history), the reference tables, object poses measured once at the episode start (+noise), the measured
holder offsets and the re-anchoring arm command delta. The teacher sees the full state (its training observation).
Rollouts: each world executes the teacher's action with probability beta (drawn per episode; beta decays to 0) else the student's; the label is
always the teacher's action (DAgger). Loss: MSE on the clamped actions. Resumable (latest.pt / state.json); node-ready (all settings in the config).

  .venv-rl-head/bin/python -I distill.py CONFIG.json [--minutes M]
config keys: task, out, teacher_config, teacher_ckpt, num_envs, device, seed, wall_hours, max_iters, steps_per_iter, buffer_iters, epochs,
minibatch, lr, lr_end, beta_iters, hidden, dr, extras (must contain 'student'), group_scale / arm_scale / finger_scale (= the teacher's), nconmax...
"""
import argparse, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
import torch, torch.nn as nn
HERE = Path(__file__).resolve().parent
import os as _os
ROOT = Path(_os.environ.get('G1PS_ROOT', str(Path(__file__).resolve().parents[1]/'deps_from_g1ps')))   # node: export G1PS_ROOT=<g1-pipette-sim checkout>
sys.path[:0] = [str(HERE), str(ROOT/'tools/rl')]
import vega_res_env as E
import vega_extras as X
import gpu_train as GT

DEF = dict(num_envs=1024, device='cuda:0', seed=0, wall_hours=.33, max_iters=10**9, steps_per_iter=24, buffer_iters=8, epochs=2, minibatch=8192,
           lr=1e-3, lr_end=2e-4, beta_iters=100, hidden=[512, 256, 128], latest_minutes=5., ckpt_minutes=20., dr=None, extras={},
           arm_scale=None, finger_scale=None, group_scale=None, reward_scale=.01, nconmax=200, njmax=1600, naconmax_per_world=400, starts=None)


class Student(nn.Module):
    """Running-normalised MLP (ELU); the normaliser freezes after `freeze_after` samples so a deployed copy is a fixed function."""
    def __init__(self, nin, nout, hidden, freeze_after=5e6):
        super().__init__(); dims = [nin]+list(hidden); L = []
        for a, b in zip(dims[:-1], dims[1:]): L += [nn.Linear(a, b), nn.ELU()]
        L.append(nn.Linear(dims[-1], nout)); self.mlp = nn.Sequential(*L)
        self.register_buffer('mu', torch.zeros(nin)); self.register_buffer('var', torch.ones(nin)); self.register_buffer('count', torch.zeros(()))
        self.freeze_after = freeze_after

    @torch.no_grad()
    def update_norm(self, x):
        if float(self.count) >= self.freeze_after: return
        n = x.shape[0]; m = x.mean(0); v = x.var(0, unbiased=False); c = self.count; tot = c+n
        d = m-self.mu; self.mu += d*n/tot; self.var = (self.var*c+v*n+d**2*c*n/tot)/tot; self.count = tot

    def forward(self, x):
        return self.mlp(((x-self.mu)/torch.sqrt(self.var+1e-6)).clamp(-10, 10))


class StudentCNN(Student):
    """(11-10) Student with the two wrist depth images (vega_extras depth cameras; int16 mm, 0 = invalid): one shared conv encoder per
    camera (the left image mirrored so both hands look alike), features concatenated with the normalised vector observation -> MLP.
    Optional auxiliary head: the held objects' axis points in the wrist frames (privileged in simulation; training signal only)."""
    def __init__(self, nin, nout, hidden, hw, feat=64, aux=0, freeze_after=5e6):
        super().__init__(nin, nout, hidden, freeze_after); H, W = hw; self.hw = (H, W)
        self.enc = nn.Sequential(nn.Conv2d(2, 16, 5, 2, 2), nn.ELU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ELU(), nn.Conv2d(32, 32, 3, 2, 1), nn.ELU(), nn.Flatten())
        nflat = 32*((H+7)//8)*((W+7)//8); self.proj = nn.Sequential(nn.Linear(nflat, feat), nn.ELU())
        dims = [nin+2*feat]+list(hidden); L = []
        for a, b in zip(dims[:-1], dims[1:]): L += [nn.Linear(a, b), nn.ELU()]
        L.append(nn.Linear(dims[-1], nout)); self.mlp = nn.Sequential(*L)
        self.aux = nn.Sequential(nn.Linear(2*feat, 128), nn.ELU(), nn.Linear(128, aux)) if aux else None

    def features(self, img):
        z = img.float()*1e-3; v = (z > 0).float(); d = ((z-.05)/.35).clamp(0, 1)*v                 # depth 5-40 cm -> 0..1, validity
        x = torch.stack([d, v], 2); x = torch.cat([x[:, :1], x[:, 1:].flip(-1)], 1)                 # (N, 2 cams, 2 ch, H, W); left mirrored
        N = x.shape[0]; f = self.proj(self.enc(x.flatten(0, 1))); return f.view(N, -1)

    def forward(self, x, img, with_aux=False):
        f = self.features(img); y = self.mlp(torch.cat([((x-self.mu)/torch.sqrt(self.var+1e-6)).clamp(-10, 10), f], 1))
        return (y, self.aux(f)) if with_aux else y


class TeacherView(GT.RslEnv):
    def get_observations(self):
        from tensordict import TensorDict
        return TensorDict({'policy': self.env.obs}, batch_size=[self.num_envs])


def load_teacher(env, cfg_path, ckpt):
    from rsl_rl.runners import OnPolicyRunner
    TC = json.loads(Path(cfg_path).read_text())
    a = SimpleNamespace(steps_per_env=TC.get('steps_per_env', 24), lr=3e-4, init_std=TC.get('init_std', .1), no_value_clip=True, deployable=False)
    hide = {k: os.environ.pop(k) for k in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK') if k in os.environ}   # (11-10) rsl_rl would start its own
    try: runner = OnPolicyRunner(TeacherView(env), GT.train_cfg(a), log_dir=None, device=env.device)   # multi-GPU training from WORLD_SIZE;
    finally: os.environ.update(hide)                                                                    # the teacher is inference only
    runner.load(str(ckpt), map_location=env.device)
    pol = runner.get_inference_policy(device=env.device); tv = TeacherView(env)
    return lambda: pol(tv.get_observations())


def main():
    p = argparse.ArgumentParser(); p.add_argument('config', type=Path); p.add_argument('--minutes', type=float); a0 = p.parse_args()
    C = {**DEF, **json.loads(a0.config.read_text())}
    # (11-10) data parallel over GPUs: one process per GPU, each seeing only its own GPU (CUDA_VISIBLE_DEVICES), started by node_distill.sh with
    # RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT. Every rank runs its own envs, teacher and buffer; gradients and the input-normaliser
    # statistics are averaged over ranks; rank 0 alone logs, checkpoints and decides when to stop. WORLD_SIZE unset = the single-GPU path.
    NW = int(os.environ.get('WORLD_SIZE', 1)); RK = int(os.environ.get('RANK', 0)); main0 = RK == 0; dist = None
    if NW > 1:
        import datetime, torch.distributed as dist
        dist.init_process_group(os.environ.get('DISTILL_BACKEND', 'nccl'), rank=RK, world_size=NW, timeout=datetime.timedelta(minutes=30))
    rel = lambda x: (Path(x) if Path(x).is_absolute() else (a0.config.parent/x)).resolve()
    task = rel(C['task']); out = rel(C['out']); out.mkdir(parents=True, exist_ok=True)
    dr = C['dr'] if isinstance(C['dr'], dict) or C['dr'] is None else json.loads(rel(C['dr']).read_text())
    assert C['extras'].get('student'), 'extras.student missing'
    fstate = out/'state.json'; latest = out/'latest.pt'
    state = json.loads(fstate.read_text()) if fstate.exists() else {'it': 0, 'env_steps': 0, 'wall_s': 0., 'status': 'new', 'starts': 0, 'ckpts': []}
    if state['status'] == 'complete': print('already complete'); return 0
    if main0: (out/'done.marker').unlink(missing_ok=True)

    def save_state(**kw):
        state.update(kw)
        if main0: tmp = fstate.with_suffix('.tmp'); tmp.write_text(json.dumps(state, indent=1)+'\n'); os.replace(tmp, fstate)

    kw = dict(dr=dr, extras=C['extras'], reward_scale=C['reward_scale'], obs_mode='deployable')
    for k in ('arm_scale', 'finger_scale', 'group_scale', 'thumb_press_scale', 'plunger_weight'):   # (11-10) tracker teachers: must match training
        if C.get(k) is not None: kw[k] = C[k]
    env = E.BimanualGpuEnv(task, C['num_envs'], device=C['device'], seed=C['seed']+state['starts']+1000*RK, nconmax=C['nconmax'], njmax=C['njmax'],
                           naconmax_per_world=C['naconmax_per_world'], starts=C['starts'], **kw)
    dev = env.device; N = env.num_envs
    teacher = load_teacher(env, rel(C['teacher_config']), rel(C['teacher_ckpt']))
    PR = C.get('perturb_ramp_iters'); P0 = float((C['extras'].get('perturb') or {}).get('scale', 1.))   # (11-12) hand pushes ramped in over iterations [i0, i1]
    push_scale = lambda it: P0 if not PR else P0*min(1., max(0., (it-float(PR[0]))/max(1., float(PR[1])-float(PR[0]))))
    if getattr(env, 'x_pert', None) is not None: env.x_pert['scale'] = push_scale(state['it'])
    nin = int(env.obs_actor.shape[1]); DX = getattr(env, 'x_depth', None)
    if DX is None: stu = Student(nin, env.nu, C['hidden']).to(dev)
    else:
        aux_w = float(DX['cfg'].get('aux_inhand', 0.)); naux = 36 if aux_w > 0 else 0
        stu = StudentCNN(nin, env.nu, C['hidden'], (DX['H'], DX['W']), aux=naux).to(dev)
    if NW > 1:
        for t_ in stu.state_dict().values(): dist.broadcast(t_, 0)
    opt = torch.optim.Adam(stu.parameters(), lr=C['lr'])
    if latest.exists():
        ck = torch.load(latest, map_location=dev); stu.load_state_dict(ck['student']); opt.load_state_dict(ck['opt']); print('resumed', state['it'], flush=True)
    model = stu if NW == 1 else nn.parallel.DistributedDataParallel(stu, device_ids=[torch.device(dev).index or 0], broadcast_buffers=False)

    @torch.no_grad()
    def update_norm(x):   # Student.update_norm with the batch statistics averaged over ranks (keeps the normalisers identical)
        if float(stu.count) >= stu.freeze_after: return
        n = x.shape[0]; m = x.mean(0); s2 = (x*x).mean(0)
        if NW > 1: t_ = torch.cat([m, s2]); dist.all_reduce(t_); t_ /= NW; m, s2 = t_[:len(m)], t_[len(m):]; n *= NW
        v = (s2-m*m).clamp(min=0); c = stu.count; tot = c+n; dl = m-stu.mu
        stu.mu += dl*n/tot; stu.var = (stu.var*c+v*n+dl**2*c*n/tot)/tot; stu.count = tot
    save_state(starts=state['starts']+1, status='running', config={k: (str(v) if isinstance(v, Path) else v) for k, v in C.items()}, obs_dim=nin,
               actor_layout=env.actor_layout, teacher_obs_dim=int(env.obs.shape[1]))
    T = C['steps_per_iter']; B = C['buffer_iters']*T*N
    bx = torch.zeros(B, nin, device=dev); by = torch.zeros(B, env.nu, device=dev); filled = 0; ptr = 0
    if DX is not None:
        bimg = torch.zeros(B, 2, DX['H'], DX['W'], dtype=torch.int16, device=dev); baux = torch.zeros(B, 36, device=dev) if naux else None
    stu_act = (lambda: stu(env.obs_actor.clone())) if DX is None else (lambda: stu(env.obs_actor.clone(), DX['obs']))
    g = torch.Generator(device=dev); g.manual_seed(C['seed']+1000*state['starts']+100003*RK)
    beta = lambda it: max(0., 1.-it/max(1, C['beta_iters']))
    use_t = torch.rand(N, device=dev, generator=g) < beta(state['it'])
    flog = open(out/'progress.jsonl', 'a') if main0 else None; t_all = time.time(); wall0 = state['wall_s']; t_latest = t_ckpt = time.time(); rc = 0; reason = None
    cap = a0.minutes*60 if a0.minutes else float('inf')

    def keep_going():   # rank 0 decides, every rank follows
        f = torch.tensor([float(state['it'] < C['max_iters'] and wall0+(time.time()-t_all) < C['wall_hours']*3600 and time.time()-t_all < cap)], device=dev)
        if NW > 1: dist.broadcast(f, 0)
        return bool(f.item())

    def save_ckpt(path):
        if not main0: return
        tmp = path.with_suffix('.tmp')
        torch.save({'student': stu.state_dict(), 'opt': opt.state_dict(), 'nin': nin, 'nout': env.nu, 'hidden': C['hidden'], 'it': state['it'], 'depth': DX['cfg'] if DX is not None else None, 'aux': naux if DX is not None else 0,
                    'actor_layout': env.actor_layout, 'action_names': list(env.action_names)}, tmp); os.replace(tmp, path)

    try:
        while keep_going():
            t0 = time.time(); b = beta(state['it']); succ_s = succ_t = n_s = n_t = 0; err = 0.
            with torch.inference_mode():
                for _ in range(T):
                    at = teacher().clamp(-1, 1); xs = env.obs_actor.clone(); as_ = stu_act().clamp(-1, 1)
                    i = torch.arange(ptr, ptr+N, device=dev) % B; bx[i] = xs; by[i] = at
                    if DX is not None:
                        bimg[i] = DX['obs']
                        if naux: baux[i] = X.inhand_targets(env)/.05
                    ptr = (ptr+N) % B; filled = min(B, filled+N)
                    err += float(((as_-at)**2).mean())
                    _, _, done, ex = env.step(torch.where(use_t[:, None], at, as_))
                    d = done.nonzero().squeeze(-1)
                    if len(d):
                        sc = ex['success'][d]; ut = use_t[d]; succ_t += int((sc & ut).sum()); n_t += int(ut.sum()); succ_s += int((sc & ~ut).sum()); n_s += int((~ut).sum())
                        use_t[d] = torch.rand(len(d), device=dev, generator=g) < b
            update_norm(bx[:filled][torch.randint(filled, (min(filled, 65536),), device=dev, generator=g)])
            fr = min(1., state['it']/max(1, C['beta_iters']*3)); lr = C['lr']+(C['lr_end']-C['lr'])*fr
            for pg in opt.param_groups: pg['lr'] = lr
            losses = []; aux_losses = []
            mb = C['minibatch']; nmb = max(1, int(round(float(C['epochs'])*filled/mb))); perm = torch.cat([torch.randperm(filled, device=dev, generator=g) for _ in range(-(-nmb*mb//filled))])
            for j in range(nmb):
                idx = perm[j*mb:(j+1)*mb]
                if DX is None: loss = ((model(bx[idx])-by[idx])**2).mean()
                elif naux:
                    y, ya = model(bx[idx], bimg[idx], with_aux=True); la = ((ya-baux[idx])**2).mean(); loss = ((y-by[idx])**2).mean()+aux_w*la; aux_losses.append(float(la))
                else: loss = ((model(bx[idx], bimg[idx])-by[idx])**2).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(stu.parameters(), 1.); opt.step(); losses.append(float(loss))
            st = env.pop_stats(); state['it'] += 1; state['env_steps'] += T*N*NW; wall = wall0+(time.time()-t_all); state['wall_s'] = wall
            agg = torch.tensor([succ_s, n_s, succ_t, n_t, err, sum(losses), len(losses), sum(aux_losses), len(aux_losses)], dtype=torch.float64, device=dev)
            if NW > 1: dist.all_reduce(agg)
            succ_s, n_s, succ_t, n_t, err, ls, nl, la_, nla = agg.tolist(); n_s, n_t = int(n_s), int(n_t)
            row = {'it': state['it'], 'env_steps': state['env_steps'], 'wall_s': round(wall, 1), 'beta': round(b, 3), 'lr': lr, 'loss': ls/max(1, nl),
                   'rollout_mse_student_vs_teacher': err/(T*NW), 'episodes_student': n_s, 'success_student': succ_s/max(1, n_s), 'episodes_teacher': n_t,
                   'success_teacher': succ_t/max(1, n_t), 'steps_per_s': round(T*N*NW/(time.time()-t0)), 'success_fraction_all (rank 0)': st.get('success_fraction'),
                   'aux_loss': la_/nla if nla else None, 'gpus': NW, 'minibatches': nmb}
            if getattr(env, 'x_pert', None) is not None: row.update(X.perturb_stats(env)); env.x_pert['scale'] = push_scale(state['it'])   # (11-12)
            if not torch.isfinite(torch.as_tensor(row['loss'])): rc, reason = 3, 'non-finite loss'; break
            if main0: flog.write(json.dumps(row)+'\n'); flog.flush(); print(json.dumps(row), flush=True)
            if time.time()-t_latest >= C['latest_minutes']*60: save_ckpt(latest); save_state(); t_latest = time.time()
            if time.time()-t_ckpt >= C['ckpt_minutes']*60:
                pth = out/f"student_{state['it']}.pt"; save_ckpt(pth); state['ckpts'].append(pth.name); save_state(); t_ckpt = time.time()
    except RuntimeError as e:
        rc, reason = 4, f'RuntimeError: {e}'[:500]
    save_ckpt(latest); done = state['it'] >= C['max_iters'] or wall0+(time.time()-t_all) >= C['wall_hours']*3600
    if rc == 0 and done: save_ckpt(out/'final.pt')
    save_state(status='complete' if (rc == 0 and done) else ('aborted' if rc else 'paused'), reason=reason)
    if main0: (out/'done.marker').write_text(json.dumps({'status': state['status'], 'reason': reason, 'it': state['it'], 'wall_s': state['wall_s'], 'gpus': NW})+'\n')
    print('END', RK, state['status'], reason, flush=True)
    if NW > 1: dist.destroy_process_group()
    return rc


if __name__ == '__main__':
    sys.exit(main())
