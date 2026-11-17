"""(11-17) Policy learning through the learned world model (rl/wm_train.py), after SuperMimic (hybridrobotics/supermimic, Apache-2.0:
learners/policy_learner.py BPTT + actor-critic, runners/supermimic_runner.py online loop) for the pickup task's right arm + hand.

Policy: obs = [world-model state (74, normalised), DR context (88, normalised), reference features at 0 / 2 / 5 / 12 control steps ahead
(pipette-in-wrist position and rotation, hand and arm joint angles, as errors to the current state), episode progress, previous action];
action = the env's residual for the 27 right-side actuators in [-1, 1] (the env scales it, EMA-smooths it and adds it to the reference command).
Rewards on the model state (differentiable): pipette-in-wrist position / rotation vs the reference, hand / arm joints, action rate; soft
termination when the pipette is > 5 cm off its place in the hand.

Loop (SuperMimic): warm start by behaviour cloning on recorded teacher actions; then per round: collect real-sim transitions with the current
policy (+ noise) -> fine-tune the world model on the growing buffer -> BPTT actor updates (horizon H through the frozen model, critic bootstrap,
GAE critic fit) -> every few rounds a full-episode evaluation in the real sim (128 worlds from frame 0, the config's DR).
  cd rl; python -I wm_policy.py CONFIG.json --wm runs/wm/wm_ab.pt --data runs/wm/data_v11_a.npz --out runs/wm/pol_v11 [--rounds 60]"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, mujoco
HERE = Path(__file__).resolve().parent; import os as _os
ROOT = Path(_os.environ.get('G1PS_ROOT', str(HERE.parent/'deps_from_g1ps'))); sys.path[:0] = [str(HERE), str(HERE.parent/'tracker'), str(ROOT/'tools/rl')]
import vega_res_env as E
import wm_train as WT

OFFS = (0, 2, 5, 12)


def mlp(i, hidden, o, act=nn.ELU):
    L, c = [], i
    for h in hidden: L += [nn.Linear(c, h), act()]; c = h
    L.append(nn.Linear(c, o)); return nn.Sequential(*L)


def make_env(cfg_path, n, dr_on, seed=0, eval_mode=False):
    C = json.loads(cfg_path.read_text()); rel = lambda x: (Path(x) if Path(x).is_absolute() else (cfg_path.parent/x)).resolve(); task = rel(C['task'])
    dr = (json.loads(rel(C['dr']).read_text()) if isinstance(C.get('dr'), str) else C.get('dr')) if dr_on else None
    ex = {k: v for k, v in C.get('extras', {}).items() if k != 'perturb'}
    kw = dict(dr=dr, extras=ex, reward_scale=C.get('reward_scale', .01))
    for k in ('group_scale', 'thumb_press_scale', 'plunger_weight'):
        if C.get(k) is not None: kw[k] = C[k]
    env = E.BimanualGpuEnv(task, n, seed=123+seed, nconmax=200, njmax=1600, naconmax_per_world=400, terminate=not eval_mode, auto_reset=not eval_mode, **kw)
    return env, task


class Side:
    """the 27 right-side actuators, the world-model state from the env, the reference tables"""
    def __init__(s, env, task):
        dev = env.device; names = env.action_names; s.env = env; s.dev = dev
        s.R = torch.as_tensor([i for i, n in enumerate(names) if n.startswith('R_arm')]+[i for i, n in enumerate(names) if n.startswith('r_')], device=dev)
        nu = env.nu; s.ctx_idx = torch.cat([torch.arange(7, device=dev), 7+s.R, 7+nu+s.R, 7+3*nu+s.R]); s.ow, s.ov = env.obj_id['right'], env.obj_v['right']
        s.scale = env.scale[s.R].clone(); s.alpha = env.xcfg['ema_alpha']; s.fstep = env.frame_step
        r = np.load(task/'reference.npz'); m = mujoco.MjModel.from_xml_path(str(task/'scene.xml')); d = mujoco.MjData(m); Q = r['qpos_ref']; nf = len(Q)
        wb, ob = m.body('right_wrist').id, m.body('right_object').id; pos, rot6 = np.zeros((nf, 3)), np.zeros((nf, 6))
        for k in range(nf):
            d.qpos[:] = Q[k]; mujoco.mj_kinematics(m, d); Rw = d.xmat[wb].reshape(3, 3); Ro = d.xmat[ob].reshape(3, 3)
            pos[k] = Rw.T@(d.xpos[ob]-d.xpos[wb]); L = Rw.T@Ro; rot6[k] = np.r_[L[:, 0], L[:, 1]]
        T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)
        aqR = env.aq[s.R].cpu().numpy(); s.pos, s.rot6, s.arm, s.hand, s.nf = T(pos), T(rot6), T(Q[:, aqR[:7]]), T(Q[:, aqR[7:]]), nf
        cs = env.cmd_sim; idx = torch.arange(nf, device=dev)*env.per_frame
        s.base_first = cs[idx.clamp(max=len(cs)-1)][:, s.R].clone(); s.base_last = cs[(idx+env.substeps-1).clamp(max=len(cs)-1)][:, s.R].clone()

    def state(s):
        env = s.env; N = env.num_envs; dev = s.dev; W, O = env._poses('right'); Rw = W[:, :3, :3]; L = torch.linalg.inv(W)@O
        q, v = env.qpos[:, env.aq[s.R]], env.qvel[:, env.av[s.R]]; vo = env.qvel[:, s.ov:s.ov+3]; wo = env.qvel[:, s.ov+3:s.ov+6]
        vo_w = (Rw.transpose(1, 2)@vo[:, :, None])[:, :, 0]; wo_w = (Rw.transpose(1, 2)@(O[:, :3, :3]@wo[:, :, None]))[:, :, 0]
        gw = (Rw.transpose(1, 2)@torch.tensor([0., 0., -1.], device=dev)[None, :, None].expand(N, 3, 1))[:, :, 0]
        pl = torch.stack([env.qpos[:, env.pl_q], env.qvel[:, env.pl_v]], 1)
        return torch.cat([q[:, :7], v[:, :7], q[:, 7:], v[:, 7:], L[:, :3, 3], torch.cat([L[:, :3, 0], L[:, :3, 1]], -1), vo_w, wo_w, pl, gw], 1)

    fl = False   # (11-17) world models trained on 108-d context (rl/wm_collect.py after 11-17) also take the right finger frictionloss
    def ctx(s): return torch.cat([s.env.dr_vec[:, s.ctx_idx], s.env.mf['dof_frictionloss'][:, s.env.av[s.R[7:]]]], 1) if s.fl else s.env.dr_vec[:, s.ctx_idx]


class Agent(nn.Module):
    def __init__(s, side, wm_stats, n_obs=None):
        super().__init__(); s.side = side
        for k in ('s_mu', 's_sd', 'x_mu', 'x_sd'): s.register_buffer(k, torch.as_tensor(wm_stats[k], dtype=torch.float32))
        if n_obs is None: n_obs = s.s_mu.numel()+s.x_mu.numel()+4*36+1+27   # (11-17) state + DR context (88, or 108 with finger stiction) + 4 x 36 reference features + progress + previous action
        s.actor = mlp(n_obs, (512, 512, 256), 27); s.critic = mlp(n_obs, (512, 256), 1); s.register_buffer('mask', torch.ones(27)); s.prior = None

    def obs(s, st, ctx, k, a_prev):
        sd = s.side; fs = []
        for o in OFFS:
            j = (k+o*sd.fstep).clamp(max=sd.nf-1)
            fs += [(sd.pos[j]-st[:, 54:57])/.02, sd.rot6[j], (sd.hand[j]-st[:, 14:34])/.2, (sd.arm[j]-st[:, :7])/.1]
        return torch.cat([(st-s.s_mu)/s.s_sd, (ctx-s.x_mu)/s.x_sd, *fs, (k.float()/sd.nf)[:, None], a_prev], -1)

    def act(s, o): return torch.tanh(s.actor(o))*s.mask


def reward(side, st, k, a, a_prev):
    j = k.clamp(max=side.nf-1); dp = (st[:, 54:57]-side.pos[j]).norm(dim=-1)
    Rp, Rr = WT.rot_from6(st[:, 57:63]), WT.rot_from6(side.rot6[j]); chord = (Rp-Rr).flatten(1).norm(dim=-1)
    he = st[:, 14:34]-side.hand[j]; ae = (st[:, :7]-side.arm[j]).norm(dim=-1)
    r = (2*torch.exp(-(dp/.01)**2)+2*torch.exp(-(chord/.2)**2)+torch.exp(-(he/.1)**2).mean(-1)+torch.exp(-(ae/.05)**2)-.05*(a-a_prev).pow(2).sum(-1))/6
    return r, torch.sigmoid((dp-.05)/.005), dp, chord


def command(side, k, r_prev, r_app):
    """the commands the world model was trained on (ctrl at the first / last physics substep of the step): reference + applied residual;
    the arm residual is streamed linearly from the previous one (env arm_command), so its first substep sees about the mean of both"""
    j = k.clamp(max=side.nf-1); first = side.base_first[j]+r_app; first[:, :7] = side.base_first[j][:, :7]+.5*(r_prev[:, :7]+r_app[:, :7])
    return torch.cat([first, side.base_last[j]+r_app], -1)


class Buffer:
    """chunks of (T+1, N) real transitions (state, cmd, ctx, frame, done, act) on the GPU"""
    def __init__(s, H): s.chunks = []; s.H = H
    def add(s, z, dev):
        c = {k: (z[k].detach().to(dev).clone() if torch.is_tensor(z[k]) else torch.as_tensor(np.asarray(z[k]), device=dev)) for k in ('state', 'cmd', 'ctx', 'frame', 'done', 'act')}
        T, N = c['cmd'].shape[:2]; ok = torch.ones(T-s.H+1, N, dtype=torch.bool, device=dev)
        for i in range(s.H): ok &= ~c['done'][i:T-s.H+1+i].bool()
        c['win'] = torch.nonzero(ok); s.chunks.append(c)
    def sample_windows(s, B):
        sizes = torch.tensor([len(c['win']) for c in s.chunks], dtype=torch.float); ci = torch.multinomial(sizes, B, replacement=True); out = []
        for i in ci.unique().tolist():
            c = s.chunks[i]; n = int((ci == i).sum()); w = c['win'][torch.randint(len(c['win']), (n,), device=c['win'].device)]; out.append((c, w))
        return out
    def sample_starts(s, B, side):
        """start states with their frame, previous action, applied residual (EMA rebuilt from the previous 8 actions of the episode) and a
        command correction held over the rollout: recorded command - modelled command at that step (joint-offset subtraction, holder
        re-anchoring of the arm command, arm streaming feed-forward; constant within an episode up to the small feed-forward part)"""
        S, X, K, A, R, CC = [], [], [], [], [], []
        for c, w in s.sample_windows(B):
            t, n = w[:, 0], w[:, 1]; t = t.clamp(min=1); S.append(c['state'][t, n]); X.append(c['ctx'][t, n]); k = c['frame'][t, n].long(); K.append(k)
            ap = c['act'][t-1, n]; same = ~c['done'][t-1, n].bool(); ap = torch.where(same[:, None], ap, torch.zeros_like(ap)); A.append(ap)
            r = torch.zeros_like(ap)
            for back in range(8, 0, -1):
                tb = (t-back).clamp(min=0); r = side.alpha*side.scale*c['act'][tb, n]+(1-side.alpha)*r if side.alpha else side.scale*c['act'][tb, n]
            r = torch.where(same[:, None], r, torch.zeros_like(r)); R.append(r)
            r_t = side.alpha*side.scale*c['act'][t, n]+(1-side.alpha)*r if side.alpha else side.scale*c['act'][t, n]
            CC.append(c['cmd'][t, n]-command(side, k, r, r_t))
        return torch.cat(S), torch.cat(X), torch.cat(K), torch.cat(A), torch.cat(R), torch.cat(CC)


def wm_finetune(wm, buf, iters, H, batch, opt):
    W = {k: 1/WT.NOMINAL[k]**2 for k in WT.G}; ar = torch.arange(H, device=buf.chunks[0]['state'].device); last = 0.; wm.requires_grad_(True); wm.train()
    for _ in range(iters):
        S0, Cm, Cx, Gt = [], [], [], []
        for c, w in buf.sample_windows(batch):   # (11-17) gather per chunk, ONE rollout on the whole batch (per-chunk rollouts: 15 ms -> 231 ms/iter at 41 chunks)
            t, n = w[:, 0], w[:, 1]; idx = t[:, None]+ar[None]
            S0.append(c['state'][t, n]); Cm.append(c['cmd'][idx, n[:, None]]); Cx.append(c['ctx'][idx, n[:, None]]); Gt.append(c['state'][idx+1, n[:, None]])
        P = WT.rollout(wm, torch.cat(S0), torch.cat(Cm), torch.cat(Cx))
        gl = WT.group_loss(P, torch.cat(Gt)); loss = sum(W[k]*v*(WT.G[k].stop-WT.G[k].start) for k, v in gl.items())/74
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.); opt.step(); last = loss.item()
    return last


def gae(values, last_value, rewards, dones, gamma, lam):
    ret = torch.empty_like(values); adv = 0.; nv = last_value
    for t in reversed(range(values.shape[0])):
        nd = 1.-dones[t]; delta = rewards[t]+nd*gamma*nv-values[t]; adv = delta+nd*gamma*lam*adv; ret[t] = adv+values[t]; nv = values[t]
    return ret


def bptt_update(agent, wm, side, buf, B, H, gamma, lam, opt_a, opt_c, bc_reg=0.):
    st, ctx, k, a_prev, r_app, corr = buf.sample_starts(B, side); k = k.clamp(max=side.nf-1-2*side.fstep)
    for p in wm.parameters(): p.requires_grad_(False)
    ret = 0.; surv = torch.ones(len(st), device=st.device); roll = []; dps = []; reg = 0.
    for t in range(H):
        o = agent.obs(st, ctx, k, a_prev); a = agent.act(o)
        if bc_reg > 0 and agent.prior is not None:
            with torch.no_grad(): ap_ = torch.tanh(agent.prior(o))*agent.mask
            reg = reg+(a-ap_).pow(2).sum(-1)
        r_new = side.scale*a; r_next = side.alpha*r_new+(1-side.alpha)*r_app if side.alpha else r_new
        cmd = command(side, k, r_app, r_next)+corr; st2 = wm(st, cmd, ctx); k2 = (k+side.fstep).clamp(max=side.nf-1)
        rw, pterm, dp, ch = reward(side, st2, k2, a, a_prev); alive_t = (k < side.nf-1).float(); rw = rw*alive_t
        roll.append((o.detach(), rw.detach(), torch.maximum(pterm, 1-alive_t).detach())); dps.append(dp.detach())
        ret = ret+gamma**t*surv*rw; surv = surv*(1-pterm)*alive_t
        st, k, a_prev, r_app = st2, k2, a, r_next
    o_last = agent.obs(st, ctx, k, a_prev)
    agent.critic.requires_grad_(False); v_last = agent.critic(o_last).squeeze(-1); ret = ret+gamma**H*surv*v_last   # gradient through the last state, critic frozen
    loss = -ret.mean()+bc_reg*(reg/H).mean() if bc_reg > 0 else -ret.mean(); opt_a.zero_grad(set_to_none=True); loss.backward()
    for p in agent.actor.parameters():
        if p.grad is not None: p.grad.clamp_(-.1, .1)
    gn = torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.); opt_a.step(); opt_a.zero_grad(set_to_none=True); agent.critic.requires_grad_(True)
    O_ = torch.stack([x[0] for x in roll]); R_ = torch.stack([x[1] for x in roll]); D_ = torch.stack([x[2] for x in roll])
    with torch.no_grad(): V = agent.critic(O_).squeeze(-1); VL = agent.critic(o_last.detach()).squeeze(-1)
    tgt = gae(V, VL, R_, D_, gamma, lam).flatten(); O2 = O_.flatten(0, 1); perm = torch.randperm(len(tgt), device=tgt.device); cl = []
    for _ in range(3):
        for idx in perm.chunk(4):
            l = (agent.critic(O2[idx]).squeeze(-1)-tgt[idx]).pow(2).mean(); opt_c.zero_grad(set_to_none=True); l.backward(); torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.); opt_c.step(); cl.append(l.item())
    return dict(ret=float(ret.mean())/H, grad=float(gn), critic=float(np.mean(cl)), dp_mm=float(torch.stack(dps).mean())*1e3)


@torch.inference_mode()
def collect(env, side, agent, steps, noise, gen):
    N = env.num_envs; dev = env.device; S, CMD, CTX, FR, DONE, ACT = [], [], [], [], [], []; a_prev = torch.zeros(N, 27, device=dev); ctrl_first = torch.zeros(N, env.nu, device=dev)
    def hook(s_):
        if s_ == 0: ctrl_first.copy_(env.ctrl)
    env.substep_hook = hook
    for t in range(steps):
        st = side.state(); ctx = side.ctx(); k = env.frame.clone()
        a = (agent.act(agent.obs(st, ctx, k, a_prev))+noise*torch.randn(N, 27, device=dev, generator=gen)).clamp(-1, 1)
        full = torch.zeros(N, env.nu, device=dev); full[:, side.R] = a; _, _, done, x = env.step(full)
        S.append(st); CMD.append(torch.cat([ctrl_first[:, side.R], env.ctrl[:, side.R]], 1)); CTX.append(ctx); FR.append(k); DONE.append(done.clone()); ACT.append(a)
        a_prev = torch.where(done[:, None], torch.zeros_like(a), a)
    S.append(side.state()); env.substep_hook = None
    return dict(state=torch.stack(S), cmd=torch.stack(CMD), ctx=torch.stack(CTX), frame=torch.stack(FR), done=torch.stack(DONE), act=torch.stack(ACT))


@torch.inference_mode()
def evaluate(env, side, agent):
    """full episodes from frame 0, no auto reset; completion = no failure; final tilt, in-hand error to the reference end; final qpos"""
    N = env.num_envs; dev = env.device; env.terminate = False; env.auto_reset = False
    env.reset_envs(torch.ones(N, dtype=torch.bool, device=dev), frames=torch.zeros(N, dtype=torch.long, device=dev)); env._launch(env.g_kin); env.obs = env._observe()
    a_prev = torch.zeros(N, 27, device=dev); failed = torch.zeros(N, dtype=torch.bool, device=dev)
    while int(env.frame[0]) < env.nframes-1:
        st = side.state(); a = agent.act(agent.obs(st, side.ctx(), env.frame, a_prev)); full = torch.zeros(N, env.nu, device=dev); full[:, side.R] = a
        _, _, _, x = env.step(full); failed |= (x['reasons'] > 0).any(1); a_prev = a
    st = side.state(); W, O = env._poses('right'); tilt = torch.rad2deg(torch.arccos(O[:, 2, 2].abs().clamp(max=1.)))
    _, _, dp, ch = reward(side, st, env.frame, a_prev, a_prev)
    env.terminate = True; env.auto_reset = True; env.reset_envs(torch.ones(N, dtype=torch.bool, device=dev)); env._launch(env.g_kin); env.obs = env._observe()
    ok = ~failed
    return dict(completed=int(ok.sum()), upright_completed=int((ok & (tilt <= 10)).sum()), tilt_p50=float(tilt[ok].median()) if ok.any() else None,
                inhand_mm_p50=float(dp[ok].median()*1e3) if ok.any() else None, inhand_chord_p50=float(ch[ok].median()) if ok.any() else None), env.qpos.clone()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config', type=Path); ap.add_argument('--wm', type=Path, required=True); ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--rounds', type=int, default=60); ap.add_argument('--n-collect', type=int, default=256); ap.add_argument('--collect-steps', type=int, default=100)
    ap.add_argument('--wm-iters', type=int, default=500); ap.add_argument('--pol-iters', type=int, default=100); ap.add_argument('--horizon', type=int, default=32); ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--bc-iters', type=int, default=3000); ap.add_argument('--noise', type=float, default=.05); ap.add_argument('--eval-every', type=int, default=5); ap.add_argument('--n-eval', type=int, default=128)
    ap.add_argument('--fingers-only', action='store_true'); ap.add_argument('--bc-reg', type=float, default=0., help='weight of |a - a_prior|^2 (prior = the frozen behaviour-cloned actor) in the BPTT loss'); ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--gamma', type=float, default=.99); ap.add_argument('--lam', type=float, default=.95); a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True); dev = 'cuda'; t0 = time.time(); log = open(a.out/'log.jsonl', 'a'); gen = torch.Generator(device=dev); gen.manual_seed(0)
    env, task = make_env(a.config, a.n_collect, True); side = Side(env, task)
    ck = torch.load(a.wm, map_location=dev, weights_only=False); wm = WT.Dynamics(*ck['dims'], ck['hidden'], ck['stats']).to(dev); wm.load_state_dict(ck['model'])
    opt_wm = torch.optim.AdamW(wm.parameters(), lr=1e-4, weight_decay=1e-5)
    side.fl = ck['dims'][2] == len(side.ctx_idx)+20
    agent = Agent(side, ck['stats']).to(dev);
    if a.fingers_only: agent.mask[:7] = 0.
    agent = agent; opt_a = torch.optim.Adam(agent.actor.parameters(), lr=a.lr, betas=(.9, .95)); opt_c = torch.optim.Adam(agent.critic.parameters(), lr=a.lr)
    buf = Buffer(max(8, ck['horizon']))
    for p in a.data: buf.add(np.load(p), dev)
    # command model check on recorded data: the world-model input vs reference + EMA(residual)
    c = buf.chunks[0]; Tn = min(400, c['cmd'].shape[0]); r = torch.zeros(c['cmd'].shape[1], 27, device=dev); errs = []; E_ = []; Dn_ = []
    for t in range(Tn):
        rp = r; r = side.alpha*side.scale*c['act'][t]+(1-side.alpha)*r if side.alpha else side.scale*c['act'][t]
        pred = command(side, c['frame'][t].long(), rp, r); keep = ~c['done'][max(0, t-8):t+1].bool().any(0) if t >= 8 else torch.zeros_like(c['done'][0]).bool()
        if keep.any(): errs.append((pred[keep]-c['cmd'][t][keep]).abs().mean(0))
        E_.append(c['cmd'][t]-pred); Dn_.append(c['done'][t].bool())
        r = torch.where(c['done'][t].bool()[:, None], torch.zeros_like(r), r)
    E_ = torch.stack(E_); Dn_ = torch.stack(Dn_); dr_ = []
    for t in range(16, Tn):
        same = ~Dn_[t-16:t].any(0)
        if same.any(): dr_.append((E_[t, same]-E_[t-16, same]).abs().mean(0))
    dr_ = torch.stack(dr_).mean(0); print(f'command correction drift over 16 steps (same episode): arm {float(dr_[:7].mean()):.4f}/{float(dr_[27:34].mean()):.4f} rad, fingers {float(dr_[7:27].mean()):.4f}/{float(dr_[34:].mean()):.4f} rad', flush=True)
    e = torch.stack(errs).mean(0); print(f'command model check: mean |error| arm first/last {float(e[:7].mean()):.4f}/{float(e[27:34].mean()):.4f} rad, fingers {float(e[7:27].mean()):.4f}/{float(e[34:].mean()):.4f} rad', flush=True)
    # behaviour cloning warm start on the recorded teacher actions
    opt_bc = torch.optim.Adam(agent.actor.parameters(), lr=1e-3)
    for it in range(a.bc_iters):
        # target: the action recorded at that step (sample_starts uses t, the action at t is act[t])
        S_, X_, K_, A_, T_ = [], [], [], [], []
        for cc, w in buf.sample_windows(4096):
            t, n = w[:, 0].clamp(min=1), w[:, 1]; same = ~cc['done'][t-1, n].bool(); ap = torch.where(same[:, None], cc['act'][t-1, n], torch.zeros_like(cc['act'][t-1, n]))
            S_.append(cc['state'][t, n]); X_.append(cc['ctx'][t, n]); K_.append(cc['frame'][t, n].long()); A_.append(ap); T_.append(cc['act'][t, n])
        o = agent.obs(torch.cat(S_), torch.cat(X_), torch.cat(K_), torch.cat(A_)); l = (agent.act(o)-torch.cat(T_)).pow(2).mean()
        opt_bc.zero_grad(set_to_none=True); l.backward(); opt_bc.step()
        if it % 500 == 0: print(f'bc {it}: mse {l.item():.4f}', flush=True)
    import copy; agent.prior = copy.deepcopy(agent.actor).requires_grad_(False)
    eval_env, _ = make_env(a.config, a.n_eval, True, seed=7, eval_mode=False)
    side_e = Side(eval_env, task); side_e.fl = side.fl
    agent.side = side_e; ev, q = evaluate(eval_env, side_e, agent); agent.side = side
    rec = dict(round=0, phase='after BC', wall_s=round(time.time()-t0), **ev); print(json.dumps(rec), flush=True); log.write(json.dumps(rec)+'\n'); log.flush()
    env.reset_envs(torch.ones(env.num_envs, dtype=torch.bool, device=dev)); env._launch(env.g_kin); env.obs = env._observe()
    for rd in range(1, a.rounds+1):
        z = collect(env, side, agent, a.collect_steps, a.noise, gen); buf.add({k: v for k, v in z.items()}, dev)
        lw = wm_finetune(wm, buf, a.wm_iters, buf.H, a.batch, opt_wm); stats = []
        for _ in range(a.pol_iters): stats.append(bptt_update(agent, wm, side, buf, a.batch, a.horizon, a.gamma, a.lam, opt_a, opt_c, a.bc_reg))
        rec = dict(round=rd, wall_s=round(time.time()-t0), wm_loss=round(lw, 4), transitions=int(sum(c_['cmd'].shape[0]*c_['cmd'].shape[1] for c_ in buf.chunks)),
                   **{k: round(float(np.mean([s_[k] for s_ in stats])), 4) for k in stats[0]})
        if rd % a.eval_every == 0 or rd == a.rounds:
            agent.side = side_e; ev, q = evaluate(eval_env, side_e, agent); agent.side = side; rec.update(ev)
            torch.save({'agent': agent.state_dict(), 'wm': wm.state_dict(), 'wm_ck': {k: ck[k] for k in ('dims', 'hidden', 'stats')}, 'round': rd}, a.out/f'agent_r{rd}.pt')
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec)+'\n'); log.flush()
    torch.save({'agent': agent.state_dict(), 'wm': wm.state_dict(), 'wm_ck': {k: ck[k] for k in ('dims', 'hidden', 'stats')}}, a.out/'final.pt'); print('done', flush=True)


if __name__ == '__main__':
    main()
