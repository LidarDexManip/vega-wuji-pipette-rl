"""record qpos of a few worlds of a pickup rollout (teacher / student / zero residual) for offline rendering (11-14)
  cd rl; python -I record_pickup.py CONFIG --ckpt CKPT [--n 4] --out OUT.npz [--perturb] [--voc V]"""
import argparse, json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
HERE = Path(__file__).resolve().parent; import os as _os
ROOT = Path(_os.environ.get('G1PS_ROOT', str(HERE.parent/'deps_from_g1ps'))); sys.path[:0] = [str(HERE), str(HERE.parent/'tracker'), str(ROOT/'tools/rl')]
import vega_res_env as E, gpu_train as GT, vega_extras as X
p = argparse.ArgumentParser(); p.add_argument('config', type=Path); p.add_argument('--ckpt', type=Path); p.add_argument('--student', type=Path); p.add_argument('--n', type=int, default=4)
p.add_argument('--out', type=Path, required=True); p.add_argument('--perturb', action='store_true'); p.add_argument('--no-dr', action='store_true'); p.add_argument("--voc", type=float); p.add_argument("--start-frame", type=int, default=0)
p.add_argument('--full', action='store_true', help='(11-16) also save qvel and ctrl per step (CPU replay of contact forces: mj_forward at the recorded state)')
p.add_argument('--student-env', action='store_true', help='(11-20) with --ckpt and a STUDENT config: the teacher in the deployable student env '
               '(same seed -> the same worlds / pushes as a --student run of that config, for side-by-side videos)')
p.add_argument('--aux-log', action='store_true', help='(11-20) with a depth --student that has the aux head: save aux_err_mm[t, w], the error of its '
               'in-hand pipette estimate (6 axis points in the right wrist frame, mean distance) vs the truth, per step')
p.add_argument('--takeover', type=int, help='(11-20) with --student AND --ckpt TEACHER --student-env: from this reference frame on, each world is '
               'driven by the teacher (does the teacher recover from the student\'s states?)')
p.add_argument('--zero-act', nargs='+', default=[], help='(11-20) counterfactual: the policy action of every actuator whose name starts with one of '
               'these prefixes is set to 0 (that joint follows the reference command), e.g. r_middle_finger_')
a0 = p.parse_args(); C = json.loads(a0.config.read_text()); rel = lambda x: (Path(x) if Path(x).is_absolute() else (a0.config.parent/x)).resolve(); task = rel(C['task']) if not C['task'].startswith('../../tasks') or (a0.config.parent/C['task']).exists() else (HERE.parent/C['task'][6:]).resolve()
dr = None if a0.no_dr else json.loads(rel(C['dr']).read_text()) if isinstance(C.get('dr'), str) else C.get('dr')
kw = dict(dr=dr, extras={k: v for k, v in C['extras'].items() if a0.perturb or k != 'perturb'}, reward_scale=C.get('reward_scale', .01))
for k in ('group_scale', 'thumb_press_scale', 'plunger_weight'):
    if C.get(k) is not None: kw[k] = C[k]
if a0.student or a0.student_env: kw['obs_mode'] = 'deployable'
N = a0.n
if a0.voc is not None:
    import tracker_env as TE; env = TE.BimanualGpuEnv(task, N, seed=123, nconmax=200, njmax=1600, naconmax_per_world=400, terminate=False, auto_reset=False, voc_scale=a0.voc, voc_kp=2000., voc_kd=90., **kw)
else: env = E.BimanualGpuEnv(task, N, seed=123, nconmax=200, njmax=1600, naconmax_per_world=400, terminate=False, auto_reset=False, **kw)
pol = None
if a0.ckpt and a0.student_env:   # the teacher reads env.obs (privileged), as in distill; the env still builds the student's inputs (same RNG use)
    import distill as D; pol = D.load_teacher(env, rel(C['teacher_config']), a0.ckpt)
elif a0.ckpt:
    a = SimpleNamespace(steps_per_env=24, lr=3e-4, init_std=.1, no_value_clip=True, deployable=False); venv, runner = GT.make_runner(env, a, None)
    runner.load(str(a0.ckpt), map_location=env.device); pp = runner.get_inference_policy(device=env.device); pol = lambda: pp(env.get_observations())
if a0.student:
    import distill as D
    ck = torch.load(a0.student, map_location=env.device); DX = env.x_depth
    stu = D.StudentCNN(ck['nin'], ck['nout'], ck['hidden'], (DX['H'], DX['W']), aux=ck.get('aux', 0)).to(env.device) if ck.get('depth') else D.Student(ck['nin'], ck['nout'], ck['hidden']).to(env.device)
    stu.load_state_dict(ck['student']); stu.eval(); pol_teacher = pol; AUX = {}
    if a0.takeover is not None: assert pol_teacher is not None and a0.student_env, '--takeover needs --ckpt TEACHER --student-env'
    if ck.get('depth') and a0.aux_log and ck.get('aux'):
        def pol():
            y, ya = stu(env.obs_actor, env.x_depth['obs'], with_aux=True); AUX['pred'] = ya; return y
    else: pol = (lambda: stu(env.obs_actor, env.x_depth['obs'])) if ck.get('depth') else (lambda: stu(env.obs_actor))
    if ck.get('direct'):   # (11-18) direct student: every world executes its absolute command and its own "go" at the descent gate
        assert getattr(env, 'x_direct', False), 'config must have extras.student.direct'
        mu_, sd_ = ck['cmd_mu'].to(env.device), ck['cmd_sd'].to(env.device); pol_d = pol; env.x_dmask = torch.ones(N, dtype=torch.bool, device=env.device)
        def pol():
            y = pol_d(); env.x_dcmd = mu_+sd_*y[:, :env.nu]; env.x_go = y[:, env.nu] > 0; return torch.zeros(N, env.nu, device=env.device)
    elif ck.get('residual_go'):   # (11-20) residual student with its own "go": no privileged servo / gate truth in any world
        pol_r = pol; env.x_smask = torch.ones(N, dtype=torch.bool, device=env.device)
        def pol():
            y = pol_r(); env.x_go = y[:, env.nu] > 0; return y[:, :env.nu].clamp(-1, 1)
dev = env.device; env.reset_envs(torch.ones(N, dtype=torch.bool, device=dev), frames=torch.full((N,), a0.start_frame, dtype=torch.long, device=dev)); env._launch(env.g_kin); env.obs = env._observe()
if env.obs_mode == 'deployable': env._observe_deployable()
nfr = env.nframes; qs = [env.qpos.cpu().numpy().copy()]; fr = [env.frame.cpu().numpy().copy()]; ff = np.full(N, -1); aux_err = []
vs = [env.qvel.cpu().numpy().copy()]; cs = [env.ctrl.cpu().numpy().copy()]
G_ = (C['extras'].get('event_gate') or {}); n_steps = (nfr-1)//env.frame_step+int(round(float(G_.get('cap_s', 0.))/.02))*len(G_.get('events', []))   # (11-18) gate holds
for i in range(n_steps):
    if bool((env.frame >= nfr-1).all()): break
    with torch.inference_mode():
        act = pol().clamp(-1, 1) if pol else torch.zeros(N, env.nu, device=dev)
        if a0.student and AUX.get('pred') is not None:   # (11-20) in-hand estimate error (right pipette: the first 6 x 3 targets, units of 5 cm)
            tgt = X.inhand_targets(env)/.05; e_ = (AUX['pred'][:, :18]-tgt[:, :18]).view(N, 6, 3).norm(dim=-1).mean(1)*50.; aux_err.append(e_.cpu().numpy())
        if a0.student and a0.takeover is not None:   # (11-20) the teacher drives every world from the takeover frame on
            act = torch.where((env.frame >= a0.takeover)[:, None], pol_teacher().clamp(-1, 1), act)
        if a0.zero_act: act[:, [i for i, n in enumerate(env.action_names) if n.startswith(tuple(a0.zero_act))]] = 0.   # (11-20) counterfactual
        _, _, _, ex = env.step(act)
    f = (ex['reasons'] > 0).any(1).cpu().numpy(); k = env.frame.cpu().numpy(); ff = np.where((ff < 0) & f, k, ff)
    qs.append(env.qpos.cpu().numpy().copy()); fr.append(k.copy())
    if a0.full: vs.append(env.qvel.cpu().numpy().copy()); cs.append(env.ctrl.cpu().numpy().copy())
ext = dict(qvel=np.array(vs), ctrl=np.array(cs)) if a0.full else {}
if aux_err: ext["aux_err_mm"] = np.array(aux_err)   # (11-20) aux_err_mm[i] is at the state of qpos[i] (frame[i])
# (11-17) per-world conditions of this recording (one reset at the start): DR vector, robot base height, arm lag, holder placement and its measured value
if getattr(env, 'dr_on', False): ext['dr_vec'] = env.dr_vec.cpu().numpy()
if getattr(env, 'base_offz', None) is not None: ext['base_z'] = env.base_offz.cpu().numpy()
if getattr(env, 'x_lag', None) is not None: ext.update(lag_d=env.x_lag['d'].cpu().numpy(), lag_tau=env.x_lag['tau'].cpu().numpy(), lag_alpha=env.x_lag['alpha'].cpu().numpy())
for nm_, h_ in (getattr(env, 'x_hold', None) or {}).items():
    for k_ in ('d', 'th', 'dm', 'thm'):
        if k_ in h_: ext[f'hold_{nm_}_{k_}'] = h_[k_].cpu().numpy()
np.savez_compressed(a0.out, qpos=np.array(qs), frame=np.array(fr), first_fail=ff, task=str(task), **ext); print('wrote', a0.out, 'first fail frames', ff)
