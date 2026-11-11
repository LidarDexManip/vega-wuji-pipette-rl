"""Record RL policy rollouts for videos (layer: RL env, MuJoCo Warp physics; deterministic actor mean). All worlds start at frame 0 with fixed
per-world hole pair and placement offsets; the env's DR (config) is on. Per world: qpos at every control step, the task outcome (reach end,
tube / pipette seated at the end, tube end lateral / height / tilt vs its seated reference), and the case spec. Writes OUT/world_<i>.npz + OUT/summary.json.
  python record_rollouts.py CONFIG.json CKPT OUT --worlds SPEC.json
  --student PT: a distilled deployable student (distill.py checkpoint; the config must carry extras.student) instead of the teacher CKPT
  (pass any path as CKPT); a depth student also stores the depth images it was given (int16 mm, every render) in world_<i>.npz depth / depth_t.
SPEC.json: list of {"pair": [pick, place], "rack": [dx_mm, dy_mm, dyaw_deg], "stand": [dx_mm, dy_mm, dyaw_deg]} (pair must be in the config's hole_pairs)"""
import argparse, json, math, os, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
HERE = Path(__file__).resolve().parent; ROOT = Path(os.environ.get('G1PS_ROOT', str(Path(__file__).resolve().parents[1]/'deps_from_g1ps')))
sys.path[:0] = [str(HERE), str(ROOT/'tools/rl')]
import vega_res_env as E
import gpu_train as GT
p = argparse.ArgumentParser(); p.add_argument('config', type=Path); p.add_argument('ckpt', type=Path); p.add_argument('out', type=Path); p.add_argument('--worlds', type=Path, required=True)
p.add_argument('--zero', action='store_true', help='zero residual (no policy), same perturbations'); p.add_argument('--no-dr', action='store_true', help='physical DR off (holders / hole pairs as specified)')
p.add_argument('--student', type=Path, help='(11-11) distilled deployable student instead of the teacher CKPT')
p.add_argument('--perturb', action='store_true', help='(11-12) keep the config\'s extras.perturb hand pushes; default: off')
p.add_argument('--init-noise', type=float, nargs=2, default=[0., 0.], metavar=('ARM_RAD', 'FINGER_RAD'), help='(11-10) uniform +- offset of every arm / finger joint at the start (the reference and its clock start unchanged)')
a0 = p.parse_args(); C = json.loads(a0.config.read_text()); rel = lambda x: (Path(x) if Path(x).is_absolute() else (a0.config.parent/x)).resolve()
task = rel(C['task']); dr = json.loads(rel(C['dr']).read_text()) if isinstance(C.get('dr'), str) else C.get('dr')
if a0.no_dr: dr = None
S = json.loads(a0.worlds.read_text()); N = len(S); r = np.load(task/'reference.npz'); t = r['timestamps']; nfr = len(t)
kw = dict(dr=dr, extras={k_: v_ for k_, v_ in C.get('extras', {}).items() if a0.perturb or k_ != 'perturb'}, reward_scale=C.get('reward_scale', .01))   # (11-12) pushes off unless --perturb
for k in ('arm_scale', 'finger_scale', 'group_scale', 'thumb_press_scale', 'plunger_weight'):   # (11-10) tracker configs: must match training
    if C.get(k) is not None: kw[k] = C[k]
if a0.student: kw['obs_mode'] = 'deployable'
env = E.BimanualGpuEnv(task, N, seed=321, nconmax=C.get('nconmax', 200), njmax=C.get('njmax', 1600), naconmax_per_world=C.get('naconmax_per_world', 400), terminate=False, auto_reset=False, **kw)
a = SimpleNamespace(steps_per_env=C.get('steps_per_env', 24), lr=3e-4, init_std=C.get('init_std', .1), no_value_clip=True, deployable=False)
if a0.zero: pol = lambda o: torch.zeros(N, env.nu, device=env.device)
elif a0.student:   # (11-11) as eval_rr.py
    import distill as D
    ck = torch.load(a0.student, map_location=env.device); DX = env.x_depth if ck.get('depth') else None
    stu = (D.StudentCNN(ck['nin'], ck['nout'], ck['hidden'], (DX['H'], DX['W']), aux=ck.get('aux', 0)) if DX is not None else D.Student(ck['nin'], ck['nout'], ck['hidden'])).to(env.device)
    stu.load_state_dict(ck['student']); stu.eval(); assert ck['nin'] == env.obs_actor.shape[1], (ck['nin'], env.obs_actor.shape)
    pol = (lambda o: stu(env.obs_actor, env.x_depth['obs'])) if DX is not None else (lambda o: stu(env.obs_actor))
if not (a0.zero or a0.student): venv, runner = GT.make_runner(env, a, None); runner.load(str(a0.ckpt), map_location=env.device); pol = runner.get_inference_policy(device=env.device)
dev = env.device; pairs = [tuple(x) for x in C['extras']['holder']['hole_pairs']]
env.x_hole_fixed = torch.as_tensor([pairs.index(tuple(s['pair'])) for s in S], device=dev)
fx = lambda key: (torch.as_tensor([[s[key][0]*1e-3, s[key][1]*1e-3] for s in S], dtype=torch.float32, device=dev), torch.as_tensor([math.radians(s[key][2]) for s in S], dtype=torch.float32, device=dev))
env.x_hold_fixed = {'rack': fx('rack'), 'stand': fx('stand')}
env.reset_envs(torch.ones(N, dtype=torch.bool, device=dev), frames=torch.zeros(N, dtype=torch.long, device=dev))
if any(a0.init_noise):   # (11-10) start the robot off its reference pose; joint limits respected
    gi = torch.Generator(device=dev); gi.manual_seed(777); lim = torch.where(env.finger_mask, a0.init_noise[1], a0.init_noise[0])
    jid = env.mjm.actuator_trnid[:, 0]; jr = torch.as_tensor(env.mjm.jnt_range[jid], dtype=torch.float32, device=dev)
    env.qpos[:, env.aq] = (env.qpos[:, env.aq]+lim*(2*torch.rand(N, env.nu, device=dev, generator=gi)-1)).clamp(jr[:, 0], jr[:, 1])
env._launch(env.g_kin); env.obs = env._observe()
Q, T, U = [env.qpos.cpu().numpy().copy()], [float(t[0])], [env.ctrl.cpu().numpy().copy()]; first_fail = torch.full((N,), -1., device=dev)
fail_why = torch.zeros(N, 2, dtype=torch.long, device=dev)   # (11-10) per object [pipette, tube] at the first failure: 1 + index into E.FAILURES, 0 = ok
press = torch.as_tensor(r['phase'] == 2, device=dev); pl_ref = torch.as_tensor(r['plunger'], device=dev).float(); pl_max = torch.zeros(N, device=dev); pl_s = torch.zeros(N, device=dev)
DEP, DEPT = [], []; save_dep = bool(a0.student) and getattr(env, 'x_depth', None) is not None
for i in range((nfr-1)//env.frame_step):
    with torch.inference_mode(): act = pol(env.get_observations()).clamp(-1, 1); _, _, _, ex = env.step(act)
    k = env.frame.clamp(max=nfr-1); f_any = (ex['reasons'] > 0).any(1); nw = f_any & (first_fail < 0); first_fail[nw] = t[0]+env.frame[nw].float()*.01; fail_why[nw] = ex['reasons'][nw]
    depth = -env.qpos[:, env.pl_q]; inp = press[k] & (first_fail < 0); pl_max = torch.where(inp, torch.maximum(pl_max, depth), pl_max)   # (11-10) eval_rr.py press rule
    pl_s += (inp & (depth >= .9*pl_ref.max())).float()*E.CONTROL_DT
    Q.append(env.qpos.cpu().numpy().copy()); T.append(float(t[int(k[0])])); U.append(env.ctrl.cpu().numpy().copy())
    if save_dep and (not DEP or env.x_depth['tick'] % int(env.x_depth['cfg']['every']) == 0): DEP.append(env.x_depth['obs'].cpu().numpy().copy()); DEPT.append(len(T)-1)   # the image the student acts on from frame len(T)-1 on
Q = np.stack(Q, 1); T = np.array(T); U = np.stack(U, 1)
kend = torch.full((N,), nfr-1, device=dev, dtype=torch.long); out = []; a0.out.mkdir(parents=True, exist_ok=True)
for s_, name in (('left', 'tube'), ('right', 'pipette')):
    _, O = env._poses(s_); Or = E.X.ref_o(env, s_, kend)
    pe = (O[:, :3, 3]-Or[:, :3, 3]).norm(dim=-1); ae = torch.arccos((O[:, :3, 2]*Or[:, :3, 2]).sum(-1).clamp(-1, 1))
    for w in range(N):
        if len(out) <= w: out.append({})
        out[w][f'{name}_end_err_mm'] = round(float(pe[w])*1e3, 2); out[w][f'{name}_end_tilt_deg'] = round(math.degrees(float(ae[w])), 2)
        if name == 'tube': out[w]['tube_end_lat_mm'] = round(float((O[w, :2, 3]-Or[w, :2, 3]).norm())*1e3, 2); out[w]['tube_end_dz_mm'] = round(float(O[w, 2, 3]-Or[w, 2, 3])*1e3, 2)
for w in range(N):
    ok = bool(first_fail[w] < 0); o = out[w]
    o['fail_reason'] = [None if int(x) == 0 else E.FAILURES[int(x)-1] for x in fail_why[w]]   # [pipette, tube]
    o['press_ok'] = bool(pl_max[w] >= .5*pl_ref.max() and pl_s[w] >= 1.5); o['press_max_mm'] = round(float(pl_max[w])*1e3, 2); o['press_s_at_90pct'] = round(float(pl_s[w]), 2)
    o.update(spec=S[w], reach_end=ok, first_fail_s=round(float(first_fail[w]), 2), tube_seated=ok and o['tube_end_err_mm'] <= 5 and o['tube_end_tilt_deg'] <= 3,
             pipette_seated=ok and o['pipette_end_err_mm'] <= 5 and o['pipette_end_tilt_deg'] <= 3)
    np.savez_compressed(a0.out/f'world_{w}.npz', qpos=Q[w], time=T, ctrl=U[w], **({'depth': np.stack([x[w] for x in DEP]), 'depth_t': np.array(DEPT)} if DEP else {}))
(a0.out/'summary.json').write_text(json.dumps({'layer': 'RL env rollouts (MuJoCo Warp physics), '+('ZERO residual' if a0.zero else 'policy deterministic mean')+(', DR off' if a0.no_dr else ''), 'config': str(a0.config), 'ckpt': str(a0.student or a0.ckpt), 'actor': 'student (deployable obs)' if a0.student else 'teacher', 'task': str(task), 'worlds': out}, indent=1))
for w, o in enumerate(out): print(w, o['spec'], 'end', o['reach_end'], 'tube', o['tube_seated'], o['tube_end_lat_mm'], o['tube_end_dz_mm'], o['tube_end_tilt_deg'], 'pip', o['pipette_seated'])
