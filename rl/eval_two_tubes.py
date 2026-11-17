"""Two-tube composite task (tracker/build_two_tube_task.py): teacher / student / zero residual, whole task from the start (11-13).

Same env, DR, seed (123), deterministic actor mean and rules as eval_rr.py (press: max depth >= 50 % of the reference and >= 90 % depth for
>= 1.5 s; tip: the mesh tip in the bore, >= 20 mm deep, for >= 1.5 s; seat: 5 mm / 3 deg tube or pipette axis), applied per press window and
per object:
  cycle 1: press 1 + tip 1 ok, tube 1 seated in hole 1 just before the handover;  cycle 2: press 2 + tip 2 ok;
  end: tube 1 still seated in hole 1 (body tube2 after the handover), tube 2 seated in hole 2, pipette seated in the stand.
A world counts as a failure from the first trainer failure on (an object > 5 cm / 45 deg off its reference, as in eval_rr.py).
The config's hole_pairs are pinned to the reference's own hole (the composite already places tube 2); pushes off unless --perturb.

  cd rl; python -I eval_two_tubes.py CONFIG.json (--ckpt TEACHER.pt | --student STUDENT.pt | neither = zero residual) --task ../tasks/task_d02_two_tubes [--n 64] [--out OUT.json]
"""
import argparse, json, math, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
HERE = Path(__file__).resolve().parent
import os as _os
ROOT = Path(_os.environ.get('G1PS_ROOT', str(HERE.parent/'deps_from_g1ps')))
sys.path[:0] = [str(HERE), str(ROOT/'tools/rl')]
import vega_res_env as E
import gpu_train as GT

p = argparse.ArgumentParser(); p.add_argument('config', type=Path); p.add_argument('--ckpt', type=Path); p.add_argument('--student', type=Path)
p.add_argument('--task', type=Path, required=True); p.add_argument('--n', type=int, default=64); p.add_argument('--out', type=Path); p.add_argument('--no-dr', action='store_true')
p.add_argument('--perturb', action='store_true'); p.add_argument('--mount-rand', type=float, nargs=2, metavar=('XYZ_MM', 'RPY_DEG'))
p.add_argument('--progress', choices=['source', 'naive'], default='source', help='student clip_progress input: the source frame\'s (default) or frame / last of the long clip')
p.add_argument('--naconmax-per-world', type=int, default=1000, help='contact / broadphase budget per world (the parked tube in the rack adds broadphase pairs; 400 overflows)')
p.add_argument('--no-remeasure', action='store_true', help='student: keep the start-of-episode tube estimate after the handover (no second head-camera look)')
p.add_argument('--hide-tube2', action='store_true', help='ablation: the second tube is invisible to the wrist depth cameras (its visual geom out of the rendered groups); physics unchanged')
a0 = p.parse_args(); t_start = time.time()
if a0.hide_tube2:
    _lm = E.X.load_model
    def _load_hidden(path, extras):
        m_ = _lm(path, extras); b2 = m_.body('tube2').id
        for g_ in range(m_.ngeom):
            if int(m_.geom_bodyid[g_]) == b2: m_.geom_group[g_] = 4
        return m_
    E.X.load_model = _load_hidden
C = json.loads(a0.config.read_text()); rel = lambda x: (Path(x) if Path(x).is_absolute() else (a0.config.parent/x)).resolve()
if a0.mount_rand is not None and (C.get('extras') or {}).get('student', {}).get('depth'):
    C['extras']['student']['depth']['mount_rand'] = ({'xyz_mm': [a0.mount_rand[0]]*3, 'rpy_deg': [a0.mount_rand[1]]*3} if any(a0.mount_rand) else None)
task = a0.task.resolve(); tj = json.loads((task/'task.json').read_text()); CP = tj['composite']; h1 = tj['ref_holes'][0]
if (C.get('extras') or {}).get('holder', {}).get('hole_pairs'): C['extras']['holder']['hole_pairs'] = [[h1, h1]]
dr = None if a0.no_dr else (C.get('dr') if isinstance(C.get('dr'), dict) or C.get('dr') is None else json.loads(rel(C['dr']).read_text()))
r = np.load(task/'reference.npz'); t = r['timestamps']; nfr = len(t); N = a0.n
kw = dict(dr=dr, extras={k_: v_ for k_, v_ in C.get('extras', {}).items() if a0.perturb or k_ != 'perturb'}, reward_scale=C.get('reward_scale', .01))
for k in ('arm_scale', 'finger_scale', 'group_scale', 'thumb_press_scale', 'plunger_weight'):
    if C.get(k) is not None: kw[k] = C[k]
if a0.student: kw['obs_mode'] = 'deployable'
env = E.BimanualGpuEnv(task, N, seed=123, nconmax=C.get('nconmax', 200), njmax=C.get('njmax', 1600), naconmax_per_world=a0.naconmax_per_world,
                       terminate=False, auto_reset=False, **kw)
assert env.x_multi is not None, 'task.json has no composite block'
if a0.no_remeasure: env.x_multi['remeasure'] = False
if a0.progress == 'naive': env.clip_prog = None
policy = None
if a0.ckpt:
    a = SimpleNamespace(steps_per_env=C.get('steps_per_env', 24), lr=3e-4, init_std=C.get('init_std', .1), no_value_clip=True, deployable=False)
    venv, runner = GT.make_runner(env, a, None); runner.load(str(a0.ckpt), map_location=env.device); pol = runner.get_inference_policy(device=env.device)
    policy = lambda: pol(env.get_observations())
if a0.student:
    import distill as D
    ck = torch.load(a0.student, map_location=env.device)
    if ck.get('depth'):
        DX = env.x_depth; assert DX is not None, 'student uses depth images: the config needs extras.student.depth'
        stu = D.StudentCNN(ck['nin'], ck['nout'], ck['hidden'], (DX['H'], DX['W']), aux=ck.get('aux', 0)).to(env.device)
    else: stu = D.Student(ck['nin'], ck['nout'], ck['hidden']).to(env.device)
    stu.load_state_dict(ck['student']); stu.eval()
    assert ck['nin'] == env.obs_actor.shape[1], (ck['nin'], env.obs_actor.shape)
    policy = (lambda: stu(env.obs_actor, env.x_depth['obs'])) if ck.get('depth') else (lambda: stu(env.obs_actor))
dev = env.device
env.reset_envs(torch.ones(N, dtype=torch.bool, device=dev), frames=torch.zeros(N, dtype=torch.long, device=dev)); env._launch(env.g_kin); env.obs = env._observe()
if env.obs_mode == 'deployable': env._observe_deployable()
steps = (nfr-1)//env.frame_step; SW = CP['swap_frame']; PW = CP['press_windows']; seg = CP['segments']
pl_ref = torch.as_tensor(r['plunger'], device=dev).float(); plmax_ref = float(pl_ref.max())
TIPM = torch.as_tensor(r['pip_tip_local'], dtype=torch.float32, device=dev)
first_fail = torch.full((N,), -1, dtype=torch.long, device=dev); fail_reason = torch.zeros(N, 2, dtype=torch.long, device=dev)
pl_max = torch.zeros(2, N, device=dev); pl_s = torch.zeros(2, N, device=dev); tip_s = torch.zeros(2, N, device=dev); tip_lat = torch.zeros(2, N, device=dev); tip_n = torch.zeros(2, N, device=dev)
pre_swap = {'pos_mm': torch.full((N,), float('nan'), device=dev), 'tilt_deg': torch.full((N,), float('nan'), device=dev)}
q2 = env.x_multi['q2']
def seat_err(O, Oref):
    return (O[:, :3, 3]-Oref[:, :3, 3]).norm(dim=-1), torch.arccos((O[:, :3, 2]*Oref[:, :3, 2]).sum(-1).clamp(-1, 1))
for i in range(steps):
    with torch.inference_mode():
        k_prev = env.frame.clone()
        if (k_prev < SW).any():   # tube 1 vs its reference in hole 1, the last control step before the handover
            k_ = k_prev.clamp(max=nfr-1); _, Ot = env._poses('left'); pe, ae = seat_err(Ot, E.X.ref_o(env, 'left', k_)); bef = k_prev < SW
            pre_swap['pos_mm'] = torch.where(bef, pe*1e3, pre_swap['pos_mm']); pre_swap['tilt_deg'] = torch.where(bef, torch.rad2deg(ae), pre_swap['tilt_deg'])
        act = policy().clamp(-1, 1) if policy is not None else torch.zeros(N, env.nu, device=dev)
        obs, rew, done, ex = env.step(act)
    k = env.frame.clamp(max=nfr-1); live = first_fail < 0; f_any = (ex['reasons'] > 0).any(1)
    newly = f_any & live; first_fail[newly] = k[newly]; fail_reason[newly] = ex['reasons'][newly].long(); live = first_fail < 0
    depth = -env.qpos[:, env.pl_q]; _, Op = env._poses('right'); _, Ou = env._poses('left')
    tp = Op[:, :3, :3]@TIPM+Op[:, :3, 3]; rl = ((tp-Ou[:, :3, 3])[:, None, :]@Ou[:, :3, :3])[:, 0]; rr = rl[:, :2].norm(dim=-1); dp = .0638-rl[:, 2]
    inb = (rr < .0129-.0018) & (dp >= .02)
    for j, (k0, k1) in enumerate(PW):
        inp = (k >= k0) & (k <= k1) & live
        pl_max[j] = torch.where(inp, torch.maximum(pl_max[j], depth), pl_max[j]); pl_s[j] += (inp & (depth >= .9*plmax_ref)).float()*E.CONTROL_DT
        tip_s[j] += (inp & inb).float()*E.CONTROL_DT; tip_lat[j] += torch.where(inp, rr, torch.zeros_like(rr)); tip_n[j] += inp.float()
end_ok = (first_fail < 0).cpu().numpy()
kend = torch.full((N,), nfr-1, dtype=torch.long, device=dev); out_ = {}
_, Op = env._poses('right'); _, Ou = env._poses('left')
pe_p, ae_p = seat_err(Op, E.X.ref_o(env, 'right', kend)); pe_u, ae_u = seat_err(Ou, E.X.ref_o(env, 'left', kend))
# tube 1 at the end: body tube2 vs its reference pose (qpos_ref) under the full rack transform
T1 = torch.eye(4, device=dev).repeat(N, 1, 1); T1[:, :3, :3] = env.xmat[:, env.mjm.body('tube2').id]; T1[:, :3, 3] = env.xpos[:, env.mjm.body('tube2').id]
qr2 = env.qpos_ref[nfr-1, q2:q2+7]; Tr = torch.eye(4, device=dev).repeat(N, 1, 1); Tr[:, :3, 3] = qr2[:3]; Tr[:, :3, :3] = E.X._qmat(qr2[3:7])[None]
if getattr(env, 'x_hold', None):
    h = env.x_hold['rack']; Tr = E.X._H(torch.ones(N, device=dev), h['d'], h['th'], h['c'])@Tr
pe_1, ae_1 = seat_err(T1, Tr)
ok = lambda pe, ae: ((pe <= .005) & (ae <= math.radians(3))).cpu().numpy()
seat_pip, seat_t2, seat_t1_end = ok(pe_p, ae_p) & end_ok, ok(pe_u, ae_u) & end_ok, ok(pe_1, ae_1)
seat_t1_swap = ((pre_swap['pos_mm'] <= 5) & (pre_swap['tilt_deg'] <= 3)).cpu().numpy()
press_ok = ((pl_max >= .5*plmax_ref) & (pl_s >= 1.5)).cpu().numpy(); tip_ok = (tip_s >= 1.5).cpu().numpy()
ff = first_fail.cpu().numpy(); fs_ = ff >= 0
cyc1 = press_ok[0] & tip_ok[0] & seat_t1_swap & ~(fs_ & (ff < SW))
cyc2 = press_ok[1] & tip_ok[1]
full = end_ok & cyc1 & cyc2 & seat_t1_end & seat_t2 & seat_pip
def seg_of(kf):
    for nm, (s0, s1) in seg.items():
        if s0 <= kf <= s1: return nm
    return 'end'
FN = E.FAILURES; frn = fail_reason.cpu().numpy()
fails = [None if f < 0 else {'t_s': round(float(t[f]-t[0]), 2), 'segment': seg_of(int(f)), 'object': 'pipette' if frn[w, 0] else 'tube', 'reason': FN[int(frn[w, 0] or frn[w, 1])-1]} for w, f in enumerate(ff)]
cnt = {}
for x in fails:
    if x: key = f"{x['segment']}: {x['object']} {x['reason']}"; cnt[key] = cnt.get(key, 0)+1
c_ = lambda x: int(np.sum(x)); tl = (tip_lat/tip_n.clamp_min(1)).cpu().numpy()*1e3
res = {'task': str(task), 'holes': CP['holes'], 'actor': 'student (deployable obs)' if a0.student else ('teacher (full state)' if a0.ckpt else 'zero residual'),
       'ckpt': str(a0.ckpt or a0.student) if (a0.ckpt or a0.student) else None, 'n': N, 'dr': None if dr is None else 'on', 'pushes': bool(a0.perturb),
       'camera_mount_rand': ((C.get('extras') or {}).get('student') or {}).get('depth', {}).get('mount_rand') if isinstance(((C.get('extras') or {}).get('student') or {}).get('depth'), dict) else None,
       'student_progress_input': a0.progress if a0.student else None, 'tube2_hidden_from_cameras': bool(a0.hide_tube2), 'student_second_look': (not a0.no_remeasure) if a0.student else None,
       'FULL_SUCCESS (both cycles, all three seated, no failure)': c_(full),
       'cycle1_ok (press + tip, tube 1 seated before the handover)': c_(cyc1), 'cycle2_ok (press + tip)': c_(cyc2),
       'reach_end_without_failure': c_(end_ok), 'press_ok [1, 2]': [c_(press_ok[0]), c_(press_ok[1])], 'tip_ok [1, 2]': [c_(tip_ok[0]), c_(tip_ok[1])],
       'press_and_tip_ok [1, 2]': [c_(press_ok[0] & tip_ok[0]), c_(press_ok[1] & tip_ok[1])],
       'tube1_seated [before handover, at the end]': [c_(seat_t1_swap), c_(seat_t1_end)], 'tube2_seated_end': c_(seat_t2), 'pipette_seated_end': c_(seat_pip),
       'tip_off_axis_mm_mean_p50 [1, 2]': [round(float(np.median(tl[0])), 1), round(float(np.median(tl[1])), 1)],
       'first_failures (segment: object reason)': dict(sorted(cnt.items(), key=lambda x: -x[1])), 'swaps': env.x_multi['n_swaps'], 'wall_s': round(time.time()-t_start, 1),
       'rules': 'eval_rr.py rules per press window / object; cycle 1 also requires no failure before the handover',
       'per_world': {'first_failure': fails, 'tube1_err_before_handover_mm': pre_swap['pos_mm'].cpu().numpy().round(2).tolist(),
                     'end_err_mm [pipette, tube2, tube1]': np.stack([pe_p.cpu().numpy(), pe_u.cpu().numpy(), pe_1.cpu().numpy()], 1).__mul__(1e3).round(2).tolist(),
                     'tip_in_bore_s [1, 2]': tip_s.cpu().numpy().T.round(2).tolist(), 'press_s [1, 2]': pl_s.cpu().numpy().T.round(2).tolist()}}
print(json.dumps({k_: v_ for k_, v_ in res.items() if k_ != 'per_world'}))
if a0.out: a0.out.parent.mkdir(parents=True, exist_ok=True); a0.out.write_text(json.dumps(res, indent=1))
