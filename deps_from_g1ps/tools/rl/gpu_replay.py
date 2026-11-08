"""Deterministic rollouts of the bimanual GPU env (tools/rl/gpu_env.py): the zero-residual baseline, or a policy (tools/rl/gpu_train.py eval).

One world per start (x --repeat). Per start: when each object first fails the trainer's thresholds (YS's order; the rollout runs on to
the end of the window with --no-terminate so both objects get a time), the object / wrist errors over time, whether the press happens
(max plunger depth in the press window vs the reference's), and at the window end whether each object is back in place (the
check_physics_seat pose rule: position <= 5 mm, axis <= 3 deg vs the reference end pose). --record WORLD keeps that world's qpos
at every physics step for tools/rl/render_rollout.py.

  .venv-rl/bin/python tools/rl/gpu_replay.py TASK --out DIR [--starts 3.033 4.5 ...] [--no-terminate] [--record 0]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_env import BimanualGpuEnv, FAILURES, SIDES, CONTROL_DT, rot_angle, refuse_backport_on_head

SPREAD_S = (3.0333, 3.6833, 4.3333, 5.5333, 7.0333, 9.0333, 11.0333, 11.7333, 12.5333, 14.5333, 17.0333, 20.0333, 22.5333)
SEAT_POS_M, SEAT_DEG = .005, 3.


def start_frames(env, starts_s):
    t = env.times; fr = []
    for s in starts_s:
        k = int(np.argmin(np.abs(t-s))); k -= k % env.frame_step; fr.append(min(k, env.nframes-1-env.frame_step))
    return fr


def rollout(env, starts_s, policy=None, repeat=1, terminate=True, record=None, log=print):
    frames = start_frames(env, starts_s)*repeat; N = env.num_envs
    if len(frames) != N: raise ValueError(f'{len(frames)} starts x repeat for {N} worlds')
    env.terminate = terminate; env.auto_reset = False
    fr = torch.as_tensor(frames, device=env.device)
    env.reset_envs(torch.ones(N, dtype=torch.bool, device=env.device), frames=fr); env._launch(env.g_kin); env.obs = env._observe()
    alive = torch.ones(N, dtype=torch.bool, device=env.device); rows = []; qrec = []
    if record is not None:
        qrec.append(env.qpos[record].cpu().numpy().copy())
        env.substep_hook = lambda s: qrec.append(env.qpos[record].cpu().numpy().copy())
    steps = int(max((env.nframes-1-f)//env.frame_step for f in frames)); obs = env.obs; t0 = time.time()
    for i in range(steps):
        with torch.inference_mode():
            a = torch.zeros(N, env.nu, device=env.device) if policy is None else policy(env.get_observations())
        obs, rew, done, ex = env.step(a); inf = ex['info']
        rows.append({'frame': env.frame.clone(), 'alive': alive.clone(), 'reasons': ex['reasons'].clone(), 'reward': rew.clone(),
                     **{f'{s}_{k}': inf[s][k].clone() for s in SIDES for k in ('object_position_m', 'object_orientation_rad', 'relative_position_m',
                                                                                 'axis_max_m', 'wrist_position_m', 'wrist_orientation_rad')},
                     'plunger': inf['plunger_depth_m'].clone(), 'plunger_ref': inf['plunger_ref_m'].clone()})
        alive &= ~done
        if not alive.any(): break
    env.substep_hook = None
    R = {k: torch.stack([r[k] for r in rows]).cpu().numpy() for k in rows[0]}
    log(f'rollout: {len(rows)} control steps x {N} worlds in {time.time()-t0:.1f} s')
    out = []; t_of = lambda f: float(env.times[min(int(f), env.nframes-1)])
    end_frame = env.nframes-1
    O_end = {s: env.O_ref[s][end_frame].cpu().numpy() for s in SIDES}
    for w in range(N):
        live = R['alive'][:, w]; nlive = int(live.sum()); last = max(nlive-1, 0)
        rec = {'world': w, 'start_s': round(t_of(frames[w]), 3), 'steps': nlive, 'end_s': round(t_of(R['frame'][last, w]), 3),
               'survived_s': round(nlive*CONTROL_DT, 2)}
        reached_end = int(R['frame'][last, w]) >= end_frame
        any_fail = (R['reasons'][:nlive, w] > 0).any(1)
        rec['success'] = bool(reached_end and not any_fail.any()) if terminate else bool(reached_end and not any_fail.any())
        rec['first_failure_s'] = round(t_of(R['frame'][int(np.argmax(any_fail)), w]), 3) if any_fail.any() else None
        for j, (s, name) in enumerate(zip(SIDES, ('pipette', 'tube'))):
            rs = R['reasons'][:nlive, w, j]; bad = rs > 0
            rec[f'{name}_first_fail_s'] = round(t_of(R['frame'][int(np.argmax(bad)), w]), 3) if bad.any() else None
            rec[f'{name}_first_fail_reason'] = FAILURES[int(rs[int(np.argmax(bad))])-1] if bad.any() else None
            e = R[f'{s}_object_position_m'][:nlive, w]
            rec[f'{name}_world_err_mm'] = {'p50': round(float(np.median(e))*1e3, 1), 'max': round(float(e.max())*1e3, 1),
                                           'every_1s': [round(float(x)*1e3, 1) for x in e[::20]]}
            rec[f'{name}_axis_max_mm_max'] = round(float(R[f'{s}_axis_max_m'][:nlive, w].max())*1e3, 1)
            we = R[f'{s}_wrist_position_m'][:nlive, w]
            rec[f'wrist_{s}_err_mm'] = {'p50': round(float(np.median(we))*1e3, 1), 'p95': round(float(np.percentile(we, 95))*1e3, 1), 'max': round(float(we.max())*1e3, 1)}
        fr_w = R['frame'][:nlive, w].clip(max=end_frame); press = env.press.cpu().numpy()[fr_w]
        if press.any():
            rec['press'] = {'max_depth_mm': round(float(R['plunger'][:nlive, w][press].max())*1e3, 2), 'ref_max_depth_mm': round(float(R['plunger_ref'][:nlive, w][press].max())*1e3, 2)}
            rec['press']['happens'] = rec['press']['max_depth_mm'] >= .5*rec['press']['ref_max_depth_mm']
        if reached_end:
            seat = {}
            for s, name in zip(SIDES, ('pipette', 'tube')):
                W, O = env._poses(s); O = O[w].cpu().numpy()
                pe = float(np.linalg.norm(O[:3, 3]-O_end[s][:3, 3])); ae = float(np.degrees(np.arccos(np.clip(O[:3, 2]@O_end[s][:3, 2], -1, 1))))
                seat[name] = {'pos_err_mm': round(pe*1e3, 1), 'axis_err_deg': round(ae, 1), 'seated': pe <= SEAT_POS_M and ae <= SEAT_DEG}
            rec['end_pose'] = seat
        out.append(rec)
    return out, R, (np.array(qrec) if record is not None else None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('task', type=Path); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--starts', type=float, nargs='*', default=list(SPREAD_S)); p.add_argument('--repeat', type=int, default=1)
    p.add_argument('--no-terminate', action='store_true'); p.add_argument('--record', type=int); p.add_argument('--no-feedforward', action='store_true')
    p.add_argument('--gjk-backport', action='store_true', help='patched mujoco_warp collision_gjk, as the physics worker; default off')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    if a.gjk_backport:   # same patched collision_gjk as the physics worker's --gjk-backport; before the env imports mujoco_warp
        refuse_backport_on_head()
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from g1_pipette_sim.retarget.gpu.mjw_backport import install_gjk_backport
        install_gjk_backport()
    env = BimanualGpuEnv(a.task, len(a.starts)*a.repeat, feedforward=not a.no_feedforward, naconmax_per_world=600, njmax=1200)
    res, R, q = rollout(env, a.starts, repeat=a.repeat, terminate=not a.no_terminate, record=a.record)
    tag = ('zero_noterm' if a.no_terminate else 'zero')+('_noff' if a.no_feedforward else '')
    (a.out/f'{tag}.json').write_text(json.dumps({'mode': tag, 'starts': a.starts, 'repeat': a.repeat, 'results': res,
                                                   'peak_stats': env.stats_peak.tolist(), 'sizes': env.cfg, **({'overflow': env.overflow_report()} if env.ovf is not None else {})}, indent=1)+'\n')
    np.savez_compressed(a.out/f'{tag}_series.npz', **R)
    if q is not None: np.savez_compressed(a.out/f'{tag}_world{a.record}_qpos.npz', qpos=q, dt=env.dt, start_frame=start_frames(env, a.starts)[a.record % len(a.starts)])
    for r in res:
        print(r['start_s'], 'survived', r['survived_s'], 's success', r['success'], 'pipette fail', r['pipette_first_fail_s'], r['pipette_first_fail_reason'],
              '| tube fail', r['tube_first_fail_s'], r['tube_first_fail_reason'], '| wristR', r['wrist_right_err_mm'], 'wristL', r['wrist_left_err_mm'],
              '| press', r.get('press'), '| end', r.get('end_pose'))


if __name__ == '__main__':
    main()
