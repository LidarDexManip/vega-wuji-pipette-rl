"""TRACKER (11-10): one command from a retargeted do-as-i-do demo episode to a trainable residual-RL task + training configs.
Method and the reason for each step: docs/DEMO_TO_TASK.md.

Steps (--steps, default all, in this order):
  download  the episode's files needed here from HF (ytung/pipette-tasks, episodes/<episode>) into --data/<episode> (skipped if present):
            reconstruction/{calibration, scene, provenance}.json, depth/000000.png, rgb.mp4, {pipette,tube}/{poses.npz, mesh_local.ply},
            retargeting/{right_pipette,left_tube}/{trajectory.npz, summary.json, command.json}
  build     tracker/build_demo_task.py -> tasks/task_<name>_noprior   (frames, ground-truth holders, references, IK; build_report.json)
  labels    tracker/build_kin_contact_ref.py (contact_ref.npz from the kinematic reference: the env's contact terms / wrench gate)
  audit     rl/audit_starts.py (valid_starts.npy: one zero-residual step from every 2nd frame)
  press     tracker/add_press_prior.py -> tasks/task_<name> (thumb press force in the press window); copies the reports, adds
            tracker/build_kin_contact_labels.py's eval labels
  check     tracker/check_demo_task.py (zero-residual rollouts at VOC 1 / 0, montage, demo-camera overlays) -> tasks/task_<name>/check/
  config    tracker/configs/kin_<name>_s5.json (1 GPU) and kin_<name>_s5_g3.json (3 GPUs) from KA's config (tracker/configs/kin_tc_s5.json):
            the new task, every rack hole back into itself as the hole pairs

  .venv-rl-head/bin/python -I tracker/demo_to_task.py pipette_depth_02_tapir d02_straight [--steps build,labels,...] [--force] [-- BUILD_ARGS]
BUILD_ARGS after '--' go to build_demo_task.py (e.g. -- --fwd 0.30 --hole r1c1). --force replaces existing task dirs / configs of <name>.
"""
import argparse, json, shutil, subprocess, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent; V = HERE.parent; PY = sys.executable
HF = 'https://huggingface.co/datasets/ytung/pipette-tasks/resolve/main/episodes'
FILES = ['reconstruction/calibration.json', 'reconstruction/scene.json', 'reconstruction/provenance.json', 'reconstruction/depth/000000.png',
         'reconstruction/rgb.mp4'] + [f'reconstruction/{o}/{f}' for o in ('pipette', 'tube') for f in ('poses.npz', 'mesh_local.ply')] + \
        [f'retargeting/{h}/{f}' for h in ('right_pipette', 'left_tube') for f in ('trajectory.npz', 'summary.json', 'command.json')]
STEPS = ('download', 'build', 'labels', 'audit', 'press', 'check', 'config')


def run(cmd, cwd=V):
    print('$', ' '.join(str(c) for c in cmd), flush=True); t0 = time.time()
    p = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    tail = '\n'.join(l for l in (p.stdout+p.stderr).splitlines() if not any(s in l.lower() for s in ('warn', 'module ', 'took ', 'epa horizon', 'to disable')))[-1500:]
    if p.returncode: raise SystemExit(f'failed ({p.returncode}):\n{tail}')
    print(f'  ok in {time.time()-t0:.0f} s', flush=True); return tail


def main():
    argv = sys.argv[1:]; extra = argv[argv.index('--')+1:] if '--' in argv else []; argv = argv[:argv.index('--')] if '--' in argv else argv
    ap = argparse.ArgumentParser(); ap.add_argument('episode', help='episode name on HF, or a local episode directory'); ap.add_argument('name', help='task name: tasks/task_<name>')
    ap.add_argument('--data', type=Path, default=Path.home()/'Projects/datasets/pipette-tasks'); ap.add_argument('--steps', default=','.join(STEPS))
    ap.add_argument('--force', action='store_true'); a = ap.parse_args(argv); steps = a.steps.split(',')
    bad = [s for s in steps if s not in STEPS]
    if bad: raise SystemExit(f'unknown steps {bad}; choose from {STEPS}')
    ep = Path(a.episode) if Path(a.episode).is_dir() else a.data/a.episode
    noprior, final = V/f'tasks/task_{a.name}_noprior', V/f'tasks/task_{a.name}'
    cfgs = {1: HERE/f'configs/kin_{a.name}_s5.json', 3: HERE/f'configs/kin_{a.name}_s5_g3.json'}
    if 'download' in steps:
        ep.mkdir(parents=True, exist_ok=True); n = 0
        for f in FILES:
            dst = ep/f
            if dst.exists(): continue
            dst.parent.mkdir(parents=True, exist_ok=True); tmp = dst.with_suffix(dst.suffix+'.part')
            with urllib.request.urlopen(f'{HF}/{ep.name}/{f}', timeout=120) as r, open(tmp, 'wb') as fo: shutil.copyfileobj(r, fo)
            tmp.rename(dst); n += 1
        print(f'download: {n} new files in {ep}', flush=True)
    if 'build' in steps:
        if noprior.exists() and not a.force: raise SystemExit(f'{noprior} exists (--force to rebuild)')
        run([PY, '-I', HERE/'build_demo_task.py', ep, noprior.relative_to(V), '--force', *extra])
    if 'labels' in steps: run([PY, '-I', HERE/'build_kin_contact_ref.py', noprior])
    if 'audit' in steps: run([PY, '-I', V/'rl/audit_starts.py', noprior])
    if 'press' in steps:
        if final.exists():
            if not a.force: raise SystemExit(f'{final} exists (--force to replace)')
            shutil.rmtree(final)
        run([PY, '-I', HERE/'add_press_prior.py', noprior, final])
        for f in ('build_report.json', 'wrist_targets.npz'): shutil.copy2(noprior/f, final/f)
        run([PY, '-I', HERE/'build_kin_contact_labels.py', final])
    if 'check' in steps: print(run([PY, '-I', HERE/'check_demo_task.py', final, '--episode', ep]))
    if 'config' in steps:
        base = json.loads((HERE/'configs/kin_tc_s5.json').read_text()); holes = [f'r{r}c{c}' for r in (0, 1) for c in range(5)]
        for g, p in cfgs.items():
            if p.exists() and not a.force: print(f'config {p} exists, kept'); continue
            c = json.loads(json.dumps(base)); c['task'] = f'../../tasks/task_{a.name}'; c['out'] = f'../runs/kin_{a.name}_s5' + ('_g3' if g > 1 else '')
            c['extras']['holder']['hole_pairs'] = [[h, h] for h in holes]
            if g > 1: c['max_env_steps'] = 6e8; c['ckpt_steps'] = 5e7; c['tracker']['voc_zero_by_steps'] = [1e8, 3e8]
            c['_note'] = (f'{time.strftime("%m-%d")} generated by tracker/demo_to_task.py from {ep.name}: KA\'s config and reward on tasks/task_{a.name} '
                          '(docs/DEMO_TO_TASK.md), every rack hole back into itself as hole pairs' +
                          ('; 3 GPUs data parallel (node_tracker_ddp.sh): 6e8 env steps over all GPUs, VOC zero by [1e8, 3e8], checkpoints every 5e7' if g > 1 else ''))
            p.write_text(json.dumps(c, indent=1)+'\n'); print('config', p, flush=True)
    rep = json.loads((final/'build_report.json').read_text()) if (final/'build_report.json').exists() else {}
    chk = json.loads((final/'check/check.json').read_text()) if (final/'check/check.json').exists() else {}
    print(json.dumps({'task': str(final), 'midair': rep.get('midair_search', {}).get('chosen'), 'ik_pos_mm_max': {s: v['pos_mm_p50_max'][1] for s, v in rep.get('ik', {}).items()},
                      'penetrations_gt3mm': {k: v['worst_mm'] for k, v in rep.get('reference_penetrations_gt3mm', {}).items()},
                      'smoothness': rep.get('smoothness'), 'zero_residual_pass_voc1': chk.get('pass_voc1'), 'check_images': str(final/'check')}, indent=1, default=float))
    print(f'train (node, 1 GPU):  cd tracker && CUDA_VISIBLE_DEVICES=0 ../.venv-rl-head/bin/python -I train_tracker.py configs/{cfgs[1].name}\n'
          f'train (node, 3 GPUs): GPUS="0 3 4" ./node_tracker_ddp.sh tracker/configs/{cfgs[3].name}   (sync the task dir and configs to the node first)')


if __name__ == '__main__':
    main()
