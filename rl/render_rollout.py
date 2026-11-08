"""Render one recorded RL rollout (record_rollouts.py world_<i>.npz) with render/render_vega_visual.py (LIDAR visual furniture).
Builds a throw-away stage folder in the renderer's input format (qpos as the 'physics' source, reference clock) and, for a placement offset,
a scene copy whose rack / stand proxies are moved exactly as the RL env moved them (rotation by dyaw about the holder's vertical centre line,
then translation; holder centre = mean of its static geoms at z = 0, as vega_extras.holder_init). The renderer's auto layout then moves the
visual meshes with the proxies. Layer label: RL POLICY, RL-env physics.
  python render_rollout.py ROLL_DIR WORLD TASK_DIR OUT.mp4 --label "line 1" --label "line 2" """
import argparse, json, math, shutil, subprocess, sys
from pathlib import Path
import numpy as np, mujoco
HERE = Path(__file__).resolve().parent; REND = HERE.parent/'render/render_vega_visual.py'
PY = sys.executable   # (public copy) was the local .venv-rl-head interpreter
p = argparse.ArgumentParser(); p.add_argument('roll', type=Path); p.add_argument('world', type=int); p.add_argument('task', type=Path); p.add_argument('out', type=Path)
p.add_argument('--label', action='append', default=[]); p.add_argument('--work', type=Path, required=True); p.add_argument('--fps', type=float, default=30.); a = p.parse_args()
summ = json.loads((a.roll/'summary.json').read_text())['worlds'][a.world]; z = np.load(a.roll/f'world_{a.world}.npz'); Q, T = z['qpos'], z['time']
W = a.work; shutil.rmtree(W, ignore_errors=True); off = W/'official/wuji/bimanual/pipette_g1_run01/0'; off.mkdir(parents=True)
np.savez(off/'trajectory_mjwp.npz', qpos=Q[None].astype(np.float64), time=T[None].astype(np.float64))
np.savez(W/'actual_reference.npz', qpos=Q, timestamps=T, warmup_steps=np.array(1200), demonstration_end=np.array(float(T[-1])))
(W/'run.json').write_text(json.dumps({'sim_dt': 0.0025, 'window_physics_s': [float(T[0]), float(T[-1])]}))
scene = a.task/'scene.xml'; spec = summ['spec']
if any(abs(x) > 1e-9 for x in spec['rack']+spec['stand']):
    m0 = mujoco.MjModel.from_xml_path(str(scene)); d0 = mujoco.MjData(m0); mujoco.mj_kinematics(m0, d0)
    sp = mujoco.MjSpec.from_file(str(scene)); moved = {}
    for name, pre in (('stand', ('furn_stand_',)), ('rack', ('furn_rackv_', 'furn_rack_plate_'))):
        ids = [i for i in range(m0.ngeom) if m0.geom_bodyid[i] == 0 and (m0.geom(i).name or '').startswith(pre)]
        c = d0.geom_xpos[ids].mean(0); c[2] = 0.; dx, dy, dyaw = spec[name]; th = math.radians(dyaw); R = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])
        qz = np.array([math.cos(th/2), 0, 0, math.sin(th/2)])
        for g in sp.worldbody.geoms:
            if not (g.name or '').startswith(pre): continue
            # transform the geom's OWN (pre-compile) pos / quat: a mesh geom's compiled position adds its mesh-frame offset, which then
            # rotates with the new quat, so the compiled pose ends up = R (xpos - c) + c + d as in the RL env (all proxies here: pos + quat)
            pn = R@(np.array(g.pos)-c)+c+np.array([dx*1e-3, dy*1e-3, 0.]); qn = np.zeros(4); mujoco.mju_mulQuat(qn, qz, np.array(g.quat, float))
            g.pos = pn; g.quat = qn; moved[name] = moved.get(name, 0)+1
    sp.compile(); xml = sp.to_xml(); scene = a.task/f'_render_scene_{a.out.stem}.xml'; scene.write_text(xml)   # next to the task scene: same relative asset paths
    print('moved proxies', moved, '->', scene)
tmp = W/(a.out.stem+'_physics.mp4')
cmd = [PY, '-I', str(REND), '--stage', str(W), '--scene', str(scene), '--out', str(tmp), '--source', 'physics', '--hands', '--overwrite', '--fps', str(a.fps)]   # 50 fps for 50 Hz rollouts: one row per frame (30 fps picks rows 2,1,2,2,1 apart = judder)
for ln in a.label: cmd += ['--label', ln]
print(' '.join(cmd)); rc = subprocess.run(cmd).returncode
if rc == 0 and tmp.exists(): shutil.copy2(tmp, a.out); print('wrote', a.out)
else: sys.exit(f'render failed rc={rc}')
