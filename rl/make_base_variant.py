"""RL-base variants for the finger-jitter diagnosis (11-06; layer: RL reference / base command). Copies TASK to OUT and edits reference.npz:
  --no-finger-ff        finger ctrl minus the stage's velocity feed-forward kv/kp * qdot_ref (ctrl_sim and controls; the squeeze / preload stays)
  --ring-abd DEG T0 T1  left ring abduction offset DEG (cosine ramp T0..T1, then held) on ctrl and qpos_ref: separates the ring from the pinky
                        (reference ring-pinky -3.1 mm self-penetration while holding; ring -8 deg only -> +2.0 mm, pinky / rack unchanged)
  python make_base_variant.py TASK OUT [--no-finger-ff] [--ring-abd DEG T0 T1]"""
import json, shutil, sys
from pathlib import Path
import numpy as np, mujoco
src, out = Path(sys.argv[1]), Path(sys.argv[2]); a = sys.argv[3:]
if out.exists(): raise SystemExit(f'{out} exists')
shutil.copytree(src, out); z = dict(np.load(out/'reference.npz', allow_pickle=True)); m = mujoco.MjModel.from_xml_path(str(out/'scene.xml'))
t = z['timestamps']; cs = z['ctrl_sim']; ts = z['sim_times'] if 'sim_times' in z else np.arange(len(cs))*float(z['sim_dt']); rep = {'source': str(src)}
fr = np.searchsorted(t, z['timestamps'][0]+np.arange(len(z['controls']))*0+0) if False else None
if '--no-finger-ff' in a:
    rep['no_finger_ff'] = {}
    for i in range(m.nu):
        jn = m.joint(int(m.actuator_trnid[i, 0])).name
        if not jn.startswith(('l_', 'r_')): continue
        v = int(m.jnt_dofadr[m.actuator_trnid[i, 0]]); kk = -m.actuator_biasprm[i, 2]/m.actuator_gainprm[i, 0]
        ff_100 = kk*z['qvel_ref'][:, v]; ff_s = np.interp(ts, t, ff_100); cs[:, i] -= ff_s; z['controls'][:, i] -= kk*z['qvel_ref'][:len(z['controls']), v] if len(z['controls']) == len(t) else np.interp(np.linspace(t[0], t[-1], len(z['controls'])), t, ff_100)
        rep['no_finger_ff'][jn] = round(float(np.degrees(np.abs(ff_s).max())), 3)
if '--ring-abd' in a:
    k = a.index('--ring-abd'); deg, t0, t1 = float(a[k+1]), float(a[k+2]), float(a[k+3]); i = m.actuator('l_ring_finger_mcp_abd').id; q = int(m.jnt_qposadr[m.joint('l_ring_finger_mcp_abd').id])
    w100 = .5-.5*np.cos(np.pi*np.clip((t-t0)/(t1-t0), 0, 1)); ws = np.interp(ts, t, w100); off = np.radians(deg)
    cs[:, i] += off*ws; z['qpos_ref'][:, q] += off*w100
    if len(z['controls']) == len(t): z['controls'][:, i] += off*w100
    rep['ring_abd'] = {'deg': deg, 'ramp_s': [t0, t1]}
z['ctrl_sim'] = cs; np.savez(out/'reference.npz', **z)
tj = json.loads((out/'task.json').read_text()); tj['base_variant'] = rep; (out/'task.json').write_text(json.dumps(tj, indent=1)); print(json.dumps(rep)[:400])
