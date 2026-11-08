"""Per-frame label of the ACTIVE REFERENCE finger event per hand (layer: kinematic reference of the RL task, task_T1k_still3_noff).
IDLE = every finger joint of that hand has reference speed < 0.005 rad/s (the idle-finger metric's threshold); otherwise the event kind:
press thumb (right, press-thumb LP window; 'onto / off the cap' = thumb moving from 9.5 s to the window / 0.4 s after it), release open
(from the hand's first post-release segment start), closing (0.6 s before each latch = rising edge of the held gate, to 0.9 s after: the
settle), pre-shape (before the first latch), else HOLD (~still if < 3 deg/s, else moves). Second line: reference speed + moving fingers.
  python finger_event_labels.py TASK_DIR ROLL_NPZ OUT.json"""
import json, sys
from pathlib import Path
import numpy as np
task, roll, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
r = np.load(task/'reference.npz'); T = json.loads((task/'task.json').read_text()); fs = T['finger_smoothing']
names = [str(x) for x in r['qpos_joint_names']]; t = r['timestamps']; q = r['qpos_ref']
press = fs['_press_thumb_lp']['window_s']; segs = fs['_fingers_still']['joints']
tr = np.load(roll)['time']; res = {'layer': 'kinematic reference (task_T1k_still3_noff); labels per rollout frame', 'time': tr.tolist()}
for side, pre, gate in (('right', 'r_', 'gate_pipette_held'), ('left', 'l_', 'gate_tube_held')):
    ids = [i for i, n in enumerate(names[:61]) if n.startswith(pre)]; assert len(ids) == 20, (side, len(ids))
    v = np.abs(np.gradient(q[:, ids], t, axis=0)).max(1); vs = np.interp(tr, t, v)
    g = r[gate].astype(int); latches = t[1:][np.diff(g) == 1].tolist() + ([float(t[0])] if g[0] else [])
    rel = min(s[0] for j, x in segs.items() if j.startswith(pre) for s in x['segments [t0, t1, dq_deg, kind, T_s]'] if s[3] == 'post-release')
    fing = ['thumb', 'index', 'middle', 'ring', 'pinky']; nm = [names[i] for i in ids]
    Fv = np.stack([np.interp(tr, t, np.abs(np.gradient(q[:, [x for x in ids if f in names[x]]], t, axis=0)).max(1)) for f in fing], 1)
    lab, sub = [], []
    for k, tk in enumerate(tr):
        mv = [f'{fing[f]} {np.degrees(Fv[k, f]):.0f}' for f in np.argsort(-Fv[k]) if np.degrees(Fv[k, f]) >= 3.][:3]
        sub.append('ref finger speed max %.1f deg/s' % np.degrees(vs[k]) + (('  (' + ', '.join(mv) + ')') if mv else ''))
        if vs[k] < .005: lab.append('IDLE (reference still)')
        elif side == 'right' and press[0] <= tk <= press[1]: lab.append('press thumb')
        elif side == 'right' and (9.5 <= tk < press[0] or press[1] < tk <= press[1]+.4) and np.degrees(Fv[k, 0]) >= 3.: lab.append('press thumb (onto / off the cap)')
        elif tk >= rel: lab.append('release open')
        elif any(L-.6 <= tk <= L+.9 for L in latches): lab.append('closing (incl. settle after latch)')
        elif latches and tk < min(latches): lab.append('pre-shape')
        elif np.degrees(vs[k]) < 3.: lab.append('HOLD (reference ~still, < 3 deg/s)')
        else: lab.append('HOLD (reference moves)')
    res[side] = {'labels': lab, 'sub': sub, 'latches_s': latches, 'release_s': rel, 'ref_speed_max_rad_s': np.round(vs, 4).tolist()}
    runs = []; 
    for k, x in enumerate(lab):
        if not runs or runs[-1][0] != x: runs.append([x, tr[k], tr[k]])
        runs[-1][2] = tr[k]
    print(side, 'latches', np.round(latches, 2), 'release', rel); [print('   %-26s %.2f-%.2f' % (a, b, c)) for a, b, c in runs if c-b >= .02 or a != 'IDLE (reference still)']
out.write_text(json.dumps(res))
