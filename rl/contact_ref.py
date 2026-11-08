"""Layer: REFERENCE LABELS from the PHYSICS demo (CPU mj_forward on the task's executed states; no new rollouts). Real-robot line, Vega DAID scene.
Per 100 Hz frame, per hand link body (descendants of right_wrist / left_wrist that own a colliding geom): does it push on its own side's object
(right -> pipette subtree incl. carrier / plunger, left -> tube) with normal force > F_ALPHA x object weight; force-weighted mean contact point
in the object frame; per object the required wrench direction u = w_req / |w_req| from the reference object motion (Jiming T3).
Also the per-frame FREE-FINGER mask: finger joints whose finger has no labelled contact within +-0.3 s (naturalness: these should not move).

  .venv-rl-head/bin/python -I contact_ref.py TASK
"""
import json, sys
from pathlib import Path
import numpy as np, mujoco
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.spatial.transform import Rotation

F_ALPHA, L_T, WIN = .1, .05, 30
task = Path(sys.argv[1]); m = mujoco.MjModel.from_xml_path(str(task/'scene.xml')); d = mujoco.MjData(m); r = np.load(task/'reference.npz')
Q, V = r['qpos_ref'], r['qvel_ref']; F = len(Q)


def subtree(root):
    s = {root}
    for b in range(m.nbody):
        p = b
        while p > 0:
            if p in s: s.add(b); break
            p = int(m.body_parentid[p])
    return s


hand = {0: subtree(m.body('right_wrist').id), 1: subtree(m.body('left_wrist').id)}
coll = {int(m.geom_bodyid[g]) for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]}
links = sorted({m.body(b).name for s in (0, 1) for b in hand[s] if b in coll}); li = {n: i for i, n in enumerate(links)}
side_of_link = np.array([0 if n.startswith('right_') else 1 for n in links])
pip = subtree(m.body('right_object').id) | subtree(m.body('plunger_carrier').id); tube = subtree(m.body('left_object').id); objs = (pip, tube)
OBJ = ('right_object', 'left_object'); mass = [sum(m.body_mass[b] for b in pip), sum(m.body_mass[b] for b in tube)]
mask = np.zeros((F, len(links)), bool); pts = np.zeros((F, len(links), 3)); force = np.zeros((F, len(links)))
for k in range(F):
    d.qpos[:] = Q[k]; d.qvel[:] = V[k]; mujoco.mj_forward(m, d); acc = {}
    for i in range(d.ncon):
        c = d.contact[i]; b1, b2 = int(m.geom_bodyid[c.geom[0]]), int(m.geom_bodyid[c.geom[1]]); n1, n2 = m.body(b1).name, m.body(b2).name
        if n1 in li and b2 in objs[side_of_link[li[n1]]]: L, s = li[n1], side_of_link[li[n1]]
        elif n2 in li and b1 in objs[side_of_link[li[n2]]]: L, s = li[n2], side_of_link[li[n2]]
        else: continue
        f6 = np.zeros(6); mujoco.mj_contactForce(m, d, i, f6)
        if f6[0] < F_ALPHA*mass[s]*9.81: continue
        bo = m.body(OBJ[s]).id; loc = d.xmat[bo].reshape(3, 3).T@(c.pos-d.xpos[bo])
        a = acc.setdefault(L, [np.zeros(3), 0.]); a[0] += f6[0]*loc; a[1] += f6[0]
    for L, (p, w) in acc.items(): mask[k, L] = True; pts[k, L] = p/w; force[k, L] = w
u = np.zeros((F, 2, 6)); wn = np.zeros((F, 2))
for j, o in enumerate(OBJ):
    a = m.jnt_qposadr[m.body_jntadr[m.body(o).id]]; P = gaussian_filter1d(Q[:, a:a+3], 5, axis=0); acc = np.gradient(np.gradient(P, .01, axis=0), .01, axis=0)
    rv = np.unwrap(Rotation.from_quat(Q[:, a+3:a+7], scalar_first=True).as_rotvec(), axis=0); rv = gaussian_filter1d(rv, 5, axis=0)
    al = np.gradient(np.gradient(rv, .01, axis=0), .01, axis=0); I = float(np.mean(m.body_inertia[m.body(o).id]))
    w = np.c_[mass[j]*(acc-np.array([0, 0, -9.81])), I*al/L_T]; n_ = np.linalg.norm(w, axis=1); u[:, j] = w/np.maximum(n_, 1e-9)[:, None]; wn[:, j] = n_
# free-finger mask over the actuated finger joints: finger = thumb / index / middle / ring / pinky of a side
acts = [m.actuator(i).name for i in range(m.nu)]; fing_acts = [i for i, n in enumerate(acts) if n.startswith(('r_', 'l_'))]
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
def finger_of(name): return next((f for f in FINGERS if f in name), None)
touch = {}
for s, pre, lp in ((0, 'r_', 'right_'), (1, 'l_', 'left_')):
    for f in FINGERS:
        cols = [li[n] for n in links if n.startswith(lp) and f in n]
        t_ = mask[:, cols].any(1) if cols else np.zeros(F, bool)
        touch[(pre, f)] = maximum_filter1d(t_.astype(np.uint8), size=2*WIN+1).astype(bool)
free = np.zeros((F, len(fing_acts)), bool)
for j, i in enumerate(fing_acts):
    n = acts[i]; f = finger_of(n); free[:, j] = ~touch[(n[:2], f)]
np.savez_compressed(task/'contact_ref.npz', links=np.array(links), side_of_link=side_of_link, mask=mask, pts=pts, force=force, u_req=u, w_req_norm=wn,
                    free_mask=free, fing_acts=np.array(fing_acts))
summ = {'layer': 'reference labels from the physics demo (force > %.2f x object weight)' % F_ALPHA, 'links': len(links), 'frames': F,
        'contact_fraction_per_finger': {f'{pre}{f}': round(float(touch[(pre, f)].mean()), 3) for (pre, f) in touch},
        'free_fraction_left_ring_pinky': round(float(free[:, [j for j, i in enumerate(fing_acts) if acts[i].startswith(('l_ring', 'l_pinky'))]].mean()), 3)}
(task/'contact_ref.json').write_text(json.dumps(summ, indent=1)); print(json.dumps(summ))
