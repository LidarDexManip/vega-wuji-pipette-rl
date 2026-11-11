"""TRACKER (11-10): contact_ref.npz for a KINEMATIC-reference task (no physics demo), in rl/contact_ref.py's format, so the env's contact terms
and the wrench-margin gate work unchanged. Layer: REFERENCE LABELS (CPU mj_geomDistance on qpos_ref; no rollouts, no physics).

Same as rl/contact_ref.py except the contact test: a hand link is "in contact" with its own side's object (right -> pipette subtree incl.
carrier / plunger, left -> tube) when the signed distance over the scene's explicit collision pairs is <= TH_MM (instead of a normal force
> 0.1 x the object's weight in the physics demo); its point = the midpoint of the closest-point segment, in the object frame; force = 1.
Required wrench (u_req, w_req_norm), free-finger mask and fing_acts: as rl/contact_ref.py, from the reference object motion.

  .venv-rl-head/bin/python -I tracker/build_kin_contact_ref.py TASK [--th-mm 2]
"""
import argparse, json
from pathlib import Path
import numpy as np, mujoco
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.spatial.transform import Rotation

L_T, WIN, DMAX = .05, 30, .012
ap = argparse.ArgumentParser(); ap.add_argument('task', type=Path); ap.add_argument('--th-mm', type=float, default=2.); a = ap.parse_args()
task = a.task; m = mujoco.MjModel.from_xml_path(str(task/'scene.xml')); d = mujoco.MjData(m); r = np.load(task/'reference.npz')
Q = r['qpos_ref'].astype(float); F = len(Q)


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
pairs = []
for i in range(m.npair):
    g1, g2 = int(m.pair_geom1[i]), int(m.pair_geom2[i])
    for ga, gb in ((g1, g2), (g2, g1)):
        n = m.body(int(m.geom_bodyid[ga])).name; ob = int(m.geom_bodyid[gb])
        if n in li and ob in objs[side_of_link[li[n]]]: pairs.append((li[n], ga, gb))
pairs = np.array(pairs, dtype=np.int64); rb = m.geom_rbound
mask = np.zeros((F, len(links)), bool); pts = np.zeros((F, len(links), 3)); force = np.zeros((F, len(links))); dmin = np.full((F, len(links)), np.inf)
ft = np.zeros(6)
for k in range(F):
    d.qpos[:] = Q[k]; mujoco.mj_kinematics(m, d)
    c = np.linalg.norm(d.geom_xpos[pairs[:, 1]]-d.geom_xpos[pairs[:, 2]], axis=1)-rb[pairs[:, 1]]-rb[pairs[:, 2]]
    for j in np.flatnonzero(c < DMAX):
        l_, ga, gb = pairs[j]; dist = mujoco.mj_geomDistance(m, d, int(ga), int(gb), DMAX, ft)
        if dist < dmin[k, l_]:
            dmin[k, l_] = dist; s = side_of_link[l_]; bo = m.body(OBJ[s]).id
            pts[k, l_] = d.xmat[bo].reshape(3, 3).T@((ft[:3]+ft[3:])/2-d.xpos[bo])
mask = dmin <= a.th_mm*1e-3; force[mask] = 1.; pts[~mask] = 0.
u = np.zeros((F, 2, 6)); wn = np.zeros((F, 2))
for j, o in enumerate(OBJ):
    ad = m.jnt_qposadr[m.body_jntadr[m.body(o).id]]; P = gaussian_filter1d(Q[:, ad:ad+3], 5, axis=0); acc = np.gradient(np.gradient(P, .01, axis=0), .01, axis=0)
    rv = np.unwrap(Rotation.from_quat(Q[:, ad+3:ad+7], scalar_first=True).as_rotvec(), axis=0); rv = gaussian_filter1d(rv, 5, axis=0)
    al = np.gradient(np.gradient(rv, .01, axis=0), .01, axis=0); I = float(np.mean(m.body_inertia[m.body(o).id]))
    w = np.c_[mass[j]*(acc-np.array([0, 0, -9.81])), I*al/L_T]; n_ = np.linalg.norm(w, axis=1); u[:, j] = w/np.maximum(n_, 1e-9)[:, None]; wn[:, j] = n_
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
    n = acts[i]; free[:, j] = ~touch[(n[:2], finger_of(n))]
np.savez_compressed(task/'contact_ref.npz', links=np.array(links), side_of_link=side_of_link, mask=mask, pts=pts, force=force, u_req=u, w_req_norm=wn,
                    free_mask=free, fing_acts=np.array(fing_acts), dmin=dmin)
held = {0: r['gate_pipette_held'].astype(bool), 1: r['gate_tube_held'].astype(bool)}
summ = {'layer': f'kinematic reference labels (signed distance <= {a.th_mm} mm, explicit pairs; no physics)', 'links': len(links), 'frames': F, 'pairs': len(pairs),
        'contact_fraction_per_finger': {f'{pre}{f}': round(float(touch[(pre, f)].mean()), 3) for (pre, f) in touch},
        'links_in_contact_while_held_p50': {('right' if s == 0 else 'left'): float(np.median(mask[held[s]][:, side_of_link == s].sum(1))) if held[s].any() else 0. for s in (0, 1)},
        'deepest_penetration_mm_while_held': {('right' if s == 0 else 'left'): round(float(dmin[held[s]][:, side_of_link == s].min())*1e3, 1) if held[s].any() else None for s in (0, 1)}}
(task/'contact_ref.json').write_text(json.dumps(summ, indent=1)); print(json.dumps(summ))
