"""TRACKER (11-06): contact labels from the task's own KINEMATIC reference (qpos_ref), by distance. EVAL ONLY (no training change).
Layer: REFERENCE LABELS (CPU mj_geomDistance on the reference states; no rollouts, no physics, no MPC data).

Per 100 Hz frame, per hand link (same 42-link order as contact_ref.npz): signed min distance between the link's collider geoms and its own
side's object (right -> pipette body + plunger, left -> tube), over the scene's explicit collision pairs (the env's contact model).
Labels at 2 / 5 / 10 mm. Also: which object part is nearest (pipette body / plunger / tube), for the video-grasp consistency check:
  left tube = thumb-vs-index pinch (+ middle allowed); ring, pinky and palm on the tube contradict the video.
  right pipette = four-finger hold (index, middle, ring, pinky on the body); the thumb only presses (plunger), so thumb on the body contradicts.

  python -I build_kin_contact_labels.py TASK_DIR
writes TASK_DIR/contact_ref_kin.npz and TASK_DIR/contact_ref_kin.json
"""
import json, sys, time
from pathlib import Path
import numpy as np, mujoco

TH_MM = (2, 5, 10); DMAX = .012
task = Path(sys.argv[1]).resolve()
m = mujoco.MjModel.from_xml_path(str(task/'scene.xml')); d = mujoco.MjData(m)
r = np.load(task/'reference.npz', allow_pickle=True); Q = r['qpos_ref']; F = len(Q)
z = np.load(task/'contact_ref.npz', allow_pickle=True); links = [str(x) for x in z['links']]; li = {n: i for i, n in enumerate(links)}
side_of_link = np.asarray(z['side_of_link']); L = len(links)


def subtree(root):
    s = {root}
    for b in range(m.nbody):
        p = b
        while p > 0:
            if p in s: s.add(b); break
            p = int(m.body_parentid[p])
    return s


body_part = {}   # object body id -> part code: 1 pipette body, 2 plunger, 3 tube
for b in subtree(m.body('right_object').id): body_part[b] = 1
for b in subtree(m.body('plunger_carrier').id) | {m.body('pipette_plunger').id}: body_part[b] = 2
for b in subtree(m.body('left_object').id): body_part[b] = 3
pairs = []   # (link index, hand geom, object geom, part)
for i in range(m.npair):
    g1, g2 = int(m.pair_geom1[i]), int(m.pair_geom2[i])
    for ga, gb in ((g1, g2), (g2, g1)):
        n = m.body(int(m.geom_bodyid[ga])).name; ob = int(m.geom_bodyid[gb])
        if n in li and ob in body_part:
            part = body_part[ob]; s = 0 if part in (1, 2) else 1
            if side_of_link[li[n]] == s: pairs.append((li[n], ga, gb, part))
pairs = np.array(pairs, dtype=np.int64)
print(f'{len(pairs)} hand-object collision pairs, {F} frames', flush=True)
dmin = np.full((F, L), np.inf); part = np.zeros((F, L), np.int8); t0 = time.time()
rb = m.geom_rbound
for k in range(F):
    d.qpos[:] = Q[k]; mujoco.mj_kinematics(m, d)
    c = np.linalg.norm(d.geom_xpos[pairs[:, 1]]-d.geom_xpos[pairs[:, 2]], axis=1)-rb[pairs[:, 1]]-rb[pairs[:, 2]]
    for j in np.flatnonzero(c < DMAX):
        l_, ga, gb, pt = pairs[j]; dist = mujoco.mj_geomDistance(m, d, int(ga), int(gb), DMAX, None)
        if dist < dmin[k, l_]: dmin[k, l_] = dist; part[k, l_] = pt
print(f'distances done in {time.time()-t0:.1f} s', flush=True)
masks = {f'mask_{t}mm': dmin <= t*1e-3 for t in TH_MM}
np.savez_compressed(task/'contact_ref_kin.npz', links=np.array(links), side_of_link=side_of_link, dmin=dmin, part=part, **masks,
                    layer=np.array('kinematic-reference labels: signed min distance link<->own object over the explicit collision pairs (qpos_ref)'))

# ---- consistency with the video grasp, and side-by-side with the physics-demo labels (held frames only) ----
held = {0: r['gate_pipette_held'].astype(bool), 1: r['gate_tube_held'].astype(bool)}; ph = r['phase'].astype(int); pn = [str(x) for x in r['phase_names']]
demo = np.asarray(z['mask']).astype(bool)
fing = lambda side, f: [i for i, n in enumerate(links) if n.startswith(('right_' if side == 0 else 'left_')+f)]
out = {'layer': 'kinematic-reference labels (distance) vs the physics-demo labels (force); held frames of that hand only', 'thresholds_mm': list(TH_MM),
       'held_frames': {'right (pipette)': int(held[0].sum()), 'left (tube)': int(held[1].sum())}, 'per_phase': {}}
for p, name in enumerate(pn):
    row = {}
    for t in TH_MM:
        M = masks[f'mask_{t}mm']; flags = {}
        hl = held[1] & (ph == p); hr = held[0] & (ph == p)
        if hl.any():
            anyf = lambda f, pt=None: (M[:, fing(1, f)] & ((part[:, fing(1, f)] == pt) if pt else True)).any(1)
            flags['left: thumb NOT touching tube (pinch needs it)'] = round(float((~anyf('thumb'))[hl].mean()), 3)
            flags['left: index NOT touching tube (pinch needs it)'] = round(float((~anyf('index'))[hl].mean()), 3)
            flags['left: ring touching tube (video: no)'] = round(float(anyf('ring')[hl].mean()), 3)
            flags['left: pinky touching tube (video: no)'] = round(float(anyf('pinky')[hl].mean()), 3)
            flags['left: palm touching tube (video: no)'] = round(float(anyf('wrist')[hl].mean()), 3)
            flags['left: middle touching tube (allowed)'] = round(float(anyf('middle')[hl].mean()), 3)
        if hr.any():
            anyr = lambda f, pt: (M[:, fing(0, f)] & (part[:, fing(0, f)] == pt)).any(1)
            for f in ('index', 'middle', 'ring', 'pinky'):
                flags[f'right: {f} NOT on pipette body (four-finger hold needs it)'] = round(float((~anyr(f, 1))[hr].mean()), 3)
            flags['right: thumb on pipette BODY (video: thumb only presses)'] = round(float(anyr('thumb', 1)[hr].mean()), 3)
            flags['right: thumb on plunger (press / light rest: allowed)'] = round(float(anyr('thumb', 2)[hr].mean()), 3)
            flags['right: a non-thumb finger on the plunger (video: no)'] = round(float(np.any([anyr(f, 2) for f in ('index', 'middle', 'ring', 'pinky')], 0)[hr].mean()), 3)
            flags['right: palm on pipette (info)'] = round(float((anyr('wrist', 1) | anyr('wrist', 2))[hr].mean()), 3)
        row[f'{t}mm'] = flags
    # per link: fraction of held frames labelled touching, kinematic (5 mm) vs physics demo
    per_link = []
    for l_, n in enumerate(links):
        h = held[side_of_link[l_]] & (ph == p)
        if not h.any(): continue
        a, b = float(masks['mask_5mm'][h, l_].mean()), float(demo[h, l_].mean())
        if a > .05 or b > .05: per_link.append([n, round(b, 3), round(a, 3), round(float(np.median(dmin[h, l_][np.isfinite(dmin[h, l_])]))*1e3, 1) if np.isfinite(dmin[h, l_]).any() else None])
    row['links (name, demo-label frac, kin-5mm frac, median dist mm)'] = per_link
    out['per_phase'][name] = row
(task/'contact_ref_kin.json').write_text(json.dumps(out, indent=1))
print(json.dumps({p: v['5mm'] for p, v in out['per_phase'].items()}, indent=1))
