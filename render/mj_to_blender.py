"""(11-10) Export a recorded RL rollout (rl/record_rollouts.py world_<i>.npz) for Blender: the visual geometry of the robot and the
two hand-held objects plus their world poses at every rendered frame. Layer: kinematic replay of recorded qpos (mj_kinematics), no physics.

  .venv-rl-head/bin/python render/mj_to_blender.py SCENE.xml ROLLOUT_DIR OUT_DIR --world 2 [--stride 2] [--t0 T --t1 T]

Writes OUT_DIR/
  meshes.npz   v<k> (n, 3) float32 vertices in the geom frame, f<k> (m, 3) int32 faces, for every exported mesh geom k
  scene.json   bodies (name, parent world index), geoms (body, type, size, local pos / quat wxyz, rgba, mesh key), fps, the
               furniture anchors the Blender side aligns its props to (rack plate box, stand base), the head camera axes
  poses.npz    pos (F, B, 3), quat (F, B, 4) wxyz world poses of the exported bodies (sign-continuous), time (F,)
Exported: geom group 1 on non-world bodies (robot visuals + the pipette / tube visuals); group 0 (hand collision meshes) and group 3
(collision / furniture proxies) are left out. Furniture comes from the Blender scene, placed on the MuJoCo anchors.
"""
import argparse, json
from pathlib import Path
import numpy as np, mujoco

p = argparse.ArgumentParser(); p.add_argument('scene', type=Path); p.add_argument('run', type=Path); p.add_argument('out', type=Path)
p.add_argument('--world', type=int, default=0); p.add_argument('--stride', type=int, default=2, help='50 Hz control steps per rendered frame')
p.add_argument('--t0', type=float); p.add_argument('--t1', type=float); a = p.parse_args()
m = mujoco.MjModel.from_xml_path(str(a.scene)); d = mujoco.MjData(m)
z = np.load(a.run/f'world_{a.world}.npz'); Q, T = z['qpos'], z['time']
info = json.loads((a.run/'summary.json').read_text())['worlds'][a.world]
assert not any(info['spec']['rack']) and not any(info['spec']['stand']), 'holder offsets are not replayed yet: use worlds with zero rack / stand offsets'
sel = np.arange(0, len(Q), a.stride)
if a.t0 is not None: sel = sel[T[sel] >= a.t0]
if a.t1 is not None: sel = sel[T[sel] <= a.t1]
TYPES = {2: 'sphere', 3: 'capsule', 5: 'cylinder', 6: 'box', 7: 'mesh'}

G = [i for i in range(m.ngeom) if m.geom_group[i] == 1 and m.geom_bodyid[i] != 0 and m.geom_type[i] in TYPES]
B = sorted({int(m.geom_bodyid[i]) for i in G} | {m.body('zed_left_camera').id, m.body('left_object').id, m.body('right_object').id})
bidx = {b: k for k, b in enumerate(B)}
a.out.mkdir(parents=True, exist_ok=True); meshes = {}; geoms = []
for i in G:
    rgba = m.mat_rgba[m.geom_matid[i]] if m.geom_matid[i] >= 0 else m.geom_rgba[i]
    g = dict(name=m.geom(i).name, body=bidx[int(m.geom_bodyid[i])], type=TYPES[int(m.geom_type[i])], size=m.geom_size[i].tolist(),
             pos=m.geom_pos[i].tolist(), quat=m.geom_quat[i].tolist(), rgba=[float(x) for x in rgba], mesh=None)
    if g['type'] == 'mesh':
        k = int(m.geom_dataid[i]); key = f'{k}'
        if key not in meshes:
            va, vn, fa, fn = m.mesh_vertadr[k], m.mesh_vertnum[k], m.mesh_faceadr[k], m.mesh_facenum[k]
            meshes[f'v{key}'] = m.mesh_vert[va:va+vn].astype(np.float32); meshes[f'f{key}'] = m.mesh_face[fa:fa+fn].astype(np.int32)
        g['mesh'] = key; g['mesh_name'] = m.mesh(k).name
    geoms.append(g)
np.savez_compressed(a.out/'meshes.npz', **meshes)

P = np.zeros((len(sel), len(B), 3), np.float32); R = np.zeros((len(sel), len(B), 4), np.float32)
for f, s in enumerate(sel):
    d.qpos[:] = Q[s]; mujoco.mj_kinematics(m, d); P[f] = d.xpos[B]; R[f] = d.xquat[B]
for f in range(1, len(sel)):   # q and -q are the same rotation; keep the sign continuous so keyframe interpolation never flips
    R[f] *= np.where(np.sum(R[f]*R[f-1], -1, keepdims=True) < 0, -1., 1.)
np.savez_compressed(a.out/'poses.npz', pos=P, quat=R, time=T[sel].astype(np.float32))

# ---- anchors for the Blender props (MuJoCo world, metres): rack = bbox of the rack top plates, stand = its base plate
d.qpos[:] = Q[0]; mujoco.mj_kinematics(m, d)
def box_bbox(prefix):
    ids = [i for i in range(m.ngeom) if (m.geom(i).name or '').startswith(prefix)]
    lo = np.min([d.geom_xpos[i]-np.abs(d.geom_xmat[i].reshape(3, 3)) @ m.geom_size[i] for i in ids], 0)
    hi = np.max([d.geom_xpos[i]+np.abs(d.geom_xmat[i].reshape(3, 3)) @ m.geom_size[i] for i in ids], 0)
    return lo.tolist(), hi.tolist()
anchors = {'rack_plate': box_bbox('furn_rack_plate_'), 'stand_base': box_bbox('furn_stand_base'), 'table_top': box_bbox('furn_table_0')}
# head camera: the zed_left_camera body axis that points at the hands' work area, and the one closest to world up
c = m.body('zed_left_camera').id; Rc = d.xmat[c].reshape(3, 3); look = np.array([.2, 0., .78])-d.xpos[c]; look /= np.linalg.norm(look)
ax = [s*Rc[:, j] for j in range(3) for s in (1, -1)]; names = [f'{"+-"[k % 2]}{"xyz"[k//2]}' for k in range(6)]
fwd = int(np.argmax([v @ look for v in ax])); up = max((k for k in range(6) if k//2 != fwd//2), key=lambda k: ax[k][2])
scene = dict(source=dict(scene=str(a.scene), run=str(a.run), world=a.world, outcome={k: info[k] for k in ('reach_end', 'tube_seated', 'pipette_seated', 'first_fail_s')}),
             fps=50./a.stride, frames=len(sel), bodies=[m.body(b).name for b in B], geoms=geoms, anchors=anchors,
             head_camera=dict(body=bidx[c], forward=names[fwd], up=names[up], forward_dot=float(ax[fwd] @ look)))
(a.out/'scene.json').write_text(json.dumps(scene, indent=1))
nv = sum(len(v) for k, v in meshes.items() if k[0] == 'v')
print(f'{len(geoms)} geoms ({sum(g["type"] == "mesh" for g in geoms)} mesh, {nv} verts) on {len(B)} bodies, {len(sel)} frames at {50/a.stride:g} fps; '
      f'head camera forward {names[fwd]} (cos {ax[fwd] @ look:.2f} to the work area), up {names[up]}')
print('anchors', json.dumps(anchors))
