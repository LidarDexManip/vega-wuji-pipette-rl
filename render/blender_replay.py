"""(11-10) Photoreal replay of an exported RL rollout (render/mj_to_blender.py) inside the user's Blender lab scene
("LIDAR Environment.blend", not included: room scan, table, tube rack, pipette stand, centrifuge, tip box). Layer: render only; the motion
is the recorded MuJoCo qpos, nothing is simulated here.

  blender -b "LIDAR Environment.blend" --python render/blender_replay.py -- EXPORT_DIR OUT_DIR
      [--cameras head overview] [--frames START END STEP | --stills T1 T2 ...] [--res 960 540] [--engine eevee|cycles] [--samples N]
      [--head-hfov 90] [--save OUT.blend] [--no-render]

Scene alignment (MuJoCo world, metres):
  * the Blender scene is modelled at ~2x scale (its table is 1.539 x 3.069 x 1.44 vs MuJoCo 0.768 x 1.536 x 0.72); all of it goes under one
    empty with scale = MuJoCo / Blender table width and an offset that puts both table tops (centre, height) together;
  * props the robot touches are moved onto the MuJoCo furniture (the layout in the .blend differs from the sim by 5-10 cm):
    tube rack ('Cylinder Base') xy bbox centre -> MuJoCo rack plate bbox centre; pipette stand -> MuJoCo stand base;
  * the hand-held objects follow the MuJoCo bodies: the sim pipette meshes (MuJoCo visual geoms); the Blender 50 mL tube
    ('Cylinder' + 'Cylinder_Cap') is scaled per axis onto the MuJoCo tube visual and parented to the tube body. The Blender 'Pipette'
    (a different, smaller model without a moving plunger) is hidden.
Cameras: 'head' = the robot's ZED left camera body (MuJoCo optical frame from the export; horizontal FOV --head-hfov, a placeholder
until the real intrinsics are known); 'overview' = fixed, across the table facing the robot. Frames -> OUT_DIR/<camera>/%04d.png.
"""
import json, math, sys, time
from pathlib import Path
import numpy as np, bpy
from mathutils import Matrix, Quaternion, Vector

argv = sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
import argparse
p = argparse.ArgumentParser(prog='blender_replay'); p.add_argument('export', type=Path); p.add_argument('out', type=Path)
p.add_argument('--cameras', nargs='+', default=['head', 'overview']); p.add_argument('--frames', type=int, nargs=3)
p.add_argument('--stills', type=float, nargs='+', help='render single frames at these rollout times (s)')
p.add_argument('--res', type=int, nargs=2, default=[960, 540]); p.add_argument('--engine', default='eevee', choices=['eevee', 'cycles'])
p.add_argument('--samples', type=int, default=32); p.add_argument('--head-hfov', type=float, default=90.)
p.add_argument('--head-pitch', type=float, default=0., help='extra downward tilt of the head camera (deg); the sim head has no joints and its ZED looks level')
p.add_argument('--keep-scan', action='store_true', help='keep the room scan geometry on the table top and where the robot stands')
p.add_argument('--light', type=float, default=1., help='scale on the added ceiling lights and world light')
p.add_argument('--save', type=Path); p.add_argument('--no-render', action='store_true'); a = p.parse_args(argv)
S = json.loads((a.export/'scene.json').read_text()); MZ = np.load(a.export/'meshes.npz'); PZ = np.load(a.export/'poses.npz')
sc = bpy.context.scene; col = sc.collection; t_start = time.time()


def world_bbox(objs):
    P = np.array([(o.matrix_world @ Vector(c))[:] for o in objs for c in o.bound_box]); return P.min(0), P.max(0)


def world_verts(o):
    M = np.array(o.matrix_world); V = np.array([v.co[:] for v in o.data.vertices]); return V @ M[:3, :3].T + M[:3, 3]


# ---------------------------------------------------------------- environment -> MuJoCo world
OB_ = bpy.data.objects
class _Names(dict):   # the .blend has names with control characters ('\x08Cylinder Base'): look objects up by the printable name
    def __getitem__(self, k): return next(o for o in OB_ if ''.join(c for c in o.name if c.isprintable()) == k)
    def __iter__(self): return iter(OB_)
    def new(self, *x): return OB_.new(*x)
OB = _Names()
env_objs = [o for o in OB_ if o.parent is None]
lo_t, hi_t = world_bbox([OB['Table_new']]); A = S['anchors']; tl, th = np.array(A['table_top'][0]), np.array(A['table_top'][1])
s = float((th[0]-tl[0])/(hi_t[0]-lo_t[0]))
off = np.array([(tl[0]+th[0])/2-s*(lo_t[0]+hi_t[0])/2, (tl[1]+th[1])/2-s*(lo_t[1]+hi_t[1])/2, th[2]-s*hi_t[2]])
root = OB.new('lidar_env_root', None); col.objects.link(root); root.scale = (s, s, s); root.location = off.tolist()
for o in env_objs: o.parent = root; o.matrix_parent_inverse = Matrix.Identity(4)
bpy.context.view_layer.update()
print(f'environment scale {s:.4f}, offset {off.round(4).tolist()}')


def move_xy(objs, target_xy, ref):
    lo, hi = world_bbox(ref); dw = np.array([target_xy[0]-(lo[0]+hi[0])/2, target_xy[1]-(lo[1]+hi[1])/2, 0.])
    for o in objs: o.location = (Vector(o.location) + Vector(dw/s))
    bpy.context.view_layer.update(); return dw


rk = np.array(A['rack_plate']); st = np.array(A['stand_base'])
d_rack = move_xy([OB['Cylinder Base'], OB['Cylinder'], OB['Cylinder_Cap']], (rk[:, :2].mean(0)), [OB['Cylinder Base']])
d_stand = move_xy([OB['Pipette Stand'], OB['Pipette']], st[:, :2].mean(0), [OB['Pipette Stand']])
print(f'moved rack by {np.round(d_rack[:2]*1e3, 1).tolist()} mm, stand by {np.round(d_stand[:2]*1e3, 1).tolist()} mm (onto the MuJoCo layout)')
for n_ in ('Cylinder Base', 'Pipette Stand'):
    lo, hi = world_bbox([OB[n_]]); print(f'  {n_}: bottom z {lo[2]:.4f}, top z {hi[2]:.4f}')
OB['Pipette'].hide_render = OB['Pipette'].hide_viewport = True

# the room scan contains the real table-top objects (rack, stand + pipette, ...) at the real layout and, where the sim robot stands, the
# scanned robot / chairs: delete scan vertices above the table top and in the robot's space (MuJoCo world boxes, the floor is kept)
if not a.keep_scan:
    import bmesh
    tt = np.array(A['table_top']); bx = [((tt[0, 0], tt[1, 0]), (tt[0, 1], tt[1, 1]), (tt[1, 2]-.02, 3.)),    # the scanned table surface and everything on it (the modelled 'Table_new' top shows instead)
                                       ((tt[1, 0]+.002, 1.7), (-0.8, 0.8), (.04, 3.))]                       # robot side of the table
    so = OB['Environment_original']; bm = bmesh.new(); bm.from_mesh(so.data); M = np.array(so.matrix_world)
    V = np.array([v.co[:] for v in bm.verts]) @ M[:3, :3].T + M[:3, 3]; cut = np.zeros(len(V), bool)
    for (x0, x1), (y0, y1), (z0, z1) in bx: cut |= (V[:, 0] > x0) & (V[:, 0] < x1) & (V[:, 1] > y0) & (V[:, 1] < y1) & (V[:, 2] > z0) & (V[:, 2] < z1)
    bm.verts.ensure_lookup_table(); bmesh.ops.delete(bm, geom=[bm.verts[i] for i in np.flatnonzero(cut)], context='VERTS'); bm.to_mesh(so.data); bm.free()
    print(f'room scan: removed {cut.sum()} of {len(V)} vertices (table top objects, robot space)')

# ---------------------------------------------------------------- MuJoCo bodies and visual geoms
mats = {}


def material(rgba):
    key = tuple(round(x, 3) for x in rgba)
    if key in mats: return mats[key]
    mt = bpy.data.materials.new(f'mj_{len(mats)}'); nt = mt.node_tree; bs = nt.nodes['Principled BSDF']
    bs.inputs['Base Color'].default_value = (*key[:3], 1.); bs.inputs['Roughness'].default_value = .45
    if key[3] < .99: bs.inputs['Transmission Weight'].default_value = 1.; bs.inputs['Roughness'].default_value = .05
    mats[key] = mt; return mt


body_obj = []
for k, n_ in enumerate(S['bodies']):
    e = OB.new(f'mj_{n_}', None); col.objects.link(e); e.rotation_mode = 'QUATERNION'; e.empty_display_size = .02; body_obj.append(e)
mesh_cache = {}
for g in S['geoms']:
    if g['type'] == 'mesh':
        me = mesh_cache.get(g['mesh'])
        if me is None:
            V, F = MZ[f'v{g["mesh"]}'], MZ[f'f{g["mesh"]}']; me = bpy.data.meshes.new(g.get('mesh_name', g['mesh']))
            me.from_pydata(V.tolist(), [], F.tolist()); me.shade_smooth(); me.set_sharp_from_angle(angle=math.radians(35)); mesh_cache[g['mesh']] = me
        o = OB.new(g['name'], me)
    else:
        sz = g['size']
        if g['type'] == 'box': bpy.ops.mesh.primitive_cube_add(size=2); o = bpy.context.object; o.scale = sz
        elif g['type'] == 'sphere': bpy.ops.mesh.primitive_uv_sphere_add(radius=sz[0]); o = bpy.context.object
        else:
            bpy.ops.mesh.primitive_cylinder_add(radius=sz[0], depth=2*sz[1], vertices=32); o = bpy.context.object
        for c_ in o.users_collection: c_.objects.unlink(o)
        o.name = g['name']
    col.objects.link(o); o.data.materials.clear(); o.data.materials.append(material(g['rgba']))
    o.parent = body_obj[g['body']]; o.rotation_mode = 'QUATERNION'; o.location = g['pos']; o.rotation_quaternion = Quaternion(g['quat'])

# ---------------------------------------------------------------- tube: the Blender model fitted onto the MuJoCo tube visual
tb = S['bodies'].index('left_object'); tube_geom = [o for o in body_obj[tb].children if o.name.startswith('tube_env')]
if tube_geom:
    tg = tube_geom[0]; Vb = np.array([v.co[:] for v in tg.data.vertices]); Ml = np.array(tg.matrix_basis); Vb = Vb @ Ml[:3, :3].T + Ml[:3, 3]
    clo, chi = Vb.min(0), Vb.max(0)   # MuJoCo tube visual in the tube body frame (axis = body z)
    parts = [OB['Cylinder'], OB['Cylinder_Cap']]; W = np.concatenate([world_verts(o) for o in parts]); blo, bhi = W.min(0), W.max(0)
    k_ = (chi-clo)/(bhi-blo); fit = Matrix.Translation(Vector((chi+clo)/2)) @ Matrix.Diagonal(Vector((*k_, 1.))) @ Matrix.Translation(Vector(-(bhi+blo)/2))
    for o in parts:
        Mw = o.matrix_world.copy(); o.parent = body_obj[tb]; o.matrix_parent_inverse = Matrix.Identity(4); o.matrix_basis = fit @ Mw
    tg.hide_render = tg.hide_viewport = True
    print(f'tube: Blender model {np.round((bhi-blo)*1e3, 1).tolist()} mm -> MuJoCo tube {np.round((chi-clo)*1e3, 1).tolist()} mm (per-axis fit)')

# ---------------------------------------------------------------- keyframes (frame 1 = first exported sample)
P, Qt = PZ['pos'], PZ['quat']; NF = len(P); sc.frame_start, sc.frame_end = 1, NF; sc.render.fps = int(round(S['fps']))
fr = np.arange(1, NF+1, dtype=np.float32)
for k, e in enumerate(body_obj):
    e.animation_data_create(); act = bpy.data.actions.new(f'mj_{S["bodies"][k]}'); e.animation_data.action = act
    if e.animation_data.action_slot is None: e.animation_data.action_slot = act.slots.new(id_type='OBJECT', name=e.name)
    for path, arr in (('location', P[:, k]), ('rotation_quaternion', Qt[:, k])):
        for j in range(arr.shape[1]):
            fc = act.fcurve_ensure_for_datablock(e, path, index=j); fc.keyframe_points.add(NF)
            fc.keyframe_points.foreach_set('co', np.stack([fr, arr[:, j]], 1).ravel()); fc.keyframe_points.foreach_set('interpolation', np.ones(NF, np.int32))  # LINEAR
            fc.update()
print(f'animated {len(body_obj)} bodies x {NF} frames at {S["fps"]:g} fps')

# ---------------------------------------------------------------- cameras, light, render settings
AX = {'+x': (1, 0, 0), '-x': (-1, 0, 0), '+y': (0, 1, 0), '-y': (0, -1, 0), '+z': (0, 0, 1), '-z': (0, 0, -1)}
cams = {}
hc = S['head_camera']; fwd, up = Vector(AX[hc['forward']]), Vector(AX[hc['up']])
cam = OB.new('cam_head', bpy.data.cameras.new('cam_head')); col.objects.link(cam); cam.parent = body_obj[hc['body']]
R = Matrix((up.cross(-fwd), up, -fwd)).transposed()   # columns = camera X, Y (up), Z (backward) in the body frame
cam.matrix_basis = (R @ Matrix.Rotation(-math.radians(a.head_pitch), 3, 'X')).to_4x4()
cam.data.sensor_fit = 'HORIZONTAL'; cam.data.angle = math.radians(a.head_hfov); cam.data.clip_start = .01; cams['head'] = cam
cam = OB.new('cam_overview', bpy.data.cameras.new('cam_overview')); col.objects.link(cam)
cam.location = (-0.62, -0.05, 1.34); cam.rotation_euler = (Vector((0.45, -0.02, 0.92))-Vector(cam.location)).to_track_quat('-Z', 'Y').to_euler()
cam.data.lens = 32; cam.data.clip_start = .01; cams['overview'] = cam

wn = sc.world.node_tree.nodes['Background']; wn.inputs['Color'].default_value = (.82, .82, .8, 1.); wn.inputs['Strength'].default_value = .35*a.light
for i, y in enumerate((-0.45, 0.45)):
    L = bpy.data.lights.new(f'ceiling_{i}', 'AREA'); L.shape = 'RECTANGLE'; L.size, L.size_y = 1.2, .5; L.energy = 110.*a.light; L.color = (1., .97, .92)
    o = OB.new(f'ceiling_{i}', L); col.objects.link(o); o.location = (0.2, y, 2.25)
r = sc.render; r.resolution_x, r.resolution_y = a.res; r.resolution_percentage = 100; r.image_settings.file_format = 'PNG'
if a.engine == 'cycles':
    r.engine = 'CYCLES'; sc.cycles.samples = a.samples; sc.cycles.use_denoising = True
    pr = bpy.context.preferences.addons['cycles'].preferences
    for t_ in ('OPTIX', 'CUDA'):
        try: pr.compute_device_type = t_; pr.get_devices(); break
        except TypeError: continue
    for dv in pr.devices: dv.use = dv.type != 'CPU'
    sc.cycles.device = 'GPU'
else:
    r.engine = 'BLENDER_EEVEE'; sc.eevee.taa_render_samples = a.samples; sc.eevee.use_raytracing = True
print(f'scene built in {time.time()-t_start:.1f} s')
if a.save: a.save.parent.mkdir(parents=True, exist_ok=True); bpy.ops.wm.save_as_mainfile(filepath=str(a.save.resolve()), compress=True); print('saved', a.save)
if a.no_render: sys.exit(0)

T = PZ['time']
if a.stills: frames = [int(np.argmin(np.abs(T-t)))+1 for t in a.stills]
elif a.frames: frames = list(range(a.frames[0], min(a.frames[1], NF)+1, a.frames[2]))
else: frames = list(range(1, NF+1))
for cn in a.cameras:
    sc.camera = cams[cn]; od = a.out/cn; od.mkdir(parents=True, exist_ok=True); t0 = time.time()
    for f in frames:
        sc.frame_set(f); r.filepath = str((od/f'{f:04d}.png').resolve()); bpy.ops.render.render(write_still=True)
    print(f'{cn}: {len(frames)} frames in {time.time()-t0:.1f} s ({(time.time()-t0)/len(frames):.2f} s/frame)')
