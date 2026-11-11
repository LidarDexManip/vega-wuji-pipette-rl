"""(11-09) Side-by-side videos of recorded RL rollouts (record_rollouts.py world_<i>.npz; same world spec and DR seed for every policy).
One row per policy: overview | right-hand close-up (pipette) | left-hand close-up (tube). The close-up cameras follow the zero-phase
smoothed (1 s Savitzky-Golay) wrist-object midpoint from a fixed direction, so high-frequency shake shows relative to a steady camera
(slowmo_closeup.py convention: azimuth = base->objects yaw + 230, elevation -25, 0.42 m). Native 50 Hz frames, no resampling.
Render only: MuJoCo task scene proxies (scene.xml), no LIDAR visuals; layer label burned in.
  MUJOCO_GL=egl python render_compare.py SCENE.xml OUT_DIR --run DIR LABEL [--run ...] --world 2 [--slow 1] [--t0 T --t1 T] [--px 480]
  --ref TASK_DIR LABEL (rows first): a task's reference.npz qpos_ref, kinematics only (nominal holders); --trail S: wrist paths of the last S s.
  (11-11) a run whose world_<i>.npz carries 'depth' (record_rollouts.py --student, depth student) gets a 4th tile: the two wrist depth
  images exactly as the student received them (noise, holes, 40 ms latency), right wrist on top; other rows get a note tile.
Writes OUT_DIR/compare_world<W>.mp4 (all rows) and OUT_DIR/<LABEL>_world<W>.mp4 (one row each)."""
import argparse, json, re, subprocess
from pathlib import Path
import numpy as np, mujoco
from scipy.signal import savgol_filter
from PIL import Image, ImageDraw, ImageFont

p = argparse.ArgumentParser(); p.add_argument('scene', type=Path); p.add_argument('out', type=Path)
p.add_argument('--run', nargs=2, action='append', default=[], metavar=('DIR', 'LABEL')); p.add_argument('--world', type=int, default=0)
p.add_argument('--ref', nargs=2, action='append', default=[], metavar=('TASK_DIR', 'LABEL'),
               help='(rows first) a task reference.npz qpos_ref shown as is: kinematics only, no physics, nominal holder layout; 100 Hz -> 50 Hz')
p.add_argument('--trail', type=float, default=0., help='draw the last S seconds of both wrist paths (right orange, left cyan) in every tile')
p.add_argument('--slow', type=int, default=1, help='write every frame SLOW times (2 = 0.5x)'); p.add_argument('--t0', type=float); p.add_argument('--t1', type=float)
p.add_argument('--px', type=int, default=480); p.add_argument('--tag', default=''); a = p.parse_args()
m = mujoco.MjModel.from_xml_path(str(a.scene)); d = mujoco.MjData(m); S = a.px
m.vis.headlight.ambient[:] = .45; m.vis.headlight.diffuse[:] = .55; m.vis.headlight.specular[:] = .1   # the scene's own light alone is dim
r = mujoco.Renderer(m, S, S); opt = mujoco.MjvOption(); opt.geomgroup[:] = [1, 1, 1, 1, 0, 0]   # group 3 = translucent furniture proxies (rack, stand, table)
FB = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16); F1 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
B = {n: m.body(n).id for n in ('base_link', 'right_wrist', 'left_wrist', 'right_object', 'left_object')}

runs = []
for t_, lab in a.ref:
    rz_ = np.load(Path(t_)/'reference.npz'); Q = rz_['qpos_ref'][::2]   # (11-12) the reference's own clock (demo tasks start at 3.0 s, not 0)
    runs.append(dict(Q=Q, T=rz_['timestamps'][::2], lab=lab, info={'note': 'replay of reference.npz qpos_ref (no RL policy)'}))
if not a.run and not a.ref: p.error('give --run and / or --ref')
for d_, lab in a.run:
    d_ = Path(d_); z = np.load(d_/f'world_{a.world}.npz'); info = json.loads((d_/'summary.json').read_text())['worlds'][a.world]
    runs.append(dict(Q=z['qpos'], T=z['time'], lab=lab, info=info, **({'DEP': z['depth'], 'DEPT': z['depth_t']} if 'depth' in z.files else {})))
T = runs[0]['T']; sel = np.ones(len(T), bool)
if a.t0 is not None: sel &= T >= a.t0
if a.t1 is not None: sel &= T <= a.t1
idx = np.nonzero(sel)[0]


# (11-10) holder placements: move the rack / stand furniture geoms (world body) by the world's spec, as vega_extras.holder_reset does
# (rotate by yaw about the vertical through the holder's nominal geom centroid, then translate); hole pairs move only the reference
HOLD = {'stand': ('furn_stand_',), 'rack': ('furn_rackv_', 'furn_rack_plate_')}
HG = {k: np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] == 0 and (m.geom(g).name or '').startswith(v)]) for k, v in HOLD.items()}
G0 = {k: (m.geom_pos[g].copy(), m.geom_quat[g].copy()) for k, g in HG.items()}
HC = {k: np.r_[m.geom_pos[g].mean(0)[:2], 0.] for k, g in HG.items()}


def place_holders(spec):
    for k, g in HG.items():
        dx, dy, yaw = (spec or {}).get(k, [0, 0, 0]); th = np.radians(yaw); c, s_ = np.cos(th), np.sin(th)
        Rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]); p0, q0 = G0[k]
        m.geom_pos[g] = (p0-HC[k])@Rz.T+HC[k]+np.array([dx*1e-3, dy*1e-3, 0.])
        qz = np.array([np.cos(th/2), 0., 0., np.sin(th/2)]); out = np.zeros(4)
        for j, gi in enumerate(g): mujoco.mju_mulQuat(out, qz, q0[j]); m.geom_quat[gi] = out


def spec_line(info):
    sp = info.get('spec')
    if not sp: return None
    f = lambda v: f"{v[0]:+.0f},{v[1]:+.0f}mm {v[2]:+.0f}\u00b0"
    return f"tube {sp['pair'][0]}->{sp['pair'][1]}  rack {f(sp['rack'])}  stand {f(sp['stand'])}"


def pos(q, b):
    d.qpos[:] = q; mujoco.mj_kinematics(m, d); return d.xpos[B[b]].copy()


q0 = runs[0]['Q'][0]; look0 = .5*(pos(q0, 'right_object')+pos(q0, 'left_object')); base = pos(q0, 'base_link')
YAW = float(np.degrees(np.arctan2(*(look0[:2]-base[:2])[::-1])))
HM = np.mean([.5*(pos(q, 'right_wrist')+pos(q, 'left_wrist')) for q in runs[0]['Q'][::10]], 0)   # mean hand position over the rollout
OV = dict(lookat=HM, azimuth=YAW+180, elevation=-25, distance=.95)                                   # in front of the robot, facing it


def track(Q, side):
    P = np.array([.5*(pos(q, f'{side}_wrist')+pos(q, f'{side}_object')) for q in Q]); w = min(len(P)-1 - (len(P) % 2 == 0), 51) | 1
    return savgol_filter(P, w, 2, axis=0)


for R in runs: R['LR'], R['LL'] = track(R['Q'], 'right'), track(R['Q'], 'left')
NTR = int(round(a.trail*50))
if NTR:
    for R in runs: R['PW'] = {s: np.array([pos(q, f'{s}_wrist') for q in R['Q']]) for s in ('right', 'left')}
TRC = {'right': (1., .55, .1), 'left': (.1, .8, 1.)}


def add_trail(scn, R, i):
    """Capsules along the last NTR steps of each wrist path, fading out with age."""
    for s, c in TRC.items():
        P = R['PW'][s]
        for j in range(max(0, i-NTR), i):
            if scn.ngeom >= scn.maxgeom: return
            g = scn.geoms[scn.ngeom]; al = .15+.85*(j-(i-NTR))/NTR
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(), np.array([*c, al], np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, .0025, P[j], P[j+1]); scn.ngeom += 1


def cam(lookat, az, el, dist):
    c = mujoco.MjvCamera(); c.type = mujoco.mjtCamera.mjCAMERA_FREE; c.lookat[:] = lookat; c.azimuth = az; c.elevation = el; c.distance = dist; return c


def label(img, lines, color=(255, 255, 255)):
    dr = ImageDraw.Draw(img); y = 6
    for j, (txt, f) in enumerate(lines):
        bb = dr.textbbox((8, y), txt, font=f); dr.rectangle([bb[0]-3, bb[1]-2, bb[2]+3, bb[3]+2], fill=(0, 0, 0)); dr.text((8, y), txt, font=f, fill=color); y = bb[3]+6
    return img


def outcome(info):
    if 'note' in info: return info['note']
    if info['tube_seated'] and info['pipette_seated']: return 'outcome: success (tube + pipette seated)'
    if info['reach_end']: return f"outcome: reached the end, tube {'seated' if info['tube_seated'] else 'NOT seated'}, pipette {'seated' if info['pipette_seated'] else 'NOT seated'}"
    why = ', '.join(f'{o} {r_}' for o, r_ in zip(('pipette', 'tube'), info.get('fail_reason') or [None, None]) if r_)
    return f"outcome: failure at t = {info['first_fail_s']:.2f} s" + (f' ({why})' if why else '')


DW = 2*S//3 if any('DEP' in R for R in runs) else 0   # depth tile width (two 4:3 images stacked)


VIR = np.array([[253, 231, 37], [94, 201, 98], [33, 145, 140], [59, 82, 139], [68, 1, 84]], np.float32)   # viridis anchors, near -> far


def depth_rgb(z):
    """int16 mm -> RGB: 5 cm (yellow) .. 30 cm and beyond (dark purple), viridis; invalid (0) = black."""
    v = np.clip((z.astype(np.float32)-50.)/250., 0, 1)*(len(VIR)-1); i = np.minimum(v.astype(int), len(VIR)-2); f = (v-i)[..., None]
    im = ((1-f)*VIR[i]+f*VIR[i+1]).astype(np.uint8); im[z <= 0] = 0; return im


def depth_tile(R, i):
    h = S//2; tile = Image.new('RGB', (DW, S), (24, 24, 24)); dr = ImageDraw.Draw(tile)
    if 'DEP' not in R:
        for j, txt in enumerate(('no camera:', 'the teacher observes the', 'true object / contact state')): dr.text((10, 10+20*j), txt, font=F1, fill=(200, 200, 200))
        return np.asarray(tile)
    k = int(np.searchsorted(R['DEPT'], i, side='right'))-1
    for j, side in enumerate(('right wrist (pipette)', 'left wrist (tube)')):
        if k >= 0: tile.paste(Image.fromarray(depth_rgb(R['DEP'][k, j])).resize((DW, h), Image.NEAREST), (0, j*h))
        label_at = ImageDraw.Draw(tile); bb = label_at.textbbox((6, j*h+4), side, font=F1); label_at.rectangle([bb[0]-3, bb[1]-2, bb[2]+3, bb[3]+2], fill=(0, 0, 0)); label_at.text((6, j*h+4), side, font=F1, fill=(255, 255, 255))
    dr = ImageDraw.Draw(tile); txt = 'yellow 5 cm .. purple 30+ cm, black: none'; bb = dr.textbbox((6, S-44), txt, font=F1); dr.rectangle([bb[0]-3, bb[1]-2, bb[2]+3, bb[3]+2], fill=(0, 0, 0)); dr.text((6, S-44), txt, font=F1, fill=(255, 255, 255))
    txt = 'student depth input (96x72, as given)'; bb = dr.textbbox((6, S-22), txt, font=F1); dr.rectangle([bb[0]-3, bb[1]-2, bb[2]+3, bb[3]+2], fill=(0, 0, 0)); dr.text((6, S-22), txt, font=F1, fill=(255, 255, 255))
    return np.asarray(tile)


def row(R, i):
    place_holders(R['info'].get('spec')); q = R['Q'][i]; d.qpos[:] = q; mujoco.mj_kinematics(m, d); tiles = []
    for c, title in ((cam(OV['lookat'], OV['azimuth'], OV['elevation'], OV['distance']), 'overview'),
                     (cam(R['LR'][i], YAW+230, -25, .42), 'right hand (pipette), steady camera'),
                     (cam(R['LL'][i], YAW+130, -25, .42), 'left hand (tube), steady camera')):
        r.update_scene(d, camera=c, scene_option=opt)
        if NTR: add_trail(r.scene, R, i)
        img = Image.fromarray(r.render().copy())
        lines = [(title, F1)]
        if title == 'overview': lines = [(R['lab'], FB), (f"t = {R['T'][i]:5.2f} s  |  {'%.2gx slow motion' % (1/a.slow) if a.slow > 1 else 'real time'}, 50 Hz frames", F1), (outcome(R['info']), F1)] + ([(spec_line(R['info']), F1)] if spec_line(R['info']) else [])
        tiles.append(np.asarray(label(img, lines)))
    if DW: tiles.append(depth_tile(R, i))
    return np.concatenate(tiles, 1)


a.out.mkdir(parents=True, exist_ok=True); fps = 50.
def writer(path, w, h):
    return subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
                             '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', str(path)], stdin=subprocess.PIPE)
safe = lambda s: re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')
sfx = f'_world{a.world}{a.tag}'
W_ = 3*S+DW; outs = {'all': writer(a.out/f'compare{sfx}.mp4', W_, S*len(runs))}
names = [safe(R['lab'].split(' ')[0]) for R in runs]
if len(set(names)) < len(names): names = [f'row{i}_{n}' for i, n in enumerate(names)]   # first words collide: keep one file per row
for R, nm in zip(runs, names): outs[R['lab']] = writer(a.out/f'{nm}{sfx}.mp4', W_, S)
for n, i in enumerate(idx):
    rows = [row(R, i) for R in runs]
    for _ in range(a.slow):
        outs['all'].stdin.write(np.concatenate(rows, 0).tobytes())
        for R, rw in zip(runs, rows): outs[R['lab']].stdin.write(rw.tobytes())
    if n % 200 == 0: print(f'frame {n}/{len(idx)}', flush=True)
for o in outs.values(): o.stdin.close(); o.wait()
print('wrote', sorted(str(x) for x in a.out.glob(f'*{sfx}.mp4')), f'{len(idx)} frames x{a.slow} at 50 fps')
