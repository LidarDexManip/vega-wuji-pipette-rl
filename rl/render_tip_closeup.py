"""(11-12) Close-up video of the pipette tip vs the tube (render only; MuJoCo task scene). The tube is drawn see-through and two horizontal
cameras 90 deg apart look at its rim (following the tube), so a tip INSIDE the bore and a tip BESIDE the tube look different (the old views
from above could not tell). Burned in per row: time, tip off the tube axis (mm), tip depth below the rim (mm), in bore / beside the tube.
One row per source: a task reference (--ref TASK LABEL, reference.npz qpos_ref on its own clock) or recorded rollouts (--run DIR LABEL,
record_rollouts.py world_<W>.npz; holder placements from its summary.json are applied as in render_compare.py).
  MUJOCO_GL=egl python render_tip_closeup.py SCENE.xml OUT.mp4 --ref tasks/task_x LABEL [--run DIR LABEL --world W] [--t0 10 --t1 16] [--slow 2]
"""
import argparse, json, subprocess
from pathlib import Path
import numpy as np, mujoco
from PIL import Image, ImageDraw, ImageFont

TOP, RIN, ROUT = .0638, .0129, .0144
p = argparse.ArgumentParser(); p.add_argument('scene', type=Path); p.add_argument('out', type=Path)
p.add_argument('--ref', nargs=2, action='append', default=[], metavar=('TASK_DIR', 'LABEL')); p.add_argument('--run', nargs=2, action='append', default=[], metavar=('DIR', 'LABEL'))
p.add_argument('--world', type=int, default=0); p.add_argument('--t0', type=float); p.add_argument('--t1', type=float); p.add_argument('--slow', type=int, default=1)
p.add_argument('--px', type=int, default=420); p.add_argument('--dist', type=float, default=.26); p.add_argument('--tube-alpha', type=float, default=.25); a = p.parse_args()
m = mujoco.MjModel.from_xml_path(str(a.scene)); d = mujoco.MjData(m); S = a.px
m.vis.headlight.ambient[:] = .5; m.vis.headlight.diffuse[:] = .5; m.vis.headlight.specular[:] = .1
for g in range(m.ngeom):   # see-through tube (its visual mesh)
    if m.geom_bodyid[g] == m.body('left_object').id and m.geom_group[g] == 1: m.geom_rgba[g] = [1., .55, .15, a.tube_alpha]   # orange, see-through
    if m.geom_bodyid[g] == m.body('left_object').id and m.geom_group[g] == 1: m.geom_matid[g] = -1
r = mujoco.Renderer(m, S, S); opt = mujoco.MjvOption(); opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14); FB = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 15)
bp, bu, bb = m.body('right_object').id, m.body('left_object').id, m.body('base_link').id

rows = []
for t_, lab in a.ref:
    z = np.load(Path(t_)/'reference.npz'); tip = z['pip_tip_local'] if 'pip_tip_local' in z.files else np.array([-.0077, 0, -.127])
    rows.append(dict(Q=z['qpos_ref'][::2], T=z['timestamps'][::2], lab=lab, spec=None, tip=tip))
for d_, lab in a.run:
    d_ = Path(d_); z = np.load(d_/f'world_{a.world}.npz'); info = json.loads((d_/'summary.json').read_text())['worlds'][a.world]
    rows.append(dict(Q=z['qpos'], T=z['time'], lab=lab, spec=info.get('spec'), tip=np.array([-.0077, 0, -.127])))
if not rows: p.error('give --ref and / or --run')
HOLD = {'stand': ('furn_stand_',), 'rack': ('furn_rackv_', 'furn_rack_plate_')}
HG = {k: np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] == 0 and (m.geom(g).name or '').startswith(v)]) for k, v in HOLD.items()}
G0 = {k: (m.geom_pos[g].copy(), m.geom_quat[g].copy()) for k, g in HG.items()}; HC = {k: np.r_[m.geom_pos[g].mean(0)[:2], 0.] for k, g in HG.items()}


def place_holders(spec):
    for k, g in HG.items():
        dx, dy, yaw = (spec or {}).get(k, [0, 0, 0]); th = np.radians(yaw); c, s_ = np.cos(th), np.sin(th); Rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.]])
        p0, q0 = G0[k]; m.geom_pos[g] = (p0-HC[k])@Rz.T+HC[k]+np.r_[dx*1e-3, dy*1e-3, 0.]; qz = np.r_[np.cos(th/2), 0, 0, np.sin(th/2)]
        for i, gi in enumerate(g): mujoco.mju_mulQuat(m.geom_quat[gi], qz, q0[i])


T = rows[0]['T']; sel = np.ones(len(T), bool)
if a.t0 is not None: sel &= T >= a.t0
if a.t1 is not None: sel &= T <= a.t1
idx = np.nonzero(sel)[0]
W, H = 2*S, S*len(rows); a.out.parent.mkdir(parents=True, exist_ok=True)
ff = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', '50', '-i', '-', '-pix_fmt', 'yuv420p',
                       '-c:v', 'libx264', '-crf', '18', str(a.out)], stdin=subprocess.PIPE)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.distance = a.dist; cam.elevation = -8.
for n, i in enumerate(idx):
    tiles = []
    for R in rows:
        place_holders(R['spec']); j = int(np.argmin(np.abs(R['T']-T[i]))); d.qpos[:] = R['Q'][j]; mujoco.mj_kinematics(m, d)
        Ru, pu, Rp, pp = d.xmat[bu].reshape(3, 3), d.xpos[bu], d.xmat[bp].reshape(3, 3), d.xpos[bp]
        rel = Ru.T@(Rp@R['tip']+pp-pu); off, dep = np.hypot(rel[0], rel[1])*1e3, (TOP-rel[2])*1e3
        state = 'IN the bore' if (off < (RIN-.0018)*1e3 and dep > 0) else ('BESIDE the tube (below rim)' if dep > 0 and off > RIN*1e3 else ('in the wall' if dep > 0 else 'above the rim'))
        fwd = d.xmat[bb].reshape(3, 3)[:, 0]; az0 = np.degrees(np.arctan2(fwd[1], fwd[0]))
        cam.lookat[:] = Ru@np.r_[0, 0, TOP-.02]+pu
        row = []
        for k, daz in enumerate((180., 90.)):   # facing the robot, then from its side
            cam.azimuth = az0+daz; r.update_scene(d, cam, opt); im = Image.fromarray(r.render()); dr = ImageDraw.Draw(im)
            lines = [(R['lab'], FB), (f"t = {T[i]:5.2f} s{'  (%.2gx slow)' % (1/a.slow) if a.slow > 1 else ''}", F)] if k == 0 else [(('view 2: 90 deg to the side'), F)]
            if k == 0: lines += [(f'tip off axis {off:5.1f} mm  depth {dep:6.1f} mm', F), (state, FB)]
            y = 4
            for txt, f_ in lines:
                bb_ = dr.textbbox((6, y), txt, font=f_); dr.rectangle([bb_[0]-3, bb_[1]-2, bb_[2]+3, bb_[3]+2], fill=(0, 0, 0)); dr.text((6, y), txt, font=f_, fill=(255, 255, 255) if 'BESIDE' not in txt and 'wall' not in txt else (255, 90, 90)); y += 19
            row.append(np.asarray(im))
        tiles.append(np.concatenate(row, 1))
    frame = np.concatenate(tiles, 0).astype(np.uint8)
    for _ in range(a.slow): ff.stdin.write(frame.tobytes())
    if n % 100 == 0: print(f'frame {n}/{len(idx)}', flush=True)
ff.stdin.close(); ff.wait(); print('wrote', a.out, len(idx), 'frames')
