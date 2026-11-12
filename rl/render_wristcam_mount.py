"""(11-12) Picture of the student's wrist depth cameras on the robot (render only). Draws, on the task reference pose:
the camera body (a D405-sized box), its viewing axis, its frustum (near 7 cm / far 15 cm = the D405 working range, at the student's fov),
the wrist frame axes (x red, y green, z blue), and the OLD forearm mount's viewing axis (grey) for comparison. Panels: overview, two
close-ups of the right hand (from the thumb side and from the palm side), and what each camera sees (colour + depth) at the pipette /
tube grasp and the tip insertion. --orbit OUT.mp4 also writes a turn-table video around the right hand.
  MUJOCO_GL=egl python render_wristcam_mount.py configs/student_d02v3_ego.json OUT.png [--orbit OUT.mp4]
"""
import argparse, json, math, subprocess
from pathlib import Path
import numpy as np, mujoco
from PIL import Image, ImageDraw, ImageFont
import vega_extras as X

p = argparse.ArgumentParser(); p.add_argument('config', type=Path); p.add_argument('out', type=Path); p.add_argument('--orbit', type=Path)
p.add_argument('--near', type=float, default=.07); p.add_argument('--far', type=float, default=.15)
p.add_argument('--samples', type=int, default=24, help='camera poses drawn from extras.student.depth.mount_rand (if the config has it), thin lines')
a = p.parse_args()
C = json.loads(a.config.read_text()); task = (a.config.parent/C['task']).resolve(); dc = X.depth_cfg(C['extras'])
m = X.load_model(task/'scene.xml', C['extras']); d = mujoco.MjData(m)
m.vis.headlight.ambient[:] = .45; m.vis.headlight.diffuse[:] = .55; m.vis.headlight.specular[:] = .1
z = np.load(task/'reference.npz'); T = z['timestamps']; Q = z['qpos_ref']
F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15); FB = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16)
OLD_TGT = [-.01, .045, -.08]   # the forearm mount's target (DEPTH_DEFAULT) before the yaw23 change
MR = dc.get('mount_rand'); rng = np.random.default_rng(0)


def _R(ax, th):
    c, s_ = math.cos(th), math.sin(th); R = np.eye(3); i, k = [q for q in range(3) if q != ax]; R[i, i] = R[k, k] = c; R[i, k] = -s_; R[k, i] = s_; return R


# the same draw as vega_extras.depth_reset (offset uniform along the wrist axes; Rz(roll) Rx(pitch) Ry(yaw) in the camera frame)
SAMP = [] if not MR else [((2*rng.random(3)-1)*np.array(MR['xyz_mm'])*1e-3, (2*rng.random(3)-1)*np.radians(MR['rpy_deg'])) for _ in range(a.samples)]
W_, H_ = dc['res']; FOVY = math.radians(float(m.cam_fovy[m.camera('wristcam_right').id])); TX, TY = math.tan(FOVY/2)*W_/H_, math.tan(FOVY/2)


def first(gate, pad=0.):
    i = int(np.argmax(z[gate] > .5)); return float(T[i])+pad


def mid(gate):
    i = np.nonzero(z[gate] > .5)[0]; return float(T[i[len(i)//2]])


EV = {'pipette grasp': first('gate_pipette_held', -.25), 'tube grasp': first('gate_tube_held', -.25), 'tip insertion': mid('gate_tip_in_tube')}
T_OVER = first('gate_pipette_held', -.8)   # right hand open, just before the pipette


def pose(t):
    d.qpos[:] = Q[int(np.argmin(np.abs(T-t)))]; mujoco.mj_forward(m, d)


def cam_frame(side):
    c = m.camera(f'wristcam_{side}').id; return d.cam_xpos[c].copy(), d.cam_xmat[c].reshape(3, 3).copy()


def wrist(side):
    b = m.body(f'{side}_wrist').id; return d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()


def add(scn, typ, rgba, size=(0, 0, 0), pos=(0, 0, 0), mat=np.eye(3)):
    g = scn.geoms[scn.ngeom]; scn.ngeom += 1
    mujoco.mjv_initGeom(g, typ, np.asarray(size, float), np.asarray(pos, float), np.asarray(mat, float).ravel(), np.asarray(rgba, np.float32)); return g


def seg(scn, typ, a_, b_, w, rgba):
    g = add(scn, typ, rgba); mujoco.mjv_connector(g, typ, w, np.asarray(a_, float), np.asarray(b_, float))


def overlay(scn, sides=('right', 'left'), axes=True):
    for side in sides:
        s = 1. if side == 'right' else -1.; pc, Rc = cam_frame(side); pw, Rw = wrist(side)
        add(scn, mujoco.mjtGeom.mjGEOM_BOX, (.15, .15, .17, 1), (.021, .021, .0115), pc+Rc@[0, 0, .0115], Rc)   # D405 body behind the lens
        for dp, (r_, p_, y_) in SAMP:   # sampled installation offsets: lens position, viewing axis and far rectangle (thin)
            pq = pc+Rw@(dp*[1, s, 1]); Rq = Rc@_R(2, r_)@_R(0, p_)@_R(1, y_)
            add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, (1, .3, .8, 1), (.0025, 0, 0), pq)
            seg(scn, mujoco.mjtGeom.mjGEOM_LINE, pq, pq+Rq@[0, 0, -a.far], 1.5, (1, .3, .8, 1))
            fq = [pq+Rq@np.r_[sx*a.far*TX, sy*a.far*TY, -a.far] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            for k in range(4): seg(scn, mujoco.mjtGeom.mjGEOM_LINE, fq[k], fq[(k+1) % 4], 1., (1, .45, .85, 1))
        cn = lambda r: [pc+Rc@np.r_[sx*r*TX, sy*r*TY, -r] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        near, far = cn(a.near), cn(a.far)
        for k in range(4):
            seg(scn, mujoco.mjtGeom.mjGEOM_LINE, pc, far[k], 2.5, (1, .85, .1, 1))
            seg(scn, mujoco.mjtGeom.mjGEOM_LINE, far[k], far[(k+1) % 4], 3, (1, .85, .1, 1))
            seg(scn, mujoco.mjtGeom.mjGEOM_LINE, near[k], near[(k+1) % 4], 2, (1, .6, .1, 1))
        seg(scn, mujoco.mjtGeom.mjGEOM_ARROW, pc, pc+Rc@[0, 0, -a.far], .004, (1, .6, 0, 1))                  # current viewing axis
        po = np.array(dc['mount'][0])*[1, s, 1]; fo = np.array(OLD_TGT)*[1, s, 1]-po; fo /= np.linalg.norm(fo)
        seg(scn, mujoco.mjtGeom.mjGEOM_ARROW, pc, pc+Rw@fo*a.far, .0025, (.55, .55, .55, 1))                  # old forearm mount's axis
        if axes:   # a copy of the wrist frame AT the camera + the hand axis (wrist -z) as a white reference line
            seg(scn, mujoco.mjtGeom.mjGEOM_LINE, pc, pc-Rw[:, 2]*a.far, 3, (1, 1, 1, 1))
            for k, col in enumerate(((.9, .1, .1, 1), (.1, .75, .1, 1), (.15, .3, 1, 1))):
                seg(scn, mujoco.mjtGeom.mjGEOM_ARROW, pc, pc+Rw[:, k]*.07, .003, col)


RS = {}; opt = mujoco.MjvOption(); opt.geomgroup[:] = [0, 1, 0, 1, 0, 0]; opt_robot = mujoco.MjvOption(); opt_robot.geomgroup[:] = [0, 1, 0, 0, 0, 0]


def rend(w, h):   # one renderer per size, kept alive (re-creating them broke the EGL context)
    if (w, h) not in RS: RS[(w, h)] = mujoco.Renderer(m, h, w)
    return RS[(w, h)]


def view(lookat, fwd, dist, w=480, h=360, sides=('right', 'left'), axes=True, up=None, fovy=45., o=None):
    """free camera at lookat - dist * fwd; with `up` the image is rolled so that `up` points up (the GL camera is set directly)."""
    r = rend(w, h); fwd = np.asarray(fwd, float)/np.linalg.norm(fwd); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat; cam.distance = dist; cam.azimuth = math.degrees(math.atan2(fwd[1], fwd[0])); cam.elevation = math.degrees(math.asin(np.clip(fwd[2], -1, 1)))
    m.vis.global_.fovy = fovy; r.update_scene(d, cam, o or opt); overlay(r.scene, sides, axes)
    if up is not None:
        u = np.asarray(up, float); u = u-fwd*(u@fwd); u /= np.linalg.norm(u)
        for c in r.scene.camera: c.pos[:] = np.asarray(lookat)-fwd*dist; c.forward[:] = fwd; c.up[:] = u
    im = Image.fromarray(r.render()); m.vis.global_.fovy = 45.; return im


def own(side, w=360, h=270):
    r = rend(w, h); r.update_scene(d, f'wristcam_{side}', opt); rgb = r.render()
    r.enable_depth_rendering(); r.update_scene(d, f'wristcam_{side}', opt); dep = r.render(); r.disable_depth_rendering()
    dn = np.clip((dep-.04)/(.30-.04), 0, 1); ks = np.linspace(0, 1, 5)   # near red -> yellow -> green -> cyan -> far blue
    cols = np.array([[220, 40, 30], [245, 200, 40], [70, 190, 80], [40, 170, 210], [40, 50, 160]], float)
    dimg = np.stack([np.interp(dn, ks, cols[:, c]) for c in range(3)], -1).astype(np.uint8); dimg[dep > .6] = 30
    small = Image.fromarray(dimg).resize((w//3, h//3), Image.NEAREST); im = Image.fromarray(rgb); im.paste(small, (w-w//3-4, h-h//3-4))
    ImageDraw.Draw(im).rectangle([w-w//3-5, h-h//3-5, w-4, h-4], outline=(255, 255, 255)); return im


def label(im, lines):
    dr = ImageDraw.Draw(im); y = 5
    for txt, f_ in lines:
        bb = dr.textbbox((7, y), txt, font=f_); dr.rectangle([bb[0]-3, bb[1]-2, bb[2]+3, bb[3]+2], fill=(0, 0, 0)); dr.text((7, y), txt, font=f_, fill=(255, 255, 255))
        y = bb[3]+5
    return im


# the mount numbers (right hand; the left is mirrored in y)
pose(T_OVER); pc, Rc = cam_frame('right'); pw, Rw = wrist('right'); vc = Rw.T@(-Rc[:, 2]); loc = Rw.T@(pc-pw)
yaw_deg = math.degrees(math.atan2(-vc[0], -vc[2])); tilt_deg = math.degrees(math.asin(-vc[1])); tot = math.degrees(math.acos(-vc[2]))
po = np.array(dc['mount'][0]); fo = np.array(OLD_TGT)-po; fo /= np.linalg.norm(fo)
Ro = Rw.T@Rc@np.diag([1., -1., -1.])   # optical frame (x right, y down, z forward) in the wrist frame
rpy = np.degrees([math.atan2(Ro[2, 1], Ro[2, 2]), math.asin(-Ro[2, 0]), math.atan2(Ro[1, 0], Ro[0, 0])])
vy = math.degrees(math.atan2(-vc[1], -vc[2]))   # the tilt as seen from the thumb side (projection on the wrist y-z plane)
info = [f'camera at ({loc[0]*1e3:.0f}, {loc[1]*1e3:+.0f}, {loc[2]*1e3:+.0f}) mm in the right wrist frame (-z = toward the fingertips, -x = little finger, +y = palm); left mirrored',
        f'view axis {tot:.1f} deg off the hand axis: {yaw_deg:.1f} deg to the little finger (seen from the palm) and {vy:.1f} deg to -y (seen from the thumb; {tilt_deg:.1f} deg out of the x-z plane)',
        f'rpy of the optical frame (x right, y down, z forward) in right_wrist, R = Rz(y) Ry(p) Rx(r): ({rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}) deg',
        f'old forearm mount: {math.degrees(math.atan2(-fo[0], -fo[2])):.1f} deg to the little finger, {math.degrees(math.atan2(-fo[1], -fo[2])):.1f} deg to -y;  '
        f'fov {math.degrees(2*math.atan(TX)):.0f} x {math.degrees(FOVY):.0f} deg, {W_}x{H_} px']
if MR: info.append(f"magenta: {len(SAMP)} sampled installation offsets (mount_rand: +-{MR['xyz_mm']} mm along the wrist axes, +-{MR['rpy_deg']} deg roll/pitch/yaw about the camera axes)")
print('\n'.join(info))

panels = []
pose(T_OVER); bR = d.xmat[m.body('base_link').id].reshape(3, 3); pr, _ = wrist('right'); pl, _ = wrist('left')
ov = view((pr+pl)/2+[0, 0, -.06], -bR[:, 0]*.75-bR[:, 1]*.45-[0, 0, .5], 1.25, axes=False)
panels.append(label(ov, [('overview (reference pose, %.1f s)' % T_OVER, FB), ('yellow: frustum 7-15 cm, orange: view axis', F), ('grey: old forearm mount axis', F)]))
look = pc+Rw@[0, -.03, -.07]
th = view(look, -Rw[:, 0], 1.4, up=Rw[:, 2], fovy=13., sides=('right',), o=opt_robot)
panels.append(label(th, [('right hand seen from the THUMB side (+x)', FB), (f'view axis {vy:.1f} deg off the hand axis (white) toward -y', F),
                         ('at the camera: wrist x red  y green  z blue', F)]))
pa = view(look, -Rw[:, 1], 1.4, up=Rw[:, 2], fovy=13., sides=('right',), o=opt_robot)
panels.append(label(pa, [('right hand seen from the PALM side (+y)', FB), (f'view axis {yaw_deg:.1f} deg to the little finger (-x)', F),
                         (f'grey: old mount, {math.degrees(math.atan2(-fo[0], -fo[2])):.1f} deg', F)]))
row2 = []
for side, ev in (('right', 'pipette grasp'), ('right', 'tip insertion'), ('left', 'tube grasp'), ('left', 'tip insertion')):
    pose(EV[ev]); row2.append(label(own(side), [(f'{side} camera: {ev} ({EV[ev]:.1f} s)', FB), ('inset: depth 4-30 cm', F)]))
top = np.concatenate([np.asarray(x) for x in panels], 1); bot = np.concatenate([np.asarray(x) for x in row2], 1)
hdr = Image.new('RGB', (top.shape[1], 30+17*len(info)), (25, 25, 28)); dr = ImageDraw.Draw(hdr)
dr.text((10, 6), f'Student wrist depth cameras (yaw23 mount)  -  {a.config.name}', font=FB, fill=(255, 255, 255))
for k, txt in enumerate(info): dr.text((10, 28+k*17), txt, font=F, fill=(220, 220, 220))
img = np.concatenate([np.asarray(hdr), top, bot], 0); a.out.parent.mkdir(parents=True, exist_ok=True); Image.fromarray(img).save(a.out); print('wrote', a.out)

if a.orbit:   # turn-table around the right hand at the overview pose
    pose(T_OVER); pw, Rw = wrist('right'); pc, _ = cam_frame('right'); look = pc+Rw@[0, -.03, -.07]; n = 240
    ff = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '640x480', '-r', '40', '-i', '-', '-pix_fmt', 'yuv420p',
                           '-c:v', 'libx264', '-crf', '18', str(a.orbit)], stdin=subprocess.PIPE)
    for i in range(n):
        th_ = 2*math.pi*i/n; fwd = -(math.cos(th_)*Rw[:, 0]+math.sin(th_)*Rw[:, 1])-.25*Rw[:, 2]
        im = label(view(look, fwd, 1.3, 640, 480, sides=('right',), up=Rw[:, 2], fovy=16., o=opt_robot), [('right wrist camera: frustum 7-15 cm (yellow), view axis (orange)', F), ('grey: old mount axis;  white: hand axis', F),
                                                                       ('at the camera: wrist x red  y green  z blue', F)])
        ff.stdin.write(np.asarray(im).tobytes())
    ff.stdin.close(); ff.wait(); print('wrote', a.orbit)
