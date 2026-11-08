"""Slow-motion right-hand close-up, side by side: RL policy rollout (recorded qpos, 50 Hz) | T1_k physics demo (trajectory_mjwp, 400 Hz -> every
8th step = the same 50 Hz instants grid). Native 50 Hz frames, NO resampling, played at 12.5 fps = 0.25x real time. Camera: looks at the
zero-phase smoothed (1 s Savitzky-Golay) right-wrist path from a fixed direction, so the high-frequency shake shows relative to a smooth
camera. Layer label burned in per panel. Render only (task scene proxies; no LIDAR visuals).
  python slowmo_closeup.py POLICY_WORLD.npz DEMO_STAGE_DIR SCENE.xml OUT.mp4 --t0 7 --t1 12"""
import argparse, glob, subprocess
from pathlib import Path
import numpy as np, mujoco
from scipy.signal import savgol_filter
from PIL import Image, ImageDraw, ImageFont
p = argparse.ArgumentParser(); p.add_argument('policy', type=Path); p.add_argument('demo', type=Path); p.add_argument('scene', type=Path); p.add_argument('out', type=Path)
p.add_argument('--t0', type=float, default=7.); p.add_argument('--t1', type=float, default=12.); p.add_argument('--labels', nargs=2, default=['RL POLICY (teacher R2 s2), RL-env physics', 'T1_k PHYSICS DEMO (DAID MPC), seed 0'])
a = p.parse_args()
m = mujoco.MjModel.from_xml_path(str(a.scene)); d = mujoco.MjData(m)
z = np.load(a.policy); QP, TP = z['qpos'], z['time']
f = sorted(glob.glob(str(a.demo/'official/*/*/*/0/trajectory_mjwp.npz')))[0]; zd = np.load(f); QD = zd['qpos'].reshape(-1, m.nq); TD = zd['time'].reshape(-1)
sel_p = (TP >= a.t0) & (TP <= a.t1); grid = TP[sel_p]
QD50 = np.stack([QD[int(np.argmin(np.abs(TD-t)))] for t in grid])          # the demo's physics state at the same instants (no interpolation)
QP50 = QP[sel_p]; wr = m.body('right_wrist').id; ob = m.body('right_object').id
d.qpos[:] = QP50[0]; mujoco.mj_kinematics(m, d); look0 = np.mean([d.xpos[m.body(n).id] for n in ('right_object', 'left_object')], 0); base = m.body('base_link').id
YAW = float(np.degrees(np.arctan2(*(look0[:2]-d.xpos[base][:2])[::-1])))       # same camera convention as render_vega_visual (hand cam: yaw + 230, el -25, 0.42 m)
def wrist_path(Q):
    P = []
    for q in Q: d.qpos[:] = q; mujoco.mj_kinematics(m, d); P.append(.5*(d.xpos[wr]+d.xpos[ob]))
    P = np.array(P); return savgol_filter(P, min(len(P)-1 - (len(P) % 2 == 0), 51) | 1, 2, axis=0)   # ~1 s window at 50 Hz
LP, LD = wrist_path(QP50), wrist_path(QD50)
W, H = 640, 640; r = mujoco.Renderer(m, H, W); opt = mujoco.MjvOption(); opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
F1 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
def cam(lk):
    c = mujoco.MjvCamera(); c.type = mujoco.mjtCamera.mjCAMERA_FREE; c.lookat[:] = lk; c.azimuth = YAW+230; c.elevation = -25; c.distance = .42; return c
frames = []
for i, t in enumerate(grid):
    tiles = []
    for Q, LK, lab in ((QP50, LP, a.labels[0]), (QD50, LD, a.labels[1])):
        d.qpos[:] = Q[i]; mujoco.mj_kinematics(m, d); r.update_scene(d, camera=cam(LK[i]), scene_option=opt); img = Image.fromarray(r.render().copy()); dr = ImageDraw.Draw(img)
        for j, ln in enumerate([lab, f'right hand close-up, carry | t = {t-3.0:5.2f} s demo time | 0.25x slow motion, native 50 Hz frames']):
            bb = dr.textbbox((8, 6+20*j), ln, font=F1); dr.rectangle([bb[0]-3, bb[1]-2, bb[2]+3, bb[3]+2], fill=(0, 0, 0)); dr.text((8, 6+20*j), ln, font=F1, fill=(255, 255, 255))
        tiles.append(np.asarray(img))
    frames.append(np.concatenate(tiles, 1))
proc = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{2*W}x{H}', '-r', '12.5', '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-r', '12.5', str(a.out)], stdin=subprocess.PIPE)
for fr in frames: proc.stdin.write(fr.tobytes())
proc.stdin.close(); proc.wait(); print('wrote', a.out, len(frames), 'frames at 12.5 fps =', len(frames)/12.5, 's video for', grid[-1]-grid[0], 's real')
