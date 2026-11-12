"""TRACKER (11-10): a kinematic-reference task from one do-as-i-do RGB-D demo episode (HF dataset ytung/pipette-tasks, episodes/<name>),
with the robot square to the table and ground-truth holders / holes. Layer: TASK EXPORT (no physics run). Method and the reasons for each
step: docs/DEMO_TO_TASK.md. The whole pipeline (download, this script, contact labels, start audit, press prior, checks, training config):
tracker/demo_to_task.py.

Inputs (episode dir): reconstruction/{calibration.json, depth/000000.png, pipette/poses.npz, tube/poses.npz},
retargeting/{right_pipette,left_tube}/trajectory.npz. Template (robot, Wuji hands, pipette + plunger, tube, table, rack, stand and their
seated relations, latched grasps): tasks/task_kin_noprior_press.

Stages (main() runs them in order; every number they decide is in OUT/build_report.json)
  1 frames      camera -> table (RANSAC plane on the first depth frame, x perpendicular to the far table edge, origin = camera foot point)
                -> world (robot base yaw 180 deg = square to the table, at the template's distance; demo origin --fwd m ahead of the base)
  2 holders     stand: rigid move of the template's stand with its seated pipette onto the demo pipette's static start xy / spin;
                rack: translation putting hole --hole under the demo tube's static start rim centre; tube placed back into the same hole
  3 events      lift / seat times of both objects from their demo heights
  4 grasps      one latched grasp per hand: --grasp template (default; rigid, thumb on the cap, compatible with the holders) or demo
  5 objects     seated poses in the holders; extraction / insertion straight along the hole axis until clear (insertion depth + 1 cm), then a
                --blend-s blend to the mid-air pose = the demo's smoothed position and axis direction (not its spin) + a constant retarget;
                a held object never passes lower than 1.5 cm above a holder top or 2 cm above the table; (11-12) the pipette goes into the tube
                like a peg into a hole (tip_insert): over the tube 15 mm above the rim, axes aligned, straight down the axis to the demo depth
                (<= 75 mm below the rim), straight back up; tip_path_check: the lower pipette never inside the tube wall, tip in the bore
  6 wrists      wrist = object . grasp^-1 while held; hover 12 cm back along the palm normal + 6 cm up, 6 cm / 2 cm at -0.6 s, at the grasp
                pose at lift - 0.8 s; after the seat open and retreat 8 cm / 4 cm
  7 arms        the mid-air retarget (shift down / forward / left, tilt scale, spin of each object about its axis) is searched for arm
                reachability with CONTINUOUS IK along the whole path (batched over candidates and 3 collision-free start postures);
                feasible = max(mm + deg) <= 4 over the path, then the smallest change; final arm path: continuous IK at 100 Hz
  8 hands       fingers open -> latched (close over [lift-0.8, lift-0.2] s, open 0.1-0.5 s after the seat); an open thumb that cuts its
                object by > 2 mm is replaced by the nearest posture with >= 5 mm clearance; right thumb tip on the template's cap point
                while held; press window = tip-in-tube window, plunger 0 -> 13.4 mm with 0.4 s ramps
  9 outputs     reference.npz (env format: qpos / qvel / controls at 100 Hz, ctrl_sim at 400 Hz, phase, gates, latch mask), scene.xml,
                task.json ('ref_holes' for vega_extras' hole pairs), build_report.json, wrist_targets.npz (wrist / object targets, checks)

  .venv-rl-head/bin/python -I tracker/build_demo_task.py EPISODE_DIR tasks/task_<name>_noprior [--fwd 0.30] [--hole r1c1] [--force]
OUT must sit in tasks/ (the scene's asset paths are relative).
"""
import argparse, json, math, re, sys, time, hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
import numpy as np, mujoco, torch
from PIL import Image
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, binary_closing, binary_opening
from scipy.spatial.transform import Rotation as Rot, Slerp

HERE = Path(__file__).resolve().parent; V = HERE.parent
sys.path.insert(0, str(V/'rl')); sys.path.insert(0, str(V/'common')); sys.path.insert(0, str(HERE))
TEMPLATE = V/'tasks/task_kin_noprior_press'
FPS = 100.; SIM_DT = .0025
HOLD = {'stand': ('furn_stand_',), 'rack': ('furn_rackv_', 'furn_rack_plate_')}
DEPTH = {'stand': .1061, 'rack': .0533}           # insertion depths (as vega_extras.holder_init)
PIP_TIP = np.array([-.0077, 0., -.127])           # pipette tip in the sim pipette frame (11-12: the mesh axis is at x = -7.7 mm; was (0, 0, -.127),
                                                  # which put the real tip 7.7 mm off the tube axis; checked against the mesh in main)
TUBE_RIN, TUBE_ROUT = .0129, .0144                # tube bore / outer radius
INSERT = dict(pre=.015, v_max=.25, t_min=.4, approach_s=.7, min_depth=.02, merge_s=.6)   # (11-12) tip insertion: see tip_insert
RETIME = dict(factor=2., ramp_s=.2, pad_s=.1)     # (11-12, user: "a little bit too fast") the three insertions (pipette into the stand, tube into
                                                  # the rack, pipette tip into the tube) run on a clock slowed by `factor` (same path, half the
                                                  # speed), min-jerk ramps of ramp_s outside each segment +- pad_s; everything else unchanged
TUBE_TOP, TUBE_BOTTOM = .0638, -.0522             # tube rim / bottom in the sim tube frame (inner radius 12.9 mm)
TIP_MAX_DEPTH = .075                              # pipette shaft radius at the rim <= 11.9 mm down to here
THUMB_ON_CAP = np.array([.2120, .1829, .5855])    # right_thumb_tip in the pipette_plunger frame, template press (mean, std <= 1 mm)
PLUNGER_MAX = .0134
DATA_TO_SIM = {'pipette': ('rotz', -math.pi/2),   # sim pipette frame = data mesh frame . Rz(-90 deg) (identical meshes)
               'tube': ('shift', .0687-TUBE_TOP)}  # different tube meshes: rims aligned (data mesh rim at +68.7 mm)
SEARCH = dict(dz=(0., -.05, -.10), dfwd=(0., .05, .10), dlat=(0., .05, .10), alpha=(1., .6), spins_deg=range(-180, 180, 30), feasible=4., every=4)
EXTRACT = dict(lead_s=.15, v_max=.3, t_min=.6)    # (11-11) extraction / insertion along the hole axis: min-jerk (peak speed v_max = the demo's carry
                                                  # speed), starts lead_s before the lift event; slower moves the clear-of-holder time and the mid-air blend
SMOOTH_LIMITS = dict(accel_m_s2=10., band_mm=.1, band_deg=.03)   # (11-11) reference smoothness check: max acceleration (100 Hz second difference), 8-24 Hz RMS


# ------------------------------------------------------------------------------------------------------------------- small geometry
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
def T_(R=None, p=None):
    T = np.eye(4)
    if R is not None: T[:3, :3] = R
    if p is not None: T[:3, 3] = p
    return T
def rotz(a): c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
def inv(T): Ti = np.eye(4); Ti[:3, :3] = T[:3, :3].T; Ti[:3, 3] = -T[:3, :3].T@T[:3, 3]; return Ti
def minjerk(s): s = np.clip(s, 0, 1); return s*s*s*(10-15*s+6*s*s)
def yaw_of(R): return math.atan2(R[1, 0], R[0, 0])
def bodyT(m, d, n): return T_(d.xmat[m.body(n).id].reshape(3, 3), d.xpos[m.body(n).id])


def blend_T(A, B, w):
    """pose between A (w=0) and B (w=1): position lerp, rotation slerp"""
    if w <= 0: return A.copy()
    if w >= 1: return B.copy()
    return T_(Slerp([0, 1], Rot.from_matrix([A[:3, :3], B[:3, :3]]))(w).as_matrix(), (1-w)*A[:3, 3]+w*B[:3, 3])


def resample_poses(t_src, Ts, valid, t_dst, sigma_s=.05):
    """(n,4,4) poses at t_src (valid mask) -> t_dst: linear / quaternion interpolation between valid samples, then Gaussian smoothing"""
    tv = t_src[valid]; Tv = Ts[valid]
    P = np.stack([np.interp(t_dst, tv, Tv[:, i, 3]) for i in range(3)], 1)
    q = Rot.from_matrix(Tv[:, :3, :3]).as_quat()
    for i in range(1, len(q)):
        if q[i]@q[i-1] < 0: q[i] = -q[i]
    Q = np.stack([np.interp(t_dst, tv, q[:, i]) for i in range(4)], 1)
    s = sigma_s*FPS
    if s > 0: P = gaussian_filter1d(P, s, axis=0, mode='nearest'); Q = gaussian_filter1d(Q, s, axis=0, mode='nearest')
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    out = np.tile(np.eye(4), (len(t_dst), 1, 1)); out[:, :3, :3] = Rot.from_quat(Q).as_matrix(); out[:, :3, 3] = P
    return out


def mean_T(Ts):
    U, _, Vt = np.linalg.svd(Ts[:, :3, :3].sum(0)); R = U@Vt
    if np.linalg.det(R) < 0: U[:, -1] *= -1; R = U@Vt
    return T_(R, Ts[:, :3, 3].mean(0))


def align_rotvec(a, b):
    """rotation vectors (n, 3) turning unit vectors a onto b by the shortest rotation"""
    cr = np.cross(a, b); sn = np.linalg.norm(cr, axis=-1); th = np.arctan2(sn, np.sum(a*b, -1))
    return np.where(sn[..., None] > 1e-9, cr/np.maximum(sn, 1e-12)[..., None], 0.)*th[..., None]


# ------------------------------------------------------------------------------------------------------------------- 1 frames
def table_frame(ep):
    """camera -> table: RANSAC plane on the first depth frame (normal = table z, pointing to the camera), x = the camera's optical axis
    projected, then turned to be perpendicular to the far table edge (where the table plane ends at the top of the image)"""
    cal = json.loads((ep/'reconstruction/calibration.json').read_text()); K = np.array(cal['K'])
    z = np.array(Image.open(ep/'reconstruction/depth/000000.png')).astype(float)*float(cal.get('depth_scale_m', 1e-3))
    H, W = z.shape; vv, uu = np.mgrid[0:H, 0:W]; P = np.stack([(uu-K[0, 2])*z/K[0, 0], (vv-K[1, 2])*z/K[1, 1], z], -1)
    ok = z > 0; Pf = P[ok]; rng = np.random.default_rng(0); best = (0, None)
    for _ in range(400):
        a, b, c = Pf[rng.choice(len(Pf), 3, replace=False)]; n = np.cross(b-a, c-a)
        if np.linalg.norm(n) < 1e-9: continue
        n /= np.linalg.norm(n); cnt = int((np.abs(Pf@n-n@a) < .004).sum())
        if cnt > best[0]: best = (cnt, (n, n@a))
    n, off = best[1]; Q = Pf[np.abs(Pf@n-off) < .004]; c0 = Q.mean(0); n = np.linalg.svd(Q-c0, full_matrices=False)[2][2]
    if n[1] > 0: n = -n                                       # up = towards the camera's -y (image up)
    foot = (c0@n)*n                                            # foot point of the camera centre (origin) on the plane
    fwd = np.array([0, 0, 1.]); x = fwd-(fwd@n)*n; x /= np.linalg.norm(x); R0 = np.stack([x, np.cross(n, x), n])
    inl = (np.abs(P@n-(c0@n)) < .005) & ok; pts = []
    for u in range(30, W-30, 4):                               # far edge: first run of 6 table pixels from the top of each column
        col = inl[:, u]
        for v in range(H-6):
            if col[v:v+6].all(): pts.append(R0@(P[v, u]-foot)); break
    pts = np.array(pts)[:, :2]; best = None
    for _ in range(400):
        i, j = rng.choice(len(pts), 2, replace=False); dv = pts[j]-pts[i]
        if np.linalg.norm(dv) < .05: continue
        dv /= np.linalg.norm(dv); nn = np.array([-dv[1], dv[0]]); c = int((np.abs((pts-pts[i])@nn) < .01).sum())
        if best is None or c > best[0]: best = (c, dv, pts[i])
    c, dv, p0 = best; ne = np.array([-dv[1], dv[0]])
    if ne[0] < 0: ne = -ne                                      # edge normal pointing away from the camera
    yaw = math.atan2(ne[1], ne[0]); Rc = rotz(-yaw)
    return {'R_tc': Rc@R0, 'foot': foot, 'cam_height': float(abs(c0@n)), 'edge_yaw_deg': math.degrees(yaw), 'edge_inliers': [int(c), len(pts)],
            'edge_dist_m': float((Rc@np.r_[p0, 0])[0]), 'K': K}


def load_demo(ep, TF):
    """demo object poses (sim object frames) and retargeted wrists / fingers, all in the table frame, 30 fps"""
    T_tc = T_(TF['R_tc'], -TF['R_tc']@TF['foot']); out = {}
    for name in ('pipette', 'tube'):
        z = np.load(ep/f'reconstruction/{name}/poses.npz'); kind, v = DATA_TO_SIM[name]
        S = T_(rotz(v)) if kind == 'rotz' else T_(p=[0, 0, v])
        out[name] = T_tc@z['T_camera_object']@S; out[name+'_valid'] = z['valid'].astype(bool)
    out['t'] = np.arange(len(out['pipette']))/30.
    for side, h in (('right', 'right_pipette'), ('left', 'left_tube')):
        z = np.load(ep/f'retargeting/{h}/trajectory.npz'); v = z['valid'].astype(bool)
        Wc = np.where(v[:, None, None], z['wrist_pose'], np.eye(4))      # retargeting world -> camera -> table
        out[side] = {'W': T_tc@np.linalg.inv(z['T_world_camera'])@Wc, 'valid': v, 'q': z['q'], 'names': [str(x) for x in z['joint_names']]}
    return out


def load_template():
    mt = mujoco.MjModel.from_xml_path(str(TEMPLATE/'scene.xml')); dt = mujoco.MjData(mt); rt = np.load(TEMPLATE/'reference.npz')
    dt.qpos[:] = rt['qpos_ref'][0]; mujoco.mj_kinematics(mt, dt); acc = {}
    for i in range(mt.ngeom):
        mm = re.match(r'furn_rackv_low_(r\d+c\d+)_s\d+$', mt.geom(i).name or '')
        if mm: acc.setdefault(mm.group(1), []).append(dt.geom_xpos[i, :2].copy())
    z_table = max(dt.geom_xpos[g][2]+mt.geom_size[g][2] for g in range(mt.ngeom) if (mt.geom(g).name or '').startswith('furn_table'))
    return SimpleNamespace(m=mt, d=dt, r=rt, pip_seat=bodyT(mt, dt, 'right_object'), tube_seat=bodyT(mt, dt, 'left_object'),
                           holes={k: np.mean(v, 0) for k, v in acc.items()}, z_table=z_table)


# ------------------------------------------------------------------------------------------------------------------- 2 holders, scene
def holder_geoms(m, name):
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] == 0 and (m.geom(g).name or '').startswith(HOLD[name])]


def ground_truth_holders(Pd, Ud, ev_k, tpl, hole, rep):
    """seated poses and holder transforms from the demo's static start poses (template seated relations)"""
    kp0, kp1, ku0, ku1 = ev_k
    Ps = Pd[:max(10, kp0-15)]; xy_p = np.median(Ps[:, :2, 3], 0); yaws = np.array([yaw_of(R) for R in Ps[:, :3, :3]])
    dyaw = math.atan2(np.median(np.sin(yaws)), np.median(np.cos(yaws)))-yaw_of(tpl.pip_seat[:3, :3])
    P_seat = T_(rotz(dyaw)@tpl.pip_seat[:3, :3], [xy_p[0], xy_p[1], tpl.pip_seat[2, 3]]); S_stand = P_seat@inv(tpl.pip_seat)
    rim = lambda T: T[:3, :3]@np.array([0, 0, TUBE_TOP])+T[:3, 3]
    Us = Ud[:max(10, ku0-15)]; rim_xy = np.median(np.array([rim(T) for T in Us])[:, :2], 0); h0 = tpl.holes[hole]
    S_rack = T_(p=[rim_xy[0]-h0[0], rim_xy[1]-h0[1], 0.])
    U_seat = S_rack@T_(p=[h0[0]-tpl.holes['r0c3'][0], h0[1]-tpl.holes['r0c3'][1], 0])@tpl.tube_seat   # template tube is seated in r0c3
    end_p = np.median(Pd[kp1+15:, :2, 3], 0); end_u = np.median(np.array([rim(T) for T in Ud[ku1+15:]])[:, :2], 0)
    rep['holders'] = {'stand_shift_m': S_stand[:3, 3].round(4).tolist(), 'stand_yaw_change_deg': round(math.degrees(dyaw), 2),
                      'rack_shift_m': S_rack[:3, 3].round(4).tolist(), 'hole': hole,
                      'pipette_seat_world': P_seat[:3, 3].round(4).tolist(), 'tube_seat_world': U_seat[:3, 3].round(4).tolist(),
                      'demo_end_vs_start_pipette_mm': round(float(np.linalg.norm(end_p-xy_p))*1e3, 1),
                      'demo_end_vs_start_tube_rim_mm': round(float(np.linalg.norm(end_u-rim_xy))*1e3, 1),
                      'demo_tube_tilt_in_rack_deg': round(float(np.degrees(np.arccos(np.median(Us[:, 2, 2])))), 1),
                      'demo_pipette_tilt_in_stand_deg': round(float(np.degrees(np.arccos(np.median(Ps[:, 2, 2])))), 1)}
    return P_seat, U_seat, S_stand, S_rack


def write_scene(src, dst, base_xy, base_yaw, moves, obj_init, base_z=0.):
    """template scene.xml with the robot base moved and holder geoms moved rigidly (moves: name -> 4x4 world transform)"""
    tree = ET.parse(src); root = tree.getroot()
    for b in root.iter('body'):
        if b.get('name') == 'base_link':
            b.set('pos', f'{base_xy[0]:.9f} {base_xy[1]:.9f} {base_z:.9f}'); b.set('quat', ' '.join(f'{x:.12g}' for x in (math.cos(base_yaw/2), 0, 0, math.sin(base_yaw/2))))
        if b.get('name') in obj_init:
            T = obj_init[b.get('name')]
            b.set('pos', ' '.join(f'{x:.9f}' for x in T[:3, 3])); b.set('quat', ' '.join(f'{x:.12g}' for x in Rot.from_matrix(T[:3, :3]).as_quat(scalar_first=True)))
    cnt = {k: 0 for k in moves}
    for g in root.iter('geom'):
        n = g.get('name') or ''
        for k, S in moves.items():
            if n.startswith(HOLD[k]):
                p = np.array([float(x) for x in (g.get('pos') or '0 0 0').split()]); qq = np.array([float(x) for x in (g.get('quat') or '1 0 0 0').split()])
                R2 = S[:3, :3]@Rot.from_quat(qq, scalar_first=True).as_matrix(); p2 = S[:3, :3]@p+S[:3, 3]
                g.set('pos', ' '.join(f'{x:.9f}' for x in p2)); g.set('quat', ' '.join(f'{x:.12g}' for x in Rot.from_matrix(R2).as_quat(scalar_first=True)))
                cnt[k] += 1
    tree.write(dst, encoding='unicode')
    return cnt


def holder_boxes(m, d):
    """world AABB (x0, x1, y0, y1, top z) of the stand and the rack meshes"""
    d.qpos[:] = m.qpos0; mujoco.mj_kinematics(m, d); out = []
    for hn in ('stand', 'rack'):
        V_ = np.concatenate([m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]:m.mesh_vertadr[m.geom_dataid[g]]+m.mesh_vertnum[m.geom_dataid[g]]]@d.geom_xmat[g].reshape(3, 3).T+d.geom_xpos[g]
                             for g in holder_geoms(m, hn) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH])
        out.append((V_[:, 0].min(), V_[:, 0].max(), V_[:, 1].min(), V_[:, 1].max(), V_[:, 2].max()))
    return out


# ------------------------------------------------------------------------------------------------------------------- 3 events, 4 grasps
def events(Td):
    """first / last frame where the (smoothed) object is > 15 mm above its static height"""
    zz = gaussian_filter1d(Td[:, 2, 3], 3); up = (zz-max(np.median(zz[:60]), np.median(zz[-40:]))) > .015
    return int(np.argmax(up)), len(up)-1-int(np.argmax(up[::-1]))


def template_grasps(tpl, names, rep):
    """the template's latched grasps: object-in-wrist mean over its held window (rigid within 2 mm / 0.4 deg), finger posture just before its
    press (right, plunger up) / mid-hold (left), open posture 1 s before its hold"""
    mt, dt, rt = tpl.m, tpl.d, tpl.r; G = {}; qlat = {}; qopen = {}; kpress = int(np.argmax(rt['phase'] == 2))-30
    for s, obj, gate, kl in (('right', 'right_object', 'gate_pipette_held', kpress), ('left', 'left_object', 'gate_tube_held', None)):
        H = np.flatnonzero(rt[gate]); kl = kl if kl is not None else int(H[len(H)//2]); Gs = []
        for k in H[::10]:
            dt.qpos[:] = rt['qpos_ref'][k]; mujoco.mj_kinematics(mt, dt); Gs.append(inv(bodyT(mt, dt, s+'_wrist'))@bodyT(mt, dt, obj))
        Gs = np.array(Gs); G[s] = mean_T(Gs); qa = [int(mt.jnt_qposadr[mt.joint(n).id]) for n in names[s]]; ko = max(0, int(H[0])-int(FPS))
        qlat[s] = rt['qpos_ref'][kl, qa].astype(float); qopen[s] = rt['qpos_ref'][ko, qa].astype(float)
        rep.setdefault('grasp', {})[s] = {'source': 'template latched grasp', 'latched_frame_s': round(float(rt['timestamps'][kl]), 2),
                                         'open_frame_s': round(float(rt['timestamps'][ko]), 2), 'obj_in_wrist_cm': (G[s][:3, 3]*100).round(1).tolist(),
                                         'spread_mm_max': round(float(np.linalg.norm(Gs[:, :3, 3]-G[s][:3, 3], axis=1).max())*1e3, 1)}
    return G, qlat, qopen


def demo_grasps(D, t, Wd, Od, win, rep):
    """the retargeting's grasps: object-in-wrist mean / finger median over a steady window (they change while held in pipette_depth_02)"""
    G = {}; qlat = {}; qopen = {}
    for s in ('right', 'left'):
        v = D[s]['valid']; sel = v & (t >= win[s][0]) & (t <= win[s][1])
        Gs = np.einsum('nij,njk->nik', np.linalg.inv(Wd[s][sel]), Od[s][sel]); G[s] = mean_T(Gs)
        qlat[s] = np.median(D[s]['q'][sel], 0); qopen[s] = D[s]['q'][np.argmax(v)]
        rep.setdefault('grasp', {})[s] = {'source': 'demo retargeting', 'window_s': list(win[s]), 'frames': int(sel.sum()), 'obj_in_wrist_cm': (G[s][:3, 3]*100).round(1).tolist(),
                                         'spread_mm_p90': round(float(np.percentile(np.linalg.norm(Gs[:, :3, 3]-G[s][:3, 3], axis=1), 90))*1e3, 1)}
    return G, qlat, qopen


# ------------------------------------------------------------------------------------------------------------------- 5 object references
class ObjRef:
    """one object's reference on the 100 Hz grid (see the module docstring, stage 5); build(prm) for a mid-air retarget prm =
    (dz, dfwd, tilt scale, spin about the own axis, dlat)"""
    def __init__(s, c, Tr, Tseat, depth, lift, seat, bottom):
        s.c, s.Tseat, s.depth, s.bottom = c, Tseat, depth, bottom; tg, nf = c.tg, c.nf; sg = c.smooth_s*FPS
        s.p = gaussian_filter1d(Tr[:, :3, 3], sg, axis=0, mode='nearest'); ax = gaussian_filter1d(Tr[:, :3, 2], sg, axis=0, mode='nearest')
        s.ax = ax/np.linalg.norm(ax, axis=1, keepdims=True)
        s.kl = int(np.searchsorted(tg, lift)); s.ks = min(nf-1, int(np.searchsorted(tg, seat)))
        # (11-11) extraction / insertion height along the hole axis: a minimum-jerk profile 0 -> depth + 3 cm (peak speed <= EXTRACT v_max),
        # starting lead_s before the demo's lift event and ending exactly at its seat event. The first version followed the demo's own height and
        # forced it to 0 up to the event frames; the events are the 15 mm points, so the reference jumped 6-18 mm in one frame at every lift / seat
        # (8-24 Hz wrist content 0.48 mm, 12x KA's).
        h = depth+.03; T = max(EXTRACT['t_min'], 1.875*h/EXTRACT['v_max']); t_on = tg[s.kl]-EXTRACT['lead_s']; t_off = tg[s.ks]
        s.hz = h*np.minimum(minjerk((tg-t_on)/T), minjerk((t_off-tg)/T)); s.k_on = int(np.searchsorted(tg, t_on)); s.k_off = s.ks
        clear = np.flatnonzero(s.hz >= depth+.01); s.kc1, s.kc2 = int(clear[0]), int(clear[-1])   # clear of the holder
        s.b = np.array([minjerk((tg[k]-tg[s.kc1])/c.blend_s)*minjerk((tg[s.kc2]-tg[k])/c.blend_s) for k in range(nf)])   # 0 at the holder, 1 mid-air

    def build(s, prm):
        dz, df, al, spin, dl = prm; c = s.c; nf = c.nf; ii = np.arange(nf); b = s.b
        hz = s.hz; inside = (ii < s.k_on) | (ii > s.k_off)
        zs = s.Tseat[:3, 2]; R0 = Rot.from_matrix(s.Tseat[:3, :3])
        Rm = Rot.from_rotvec(align_rotvec(np.broadcast_to(zs, s.ax.shape), s.ax)*(1-b*(1-al))[:, None])*R0*Rot.from_rotvec(np.outer(b*spin, [0, 0, 1.]))
        pm = s.p+b[:, None]*np.array([-df, -dl, dz]); ph = s.Tseat[:3, 3][None]+hz[:, None]*zs[None]
        Rb = R0*Rot.from_rotvec((R0.inv()*Rm).as_rotvec()*b[:, None])
        out = np.tile(np.eye(4), (nf, 1, 1)); out[:, :3, :3] = Rb.as_matrix(); out[:, :3, 3] = (1-b)[:, None]*ph+b[:, None]*pm; out[inside] = s.Tseat
        bot = out[:, :3, :3]@s.bottom+out[:, :3, 3]; need = np.zeros(nf)        # clearance over the holders / the table once clear of its own
        for (x0, x1, y0, y1, zt) in c.boxes:
            over = (bot[:, 0] > x0-.025) & (bot[:, 0] < x1+.025) & (bot[:, 1] > y0-.025) & (bot[:, 1] < y1+.025)
            need = np.maximum(need, np.where(over, zt+.015-bot[:, 2], 0.))
        need = np.maximum(need, c.z_table+.02-bot[:, 2]); need[(ii < s.kc1) | (ii > s.kc2)] = 0.; need = np.clip(need, 0., None)
        if need.any(): out[:, 2, 3] += gaussian_filter1d(maximum_filter1d(need, 41), 10)
        return out


def tip_in_tube(P, U, pt=PIP_TIP):
    """pipette point pt (pipette frame) in the tube frame, per frame"""
    tip = np.einsum('nij,j->ni', P[:, :3, :3], pt)+P[:, :3, 3]; return np.einsum('nji,nj->ni', U[:, :3, :3], tip-U[:, :3, 3])


def tip_window(c, P, U):
    """frames where the demo's pipette tip is in / at the tube (< 3 cm off its axis, below rim + 2 cm; gaps < INSERT merge_s closed, the longest
    run kept): mask, tip height in the tube frame (smoothed, clipped to the shaft-radius limit)"""
    rel = tip_in_tube(P, U); win = (np.linalg.norm(rel[:, :2], axis=1) < .03) & (rel[:, 2] < TUBE_TOP+.02)
    if not win.any(): return None
    dz = gaussian_filter1d(np.interp(np.arange(c.nf), np.flatnonzero(win), rel[win, 2]), 20); dz = np.clip(dz, TUBE_TOP-TIP_MAX_DEPTH, TUBE_TOP+.02)
    wb = binary_opening(binary_closing(win, np.ones(int(INSERT['merge_s']*FPS)+1)), np.ones(21))
    if not wb.any(): return None
    e = np.flatnonzero(np.diff(np.r_[0, wb.astype(int), 0])); runs = list(zip(e[::2], e[1::2])); a0, a1 = max(runs, key=lambda r: r[1]-r[0])
    wb = np.zeros(c.nf, bool); wb[a0:a1] = True
    return wb, dz


def tip_insert(c, P, U, tw, t_back=None):
    """(11-12) the pipette goes INTO the tube the way a peg goes into a hole. The old tip_fix blended the tip's position linearly onto the tube axis,
    so it came in sideways below the rim, through the wall (the demo's tracked tip is already ~7 cm deep when it gets within 3 cm of the axis), and
    in physics the tip stayed outside the tube against its wall. Now, in the tube frame (the tube moves with the left hand):
      approach  [w0-Td-Ta, w0-Td]  demo pose -> pre-insertion pose: tip on the axis INSERT pre above the rim, pipette axis on the tube axis (the
                                   demo's spin); the height blends twice as fast as the lateral position (up first, then over)
      descent   [w0-Td, w0]        straight down the axis to the demo depth (min-jerk, peak speed <= v_max)
      in tube   [w0, w1]           on the axis at the demo depth (smoothed, >= min_depth: dips in the demo do not leave the tube)
      extract   [w1, w1+Te]        straight up the axis to the pre-insertion point
      return    [w1+Te, t_back]    back to the demo pose, lateral first, the height after half the time; t_back = when the pipette's own path
                                   starts down into the stand (else Ta later: 0.7 s gave 0.69 m/s, 2x the demo's carry speed)
    Returns the new pipette poses and the frame indices / times."""
    wb, dz = tw; tg, nf = c.tg, c.nf; I = INSERT; idx = np.flatnonzero(wb); w0, w1 = int(idx[0]), int(idx[-1])
    dep = gaussian_filter1d(np.clip(TUBE_TOP-dz, I['min_depth'], TIP_MAX_DEPTH), 10)      # tip depth below the rim
    Td = max(I['t_min'], 1.875*(I['pre']+dep[w0])/I['v_max']); Te = max(I['t_min'], 1.875*(I['pre']+dep[w1])/I['v_max']); Ta = I['approach_s']
    td0 = tg[w0]-Td; ta0 = td0-Ta; te1 = tg[w1]+Te; Tr = max(Ta, (t_back-te1) if t_back else Ta); tr1 = te1+Tr; zp = TUBE_TOP+I['pre']
    tipd = tip_in_tube(P, U); RP = Rot.from_matrix(P[:, :3, :3]); Ra = Rot.from_rotvec(align_rotvec(P[:, :3, 2], U[:, :3, 2]))*RP
    out = P.copy()
    for k in range(nf):
        t = tg[k]
        if t < ta0 or t > tr1: continue
        if t < td0:
            s = (t-ta0)/Ta; wl, wz = minjerk(s), minjerk(2*s); tgt = np.r_[(1-wl)*tipd[k, :2], (1-wz)*tipd[k, 2]+wz*zp]; wr = wl
        elif t <= tg[w0]: tgt = np.r_[0., 0., zp-(I['pre']+dep[w0])*minjerk((t-td0)/Td)]; wr = 1.
        elif t <= tg[w1]: tgt = np.r_[0., 0., TUBE_TOP-dep[k]]; wr = 1.
        elif t <= te1: tgt = np.r_[0., 0., TUBE_TOP-dep[w1]+(I['pre']+dep[w1])*minjerk((t-tg[w1])/Te)]; wr = 1.
        else:
            s = (t-te1)/Tr; wl, wz = minjerk(s), minjerk(2*s-1); tgt = np.r_[wl*tipd[k, :2], (1-wz)*zp+wz*tipd[k, 2]]; wr = 1-wl
        R = (RP[k]*Rot.from_rotvec((RP[k].inv()*Ra[k]).as_rotvec()*wr)).as_matrix()
        out[k, :3, :3] = R; out[k, :3, 3] = U[k, :3, :3]@tgt+U[k, :3, 3]-R@PIP_TIP
    K = lambda tt: int(np.clip(np.searchsorted(tg, tt), 0, nf-1))
    return out, {'k': {'approach': K(ta0), 'descent': K(td0), 'in': w0, 'out': w1, 'extracted': K(te1), 'returned': K(tr1)},
                 't_s': {'approach': round(ta0, 2), 'descent': round(td0, 2), 'in': round(float(tg[w0]), 2), 'out': round(float(tg[w1]), 2),
                         'extracted': round(te1, 2), 'returned': round(tr1, 2)},
                 'depth_mm_in_out_max': [round(float(dep[w0])*1e3, 1), round(float(dep[w1])*1e3, 1), round(float(dep[w0:w1+1].max())*1e3, 1)]}


def retime_clock(tg, windows, factor, ramp, pad):
    """(11-12) new clock tn(tg): dtn/dt = f(t), f = factor inside each window +- pad, smooth (min-jerk) ramps of `ramp` outside it, 1 elsewhere"""
    f = np.ones(len(tg))
    for a_, b_ in windows:
        a_, b_ = a_-pad, b_+pad; f = np.maximum(f, 1+(factor-1)*minjerk((tg-(a_-ramp))/ramp)*minjerk(((b_+ramp)-tg)/ramp))
    return tg[0]+np.r_[0., np.cumsum(.5*(f[1:]+f[:-1])*np.diff(tg))]


def rs_lin(ts, tg, X):
    i = np.clip(np.searchsorted(tg, ts, side='right')-1, 0, len(tg)-2); w = np.clip((ts-tg[i])/(tg[i+1]-tg[i]), 0, 1).reshape(-1, *[1]*(X.ndim-1))
    return (1-w)*X[i]+w*X[i+1]


def rs_near(ts, tg, X): return X[np.clip(np.round(np.interp(ts, tg, np.arange(len(tg)))).astype(int), 0, len(tg)-1)]


def rs_T(ts, tg, T):
    """4x4 poses at times ts: position linear, rotation slerp between the neighbouring frames"""
    i = np.clip(np.searchsorted(tg, ts, side='right')-1, 0, len(tg)-2); w = np.clip((ts-tg[i])/(tg[i+1]-tg[i]), 0, 1)
    R0 = Rot.from_matrix(T[i, :3, :3]); R1 = Rot.from_matrix(T[i+1, :3, :3]); out = np.tile(np.eye(4), (len(ts), 1, 1))
    out[:, :3, :3] = (R0*Rot.from_rotvec((R0.inv()*R1).as_rotvec()*w[:, None])).as_matrix(); out[:, :3, 3] = (1-w[:, None])*T[i, :3, 3]+w[:, None]*T[i+1, :3, 3]
    return out


def pipette_lower_verts(m):
    """vertices of the pipette's lower visual meshes (tip cone, shaft) in the pipette body frame"""
    V = []
    for i in (1, 2, 3):
        g = m.geom(f'pipette_env_01_pipette_visual_{i}').id; mid = m.geom_dataid[g]; q = m.geom_quat[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
        V.append(v@Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix().T+m.geom_pos[g])
    return np.concatenate(V)


def tip_path_check(m, tg, P, U, press):
    """(11-12) the pipette vs the tube along the whole reference: frames where the lower pipette is INSIDE the tube wall (the wall collides with the
    pipette in physics; must be 0), the tip in the bore during the press, the smallest bore clearance of the inserted part"""
    V = pipette_lower_verts(m); wall = np.zeros(len(tg), bool); clear = np.full(len(tg), np.inf); bore = np.zeros(len(tg), bool)
    for k in range(len(tg)):
        v = (V@P[k, :3, :3].T+P[k, :3, 3]-U[k, :3, 3])@U[k, :3, :3]; inb = (v[:, 2] < TUBE_TOP) & (v[:, 2] > TUBE_BOTTOM); r = np.hypot(v[inb, 0], v[inb, 1])
        wall[k] = bool(((r > TUBE_RIN-3e-4) & (r < TUBE_ROUT+3e-4)).any())
        if len(r) and (r < TUBE_RIN).all(): clear[k] = TUBE_RIN-r.max()
    tip = tip_in_tube(P, U); bore = (np.hypot(tip[:, 0], tip[:, 1]) < TUBE_RIN-.0018) & (tip[:, 2] < TUBE_TOP-.005)
    e = np.flatnonzero(np.diff(np.r_[0, wall.astype(int), 0])); runs = [(round(float(tg[a]), 2), round(float(tg[b-1]), 2)) for a, b in zip(e[::2], e[1::2])]
    ins = np.isfinite(clear)
    out = {'frames_through_wall': int(wall.sum()), 'through_wall_s': runs[:6], 'tip_in_bore_press_frac': round(float(bore[press].mean()), 3) if press.any() else None,
           'min_bore_clearance_mm': round(float(clear[ins].min())*1e3, 2) if ins.any() else None}
    out['pass'] = out['frames_through_wall'] == 0 and (out['tip_in_bore_press_frac'] or 0) >= .999
    return out


# ------------------------------------------------------------------------------------------------------------------- 6 wrists, 7 arms
def wrist_path(c, side, O, G, lift, seat):
    """wrist reference and finger closure weight (0 open, 1 latched) of one hand"""
    tg, nf = c.tg, c.nf; Gi = inv(G); up = O[0][:3, 2]; reach = lift-.8; sg = 1. if side == 'right' else -1.
    kr = int(np.searchsorted(tg, reach)); Wg0 = O[kr]@Gi; nrm = -(Wg0[:3, :3]@np.array([0, sg, 0]))   # back along the palm normal (right palm +y)
    Wpre = T_(Wg0[:3, :3], Wg0[:3, 3]+.06*nrm+.02*up); Whov = T_(Wg0[:3, :3], Wg0[:3, 3]+.12*nrm+.06*up)
    kseat = min(nf-1, int(np.searchsorted(tg, seat+.5))); Wr0 = O[kseat]@Gi; nr_ = -(Wr0[:3, :3]@np.array([0, sg, 0]))
    Wret = T_(Wr0[:3, :3], Wr0[:3, 3]+.08*nr_+.04*O[-1][:3, 2])
    W = np.zeros((nf, 4, 4)); fw = np.zeros(nf); held = np.zeros(nf, bool)
    for k, tk in enumerate(tg):
        if tk < reach-1.4: W[k] = Whov
        elif tk < reach-.6: W[k] = blend_T(Whov, Wpre, minjerk((tk-(reach-1.4))/.8))
        elif tk < reach: W[k] = blend_T(Wpre, Wg0, minjerk((tk-(reach-.6))/.6))
        elif tk <= seat+.5: W[k] = O[k]@Gi; held[k] = True
        elif tk < seat+1.1: W[k] = blend_T(Wr0, Wret, minjerk((tk-(seat+.5))/.6))
        else: W[k] = Wret
        fw[k] = minjerk((tk-(lift-.8))/.6) if tk < seat+.1 else 1-minjerk((tk-(seat+.1))/.4)
    return W, fw, held


class Arms:
    """batched DLS IK of the two arms (common/arm_ik.py via vega_extras._IK) and its continuous-path use"""
    def __init__(s, m, d, rt, dev):
        import vega_extras as X
        s.m, s.d, s.dev = m, d, dev; s.ik = X._IK(m, dev)
        s.q_nom = {sd: torch.as_tensor(rt['qpos_ref'][:, s.ik.qadr[sd]].mean(0), dtype=torch.float32, device=dev) for sd in 'LR'}

    def seq(s, sd, T, q_init, every=1, iters=10, step_max=.08):
        """continuous IK along paths, batched over candidates: T (C, n, 4, 4), q_init (C, 7); each frame warm-started from the previous
        solution with the null space held at it (no branch switches). Returns q (C, n', 7), cost mm + deg (C, n')"""
        q = q_init.clone(); Q = []; cost = []; Tt = torch.as_tensor(T, dtype=torch.float32, device=s.dev)
        for j, k in enumerate(range(0, T.shape[1], every)):
            q, rp, rr = s.ik.solve(sd, Tt[:, k], q, q, iters=iters if j else 60, lam=.02, k_null=.05, step_max=step_max*every)
            Q.append(q); cost.append(rp*1e3+rr*57.3)
        return torch.stack(Q, 1), torch.stack(cost, 1)

    def starts(s, sd, T0, n=3):
        """up to n distinct start postures reaching T0 (within 1 mm+deg of the best), closest to the template posture, without arm self-collision"""
        m, d, ik = s.m, s.d, s.ik; lo, hi = ik.lo[sd], ik.hi[sd]; g0 = torch.Generator(device=s.dev).manual_seed(1)
        q0 = lo+(hi-lo)*torch.rand(64, 7, device=s.dev, generator=g0); q0[0] = s.q_nom[sd]
        qs0, rp0, rr0 = ik.solve(sd, torch.as_tensor(T0, dtype=torch.float32, device=s.dev)[None].expand(64, 4, 4), q0, q0, iters=200, lam=.03, k_null=0.)
        c0 = (rp0*1e3+rr0*57.3).cpu().numpy(); robot = {m.body(n_).id for n_ in ('base_link', 'lift_link', 'torso_flip_link')}
        arm = lambda n_: n_.startswith(('R_arm', 'L_arm', 'right_', 'left_')) and 'object' not in n_
        def self_hit(qv):
            d.qpos[:] = m.qpos0; d.qpos[ik.qadr[sd]] = qv; mujoco.mj_forward(m, d)
            for cn in d.contact[:d.ncon]:
                b1, b2 = int(m.geom_bodyid[cn.geom[0]]), int(m.geom_bodyid[cn.geom[1]]); n1, n2 = m.body(b1).name, m.body(b2).name
                if cn.dist < -.002 and ((arm(n1) and (b2 in robot or (arm(n2) and n1[0] != n2[0]))) or (arm(n2) and b1 in robot)): return True
            return False
        pick = []
        for i in np.argsort(((qs0-s.q_nom[sd])**2).sum(1).cpu().numpy()):
            if c0[i] > c0.min()+1. or self_hit(qs0[i].cpu().numpy()): continue
            if all(float((qs0[i]-qs0[j]).abs().max()) > .5 for j in pick): pick.append(i)
            if len(pick) == n: break
        return qs0[pick]


def search_midair(c, arms, objs, G, WB, Q0, rep):
    """mid-air retarget for arm reachability: per shift (dz, dfwd, alpha, dlat; shared by both objects, so the tip-in-tube relation is kept)
    and per object spin, the max over the path of the continuous-IK cost from the best of the start postures; feasible <= SEARCH['feasible'];
    choose the smallest change (|spins| deg + 1 deg per cm of shift + 30 x (1 - alpha))"""
    spins = np.radians(list(SEARCH['spins_deg'])); shifts = [(dz, df, al, dl) for dz in SEARCH['dz'] for df in SEARCH['dfwd'] for dl in SEARCH['dlat'] for al in SEARCH['alpha']]
    cost = {}
    for s_ in ('right', 'left'):
        sd = 'R' if s_ == 'right' else 'L'; held = WB[s_][2]; Gi = inv(G[s_]); Ts = []; meta = []
        for sh in shifts:
            U = objs['left'].build((sh[0], sh[1], sh[2], 0., sh[3])); tw = tip_window(c, objs['right'].build((sh[0], sh[1], sh[2], 0., sh[3])), U)
            for sp in spins:
                O = objs[s_].build((sh[0], sh[1], sh[2], sp, sh[3]))
                if s_ == 'right': O = tip_fix(O, U, tw)
                W = WB[s_][0].copy(); W[held] = np.einsum('nij,jk->nik', O[held], Gi); Ts.append(W); meta.append((sh, sp))
        Ts = np.array(Ts); S0 = Q0[s_]; C = len(Ts)
        _, cst = arms.seq(sd, np.repeat(Ts, len(S0), 0), S0.repeat(C, 1), every=SEARCH['every']); mx = cst.max(1).values.view(C, len(S0)).cpu().numpy()
        for i, (sh, sp) in enumerate(meta): cost[(s_, sh, float(sp))] = (float(mx[i].min()), int(mx[i].argmin()))
    def pick(s_, sh):
        cc = np.array([cost[(s_, sh, float(sp))][0] for sp in spins]); pen = np.where(cc <= SEARCH['feasible'], 0., 1000.+cc)+np.degrees(np.abs(spins)); i = int(np.argmin(pen))
        return pen[i], float(spins[i]), cc[i], cost[(s_, sh, float(spins[i]))][1]
    score = {sh: (lambda pr, pu: (pr[0]+pu[0]+100*(abs(sh[0])+abs(sh[1])+abs(sh[3]))+30*(1-sh[2]), pr, pu))(pick('right', sh), pick('left', sh)) for sh in shifts}
    kb = min(score, key=lambda k_: score[k_][0]); _, pr, pu = score[kb]
    rep['midair_search'] = {'method': 'continuous-path IK (warm-started frame to frame, 3 start postures, every 40 ms), max over the path of mm + deg',
                            'chosen': {'dz_m': kb[0], 'dfwd_m': kb[1], 'tilt_scale': kb[2], 'dlat_left_m': kb[3], 'spin_pipette_deg': round(math.degrees(pr[1])),
                                       'spin_tube_deg': round(math.degrees(pu[1])), 'path_cost_max_right': round(pr[2], 1), 'path_cost_max_left': round(pu[2], 1)},
                            'path_cost_max_per_spin_at_chosen_shift': {'spins_deg': np.degrees(spins).round().tolist(),
                                'right': [round(cost[('right', kb, float(sp))][0], 1) for sp in spins], 'left': [round(cost[('left', kb, float(sp))][0], 1) for sp in spins]},
                            'feasible_shifts': int(sum(1 for v in score.values() if v[0] < 1000))}
    if score[kb][0] >= 1000: raise SystemExit('no reachable mid-air retarget in the search grid: widen SEARCH or check --fwd / the grasps')
    return (kb[0], kb[1], kb[2], pr[1], pu[1], kb[3]), {'right': pr[3], 'left': pu[3]}


# ------------------------------------------------------------------------------------------------------------------- 8 hands
def fix_open_thumbs(c, m, d, Q, fing_w, qopen, qlat, names, LS, rep):
    """an open thumb that cuts its own object by > 2 mm (approach / release) -> the grid posture with >= 5 mm clearance nearest to it"""
    jq = lambda n: int(m.jnt_qposadr[m.joint(n).id]); tg, nf = c.tg, c.nf
    for s_, pre, objb in (('right', 'r_', ('right_object', 'pipette_plunger')), ('left', 'l_', ('left_object',))):
        th = [f'{pre}thumb_cmc_flex', f'{pre}thumb_cmc_abd', f'{pre}thumb_mcp', f'{pre}thumb_ip']; tq = [jq(n) for n in th]
        lo = np.array([m.jnt_range[m.joint(n).id][0] for n in th]); hi = np.array([m.jnt_range[m.joint(n).id][1] for n in th])
        tg_ = [g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name.startswith(side_name(s_)+'_thumb') and (m.geom_contype[g] or m.geom_conaffinity[g])]
        og = [g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name in objb and (m.geom_contype[g] or m.geom_conaffinity[g])]
        lift, seat = LS[s_]; openk = np.flatnonzero((fing_w[s_] < .999) & (tg > lift-2.5) & (tg < seat+1.2))[::3]
        q_l = qlat[s_][[names[s_].index(n) for n in th]]; q_o = qopen[s_][[names[s_].index(n) for n in th]]
        def clearance(qth):
            w_ = 1.
            for k in openk:
                d.qpos[:] = Q[k]; d.qpos[tq] = (1-fing_w[s_][k])*qth+fing_w[s_][k]*q_l; mujoco.mj_kinematics(m, d)
                for g1 in tg_:
                    for g2 in og: w_ = min(w_, mujoco.mj_geomDistance(m, d, g1, g2, .05, None))
            return w_
        c_o = clearance(q_o); rep.setdefault('thumb_open_clearance_mm', {})[s_] = {'template_open': round(c_o*1e3, 1)}
        if c_o >= -.002: continue
        best = None
        for f in np.linspace(0, 1, 5):
            for ab in np.linspace(0, 1, 5):
                for mi in np.linspace(0, 1, 3):
                    q_ = np.array([lo[0]+f*(hi[0]-lo[0]), lo[1]+ab*(hi[1]-lo[1]), lo[2]+mi*(hi[2]-lo[2]), q_o[3]]); c_ = clearance(q_)
                    key = (c_ < .005, float(np.abs(q_-q_o).sum()))
                    if best is None or key < best[0]: best = (key, q_, c_)
        if best[2] < .005: continue
        rep['thumb_open_clearance_mm'][s_].update(replaced_deg=np.round(np.degrees(best[1]), 1).tolist(), clearance=round(best[2]*1e3, 1))
        for k in range(nf):
            if fing_w[s_][k] < .999: Q[k, tq] = np.clip((1-fing_w[s_][k])*best[1]+fing_w[s_][k]*q_l, lo, hi)


def side_name(s): return 'right' if s == 'right' else 'left'


def thumb_on_cap(c, m, d, Q, lift, seat, rep):
    """right_thumb_tip onto the template's thumb-on-cap point (plunger frame) while the pipette is held (DLS on the 4 thumb joints)"""
    thumb = ['r_thumb_cmc_flex', 'r_thumb_cmc_abd', 'r_thumb_mcp', 'r_thumb_ip']
    ta = [int(m.jnt_qposadr[m.joint(n).id]) for n in thumb]; td = [int(m.jnt_dofadr[m.joint(n).id]) for n in thumb]
    lo = [m.jnt_range[m.joint(n).id][0] for n in thumb]; hi = [m.jnt_range[m.joint(n).id][1] for n in thumb]
    site = m.site('right_thumb_tip').id; plb = m.body('pipette_plunger').id; jac = np.zeros((3, m.nv)); res = []
    wth = gaussian_filter1d(((c.tg >= lift-.2) & (c.tg <= seat+.1)).astype(float), 10); q_prev = None
    for k in np.flatnonzero(wth > 1e-3):
        d.qpos[:] = Q[k]; q0 = Q[k, ta].copy(); qt = q0.copy() if q_prev is None else q_prev.copy()
        for _ in range(40):
            d.qpos[ta] = qt; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            e = d.xmat[plb].reshape(3, 3)@THUMB_ON_CAP+d.xpos[plb]-d.site_xpos[site]
            if np.linalg.norm(e) < 2e-4: break
            mujoco.mj_jacSite(m, d, jac, None, site); J = jac[:, td]; qt = np.clip(qt+np.clip(J.T@np.linalg.solve(J@J.T+1e-4*np.eye(3), e), -.1, .1), lo, hi)
        q_prev = qt; res.append(np.linalg.norm(e)); w = min(1., wth[k]); Q[k, ta] = (1-w)*q0+w*qt
    rep['thumb_on_cap_residual_mm_p50_max'] = [round(float(np.median(res))*1e3, 2), round(float(np.max(res))*1e3, 2)] if res else None


# ------------------------------------------------------------------------------------------------------------------- 9 outputs
def smoothness(tg, poses, Qarm):
    """(11-11) reference smoothness: per pose path the largest acceleration (a one-frame jump of 17 mm = 177 m/s^2) and the 8-24 Hz RMS of the position (what the
    eval's shake metric sees as wrist deviation when the robot cannot follow it); per arm the worst joint's 8-24 Hz RMS. Pass = SMOOTH_LIMITS."""
    from scipy.signal import butter, filtfilt
    bb = butter(2, [8/(FPS/2), 24/(FPS/2)], 'band'); out = {}
    for k, T in poses.items():
        p = T[:, :3, 3]*1e3; out[k] = {'max_accel_m_s2': round(float(np.linalg.norm(np.diff(p, 2, axis=0), axis=1).max())*1e-3*FPS**2, 2),
                                       'band_8_24Hz_mm': round(float(np.sqrt((np.linalg.norm(filtfilt(*bb, p, axis=0), axis=1)**2).mean())), 3)}
    for sd, q in Qarm.items(): out[f'arm_{sd}'] = {'band_8_24Hz_deg_worst_joint': round(float(np.sqrt((np.degrees(filtfilt(*bb, q, axis=0))**2).mean(0)).max()), 4)}
    L = SMOOTH_LIMITS
    out['pass'] = bool(all(v['max_accel_m_s2'] <= L['accel_m_s2'] and v['band_8_24Hz_mm'] <= L['band_mm'] for k, v in out.items() if k in poses) and
                       all(v['band_8_24Hz_deg_worst_joint'] <= L['band_deg'] for k, v in out.items() if k.startswith('arm_')))
    out['limits'] = L; return out


def penetrations(m, d, Q, tg, every=5, th=-.003):
    pen = {}
    for k in range(0, len(Q), every):
        d.qpos[:] = Q[k]; mujoco.mj_forward(m, d)
        for cn in d.contact[:d.ncon]:
            if cn.dist >= th: continue
            nm = [m.body(int(m.geom_bodyid[g])).name if m.geom_bodyid[g] else (m.geom(g).name or 'world')[:10] for g in cn.geom]
            pen.setdefault('|'.join(sorted(nm)), []).append((round(float(tg[k]), 2), round(float(cn.dist)*1e3, 1)))
    return {k: {'frames': len(v), 'worst_mm': min(x[1] for x in v), 'first_s': v[0][0], 'last_s': v[-1][0]} for k, v in sorted(pen.items(), key=lambda kv: min(x[1] for x in kv[1]))[:25]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('episode', type=Path); ap.add_argument('out', type=Path)
    ap.add_argument('--fwd', type=float, default=.30, help='demo origin (camera foot) in front of the robot base origin (m)')
    ap.add_argument('--lat', type=float, default=0., help='demo origin to the left of the robot base origin (m)')
    ap.add_argument('--base-x', type=float, default=.890829257623, help='robot base x (template distance to the table edge)'); ap.add_argument('--base-y', type=float, default=0.)
    ap.add_argument('--base-z', type=float, default=0., help='(11-12) robot base height above the floor (m): the robot vs the table height (template table top 0.72 m); '
                    '+ = the robot sits higher relative to the table. Holders, objects and the demo stay on the table; the IK re-solves the arms')
    ap.add_argument('--hole', default='r1c1', help='rack hole the tube is picked from and placed back into')
    ap.add_argument('--blend-s', type=float, default=1.5); ap.add_argument('--smooth-s', type=float, default=.08)
    ap.add_argument('--pre-s', type=float, default=2.5, help='reference starts this long before the pipette lift')
    ap.add_argument('--post-s', type=float, default=1.6, help='reference ends this long after the last seating')
    ap.add_argument('--midair', type=float, nargs=6, metavar=('DZ', 'DFWD', 'ALPHA', 'SPIN_P_DEG', 'SPIN_U_DEG', 'DLAT'), help='fixed mid-air retarget (default: searched)')
    ap.add_argument('--grasp', choices=['template', 'demo'], default='template', help='latched grasps: the template task\'s (default) or the retargeting\'s')
    ap.add_argument('--win-right', type=float, nargs=2, default=[7.0, 9.5]); ap.add_argument('--win-left', type=float, nargs=2, default=[10.5, 14.5])
    ap.add_argument('--force', action='store_true'); a = ap.parse_args()
    if a.out.exists() and not a.force: raise SystemExit(f'{a.out} exists')
    a.out.mkdir(parents=True, exist_ok=True); rep = {'episode': str(a.episode), 'args': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}}
    # 1 frames
    TF = table_frame(a.episode); D = load_demo(a.episode, TF); tpl = load_template(); t = D['t']
    rep['table'] = {k: TF[k] for k in ('cam_height', 'edge_yaw_deg', 'edge_inliers', 'edge_dist_m')}
    T_wt = T_(rotz(math.pi), [a.base_x-a.fwd, a.base_y-a.lat, tpl.z_table])     # table x -> world -x, table +y (left) -> world -y
    W_ = lambda X: np.einsum('ij,njk->nik', T_wt, X)
    Pd, Ud = W_(D['pipette']), W_(D['tube']); Wd = {s: W_(D[s]['W']) for s in ('right', 'left')}
    # 3 events, 2 holders, scene
    kp0, kp1 = events(Pd); ku0, ku1 = events(Ud)
    ev = {'pipette_lift': t[kp0], 'pipette_seat': t[kp1], 'tube_lift': t[ku0], 'tube_seat': t[ku1]}; rep['events_demo_s'] = {k: round(float(v), 2) for k, v in ev.items()}
    P_seat, U_seat, S_stand, S_rack = ground_truth_holders(Pd, Ud, (kp0, kp1, ku0, ku1), tpl, a.hole, rep)
    base_xy = (a.base_x, a.base_y)
    cnt = write_scene(TEMPLATE/'scene.xml', a.out/'scene.xml', base_xy, math.pi, {'stand': S_stand, 'rack': S_rack}, {'right_object': P_seat, 'left_object': U_seat}, base_z=a.base_z)
    m = mujoco.MjModel.from_xml_path(str(a.out/'scene.xml')); d = mujoco.MjData(m); rep['scene'] = {'moved_geoms': cnt, 'nq': m.nq, 'nu': m.nu}
    Vp = pipette_lower_verts(m); lo_ = Vp[:, 2] < Vp[:, 2].min()+1e-3; tip_mesh = np.r_[Vp[lo_, :2].mean(0), Vp[:, 2].min()]   # (11-12) PIP_TIP = the mesh's tip
    assert np.linalg.norm(tip_mesh-PIP_TIP) < 1e-3, ('PIP_TIP does not match the pipette mesh', tip_mesh, PIP_TIP); rep['pipette_tip_mesh_m'] = tip_mesh.round(4).tolist()
    t0 = ev['pipette_lift']-a.pre_s; t1 = max(ev['pipette_seat'], ev['tube_seat'])+a.post_s; tg = np.round(np.arange(t0, t1+1e-9, 1/FPS), 6)
    c = SimpleNamespace(tg=tg, nf=len(tg), blend_s=a.blend_s, smooth_s=a.smooth_s, z_table=tpl.z_table, boxes=holder_boxes(m, d)); nf = c.nf
    rep['holder_boxes_world'] = {hn: np.round(bx, 3).tolist() for hn, bx in zip(('stand', 'rack'), c.boxes)}
    # 4 grasps
    names = {s: D[s]['names'] for s in ('right', 'left')}
    G, qlat, qopen = template_grasps(tpl, names, rep) if a.grasp == 'template' else \
        demo_grasps(D, t, Wd, {'right': Pd, 'left': Ud}, {'right': a.win_right, 'left': a.win_left}, rep)
    # 5 objects
    objs = {'right': ObjRef(c, resample_poses(t, Pd, D['pipette_valid'], tg), P_seat, DEPTH['stand'], ev['pipette_lift'], ev['pipette_seat'], PIP_TIP),
            'left': ObjRef(c, resample_poses(t, Ud, D['tube_valid'], tg), U_seat, DEPTH['rack'], ev['tube_lift'], ev['tube_seat'], np.array([0, 0, TUBE_BOTTOM]))}
    rep['clear_of_holder_s'] = {n_: [round(float(tg[o.kc1]), 2), round(float(tg[o.kc2]), 2)] for n_, o in (('pipette', objs['right']), ('tube', objs['left']))}
    LS = {'right': (ev['pipette_lift'], ev['pipette_seat']), 'left': (ev['tube_lift'], ev['tube_seat'])}
    # 6 / 7 wrists and arms: the mid-air retarget, then the final object, wrist and arm paths
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'; arms = Arms(m, d, tpl.r, dev)
    WB = {s_: wrist_path(c, s_, objs[s_].build((0, 0, 1, 0, 0)), G[s_], *LS[s_]) for s_ in ('right', 'left')}   # held frames replaced per candidate
    Q0 = {s_: arms.starts('R' if s_ == 'right' else 'L', WB[s_][0][0]) for s_ in ('right', 'left')}
    if a.midair is None: prm, start_i = search_midair(c, arms, objs, G, WB, Q0, rep)
    else:
        dz, df, al, sp, su, dl = a.midair; prm = (dz, df, al, math.radians(sp), math.radians(su), dl); start_i = {'right': 0, 'left': 0}
        rep['midair_search'] = {'chosen': 'fixed', 'args': a.midair}
    U_ref = objs['left'].build((prm[0], prm[1], prm[2], prm[4], prm[5])); P_ref = objs['right'].build((prm[0], prm[1], prm[2], prm[3], prm[5]))
    tw = tip_window(c, P_ref, U_ref); tip_win = None; wb = np.zeros(nf, bool); ti = None
    if tw is not None:   # (11-12) peg-in-hole insertion (tip_insert) instead of the old sideways blend (tip_fix)
        P_ref, ti = tip_insert(c, P_ref, U_ref, tw, t_back=float(tg[objs['right'].kc2])); wb = tw[0]; idx = np.flatnonzero(wb); tip_win = (float(tg[idx[0]]), float(tg[idx[-1]]))
        rep['tip_in_tube'] = {'window_s': [round(tip_win[0], 2), round(tip_win[1], 2)], 'insert': ti}
    for nm, O in (('pipette', P_ref), ('tube', U_ref)):
        v = np.linalg.norm(np.gradient(O[:, :3, 3], 1/FPS, axis=0), axis=1); w_ = np.r_[(Rot.from_matrix(O[1:, :3, :3])*Rot.from_matrix(O[:-1, :3, :3]).inv()).magnitude()*FPS, 0.]
        rep.setdefault('object_speed', {})[nm] = {'lin_max_m_s': round(float(v.max()), 2), 'ang_max_deg_s': round(float(np.degrees(w_.max())), 0), 'ang_max_at_s': round(float(tg[int(np.argmax(w_))]), 2)}
    Wref = {}; fing_w = {}
    for s_, O in (('right', P_ref), ('left', U_ref)): Wref[s_], fing_w[s_], _ = wrist_path(c, s_, O, G[s_], *LS[s_])
    np.savez_compressed(a.out/'wrist_targets.npz', t=tg, right=Wref['right'], left=Wref['left'], pipette=P_ref, tube=U_ref)
    Qarm = {}; rep['ik'] = {}
    for sd, s in (('R', 'right'), ('L', 'left')):
        q, _ = arms.seq(sd, Wref[s][None], Q0[s][start_i[s]:start_i[s]+1], every=1, iters=12); Qarm[sd] = q[0].cpu().numpy()
        Tf = arms.ik.fk(sd, q[0]).cpu().numpy(); rp = np.linalg.norm(Wref[s][:, :3, 3]-Tf[:, :3, 3], axis=1)
        rr = (Rot.from_matrix(Wref[s][:, :3, :3])*Rot.from_matrix(Tf[:, :3, :3]).inv()).magnitude()
        qd = np.abs(np.gradient(gaussian_filter1d(Qarm[sd], 2, axis=0), 1/FPS, axis=0)); lo, hi = arms.ik.lo[sd].cpu().numpy(), arms.ik.hi[sd].cpu().numpy()
        rep['ik'][s] = {'pos_mm_p50_max': [round(float(np.median(rp))*1e3, 2), round(float(rp.max())*1e3, 2)], 'rot_deg_max': round(float(rr.max())*57.3, 2),
                        'frames_over_2mm_or_2deg': int(((rp > .002) | (rr > .035)).sum()), 'worst_times_s': [round(float(tg[i]), 2) for i in np.argsort(-rp)[:5]],
                        'max_step_deg': round(float(np.degrees(np.abs(np.diff(Qarm[sd], axis=0)).max())), 2),
                        'margin_to_limit_deg_min': round(float(np.degrees(np.minimum(Qarm[sd]-lo, hi-Qarm[sd]).min())), 1),
                        'max_joint_speed_rad_s': round(float(qd.max()), 2), 'frames_over_2rad_s': int((qd > 2.).any(1).sum()),
                        'fastest_times_s': [round(float(tg[i]), 2) for i in np.argsort(-qd.max(1))[:3]]}
    rep['smoothness'] = smoothness(tg, {'pipette': P_ref, 'tube': U_ref, 'wrist_right': Wref['right'], 'wrist_left': Wref['left']}, Qarm)
    # 8 hands, objects, plunger -> qpos
    jq = lambda n: int(m.jnt_qposadr[m.joint(n).id]); Q = np.tile(m.qpos0, (nf, 1)).astype(float)
    for sd in 'LR': Q[:, arms.ik.qadr[sd]] = Qarm[sd]
    for s in ('right', 'left'):
        w = fing_w[s][:, None]; qf = (1-w)*qopen[s][None]+w*qlat[s][None]
        for i, n in enumerate(names[s]): Q[:, jq(n)] = np.clip(qf[:, i], *m.jnt_range[m.joint(n).id])
    for n, Tr in (('right_object_joint', P_ref), ('left_object_joint', U_ref)):
        a_ = jq(n); Q[:, a_:a_+3] = Tr[:, :3, 3]; qq = Rot.from_matrix(Tr[:, :3, :3]).as_quat(scalar_first=True)
        for i in range(1, nf):
            if qq[i]@qq[i-1] < 0: qq[i] = -qq[i]
        Q[:, a_+3:a_+7] = qq
    eul = np.unwrap(Rot.from_matrix(P_ref[:, :3, :3]).as_euler('XYZ'), axis=0)   # plunger carrier = the pipette pose (intrinsic XYZ hinges)
    for i, n in enumerate(('pc_x', 'pc_y', 'pc_z')): Q[:, jq(n)] = P_ref[:, i, 3]
    for i, n in enumerate(('pc_rx', 'pc_ry', 'pc_rz')): Q[:, jq(n)] = eul[:, i]
    phase = np.zeros(nf, np.int8); plunger = np.zeros(nf); phase[int(np.searchsorted(tg, ev['pipette_lift'])):] = 1
    if tip_win:
        k0, k1 = int(np.searchsorted(tg, tip_win[0])), int(np.searchsorted(tg, tip_win[1])); phase[k0:k1+1] = 2; phase[k1+1:] = 3
        for k in range(k0, k1+1):
            plunger[k] = PLUNGER_MAX*min(1., .5-.5*math.cos(math.pi*min(1., (tg[k]-tg[k0])/.4)), .5-.5*math.cos(math.pi*min(1., (tg[k1]-tg[k])/.4)))
    Q[:, jq('pipette_plunger')] = -plunger
    fix_open_thumbs(c, m, d, Q, fing_w, qopen, qlat, names, LS, rep)
    thumb_on_cap(c, m, d, Q, ev['pipette_lift'], ev['pipette_seat'], rep)
    # 8b (11-12) slower insertions: resample everything onto a clock that runs RETIME factor slower through the three insertions (same paths)
    demo_t = tg.copy(); rt = RETIME
    if rt and rt.get('factor', 1.) != 1.:
        win_ = {'pipette_into_stand': (float(tg[objs['right'].kc2]), float(ev['pipette_seat'])), 'tube_into_rack': (float(tg[objs['left'].kc2]), float(ev['tube_seat']))}
        if ti: win_['tip_into_tube'] = (ti['t_s']['descent'], ti['t_s']['in'])
        tn = retime_clock(tg, list(win_.values()), rt['factor'], rt['ramp_s'], rt['pad_s']); M = lambda x: float(np.interp(x, tg, tn))
        tg2 = np.round(np.arange(tg[0], tn[-1]+1e-9, 1/FPS), 6); ts = np.interp(tg2, tn, tg)   # new 100 Hz grid; ts = the old (demo) time of each new frame
        Q = rs_lin(ts, tg, Q)
        for n_ in ('right_object_joint', 'left_object_joint'):
            a_ = int(m.jnt_qposadr[m.joint(n_).id]); Q[:, a_+3:a_+7] /= np.linalg.norm(Q[:, a_+3:a_+7], axis=1, keepdims=True)
        P_ref, U_ref = rs_T(ts, tg, P_ref), rs_T(ts, tg, U_ref); Wref = {s_: rs_T(ts, tg, W_) for s_, W_ in Wref.items()}
        fing_w = {s_: rs_lin(ts, tg, w_) for s_, w_ in fing_w.items()}; plunger = rs_lin(ts, tg, plunger); phase = rs_near(ts, tg, phase); wb = rs_near(ts, tg, wb)
        ev = {k_: M(v_) for k_, v_ in ev.items()}; tip_win = (M(tip_win[0]), M(tip_win[1])) if tip_win else None
        if ti:
            ti['t_s'] = {k_: round(M(v_), 2) for k_, v_ in ti['t_s'].items()}; ti['k'] = {k_: int(np.clip(np.searchsorted(tg2, v_-1e-6), 0, len(tg2)-1)) for k_, v_ in ti['t_s'].items()}
            rep['tip_in_tube']['window_s'] = [round(tip_win[0], 2), round(tip_win[1], 2)]
        rep['retime'] = {**rt, 'segments_demo_s': {k_: [round(x, 2) for x in v_] for k_, v_ in win_.items()}, 'segments_ref_s': {k_: [round(M(x), 2) for x in v_] for k_, v_ in win_.items()},
                         'duration_s': [round(float(tg[-1]-tg[0]), 2), round(float(tg2[-1]-tg2[0]), 2)]}
        rep['clear_of_holder_s'] = {k_: [round(M(x), 2) for x in v_] for k_, v_ in rep['clear_of_holder_s'].items()}
        tg, demo_t = tg2, ts; nf = len(tg); c.tg, c.nf = tg, nf
        Qarm = {sd: Q[:, arms.ik.qadr[sd]] for sd in 'LR'}
        rep['smoothness_before_retime'] = rep['smoothness']; rep['smoothness'] = smoothness(tg, {'pipette': P_ref, 'tube': U_ref, 'wrist_right': Wref['right'], 'wrist_left': Wref['left']}, Qarm)
        for nm, O in (('pipette', P_ref), ('tube', U_ref)):
            v = np.linalg.norm(np.gradient(O[:, :3, 3], 1/FPS, axis=0), axis=1); rep['object_speed'][nm]['lin_max_m_s_after_retime'] = round(float(v.max()), 2)
        for sd, s_ in (('R', 'right'), ('L', 'left')):
            qd = np.abs(np.gradient(gaussian_filter1d(Qarm[sd], 2, axis=0), 1/FPS, axis=0)); rep['ik'][s_]['max_joint_speed_rad_s_after_retime'] = round(float(qd.max()), 2)
        np.savez_compressed(a.out/'wrist_targets.npz', t=tg, right=Wref['right'], left=Wref['left'], pipette=P_ref, tube=U_ref, demo_time=demo_t)
    rep['events_ref_s'] = {k_: round(float(v_), 2) for k_, v_ in ev.items()}
    # 9 outputs
    import build_kin_task as BK
    acts = [m.actuator(i).name for i in range(m.nu)]; controls = Q[:, m.jnt_qposadr[m.actuator_trnid[:, 0]]].copy()
    ns = 4*(nf-1); ts = tg[0]+np.arange(ns)*SIM_DT; ctrl_sim = np.stack([np.interp(ts, tg, controls[:, i]) for i in range(m.nu)], 1)
    inr = lambda ta, tb: (tg >= ta) & (tg <= tb)
    gates = {'gate_pipette_held': inr(ev['pipette_lift']-.8, ev['pipette_seat']+.3), 'gate_tube_held': inr(ev['tube_lift']-.8, ev['tube_seat']+.3),
             'gate_pipette_in_stand': (tg <= ev['pipette_lift']) | (tg >= ev['pipette_seat']), 'gate_tube_in_rack': (tg <= ev['tube_lift']) | (tg >= ev['tube_seat']),
             'gate_press': phase == 2}
    # (11-12) the tip's own reference in the tube frame (tip_align reward / teacher observation in vega_extras; eval tip-in-tube check):
    # gate_tip_align = approach start .. extraction end; gate_tip_in_tube = the reference tip inside the bore
    tip_ref = tip_in_tube(P_ref, U_ref); ax_ref = np.einsum('nji,nj->ni', U_ref[:, :3, :3], P_ref[:, :3, 2]); ga = np.zeros(nf, bool)
    if ti: ga[ti['k']['approach']:ti['k']['extracted']+1] = True
    gates['gate_tip_align'] = ga; gates['gate_tip_in_tube'] = (np.hypot(tip_ref[:, 0], tip_ref[:, 1]) < TUBE_RIN) & (tip_ref[:, 2] < TUBE_TOP)
    rep['tip_path'] = tip_path_check(m, tg, P_ref, U_ref, phase == 2)
    gates['gate_pipette_carried'] = gates['gate_pipette_held'] & ~gates['gate_pipette_in_stand']
    for g_ in ('gate_grasp_shift_right', 'gate_grasp_shift_left', 'gate_press_follow_latch', 'gate_regrasp', 'gate_pipette_perturb', 'gate_tube_perturb',
               'gate_grasp_gate_right', 'gate_grasp_gate_left'): gates[g_] = np.zeros(nf, bool)   # env / eval compatibility (unused here)
    latch = np.zeros((nf, m.nu), bool)
    for i, n in enumerate(acts):
        if n.startswith(('r_', 'l_')): s = 'right' if n[0] == 'r' else 'left'; latch[:, i] = gates['gate_pipette_held' if s == 'right' else 'gate_tube_held'] & (fing_w[s] > .99)
    np.savez_compressed(a.out/'reference.npz', timestamps=tg, controls=controls, plunger=plunger.astype(np.float32), qpos_ref=Q.astype(np.float32),
                        qvel_ref=BK.qvel_from_qpos(m, Q, 1/FPS), ctrl_sim=ctrl_sim, ctrl_sim_verbatim=np.array(False), sim_times=ts, sim_dt=np.array(SIM_DT),
                        phase=phase, phase_names=np.array(['pre_lift', 'carry', 'press', 'post_press']), actuator_names=np.array(acts),
                        qpos_joint_names=np.array([m.joint(j).name for j in range(m.njnt)]), grasp_latch_mask=latch,
                        tip_in_tube_ref=tip_ref.astype(np.float32), pip_axis_in_tube_ref=ax_ref.astype(np.float32), pip_tip_local=PIP_TIP.astype(np.float32),
                        demo_time=demo_t, **gates)   # (11-12) demo_time: the demo's clock at each reference frame (differs after RETIME)
    rep['reference_penetrations_gt3mm'] = penetrations(m, d, Q, tg)
    tj = {'mode': 'TRACKER kinematic reference from a do-as-i-do RGB-D demo (robot square to the table, ground-truth holders)', 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
          'command': ' '.join(sys.argv), 'episode': str(a.episode), 'template': str(TEMPLATE), 'template_reference_sha': sha(TEMPLATE/'reference.npz'),
          'sim_dt': SIM_DT, 'frames_100hz': nf, 'ctrl_rows': ns, 'window_demo_s': [float(demo_t[0]), float(demo_t[-1])], 'window_ref_s': [float(tg[0]), float(tg[-1])], 'retime': rep.get('retime'),
          'phases': {'lift': float(ev['pipette_lift']), 'press_on': tip_win[0] if tip_win else None, 'press_off': tip_win[1] if tip_win else None},
          'ref_holes': [a.hole, a.hole], 'tube_slip_window_s': [float(ev['tube_lift']), float(ev['tube_seat'])], 'base_kind': 'kinematic', 'prior': 'none',
          'plunger_ref_max_mm': PLUNGER_MAX*1e3, 'robot_base': {'xy': list(base_xy), 'z': a.base_z, 'yaw_deg': 180.}, 'build': rep}
    (a.out/'task.json').write_text(json.dumps(tj, indent=1, default=float)+'\n'); (a.out/'build_report.json').write_text(json.dumps(rep, indent=1, default=float)+'\n')
    print(json.dumps({k: rep[k] for k in ('events_demo_s', 'midair_search', 'ik', 'reference_penetrations_gt3mm', 'tip_in_tube', 'tip_path') if k in rep}, default=float)[:4000])


if __name__ == '__main__':
    main()
