"""(11-11) EGO HOLDER ESTIMATE: the stand / rack poses and the tube position from ONE head-camera depth frame at the episode start, and the
error model of that estimate for the student's training (rl/vega_extras.py: extras.holder.meas_model).

Camera: the robot model's zed_left_camera (body head_l3; ZED X Mini, 50 mm baseline in the URDF) at the real robot's neck reading
NECK = (head_l1 about y, head_l2 about z, head_l3 about -y) = (0.878, 0.017, 0.386) rad (user, 11-10). Lens / mode are assumptions: 2.2 mm lens
(VFOV 80 deg), SVGA depth (960 x 600) -> CAM; change them for the real unit.

estimate(depth, K, T_wc, model) -> poses, deployable (numpy / scipy only):
  1 back-project the depth to the world with the camera pose from the robot's forward kinematics (its calibration error is not known here)
  2 table plane by RANSAC near the known table height; heights above the FITTED plane (absorbs camera pitch / roll / height error)
  3 per holder: 2 mm height map of the observed points (max height per cell) around its nominal place (training DR +- margin); the holder model
    is its top-down height map (ray cast of the scene's holder geoms). Pose = (dx, dy, dyaw about the holder centre: the env's
    parametrisation, vega_extras._H) by a coarse-to-fine search maximising agreement: model cells seen at the model height (+), seen clearly
    lower (the table through where the holder should be, -), a ring around the footprint seen tall (-); cells seen HIGHER than the model (the
    pipette / tube / a hand above the holder) are neutral
  4 tube: the rack hole (of the estimated rack) with the most points > 15 mm above the rack top; circle of the known outer radius fitted to the
    camera-facing half of the tube wall (xy; the far half shows the inner wall through the opening); rim = top of the points inside it (z)
  The pipette pose is the stand estimate applied to its seat (its slack in the stand is not observed).
characterize(task, n) -> <task>/ego/errors.npz + summary.json + overlays.png: n scenes with the training randomisation (holders +-5 cm / +-10 deg
  uniform, a random pick hole, object slack 2 mm / 3 deg), the arms at the reference's first frame, a ZED-like stereo noise model, and a per-scene
  camera pose error (head calibration: CAL_ERR). One row per scene = one JOINT sample of the stand, rack and tube errors (ERR_COLS).

  .venv-rl-head/bin/python -I perception/ego_holders.py tasks/task_d02_straight_v2 --n 400 --procs 16
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
from scipy import ndimage, optimize

ROOT = Path(__file__).resolve().parents[1]
NECK = (.878, .017, .386)
CAM = dict(W=960, H=600, fovy=80., pos=(.0365, .023, .0489))                     # zed_left_camera in head_l3 (URDF), looking along +x
NOISE = dict(baseline=.05, white_px=.08, corr_px=.12, corr_sigma_px=4., edge_bleed=.5, holes=.03, scale=.004, zmin=.1, zmax=3.)
CAL_ERR = dict(pos_mm=1.2, rot_deg=.2)                                              # per axis std of the head camera pose error
DR = dict(xy_m=.05, yaw_deg=10., obj_xy_m=.002, obj_yaw_deg=3.)                     # = the training env (holder, dr object_start_*)
HOLDERS = {'stand': ('furn_stand_',), 'rack': ('furn_rackv_', 'furn_rack_plate_')}  # = vega_extras.HOLDER geoms
SEARCH = dict(xy_m=.075, yaw_deg=13.)                                               # prior search range (DR + margin)
ERR_COLS = ['stand_dx', 'stand_dy', 'stand_dyaw', 'rack_dx', 'rack_dy', 'rack_dyaw', 'tube_dx', 'tube_dy', 'tube_dz']


def _rz(t): c, s = math.cos(t), math.sin(t); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
def _H(d, th, c):
    """the env's holder transform: rotate by th about the vertical through c, then translate by d (xy)"""
    T = np.eye(4); T[:3, :3] = _rz(th); T[:3, 3] = c-_rz(th)@c; T[:2, 3] += d; return T


# ------------------------------------------------------------------------------------------------------------------------- scene / camera
def build_model(task, neck=NECK, cam=CAM):
    import mujoco
    sp = mujoco.MjSpec.from_file(str(Path(task)/'scene.xml'))
    def q(axis, a): o = np.zeros(4); mujoco.mju_axisAngle2Quat(o, np.array(axis, float), a); return o
    for b, ax, a in (('head_l1', [0, 1, 0], neck[0]), ('head_l2', [0, 0, 1], neck[1]), ('head_l3', [0, -1, 0], neck[2])): sp.body(b).quat = q(ax, a)
    sp.body('head_l3').add_camera(name='ego', pos=list(cam['pos']), xyaxes=[0, -1, 0, 0, 0, 1], fovy=cam['fovy'])
    m = sp.compile(); m.vis.map.znear = .01/m.stat.extent
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, cam['W']); m.vis.global_.offheight = max(m.vis.global_.offheight, cam['H'])
    for i in range(m.ngeom):   # collision proxies on moving bodies overlap the visuals: out of the rendered groups (as the wrist cameras)
        if m.geom_group[i] == 3 and int(m.geom_bodyid[i]) != 0: m.geom_group[i] = 4
    return m


def intrinsics(cam=CAM):
    f = cam['H']/2/math.tan(math.radians(cam['fovy'])/2); return np.array([[f, 0, cam['W']/2], [0, f, cam['H']/2], [0, 0, 1.]])


def cam_pose(m, d):
    """T_wc with the camera frame x right, y DOWN, z forward (OpenCV), from MuJoCo's (x right, y up, looking along -z)"""
    cid = m.camera('ego').id; T = np.eye(4); T[:3, :3] = d.cam_xmat[cid].reshape(3, 3)@np.diag([1., -1., -1.]); T[:3, 3] = d.cam_xpos[cid]; return T


class Model:
    """Nominal geometry the estimator needs, read from the task scene: holder geoms / centres / top-down height maps, rack holes, table
    height, tube radius and rim. The same object serves the simulated scenes (geom placement)."""
    def __init__(s, task, m=None):
        import mujoco
        s.task = Path(task); s.m = m if m is not None else build_model(task); m = s.m; d = mujoco.MjData(m)
        r = np.load(s.task/'reference.npz'); s.q0 = r['qpos_ref'][0].astype(float); d.qpos[:] = s.q0; mujoco.mj_forward(m, d)
        s.z_table = float(max(d.geom_xpos[i, 2]+m.geom_size[i, 2] for i in range(m.ngeom) if (m.geom(i).name or '') == 'furn_table_0'))
        s.g, s.c, s.p0, s.q0g, s.maps = {}, {}, {}, {}, {}
        for nm, pre in HOLDERS.items():
            g = np.array([i for i in range(m.ngeom) if m.geom_bodyid[i] == 0 and (m.geom(i).name or '').startswith(pre)])
            c = d.geom_xpos[g].mean(0).copy(); c[2] = 0.; s.g[nm], s.c[nm] = g, c; s.p0[nm] = m.geom_pos[g].copy(); s.q0g[nm] = m.geom_quat[g].copy()
        cache = s.task/'ego'/'topdown.npz'
        if cache.exists():
            z = np.load(cache); s.maps = {nm: {k: z[f'{nm}_{k}'] for k in ('P', 'h', 'R')} for nm in HOLDERS}
        else:
            s.maps = {nm: s._topdown(m, d, s.g[nm], s.c[nm]) for nm in HOLDERS}; cache.parent.mkdir(exist_ok=True)
            np.savez_compressed(cache, **{f'{nm}_{k}': v for nm, mp_ in s.maps.items() for k, v in mp_.items()})
        import re
        acc = {}
        for i in range(m.ngeom):
            mm = re.match(r'furn_rackv_low_(r\d+c\d+)_s\d+$', m.geom(i).name or '')
            if mm: acc.setdefault(mm.group(1), []).append(d.geom_xpos[i, :2].copy())
        s.holes = {k: np.mean(v, 0) for k, v in sorted(acc.items())}
        tj = json.loads((s.task/'task.json').read_text()); s.ref_hole = (tj.get('ref_holes') or ['r0c3'])[0]
        s.rack_top = float(s.maps['rack']['h'].max())                       # above the table
        s.tube_r, s.tube_top = .0144, .0638                                # sim tube: outer radius, rim in the tube frame (TUBE_TOP)
        s.adr = {n: int(m.jnt_qposadr[m.joint(n).id]) for n in ('right_object_joint', 'left_object_joint', 'pc_x')}
        s.tube_rim_h = float(s.q0[s.adr['left_object_joint']+2]+s.tube_top-s.z_table)   # seated tube rim above the table
        pip = {m.body(b).id for b in ('right_object', 'pipette_plunger')}
        vis = [i for i in range(m.ngeom) if m.geom_group[i] in (1, 3) and (i in set(s.g['stand'].tolist()) or int(m.geom_bodyid[i]) in pip)]
        s.stand_pts = _surface_points(m, d, vis)                           # stand + seated pipette, world, nominal pose (3D refinement)

    def _topdown(s, m, d, g, c, res=.001):
        """height above the table of the holder's top surface on a 1 mm grid (ray cast from above, only this holder's geoms)"""
        import mujoco
        lo = d.geom_xpos[g, :2].min(0)-m.geom_rbound[g].max()-.01; hi = d.geom_xpos[g, :2].max(0)+m.geom_rbound[g].max()+.01
        xs = np.arange(lo[0], hi[0], res); ys = np.arange(lo[1], hi[1], res); X, Y = np.meshgrid(xs, ys, indexing='ij'); n = X.size
        grp = m.geom_group.copy(); m.geom_group[:] = 0; m.geom_group[g] = 5; gg = np.array([0, 0, 0, 0, 0, 1], np.uint8); gid = np.zeros(1, np.int32)
        dist = np.full(n, -1.); vec = np.array([0, 0, -1.])
        try:
            for i, (x, y) in enumerate(zip(X.ravel(), Y.ravel())): dist[i] = mujoco.mj_ray(m, d, np.array([x, y, s.z_table+.5]), vec, gg, 1, -1, gid)
        finally: m.geom_group[:] = grp
        h = np.where(dist >= 0, .5-dist, 0.).reshape(X.shape); h[h < .002] = 0.
        P = np.stack([X, Y], -1)[h > 0]-c[:2]; H = h[h > 0]
        mask = h > 0; ring = ndimage.binary_dilation(mask, iterations=20) & ~ndimage.binary_dilation(mask, iterations=4)
        R = np.stack([X, Y], -1)[ring]-c[:2]
        return {'P': P, 'h': H, 'R': R}

    def place(s, m, d, sample):
        """put the holders / objects of a sampled scene into the CPU model / data (geoms of body 0 and object free joints, carrier)"""
        import mujoco
        from scipy.spatial.transform import Rotation as Rot
        Hs = {}
        for nm in HOLDERS:
            dd, th = np.asarray(sample[nm][:2]), float(sample[nm][2]); Hs[nm] = _H(dd, th, s.c[nm]); g = s.g[nm]
            m.geom_pos[g] = (_rz(th)@(s.p0[nm]-s.c[nm]).T).T+s.c[nm]+np.r_[dd, 0.]
            qz = np.array([math.cos(th/2), 0, 0, math.sin(th/2)]); m.geom_quat[g] = np.array([_qmul(qz, q) for q in s.q0g[nm]])
        q = s.q0.copy()
        def pose(a): T = np.eye(4); T[:3, :3] = Rot.from_quat(q[a+3:a+7], scalar_first=True).as_matrix(); T[:3, 3] = q[a:a+3]; return T
        def put(a, T): q[a:a+3] = T[:3, 3]; q[a+3:a+7] = Rot.from_matrix(T[:3, :3]).as_quat(scalar_first=True)
        def slack(T, e): U = T.copy(); U[:3, :3] = _rz(e[2])@T[:3, :3]; U[:2, 3] += np.asarray(e[:2]); return U
        off = np.eye(4); off[:2, 3] = s.holes[sample['hole']]-s.holes[s.ref_hole]
        Tp = slack(Hs['stand']@pose(s.adr['right_object_joint']), sample['pip_slack']); put(s.adr['right_object_joint'], Tp)
        Tu = slack(Hs['rack']@off@pose(s.adr['left_object_joint']), sample['tube_slack']); put(s.adr['left_object_joint'], Tu)
        a = s.adr['pc_x']; q[a:a+3] = Tp[:3, 3]; q[a+3:a+6] = Rot.from_matrix(Tp[:3, :3]).as_euler('XYZ')
        d.qpos[:] = q; mujoco.mj_forward(m, d)
        return {'tube': Tu[:3, 3].copy(), 'pipette': Tp[:3, 3].copy()}


def _surface_points(m, d, geoms, n=40000, voxel=.002, seed=0):
    """area-uniform surface samples of mesh / box geoms (world frame at the current data), voxel-downsampled"""
    rng = np.random.default_rng(seed); tris = []
    for g in geoms:
        R = d.geom_xmat[g].reshape(3, 3); t = d.geom_xpos[g]
        if m.geom_type[g] == 7:
            mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
            f = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid]+m.mesh_facenum[mid]]; tris.append((v[f]@R.T)+t)
        elif m.geom_type[g] == 6:
            e = m.geom_size[g]; c8 = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])*e
            F = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
            tris.append((c8[np.array(F)]@R.T)+t)
    T = np.concatenate(tris); A = .5*np.linalg.norm(np.cross(T[:, 1]-T[:, 0], T[:, 2]-T[:, 0]), axis=1)
    k = rng.choice(len(T), n, p=A/A.sum()); u, v = rng.random(n), rng.random(n); fl = u+v > 1; u[fl], v[fl] = 1-u[fl], 1-v[fl]
    P = T[k, 0]+u[:, None]*(T[k, 1]-T[k, 0])+v[:, None]*(T[k, 2]-T[k, 0])
    _, i = np.unique(np.floor(P/voxel).astype(int), axis=0, return_index=True); return P[np.sort(i)]


def icp_stand(P, h, M, init, starts_deg=(-9, -6, -3, 0, 3, 6, 9), iters=20):
    """(x, y, yaw) of the stand + seated pipette by trimmed point-to-point ICP of the observed points around the stand (obs -> model
    nearest neighbours, 2D rigid update), from several yaw starts around the height-map estimate; the pipette's shape carries the yaw"""
    from scipy.spatial import cKDTree
    c = M.c['stand'][:2]; Q = M.stand_pts; tree = cKDTree(Q); d0 = np.asarray(init['d']); best = None
    sel = (np.linalg.norm(P[:, :2]-(c+d0), axis=1) < .07) & (h > .004) & (h < .30); X = P[sel]
    if len(X) < 200: return init
    for s0 in starts_deg:
        th, d = init['dth']+math.radians(s0), d0.copy()
        for it in range(iters):
            R = _rz(-th)[:2, :2]; Y = X.copy(); Y[:, :2] = (X[:, :2]-c-d)@R.T+c             # observed points in the nominal model frame
            dist, j = tree.query(Y); thr = max(.006, .02*(1-it/8)); k = dist < thr
            if k.sum() < 100: break
            A, B = Y[k, :2], Q[j[k], :2]; ma, mb = A.mean(0), B.mean(0); U, _, Vt = np.linalg.svd((A-ma).T@(B-mb))
            Rd = (U@Vt).T
            if np.linalg.det(Rd) < 0: Vt[-1] *= -1; Rd = (U@Vt).T
            td = mb-Rd@ma; dl = math.atan2(Rd[1, 0], Rd[0, 0]); th = th-dl
            d = d+_rz(th)[:2, :2]@(c-Rd@c-td)
            if abs(dl) < 1e-5 and np.linalg.norm(td) < 1e-5: break
        R = _rz(-th)[:2, :2]; Y = X.copy(); Y[:, :2] = (X[:, :2]-c-d)@R.T+c; cost = float(np.minimum(tree.query(Y)[0], .008).mean())
        if best is None or cost < best[0]: best = (cost, th, d)
    return {'dth': float(best[1]), 'd': best[2].tolist(), 'score': init['score'], 'icp_cost_mm': best[0]*1e3, 'hm': init}


def _qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])


def stereo_noise(z, K, rng, nz=NOISE):
    """ZED-like stereo depth from true z-depth (m): disparity noise (white + correlated), per-frame scale error, edge bleeding (foreground
    fattening at discontinuities), blob holes, range limits; 0 = invalid"""
    fB = K[0, 0]*nz['baseline']; bg = (z <= 0) | (z > nz['zmax']); zt = np.where(bg, nz['zmax'], z)
    n = nz['white_px']*rng.standard_normal(z.shape)
    c = ndimage.gaussian_filter(rng.standard_normal(z.shape), nz['corr_sigma_px']); n += nz['corr_px']*c/max(c.std(), 1e-9)
    zn = fB/np.maximum(fB/zt+n, 1e-3)*(1+nz['scale']*rng.standard_normal())
    jump = np.zeros(z.shape, bool); dv = np.abs(np.diff(zt, axis=0)) > .01; dh = np.abs(np.diff(zt, axis=1)) > .01
    jump[1:] |= dv; jump[:-1] |= dv; jump[:, 1:] |= dh; jump[:, :-1] |= dh
    fat = jump & (rng.random(z.shape) < nz['edge_bleed']); zn = np.where(fat, ndimage.minimum_filter(zt, 3), zn)
    hb = ndimage.gaussian_filter(rng.standard_normal(z.shape), 6.); holes = hb > np.quantile(hb, 1-nz['holes'])
    return np.where(bg | holes | (zn < nz['zmin']) | (zn > nz['zmax']), 0., zn)


# ------------------------------------------------------------------------------------------------------------------------- estimator
def backproject(z, K, T_wc, stride=1):
    v, u = np.nonzero(z[::stride, ::stride] > 0); v, u = v*stride, u*stride; zz = z[v, u]
    P = np.stack([(u+.5-K[0, 2])/K[0, 0]*zz, (v+.5-K[1, 2])/K[1, 1]*zz, zz], 1)
    return P@T_wc[:3, :3].T+T_wc[:3, 3]


def table_plane(P, z_table, rng, iters=300, tol=.003):
    S = P[np.abs(P[:, 2]-z_table) < .03]
    if len(S) < 500: return np.array([0., 0., z_table])
    best, bn = None, -1; A = np.c_[S[:, :2], np.ones(len(S))]
    for _ in range(iters):
        i = rng.choice(len(S), 3, replace=False)
        try: p = np.linalg.solve(A[i], S[i, 2])
        except np.linalg.LinAlgError: continue
        k = int((np.abs(A@p-S[:, 2]) < tol).sum())
        if k > bn: bn, best = k, p
    inl = np.abs(A@best-S[:, 2]) < tol; return np.linalg.lstsq(A[inl], S[inl, 2], rcond=None)[0]


def height_map(P, h, lo, hi, res=.002):
    n = np.ceil((hi-lo)/res).astype(int); ij = np.floor((P[:, :2]-lo)/res).astype(int); ok = (ij >= 0).all(1) & (ij < n).all(1)
    O = np.full(n, -1.); np.maximum.at(O, (ij[ok, 0], ij[ok, 1]), h[ok]); return O, O > -1


def _lookup(O, K_, lo, res, X, bilinear):
    f = (X-lo)/res-(.5 if bilinear else 0.)
    if not bilinear:
        ij = np.floor(f).astype(int); ok = (ij >= 0).all(-1) & (ij[..., 0] < O.shape[0]) & (ij[..., 1] < O.shape[1])
        i, j = np.where(ok, ij[..., 0], 0), np.where(ok, ij[..., 1], 0); return np.where(ok, O[i, j], 0.), ok & K_[i, j]
    i0 = np.floor(f).astype(int); w = f-i0; ok = (i0 >= 0).all(-1) & (i0[..., 0] < O.shape[0]-1) & (i0[..., 1] < O.shape[1]-1)
    i, j = np.where(ok, i0[..., 0], 0), np.where(ok, i0[..., 1], 0)
    kn = ok & K_[i, j] & K_[i+1, j] & K_[i, j+1] & K_[i+1, j+1]
    o = (O[i, j]*(1-w[..., 0])*(1-w[..., 1])+O[i+1, j]*w[..., 0]*(1-w[..., 1])+O[i, j+1]*(1-w[..., 0])*w[..., 1]+O[i+1, j+1]*w[..., 0]*w[..., 1])
    return o, kn


def _scores(O, K_, lo, res, mp, cand, c, sub, bilinear, sig=.006, chunk=128):
    P, h, R = mp['P'][::sub], mp['h'][::sub], mp['R'][::max(1, sub)]
    out = np.empty(len(cand))
    for a in range(0, len(cand), chunk):
        C = cand[a:a+chunk]; ct, st = np.cos(C[:, 0]), np.sin(C[:, 0])
        Rm = np.stack([np.stack([ct, -st], 1), np.stack([st, ct], 1)], 1)                          # (k, 2, 2)
        X = np.einsum('kij,pj->kpi', Rm, P)+c[:2]+C[:, None, 1:]; o, kn = _lookup(O, K_, lo, res, X, bilinear)
        e = o-h[None]; s = np.exp(-.5*(e/sig)**2)-.8*(e < -.012); s = np.where(kn & (e < .015), s, 0.)
        Y = np.einsum('kij,pj->kpi', Rm, R)+c[:2]+C[:, None, 1:]; o2, k2 = _lookup(O, K_, lo, res, Y, False)
        out[a:a+chunk] = (s.sum(1)-.5*(k2 & (o2 > .012)).sum(1)*len(P)/max(len(R), 1))/len(P)
    return out


def _grid(c0, dth, nth, dxy, nxy):
    th = c0[0]+dth*np.arange(-nth, nth+1); xy = dxy*np.arange(-nxy, nxy+1)
    T, X, Y = np.meshgrid(th, c0[1]+xy, c0[2]+xy, indexing='ij'); return np.stack([T.ravel(), X.ravel(), Y.ravel()], 1)


def fit_holder(P, h, mp, c, res=.002):
    lo = c[:2]-np.abs(mp['P']).max()-SEARCH['xy_m']-.03; hi = c[:2]+np.abs(mp['P']).max()+SEARCH['xy_m']+.03
    sel = (P[:, :2] > lo).all(1) & (P[:, :2] < hi).all(1); O, K_ = height_map(P[sel], h[sel], lo, hi, res)
    ym = math.radians(SEARCH['yaw_deg'])
    cand = _grid(np.zeros(3), math.radians(2.), int(round(ym/math.radians(2.))), .004, int(round(SEARCH['xy_m']/.004)))
    sc = _scores(O, K_, lo, res, mp, cand, c, 6, False); b = cand[np.argmax(sc)]
    cand = _grid(b, math.radians(.5), 5, .001, 6); sc = _scores(O, K_, lo, res, mp, cand, c, 2, True); b = cand[np.argmax(sc)]
    f = lambda x: -_scores(O, K_, lo, res, mp, x[None], c, 2, True)[0]
    simplex = np.stack([b, b+[math.radians(.3), 0, 0], b+[0, .0008, 0], b+[0, 0, .0008]])
    r = optimize.minimize(f, b, method='Nelder-Mead', options=dict(xatol=1e-4, fatol=1e-5, initial_simplex=simplex))
    x = r.x if -r.fun >= -f(b) else b
    return {'dth': float(x[0]), 'd': x[1:].tolist(), 'score': float(-f(x))}


def fit_tube(P, h, M, rack, cam_xy):
    """the rack hole with the most tall points within 18 mm of its axis, a circle of the tube's outer radius fitted to the camera-facing wall,
    then the hole NEAREST to the fitted centre (refit there if it differs: neighbouring holes are 36 mm apart and a tilted view spills a tube's
    wall into the neighbour's circle)"""
    T = _H(np.asarray(rack['d']), rack['dth'], M.c['rack']); W = {k: (T@np.r_[cxy, 0., 1.])[:2] for k, cxy in M.holes.items()}
    tall = h > M.rack_top+.015
    def fit(k):
        w = W[k]; sel = (np.linalg.norm(P[:, :2]-w, axis=1) < .025) & tall
        if sel.sum() < 30: return None
        Q, hq = P[sel], h[sel]; win = (hq > M.tube_rim_h-.025) & (hq < M.tube_rim_h+.012)   # rim region (a hand above the tube is excluded)
        top = float(np.percentile(hq[win], 97)) if win.sum() >= 10 else M.tube_rim_h; u = (cam_xy-w)/max(np.linalg.norm(cam_xy-w), 1e-9)
        wall = (hq < top-.006) & ((Q[:, :2]-w)@u > 0)                     # camera-facing half of the outer wall
        if wall.sum() < 15: wall = hq < top-.006
        res = optimize.least_squares(lambda x: np.linalg.norm(Q[wall, :2]-x, axis=1)-M.tube_r, w, loss='soft_l1', f_scale=.002)
        return {'n': int(sel.sum()), 'xy': res.x.tolist(), 'top_h': top}
    def wall_bins(w):   # tube evidence: 5 mm height bins between the rack top and the rim occupied at the wall radius (a finger fills a few)
        r_ = np.linalg.norm(P[:, :2]-w, axis=1); hh = h[(r_ > .010) & (r_ < .019) & tall & (h < M.tube_rim_h+.01)]
        return len(np.unique(np.floor(hh/.005))), len(hh)
    cnt = {k: wall_bins(w) for k, w in W.items()}; k = max(cnt, key=cnt.get)
    if cnt[k][0] < 4: return {'hole': None, 'n': cnt[k][1]}
    for _ in range(2):
        f = fit(k)
        if f is None: return {'hole': None, 'n': cnt[k][1]}
        k2 = min(W, key=lambda kk: np.linalg.norm(W[kk]-np.asarray(f['xy'])))
        if k2 == k: break
        k = k2
    return {'hole': k, **f}


def estimate(z, K, T_wc, M, rng=None, stride=1):
    """depth z (m, 0 invalid), intrinsics K (OpenCV), camera pose T_wc (OpenCV axes) in the scene's world frame -> holder poses (the env's
    (dx, dy, dyaw) offsets from the scene's nominal holders), tube position (world), pick hole, table plane"""
    rng = rng or np.random.default_rng(0); P = backproject(z, K, T_wc, stride)
    pl = table_plane(P, M.z_table, rng); h = P[:, 2]-(P[:, 0]*pl[0]+P[:, 1]*pl[1]+pl[2])
    out = {'table': pl.tolist()}
    for nm in HOLDERS: out[nm] = fit_holder(P, h, M.maps[nm], M.c[nm])
    out['stand'] = icp_stand(P, h, M, out['stand'])
    tb = fit_tube(P, h, M, out['rack'], T_wc[:2, 3]); out['tube_fit'] = tb
    if tb.get('hole'):
        xy = np.asarray(tb['xy']); out['tube'] = [float(xy[0]), float(xy[1]), float(xy@pl[:2]+pl[2]+tb['top_h']-M.tube_top)]
    return out


# ------------------------------------------------------------------------------------------------------------------------- error model
def sample_scene(M, rng):
    u = lambda a: a*(2*rng.random()-1)
    s = {nm: [u(DR['xy_m']), u(DR['xy_m']), math.radians(u(DR['yaw_deg']))] for nm in HOLDERS}
    s['hole'] = str(list(M.holes)[rng.integers(len(M.holes))])
    for k in ('pip_slack', 'tube_slack'): s[k] = [u(DR['obj_xy_m']), u(DR['obj_xy_m']), math.radians(u(DR['obj_yaw_deg']))]
    s['cal'] = [rng.normal(0, CAL_ERR['pos_mm']*1e-3, 3).tolist(), np.radians(rng.normal(0, CAL_ERR['rot_deg'], 3)).tolist()]
    return s


_W = {}
def _worker_init(task):
    import mujoco
    m = build_model(task); _W.update(m=m, d=mujoco.MjData(m), M=Model(task, m), r=mujoco.Renderer(m, CAM['H'], CAM['W']), K=intrinsics())
    o = mujoco.MjvOption(); o.geomgroup[:] = [1, 1, 1, 1, 0, 0]; _W['opt'] = o
    cid = m.camera('ego').id; _W['cam0'] = (m.cam_pos[cid].copy(), m.cam_quat[cid].copy())


def render_scene(sample, noisy=True, seed=0):
    import mujoco
    from scipy.spatial.transform import Rotation as Rot
    m, d, M, r, K = _W['m'], _W['d'], _W['M'], _W['r'], _W['K']; cid = m.camera('ego').id; p0, q0 = _W['cam0']
    truth = M.place(m, d, sample); T_nom = cam_pose(m, d)                          # the robot's belief of its camera pose
    dp, dr = np.asarray(sample['cal'][0]), np.asarray(sample['cal'][1]); R0 = Rot.from_quat(q0, scalar_first=True)
    m.cam_pos[cid] = p0+R0.apply(dp); m.cam_quat[cid] = (R0*Rot.from_rotvec(dr)).as_quat(scalar_first=True)   # the real camera pose
    mujoco.mj_forward(m, d); r.update_scene(d, camera='ego', scene_option=_W['opt']); r.enable_depth_rendering(); z = r.render().copy(); r.disable_depth_rendering()
    m.cam_pos[cid], m.cam_quat[cid] = p0, q0; mujoco.mj_forward(m, d)
    if noisy: z = stereo_noise(z, K, np.random.default_rng(seed))
    return z, T_nom, truth


def _one(args):
    i, seed = args; rng = np.random.default_rng(seed); M = _W['M']; s = sample_scene(M, rng); t0 = time.time()
    z, T, truth = render_scene(s, True, seed+1); est = estimate(z, _W['K'], T, M, np.random.default_rng(seed+2))
    e = [est['stand']['d'][0]-s['stand'][0], est['stand']['d'][1]-s['stand'][1], _wrap(est['stand']['dth']-s['stand'][2]),
         est['rack']['d'][0]-s['rack'][0], est['rack']['d'][1]-s['rack'][1], _wrap(est['rack']['dth']-s['rack'][2])]
    e += (np.asarray(est['tube'])-truth['tube']).tolist() if 'tube' in est else [np.nan]*3
    return dict(i=i, err=e, hole_ok=est['tube_fit'].get('hole') == s['hole'], scores=[est['stand']['score'], est['rack']['score']], s_sec=time.time()-t0,
                sample=s, est=est, truth={k: v.tolist() for k, v in truth.items()})


def _wrap(a): return float(math.atan2(math.sin(a), math.cos(a)))


def overlays(task, rows, path, k=4):
    """depth (turbo) with the estimated (green) and true (red) holder outlines and tube circles, for the first k scenes"""
    import mujoco
    from PIL import Image, ImageDraw
    _worker_init(task); M, K = _W['M'], _W['K']; tiles = []
    for row in rows[:k]:
        s = row['sample']; z, T, truth = render_scene(s, True, 7)
        Ti = np.linalg.inv(T); zz = np.clip((z-.4)/.8, 0, 1); im = (255*np.stack([zz, 1-np.abs(zz-.5)*2, 1-zz], -1)).astype(np.uint8); im[z <= 0] = 0
        img = Image.fromarray(im); dr = ImageDraw.Draw(img)
        def proj(Xw):
            Xc = Xw@Ti[:3, :3].T+Ti[:3, 3]; return np.stack([K[0, 0]*Xc[:, 0]/Xc[:, 2]+K[0, 2], K[1, 1]*Xc[:, 1]/Xc[:, 2]+K[1, 2]], 1)
        for nm in HOLDERS:
            P2 = M.maps[nm]['P'][::25]; H2 = M.maps[nm]['h'][::25]
            for (dd, th), col in (((np.asarray(s[nm][:2]), s[nm][2]), (255, 40, 40)), ((np.asarray(row['est'][nm]['d']), row['est'][nm]['dth']), (40, 255, 40))):
                X = (np.c_[P2, np.zeros(len(P2))]@_rz(th).T)+M.c[nm]+np.r_[dd, 0.]; X[:, 2] = M.z_table+H2
                for u, v in proj(X): dr.point((u, v), fill=col)
        for xyz, col in ((truth['tube'], (255, 40, 40)), (row['est'].get('tube'), (40, 255, 40))):
            if xyz is None: continue
            a = np.linspace(0, 2*np.pi, 40); X = np.c_[xyz[0]+M.tube_r*np.cos(a), xyz[1]+M.tube_r*np.sin(a), np.full(40, xyz[2]+M.tube_top)]
            dr.line([tuple(p) for p in proj(X)], fill=col, width=2)
        e = row['err']; dr.rectangle([0, 0, 700, 16], fill=(0, 0, 0))
        dr.text((4, 2), f"stand {e[0]*1e3:+.1f} {e[1]*1e3:+.1f} mm {math.degrees(e[2]):+.2f} deg | rack {e[3]*1e3:+.1f} {e[4]*1e3:+.1f} mm {math.degrees(e[5]):+.2f} deg | tube {e[6]*1e3:+.1f} {e[7]*1e3:+.1f} {e[8]*1e3:+.1f} mm  (red truth, green estimate)", fill=(255, 255, 255))
        tiles.append(np.asarray(img))
    Image.fromarray(np.concatenate(tiles, 0)).save(path)


def characterize(task, n, procs, seed=0):
    import multiprocessing as mp
    task = Path(task); out = task/'ego'; out.mkdir(exist_ok=True); t0 = time.time()
    with mp.get_context('spawn').Pool(procs, initializer=_worker_init, initargs=(str(task),)) as pool:
        rows = pool.map(_one, [(i, seed*100003+i) for i in range(n)], chunksize=2)
    E = np.array([r['err'] for r in rows], float); ok = np.array([r['hole_ok'] for r in rows]); good = ok & np.isfinite(E).all(1)
    np.savez_compressed(out/'errors.npz', errors=E[good].astype(np.float32), cols=np.array(ERR_COLS), hole_ok=ok, all_errors=E.astype(np.float32))
    st = {}
    for j, cn in enumerate(ERR_COLS):
        x = E[good, j]*(180/math.pi if 'yaw' in cn else 1e3)
        st[cn+(' deg' if 'yaw' in cn else ' mm')] = {'mean': round(float(x.mean()), 2), 'std': round(float(x.std()), 2), 'p95_abs': round(float(np.percentile(np.abs(x), 95)), 2), 'max_abs': round(float(np.abs(x).max()), 2)}
    summ = {'n': n, 'hole_id_correct': int(ok.sum()), 'used_rows': int(good.sum()), 'errors': st, 'neck': NECK, 'cam': CAM, 'noise': NOISE, 'cal_err': CAL_ERR, 'dr': DR,
            'sec_per_estimate_p50': round(float(np.median([r['s_sec'] for r in rows])), 2), 'wall_s': round(time.time()-t0, 1), 'cols': ERR_COLS,
            'worst': sorted([(r['i'], round(float(np.nanmax(np.abs(np.array(r['err'][:2]+r['err'][3:5]))))*1e3, 1)) for r in rows], key=lambda x: -x[1])[:5]}
    (out/'summary.json').write_text(json.dumps(summ, indent=1)+'\n'); json.dump(rows, open(out/'rows.json', 'w'))
    overlays(task, rows, out/'overlays.png'); return summ


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('task', type=Path); ap.add_argument('--n', type=int, default=400); ap.add_argument('--procs', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0); a = ap.parse_args()
    print(json.dumps(characterize(a.task, a.n, a.procs, a.seed), indent=1))
