"""(11-10) Can a palm-facing wrist stereo camera (256x192) see the held objects well enough for in-hand pose? Simulation, render only:
MuJoCo kinematics of recorded RL rollouts (rl/record_rollouts.py world_<i>.npz), no physics, nothing is trained.

A virtual camera is added to each hand's wrist body (MjSpec) at one or more candidate mounts (right-hand wrist frame; the left hand
is mirrored in y). Wrist frame: fingers along -z, thumb on +x, the right palm faces +y (left: -y). Intrinsics: 256x192, f = 156 px
(the D405's 640x480 fx 390 scaled to 256 px: HFOV 79 deg, VFOV 63 deg; the first 11-10 run used 135 px), baseline 18 mm, image
up = the palm normal. Per hand, per task phase (reference gates), frames before the world's first failure only:
  visibility  object pixels seen (segmentation), share of the object's own silhouette not hidden by the robot / furniture, robot
              pixels in the image, object depth, share of object pixels nearer than the stereo limits (FFS at 256x192 4.4 cm,
              D405 on-chip 7 cm);
  fit         one-frame pose fit of the object's visual mesh to simulated stereo depth, at the noise levels of NOISE (ideal / medium /
              high, each also without calibration errors: *_nocal; see NOISE for the model and its calibration on real D405 / FFS
              captures). Start = the true pose perturbed by --prior mm / deg (a rigid-grasp prior that has drifted).
              Segmentation: hand-aware masking -- pixels explained by a render of the robot at its measured joint angles (finger
              encoder noise 0.02 rad) are removed, then points within 20 mm of the prior object surface are kept.
              Fit: trimmed point-to-point ICP, then a render-and-compare search along the fitted long axis (axial_search; ICP alone
              barely moves the object along its axis, because only the end pixels carry that information -- wristcam_axial.py).
              Errors: object-origin position (mm), split along the true long axis (axial) and across it (lateral), long-axis angle
              (deg), pipette spin about its axis (deg; the tube is symmetric).

  MUJOCO_GL=egl python perception/wristcam_sim.py SCENE.xml OUT --run DIR LABEL [--run ...] [--worlds 64] [--fit-worlds 64]
      [--use forearm] [--mount NAME x y z tx ty tz ...] [--fit-stride 10] [--preview WORLD T1 T2 ...] [--procs 16]
Writes OUT/rows.json (per frame), OUT/summary.json (per run / mount / hand / phase), OUT/preview_*.png.
"""
import argparse, json, math, multiprocessing as mp, os
from pathlib import Path
import numpy as np

W, H, F = 256, 192, 156.0   # D405 640x480 intrinsics (fx 390, perception/ffs_compare.py) scaled to 256x192
FOVY = math.degrees(2*math.atan(H/2/F))
BASELINE = .018          # D405 baseline (m)
ZMIN_FFS, ZMIN_D405 = .044, .07
# reference gates (tasks/*/reference.npz): pipette held 3.83-25.36, in stand -5.19 / 23.13-; tube held 7.40-21.76, in rack -8.22 / 20.53-;
# pipette tip in tube 10.90-15.06 (press 11.63-14.29)
PHASES = {'right': [('approach', 0, 3.83), ('grasp / lift', 3.83, 5.2), ('carry', 5.2, 10.9), ('tip in tube / press', 10.9, 15.07),
                    ('carry back', 15.07, 23.13), ('insert into holder', 23.13, 25.37), ('released', 25.37, 99)],
          'left': [('approach', 0, 7.4), ('grasp / lift', 7.4, 8.23), ('carry', 8.23, 10.9), ('tip in tube / press', 10.9, 15.07),
                   ('carry back', 15.07, 20.53), ('insert into holder', 20.53, 21.77), ('released', 21.77, 99)]}
MOUNTS = {   # name: camera position, look-at target (right wrist frame, m)
    'heel': ((-.005, .035, .0), (-.01, .045, -.08)),       # on the hand base at the wrist, 1.8 cm off the palm, looking along the palm
    'forearm': ((-.005, .065, .06), (-.01, .045, -.08)),   # 6 cm back on a bracket, 4.8 cm off the palm, looking down at the grasp
}
OBJ = {'right': 'right_object', 'left': 'left_object'}
HOLD = {'stand': ('furn_stand_',), 'rack': ('furn_rackv_', 'furn_rack_plate_')}


def phase_of(side, t):
    return next(n for n, a, b in PHASES[side] if a <= t < b)


def build(scene, mounts):
    import mujoco
    sp = mujoco.MjSpec.from_file(str(scene))
    for name, (pos, tgt) in mounts.items():
        for side in ('right', 'left'):
            s = 1. if side == 'right' else -1.
            p = np.array(pos)*[1, s, 1]; f = np.array(tgt)*[1, s, 1]-p; f /= np.linalg.norm(f)
            zc = -f; xc = np.cross([0, s, 0], zc); xc /= np.linalg.norm(xc); yc = np.cross(zc, xc)
            q = np.zeros(4); mujoco.mju_mat2Quat(q, np.stack([xc, yc, zc], 1).ravel())
            sp.body(f'{side}_wrist').add_camera(name=f'{name}_{side}', pos=p, quat=q, fovy=FOVY)
    m = sp.compile()
    m.vis.map.znear = .001/m.stat.extent; m.vis.map.zfar = 5./m.stat.extent; m.vis.quality.offsamples = 0   # no MSAA: exact segmentation ids and depth
    # groups: 1 robot visuals, 2 object visuals (pipette, plunger, tube), 3 furniture; everything else (collision copies) hidden in 5
    objb = {m.body(n).id for n in ('right_object', 'left_object', 'pipette_plunger')}
    g = m.geom_group.copy(); new = np.full(m.ngeom, 5)
    for i in range(m.ngeom):
        b = int(m.geom_bodyid[i])
        if g[i] == 1: new[i] = 2 if b in objb else 1
        elif g[i] == 3 and b == 0: new[i] = 3
    m.geom_group[:] = new
    return m


def mesh_points(m, body_names, n=6000, seed=0):
    """Area-weighted surface samples of the bodies' visual meshes (group 2), in the first body's frame."""
    import mujoco
    rng = np.random.default_rng(seed); tri = []
    b0 = m.body(body_names[0]).id
    for i in range(m.ngeom):
        if m.geom_bodyid[i] != b0 or m.geom_group[i] != 2 or m.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH: continue
        k = m.geom_dataid[i]; v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k]+m.mesh_vertnum[k]]
        f = m.mesh_face[m.mesh_faceadr[k]:m.mesh_faceadr[k]+m.mesh_facenum[k]]
        R = np.zeros(9); mujoco.mju_quat2Mat(R, m.geom_quat[i]); v = v@R.reshape(3, 3).T+m.geom_pos[i]; tri.append(v[f])
    tri = np.concatenate(tri); a = .5*np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
    j = rng.choice(len(tri), n, p=a/a.sum()); u, w = rng.random((2, n)); s = u+w > 1; u[s], w[s] = 1-u[s], 1-w[s]
    t = tri[j]; return t[:, 0]+u[:, None]*(t[:, 1]-t[:, 0])+w[:, None]*(t[:, 2]-t[:, 0])


def kabsch(A, B):
    """R, t minimising |R A + t - B|."""
    ca, cb = A.mean(0), B.mean(0); U, _, Vt = np.linalg.svd((A-ca).T@(B-cb)); D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T@U.T))])
    R = Vt.T@D@U.T; return R, cb-R@ca


def icp(P, tree, M, R, t, iters=40, floor=.0015):
    """Points P (world) onto model M (object frame, KD-tree), from object pose R, t. Returns R, t, inlier count."""
    if len(P) > 1500: P = P[np.random.default_rng(1).choice(len(P), 1500, replace=False)]
    n_in = 0
    for _ in range(iters):
        dist, idx = tree.query((P-t)@R)
        thr = min(max(.01, 2*floor), max(2.5*np.median(dist), floor)); k = dist < thr; n_in = int(k.sum())
        if n_in < 20: break
        Ro, to = kabsch(M[idx[k]], P[k]); R, t = Ro, to
    return R, t, n_in


def rot(axis, ang):
    a = np.asarray(axis, float); a /= np.linalg.norm(a); K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3)+math.sin(ang)*K+(1-math.cos(ang))*K@K


def errors(side, R, t, Rg, tg):
    """Position error split along the true long axis (axial: seating depth) and across it (lateral: what an insertion needs), mm;
    long-axis angle, deg; pipette spin about the axis, deg (the tube is symmetric)."""
    e = 1e3*(t-tg); zg = Rg[:, 2]; axial = abs(e@zg); lat = float(np.linalg.norm(e-(e@zg)*zg))
    za = R[:, 2]; ax = math.degrees(math.acos(np.clip(za@zg, -1, 1)))
    spin = None
    if side == 'right':   # angle between the x axes after aligning the long axes
        Ra = rot(np.cross(za, zg), math.radians(ax)) if ax > 1e-6 else np.eye(3); xa = Ra@R[:, 0]
        spin = math.degrees(math.acos(np.clip(xa@Rg[:, 0], -1, 1)))
    return dict(pos=float(np.linalg.norm(e)), lat=lat, axial=float(axial), ang=ax, spin=spin)


_G = {}


def init(scene, mounts, prior):
    import mujoco
    from scipy.spatial import cKDTree
    m = build(scene, mounts); d = mujoco.MjData(m); r = mujoco.Renderer(m, H, W)
    HG = {k: np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] == 0 and (m.geom(g).name or '').startswith(v)]) for k, v in HOLD.items()}
    fj = [j for j in range(m.njnt) if m.body(int(m.jnt_bodyid[j])).name.startswith(('right_', 'left_')) and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    M = {s: mesh_points(m, [OBJ[s]], n=30000) for s in ('right', 'left')}   # ~0.8 mm spacing on the pipette, so point-to-point ICP is not quantised
    tgt = {s: {i for i in range(m.ngeom) if m.geom_group[i] == 2 and m.body(int(m.geom_bodyid[i])).name in
               ((OBJ[s], 'pipette_plunger') if s == 'right' else (OBJ[s],))} for s in ('right', 'left')}
    fitg = {s: {i for i in tgt[s] if m.body(int(m.geom_bodyid[i])).name == OBJ[s]} for s in tgt}
    _G.update(m=m, d=d, r=r, HG=HG, G0={k: (m.geom_pos[g].copy(), m.geom_quat[g].copy()) for k, g in HG.items()},
              HC={k: np.r_[m.geom_pos[g].mean(0)[:2], 0.] for k, g in HG.items()}, fq=np.array([m.jnt_qposadr[j] for j in fj]),
              M=M, T={s: cKDTree(M[s]) for s in M}, tgt=tgt, fitg=fitg, mounts=mounts, prior=prior,
              robot={i for i in range(m.ngeom) if m.geom_group[i] == 1})


def place_holders(spec):
    import mujoco
    m = _G['m']
    for k, g in _G['HG'].items():
        dx, dy, yaw = (spec or {}).get(k, [0, 0, 0]); th = math.radians(yaw); c, s_ = math.cos(th), math.sin(th)
        Rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]); p0, q0 = _G['G0'][k]
        m.geom_pos[g] = (p0-_G['HC'][k])@Rz.T+_G['HC'][k]+np.array([dx*1e-3, dy*1e-3, 0.])
        qz = np.array([math.cos(th/2), 0., 0., math.sin(th/2)]); out = np.zeros(4)
        for j, gi in enumerate(g): mujoco.mju_mulQuat(out, qz, q0[j]); m.geom_quat[gi] = out


def render(cam, groups, kind):
    import mujoco
    m, d, r = _G['m'], _G['d'], _G['r']; opt = mujoco.MjvOption(); opt.geomgroup[:] = [1 if i in groups else 0 for i in range(6)]
    r.update_scene(d, camera=cam, scene_option=opt)
    if kind == 'depth': r.enable_depth_rendering()
    elif kind == 'seg': r.enable_segmentation_rendering()
    out = r.render().copy(); r.disable_depth_rendering(); r.disable_segmentation_rendering(); return out


def obj_adr(side):
    m = _G['m']; return int(m.jnt_qposadr[int(m.body_jntadr[m.body(OBJ[side]).id])])


def axial_search(cam, side, R, t, zobs, valid, span=8., step=1., trunc=.01):
    """(11-10) Render-and-compare along the estimated long axis: the full scene (robot at the measured joints) is rendered with the
    object at t + s * axis for s in [-span, span] mm; cost = mean over valid pixels near the object of min(|z_obs - z_render|, 1 cm).
    ICP barely moves the object along its axis (only the end pixels carry that information); this uses the ends and the silhouette."""
    import mujoco
    from scipy.ndimage import binary_dilation
    m, d = _G['m'], _G['d']; a = obj_adr(side); q = d.qpos.copy(); qu = np.zeros(4); mujoco.mju_mat2Quat(qu, R.ravel()); ax = R[:, 2]
    def at(s):
        d.qpos[a:a+3] = t+s*1e-3*ax; d.qpos[a+3:a+7] = qu; mujoco.mj_kinematics(m, d); mujoco.mj_camlight(m, d)
    at(0.); reg = binary_dilation(segmask(render(cam, {1, 2, 3}, 'seg'), _G['fitg'][side]), iterations=12) & valid
    S = np.arange(-span, span+1e-9, step); C = []
    for s in S:
        at(s); zr = render(cam, {1, 2, 3}, 'depth'); C.append(np.mean(np.minimum(np.abs(zobs[reg]-zr[reg]), trunc)) if reg.any() else 0.)
    d.qpos[:] = q; mujoco.mj_kinematics(m, d); mujoco.mj_camlight(m, d)
    i = int(np.argmin(C)); s = S[i]
    if 0 < i < len(S)-1:   # parabolic refinement
        c0, c1, c2 = C[i-1], C[i], C[i+1]; den = c0-2*c1+c2
        if den > 0: s += step*.5*(c0-c2)/den
    return t+s*1e-3*ax


def segmask(seg, ids):
    import mujoco
    g = seg[..., 0]; return (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(g, list(ids))


def backproject(z, mask, cam):
    m, d = _G['m'], _G['d']; c = m.camera(cam).id; v, u = np.nonzero(mask); zz = z[v, u]
    pc = np.stack([(u+.5-W/2)*zz/F, -(v+.5-H/2)*zz/F, -zz], 1)
    return pc@d.cam_xmat[c].reshape(3, 3).T+d.cam_xpos[c]


# (11-10) stereo noise levels, in disparity px at 256x192 (f B = F * BASELINE = 2.81 px m). Calibrated on the 11-09 D405 snapshots
# (perception/snapshots, 640x480, 10-20 cm): on-chip vs FFS disagree by 0.2-0.67 px (640) and most of it is low-frequency (correlated),
# |dz| p50 0.6-1.1 mm / p90 3-7 mm off edges, p50 2-4 mm / p90 20-40 mm within 3 px of depth edges; on-chip valid 37-72 %, FFS 57-93 %.
#   white / corr   per-pixel and spatially correlated (Gaussian blob, sigma 3 px) disparity noise
#   edge_*         pixels within edge_px of a depth jump > 5 mm: edge_fat of them take the foreground depth (stereo fattening), of the
#                  rest edge_drop are invalid
#   holes          invalid blobs anywhere; obj_holes extra on the object's own surface (white / translucent plastic)
#   speckle        share of pixels with a uniform +-2 cm error;  scale  per-episode depth scale error (sigma)
#   extr           per-episode camera mount calibration error (mm, deg): the image comes from the true (perturbed) camera, the
#                  estimator back-projects with the nominal one
NOISE = {'ideal': dict(white=.2, corr=0., edge_px=1, edge_fat=0., edge_drop=1., holes=0., obj_holes=0., speckle=0., scale=0., extr=(0., 0.)),
         'medium': dict(white=.1, corr=.15, edge_px=1, edge_fat=.5, edge_drop=.5, holes=.10, obj_holes=.10, speckle=.01, scale=.005, extr=(1., .5)),
         'high': dict(white=.25, corr=.4, edge_px=2, edge_fat=.5, edge_drop=.6, holes=.35, obj_holes=.25, speckle=.03, scale=.01, extr=(2., 1.))}
for _l in ('medium', 'high'): NOISE[_l+'_nocal'] = dict(NOISE[_l], scale=0., extr=(0., 0.))   # stereo noise only, perfect calibration
FITS = (('ideal', 'mask'), ('medium_nocal', 'mask'), ('medium', 'mask'), ('high_nocal', 'mask'), ('high', 'mask'))


def sigma_z(z, nz):
    return math.hypot(nz['white'], nz['corr'])*z**2/(F*BASELINE)


def stereo(z, objmask, nz, rng, scale):
    """Simulated stereo depth from a true depth render (NaN = invalid)."""
    from scipy.ndimage import gaussian_filter, minimum_filter, binary_dilation
    fB = F*BASELINE; zt = np.clip(z, 1e-3, 2.); n = nz['white']*rng.normal(size=z.shape)
    if nz['corr']: g = gaussian_filter(rng.normal(size=z.shape), 3.); n += nz['corr']*g/g.std()
    zn = fB/np.maximum(fB/zt+n, 1e-3)*(1+scale)
    jump = np.zeros_like(z, bool); dz = np.abs(np.diff(z, axis=0)) > .005; jump[1:] |= dz; jump[:-1] |= dz
    dz = np.abs(np.diff(z, axis=1)) > .005; jump[:, 1:] |= dz; jump[:, :-1] |= dz
    band = binary_dilation(jump, iterations=nz['edge_px']-1) if nz['edge_px'] > 1 else jump
    fat = band & (rng.random(z.shape) < nz['edge_fat'])
    zn[fat] = (minimum_filter(zt, size=2*nz['edge_px']+1)*(1+scale))[fat]
    bad = band & ~fat & (rng.random(z.shape) < nz['edge_drop'])
    if nz['holes']: b = gaussian_filter(rng.normal(size=z.shape), 4.); bad |= b > np.quantile(b, 1-nz['holes'])
    if nz['obj_holes'] and objmask.sum() > 10:
        b = gaussian_filter(rng.normal(size=z.shape), 3.); bad |= objmask & (b > np.quantile(b[objmask], 1-nz['obj_holes']))
    if nz['speckle']: s = rng.random(z.shape) < nz['speckle']; zn[s] += rng.uniform(-.02, .02, int(s.sum()))
    bad |= (zn < ZMIN_FFS) | (z >= 2.); zn[bad] = np.nan; return zn


def observe(cam, side, lev, need_seg):
    """True depth (+ segmentation) from the camera displaced by this episode's mount calibration error of noise level lev."""
    import mujoco
    m, d = _G['m'], _G['d']; c = m.camera(cam).id; p0, q0 = m.cam_pos[c].copy(), m.cam_quat[c].copy(); dp, dq = _G['epi'][(lev, side)]['cam']
    m.cam_pos[c] = p0+dp; out = np.zeros(4); mujoco.mju_mulQuat(out, q0, dq); m.cam_quat[c] = out; mujoco.mj_camlight(m, d)
    z = render(cam, {1, 2, 3}, 'depth'); seg = render(cam, {1, 2, 3}, 'seg') if need_seg else None
    m.cam_pos[c] = p0; m.cam_quat[c] = q0; mujoco.mj_camlight(m, d); return z, seg


def frame(side, mount, fit, rng):
    """Visibility stats (and fits) of one hand's camera at the current state."""
    import mujoco
    m, d = _G['m'], _G['d']; cam = f'{mount}_{side}'
    z = render(cam, {1, 2, 3}, 'depth'); seg = render(cam, {1, 2, 3}, 'seg'); seg_o = render(cam, {2}, 'seg')
    vis = segmask(seg, _G['tgt'][side]); n_all = int(segmask(seg_o, _G['tgt'][side]).sum()); n_vis = int(vis.sum())
    zo = z[vis]; row = dict(n_vis=n_vis, n_all=n_all, robot_px=int(segmask(seg, _G['robot']).sum()),
                            z_med=float(np.median(zo)) if n_vis else None,
                            n_ffs=int((zo >= ZMIN_FFS).sum()), n_d405=int((zo >= ZMIN_D405).sum()))
    if not fit: return row
    ob = m.body(OBJ[side]).id; Rg, tg = d.xmat[ob].reshape(3, 3).copy(), d.xpos[ob].copy()
    dp, da = _G['prior']; u = rng.normal(size=3); a = rng.normal(size=3)
    t0 = tg+dp*1e-3*u/np.linalg.norm(u); R0 = rot(a, math.radians(da))@Rg
    # hand-aware masking needs the robot rendered at its MEASURED joint angles (finger encoder noise) from the nominal camera
    q = d.qpos.copy(); d.qpos[_G['fq']] += rng.normal(size=len(_G['fq']))*.02; mujoco.mj_kinematics(m, d); mujoco.mj_camlight(m, d)
    zh = render(cam, {1}, 'depth'); d.qpos[:] = q; mujoco.mj_kinematics(m, d); mujoco.mj_camlight(m, d)
    obs = {}; out = {}
    for lev, var in FITS:
        nz = NOISE[lev]
        if lev not in obs:
            zt, st = (z, seg) if tuple(nz['extr']) == (0., 0.) else observe(cam, side, lev, any(l == lev and v == 'gt' for l, v in FITS))
            obs[lev] = (stereo(zt, segmask(st if st is not None else seg, _G['fitg'][side]), nz, rng, _G['epi'][(lev, side)]['scale']), st)
        zn, st = obs[lev]; valid = np.isfinite(zn); zf = np.where(valid, zn, 0.)
        sz = sigma_z(row['z_med'] or .15, nz)
        if var == 'gt': mask = valid & segmask(st if st is not None else seg, _G['fitg'][side])
        else: mask = valid & ~(np.abs(zf-zh) < max(.004, 3*sz))
        P = backproject(zf, mask, cam)
        if var == 'mask' and len(P):
            dist, _ = _G['T'][side].query((P-t0)@R0); P = P[dist < .02]
        key = f'{lev}/{var}'
        if len(P) < 50: out[key] = None; continue
        R, t, n_in = icp(P, _G['T'][side], _G['M'][side], R0.copy(), t0.copy(), floor=max(.0015, 2*sz))
        if n_in < 20: out[key] = None; continue
        t2 = axial_search(cam, side, R, t, zf, valid, trunc=max(.01, 3*sz))
        out[key] = dict(icp=errors(side, R, t, Rg, tg), final=errors(side, R, t2, Rg, tg), n=len(P))
    row['fit'] = out; row['prior'] = errors(side, R0, t0, Rg, tg); return row


def work(job):
    import mujoco
    run, lab, w, stride, fstride, fit = job; m, d = _G['m'], _G['d']
    z = np.load(Path(run)/f'world_{w}.npz'); Q, T = z['qpos'], z['time']
    info = json.loads((Path(run)/'summary.json').read_text())['worlds'][w]; place_holders(info.get('spec'))
    tf = info['first_fail_s'] if info['first_fail_s'] >= 0 else 1e9; rows = []
    rng = np.random.default_rng(1000*w+7); _G['epi'] = {}
    for lev, nz in NOISE.items():   # per-episode draws: depth scale error, camera mount calibration error (per camera)
        for side in ('right', 'left'):
            u = rng.normal(size=3); a = rng.normal(size=3); dq = np.zeros(4)
            mujoco.mju_axisAngle2Quat(dq, a/np.linalg.norm(a), math.radians(nz['extr'][1]))
            _G['epi'][(lev, side)] = dict(scale=nz['scale']*rng.normal(), cam=(nz['extr'][0]*1e-3*u/np.linalg.norm(u), dq))
    for k in range(0, len(Q), stride):
        if T[k] >= tf: break
        d.qpos[:] = Q[k]; mujoco.mj_kinematics(m, d); mujoco.mj_camlight(m, d)
        for side in ('right', 'left'):
            for mount in _G['mounts']:
                r_ = frame(side, mount, fit and k % fstride == 0, rng)
                rows.append(dict(run=lab, world=w, t=float(T[k]), side=side, mount=mount, phase=phase_of(side, float(T[k])), **r_))
    return rows


def preview(run, w, times, out):
    import mujoco
    from PIL import Image, ImageDraw
    m, d, r = _G['m'], _G['d'], _G['r']
    z_ = np.load(Path(run)/f'world_{w}.npz'); Q, T = z_['qpos'], z_['time']
    place_holders(json.loads((Path(run)/'summary.json').read_text())['worlds'][w].get('spec'))
    tiles = []
    for t in times:
        k = int(np.argmin(np.abs(T-t))); d.qpos[:] = Q[k]; mujoco.mj_kinematics(m, d); mujoco.mj_camlight(m, d); row = []
        for side in ('right', 'left'):
            for mount in _G['mounts']:
                cam = f'{mount}_{side}'; opt = mujoco.MjvOption(); opt.geomgroup[:] = [0, 1, 1, 1, 0, 0]
                r.update_scene(d, camera=cam, scene_option=opt); rgb = r.render().copy()
                z = render(cam, {1, 2, 3}, 'depth'); vis = segmask(render(cam, {1, 2, 3}, 'seg'), _G['tgt'][side])
                zc = np.clip((z-.02)/(.25-.02), 0, 1); dep = (255*(1-zc)).astype(np.uint8); dep = np.stack([dep]*3, -1)
                dep[vis] = (.5*dep[vis]+.5*np.array([60, 200, 60])).astype(np.uint8)
                dep[vis & (z < ZMIN_FFS)] = [230, 40, 40]; dep[vis & (z >= ZMIN_FFS) & (z < ZMIN_D405)] = [240, 170, 30]
                im = Image.fromarray(np.concatenate([rgb, dep], 1)); dr = ImageDraw.Draw(im)
                dr.text((4, 2), f'{mount} {side}  t={T[k]:.1f}s  {phase_of(side, float(T[k]))}', fill=(255, 255, 0))
                dr.text((W+4, 2), f'object px {int(vis.sum())}', fill=(255, 255, 0)); row.append(np.asarray(im))
        tiles.append(np.concatenate(row, 1))
    Image.fromarray(np.concatenate(tiles, 0)).save(out)


def agg(rows):
    """Per run / mount / side / phase."""
    out = {}
    keys = sorted({(r['run'], r['mount'], r['side']) for r in rows})
    for run, mount, side in keys:
        for ph, _, _ in PHASES[side]:
            R = [r for r in rows if r['run'] == run and r['mount'] == mount and r['side'] == side and r['phase'] == ph]
            if not R: continue
            inv = [r for r in R if r['n_all'] >= 50]; vis = [r for r in R if r['n_vis'] > 0]
            e = dict(frames=len(R), in_view=len(inv)/len(R),
                     vis_px_p50=float(np.median([r['n_vis'] for r in R])),
                     unoccluded_share_p50=float(np.median([r['n_vis']/r['n_all'] for r in inv])) if inv else None,
                     robot_px_share_p50=float(np.median([r['robot_px'] for r in R]))/(W*H),
                     z_med_cm_p50=100*float(np.median([r['z_med'] for r in vis])) if vis else None,
                     near_ffs_share=float(np.mean([1-r['n_ffs']/r['n_vis'] for r in vis])) if vis else None,
                     near_d405_share=float(np.mean([1-r['n_d405']/r['n_vis'] for r in vis])) if vis else None,
                     usable_ffs=float(np.mean([r['n_ffs'] >= 300 for r in R])), usable_d405=float(np.mean([r['n_d405'] >= 300 for r in R])))
            Fr = [r for r in R if 'fit' in r]
            if Fr: e['prior'] = {k: float(np.median([r['prior'][k] for r in Fr])) for k in ('pos', 'lat', 'axial', 'ang')}
            for lev, var in FITS:
                key = f'{lev}/{var}'; ok = [r['fit'][key] for r in Fr if r['fit'].get(key) is not None]
                if not Fr: continue
                f = dict(n=len(Fr), avail=len(ok)/len(Fr))
                if ok:
                    fin = [o['final'] for o in ok]
                    for k in ('pos', 'lat', 'axial', 'ang') + (('spin',) if side == 'right' else ()):
                        x = np.array([o[k] for o in fin]); f[f'{k}_p50'] = float(np.median(x)); f[f'{k}_p90'] = float(np.percentile(x, 90))
                    f['axial_icp_only_p50'] = float(np.median([o['icp']['axial'] for o in ok]))
                    f['pos_3mm_ang_2deg'] = float(np.mean([(o['pos'] <= 3) & (o['ang'] <= 2) for o in fin]))
                    f['pts_p50'] = float(np.median([o['n'] for o in ok]))
                e['fit_'+key] = f
            out[f'{run} | {mount} | {side} | {ph}'] = e
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('scene', type=Path); p.add_argument('out', type=Path)
    p.add_argument('--run', nargs=2, action='append', default=[], metavar=('DIR', 'LABEL')); p.add_argument('--worlds', type=int, default=64)
    p.add_argument('--fit-worlds', type=int, default=64); p.add_argument('--stride', type=int, default=5, help='control steps (50 Hz) per stats frame')
    p.add_argument('--fit-stride', type=int, default=10); p.add_argument('--prior', type=float, nargs=2, default=[5., 3.], metavar=('MM', 'DEG'))
    p.add_argument('--mount', nargs=7, action='append', metavar=('NAME', 'x', 'y', 'z', 'tx', 'ty', 'tz'))
    p.add_argument('--preview', type=float, nargs='+', metavar=('WORLD', 'T')); p.add_argument('--procs', type=int, default=16)
    p.add_argument('--use', nargs='+', help='only these of the built-in mounts (e.g. forearm)')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    mounts = {x[0]: (tuple(map(float, x[1:4])), tuple(map(float, x[4:7]))) for x in a.mount} if a.mount else MOUNTS
    if a.use: mounts = {k: v for k, v in mounts.items() if k in a.use}
    if a.preview:
        init(a.scene, mounts, a.prior); w = int(a.preview[0])
        for d_, lab in a.run: preview(d_, w, a.preview[1:], a.out/f'preview_{lab}_world{w}.png')
        raise SystemExit
    jobs = [(d_, lab, w, a.stride, a.fit_stride, w < a.fit_worlds) for d_, lab in a.run for w in range(a.worlds)]
    with mp.get_context('spawn').Pool(a.procs, initializer=init, initargs=(a.scene, mounts, a.prior)) as pool:
        rows = []
        for i, R in enumerate(pool.imap_unordered(work, jobs)):
            rows += R; print(f'{i+1}/{len(jobs)} worlds done', flush=True)
    (a.out/'rows.json').write_text(json.dumps(rows))
    S = dict(camera=dict(px=[W, H], f_px=F, fovy_deg=FOVY, mounts_right_wrist_frame=mounts, stereo=dict(baseline_m=BASELINE, noise_levels=NOISE,
             zmin_ffs_m=ZMIN_FFS, zmin_d405_m=ZMIN_D405), prior_mm_deg=a.prior), runs=[x for x in a.run], phases=PHASES, table=agg(rows))
    (a.out/'summary.json').write_text(json.dumps(S, indent=1)); print('rows', len(rows), '->', a.out)
