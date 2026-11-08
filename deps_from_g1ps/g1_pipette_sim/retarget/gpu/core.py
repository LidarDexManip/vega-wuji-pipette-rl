"""Torch kinematics of the floating Wuji hand and MuJoCo Warp collision queries (GPU solver building blocks).

Conventions follow topo.py: a configuration is (wrist position, wrist rotation vector, 20 finger angles); the
solver's variables are the CPU's own source-wrist coordinates x, world wrist W = anchor @ T(x).
"""
import numpy as np
import torch

DT = torch.float64


def skew(v):
    z = torch.zeros_like(v[..., 0])
    return torch.stack([torch.stack([z, -v[..., 2], v[..., 1]], -1), torch.stack([v[..., 2], z, -v[..., 0]], -1),
                        torch.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def rotvec_quat(r):
    """Unit quaternion (w, x, y, z) of a rotation vector; smooth at 0 (forward-mode safe)."""
    t2 = (r*r).sum(-1, keepdim=True); small = t2 < 1e-10
    t = torch.sqrt(torch.where(small, torch.ones_like(t2), t2))
    s = torch.where(small, .5-t2/48., torch.sin(t/2)/t); c = torch.where(small, 1.-t2/8., torch.cos(t/2))
    return torch.cat([c, s*r], -1)


def qmul(a, b):
    aw, ax, ay, az = a.unbind(-1); bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw], -1)


def qconj(q):
    return q*torch.tensor([1., -1., -1., -1.], dtype=q.dtype, device=q.device)


def qmat(q):
    q = q/torch.linalg.norm(q, dim=-1, keepdim=True); w, x, y, z = q.unbind(-1)
    return torch.stack([torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], -1), torch.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], -1),
                        torch.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1)], -2)


def mat_quat_np(R):
    from scipy.spatial.transform import Rotation
    return np.roll(Rotation.from_matrix(R).as_quat(), 1, axis=-1)


def axis_angle(axis, q):
    """Rotation matrices about unit axes (..., 3) by angles q (...)."""
    K = skew(axis); s, c = torch.sin(q)[..., None, None], torch.cos(q)[..., None, None]
    return torch.eye(3, dtype=q.dtype, device=q.device)+s*K+(1-c)*(K@K)


class HandFK:
    """Forward kinematics of the hand below the floating wrist: body rotations and positions (world)."""
    def __init__(self, z, device):
        f = lambda k, dt=DT: torch.as_tensor(np.asarray(z[k]), dtype=dt, device=device)
        self.parent = [int(p) for p in z['body_parent']]; self.jidx = [int(j) for j in z['body_jidx']]
        self.pos = f('body_pos'); self.rot = torch.as_tensor(np.stack([qmat_np(q) for q in z['body_quat']]), dtype=DT, device=device)
        self.axis = f('body_axis'); self.nb = len(self.parent)
        self.site_body = torch.as_tensor(z['site_body'], device=device); self.site_pos = f('site_pos')
        self.coll_body = torch.as_tensor(z['coll_body'], device=device); cl = f('coll_local'); self.coll_R, self.coll_t = cl[:, :3, :3], cl[:, :3, 3]
        self.limits = f('limits')

    def __call__(self, Rw, pw, th):
        """Rw (B,3,3), pw (B,3), th (B,20) -> R (B,nb,3,3), p (B,nb,3)."""
        Rs, ps = [None]*self.nb, [None]*self.nb
        for b in range(self.nb):
            if self.parent[b] < 0:
                R, p = Rw@self.rot[b], pw+(Rw@self.pos[b])
            else:
                Rp, pp = Rs[self.parent[b]], ps[self.parent[b]]
                R, p = Rp@self.rot[b], pp+(Rp@self.pos[b])
            if self.jidx[b] >= 0:
                R = R@axis_angle(self.axis[b].expand(th.shape[0], 3), th[:, self.jidx[b]])
            Rs[b], ps[b] = R, p
        return torch.stack(Rs, 1), torch.stack(ps, 1)

    def sites(self, R, p):
        return p[:, self.site_body]+(R[:, self.site_body]@self.site_pos[..., None])[..., 0]

    def colliders(self, R, p):
        Rb, pb = R[:, self.coll_body], p[:, self.coll_body]
        return Rb@self.coll_R, pb+(Rb@self.coll_t[..., None])[..., 0]


def qmat_np(q):
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(np.roll(np.asarray(q, float), -1)).as_matrix()


def world_wrist(x, qA, A):
    """World wrist (rotation, position, quaternion) of solver coordinates x (B,26) with anchors A (B,4,4), quats qA (B,4)."""
    q = qmul(qA, rotvec_quat(x[:, 3:6])); R = qmat(q)
    p = A[:, :3, 3]+(A[:, :3, :3]@x[:, :3, None])[..., 0]
    return R, p, q


def patch_mjw_broadphase_cache():
    """Host-memory leak fix (h13, 10-29): mujoco_warp 3.4 collision_driver.nxn_broadphase / sap_broadphase call
    _broadphase_filter(m), which builds a NEW @wp.func on every call; the broadphase kernel is cached (warp_util.cache_kernel)
    by hash() of that func object, so every mjw.collision() call adds one more kernel + its 'unique' module (and the loaded
    CUDA module) to warp_util._KERNEL_CACHE, never freed: ~172 kB host RSS per collision call (gsearch.Evaluator._eval does two
    -> ~344 kB per evaluate; the h13 schedule grew to 22.8 GB and was killed). The func depends on m only through wp.static of
    geom_aabb / geom_rbound / geom_margin's leading (batch) dims and opt.broadphase_filter: memoise it on exactly those, so the
    kernel is built once per such key. Generated code is identical -> results bit-identical (h13_work/leak_repro.py)."""
    from mujoco_warp._src import collision_driver as cd
    if getattr(cd, '_g1ps_bp_cached', False): return
    import inspect
    if list(inspect.signature(cd._broadphase_filter).parameters) != ['m']:   # (11-04) upstream 4911167: keyed on ints inside a cache_kernel builder; no leak, nothing to patch (h13_work/daid_port/leak_test.py)
        cd._g1ps_bp_cached = True; return
    orig = cd._broadphase_filter; cache = {}
    def _broadphase_filter_cached(m):
        k = (int(m.geom_aabb.shape[0]), int(m.geom_rbound.shape[0]), int(m.geom_margin.shape[0]), int(m.opt.broadphase_filter))
        if k not in cache: cache[k] = orig(m)
        return cache[k]
    cd._broadphase_filter = _broadphase_filter_cached; cd._g1ps_bp_cached = True


class WarpCollider:
    """MuJoCo Warp collision queries on the exported scene: worlds = poses to test (frames, midpoints, ...).
    Geom poses are written directly (hand colliders from HandFK, props from their tracks, furniture static)."""
    def __init__(self, xml, z, nworld, nconmax=96, device='cuda'):
        import mujoco, warp as wp, mujoco_warp as mjw
        patch_mjw_broadphase_cache()
        self.wp, self.mjw = wp, mjw
        self.mjm = mujoco.MjModel.from_xml_string(xml)
        self.m = mjw.put_model(self.mjm)
        self.d = mjw.make_data(self.mjm, nworld=nworld+1, nconmax=nconmax, njmax=1)   # (+1: the canary world, see collide)
        self.nworld = nworld; self.dev = device; self.canary = nworld
        self.xpos = wp.to_torch(self.d.geom_xpos); self.xmat = wp.to_torch(self.d.geom_xmat)
        kind = np.asarray(z['geom_kind']); self.kind = torch.as_tensor(kind, device=device)
        self.ref = torch.as_tensor(np.asarray(z['geom_ref']), device=device)
        loc = torch.as_tensor(np.asarray(z['geom_local']), dtype=torch.float64, device=device)
        self.hand_g = torch.as_tensor(np.flatnonzero(kind == 0), device=device)
        self.hand_ref = self.ref[self.hand_g]
        self.static_g = torch.as_tensor(np.flatnonzero(kind == 1), device=device)
        self.prop_g = {k: torch.as_tensor(np.flatnonzero(kind == c), device=device) for k, c in (('object', 2), ('plunger', 3), ('tube', 4))}
        self.loc = loc; self.axis = torch.as_tensor(np.asarray(z['plunger_axis']), dtype=torch.float64, device=device)
        # static poses once
        S = loc[self.static_g]
        self.xpos[:, self.static_g] = S[:, :3, 3].float(); self.xmat[:, self.static_g] = S[:, :3, :3].float()
        ncoll = int(len(z['coll_geom'])); ng = self.mjm.ngeom
        # contact lookup: geom -> collider index (or -1), pair margins (collider, geom)
        self.coll_of = torch.full((ng,), -1, dtype=torch.long, device=device); self.coll_of[self.hand_g] = self.hand_ref
        self.margin = torch.zeros((ncoll, ng), dtype=torch.float64, device=device)
        for a, o, mg in np.asarray(z['pair_margins']).reshape(-1, 3):
            self.margin[int(self.coll_of[int(a)]), int(o)] = mg

    def pose(self, W, Rc, pc, T_obj, plunger, T_tube):
        """Write the geom poses of worlds 0..W-1: colliders (W,ncoll,3,3)/(W,ncoll,3), props (W,4,4), plunger depth (W,)."""
        self.xmat[:W, self.hand_g] = Rc[:, self.hand_ref].float(); self.xpos[:W, self.hand_g] = pc[:, self.hand_ref].float()
        for key, T in (('object', T_obj), ('tube', T_tube)):
            g = self.prop_g[key]
            if len(g):
                M = T[:, None]@self.loc[g][None]
                self.xmat[:W, g] = M[..., :3, :3].float(); self.xpos[:W, g] = M[..., :3, 3].float()
        g = self.prop_g['plunger']
        if len(g):
            M = T_obj[:, None]@self.loc[g][None]; off = (T_obj[:, :3, :3]@self.axis)*(-plunger[:, None])
            self.xmat[:W, g] = M[..., :3, :3].float(); self.xpos[:W, g] = (M[..., :3, 3]+off[:, None]).float()

    def _pass(self, W, margin):
        wp, mjw = self.wp, self.mjw
        self.m.geom_margin = margin
        mjw.collision(self.m, self.d)
        n = int(self.d.nacon.numpy()[0]); c = self.d.contact
        if n > self.d.naconmax: raise RuntimeError(f'contact overflow {n} > {self.d.naconmax}')
        t = lambda a: wp.to_torch(a)[:n]
        g = t(c.geom).long(); mg = self._gm[margin is self._m2]
        return dict(world=t(c.worldid).long(), geom=g, dist=t(c.dist).double()+torch.maximum(mg[g[:, 0]], mg[g[:, 1]]),
                    normal=t(c.frame)[:, 0].double().clone(), pos=t(c.pos).double().clone())

    def collide(self, W, tol=2e-4, near=6e-4):
        """Contacts of worlds 0..W-1 with their TRUE distances (MJWarp reports the distance less the pair margin):
        dict of tensors (world, geom pair, dist, normal from geom 0 to 1, midpoint).
        MJWarp's convex collision returns a bogus deep penetration for a pair whose distance is within ~0.3 mm of its
        margin (e.g. -26 mm for two fingers 6.0 mm apart, checked against mj_geomDistance and a hull-vertex test), so
        every query runs twice, with margins m and m + delta: a pair is kept when both passes agree; when they
        disagree, the pass whose margin the other pass's distance sits at is the bogus one; pairs seen once are dropped
        (beyond the first margin: no penalty needs them)."""
        if not hasattr(self, '_m2'):
            import warp as wp
            m1 = self.mjm.geom_margin.copy(); m2 = m1.copy(); m2[m1 > 0] += np.where(m1[m1 > 0] > .01, .003, .0015)
            self._m1 = self.m.geom_margin
            self._m2 = wp.array(m2[None].astype(np.float32), dtype=float)
            self._gm = {False: torch.as_tensor(m1, dtype=torch.float64, device=self.dev), True: torch.as_tensor(m2, dtype=torch.float64, device=self.dev)}
        if W < self.nworld:   # park the unused worlds' hands far away
            self.xpos[W:, self.hand_g] = 1e3
        cw = self.canary; self.xpos[cw] = self.xpos[0]; self.xmat[cw] = self.xmat[0]   # canary: world 0 again, in the last world
        c1 = self._pass(W, self._m1); c2 = self._pass(W, self._m2); self.m.geom_margin = self._m1
        for c_ in (c1, c2):   # MJWarp silently drops the later worlds' contacts when an internal pair buffer fills (10-28, gsearch
            a_, b_ = c_['world'] == 0, c_['world'] == cw   # Evaluator: ~190 of 1024 worlds read empty): the canary must see world 0's contacts
            if int(a_.sum()) != int(b_.sum()) or (a_.any() and abs(float(c_['dist'][a_].min())-float(c_['dist'][b_].min())) > 1e-6):
                raise RuntimeError(f'WarpCollider: contacts dropped (canary {int(b_.sum())} vs world 0 {int(a_.sum())} contacts, {W} worlds): raise nconmax or use fewer worlds')
            keep_ = c_['world'] != cw
            for k_ in list(c_): c_[k_] = c_[k_][keep_]
        ng = self.mjm.ngeom
        k1 = (c1['world']*ng+c1['geom'][:, 0])*ng+c1['geom'][:, 1]; k2 = (c2['world']*ng+c2['geom'][:, 0])*ng+c2['geom'][:, 1]
        o2 = torch.argsort(k2); k2s = k2[o2]
        j = torch.searchsorted(k2s, k1).clamp(max=max(len(k2s)-1, 0))
        found = (k2s[j] == k1) if len(k2s) else torch.zeros_like(k1, dtype=torch.bool)
        j2 = o2[j] if len(k2s) else j
        d1 = c1['dist']; d2 = c2['dist'][j2] if len(k2s) else d1
        g = c1['geom']; M1 = torch.maximum(self._gm[False][g[:, 0]], self._gm[False][g[:, 1]]); M2 = torch.maximum(self._gm[True][g[:, 0]], self._gm[True][g[:, 1]])
        agree = found & ((d1-d2).abs() < tol)
        use2 = found & ~agree & ((d2-M1).abs() < near)
        use1 = found & ~agree & ~use2 & ((d1-M2).abs() < near)
        # (both passes report the pair and disagree otherwise: dropped. Checked on the left hand: such pairs reported -12 to -31 mm
        # were 3.5-5.9 mm apart by mj_geomDistance; keeping the deeper value made the solver push fingers off phantom contacts)
        keep = agree | use1 | use2
        out = {k: v.clone() for k, v in c1.items()}
        if len(k2s):
            for k in ('dist', 'normal', 'pos'): out[k][use2] = c2[k][j2[use2]]
        self.stats = dict(n1=len(k1), n2=len(k2), agree=int(agree.sum()), fixed=int((use1 | use2).sum()), dropped=int((~keep).sum()),
                          dropped_deep=int((~keep & (d1 < -1e-3)).sum()))
        # canonical order (world, geom pair): MJWarp appends contacts with atomics, so their order varies run to run, and
        # every later tie-break / summation would follow it (determinism)
        out = {k: v[keep] for k, v in out.items()}
        g = out['geom']; kk = (out['world']*ng+g[:, 0])*ng+g[:, 1]; o = torch.argsort(kk, stable=True)
        return {k: v[o] for k, v in out.items()}
