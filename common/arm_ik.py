"""Batched forward kinematics and damped-least-squares IK for the two 7-DoF Vega-U arms, in torch on the GPU (layer: KINEMATICS).

Written for this project (the idea of a batched GPU IK anchored on object poses follows Weihan's online-IK design; no code taken).
The chain from 'arm_center' to '{l,r}_wrist' is read from the compiled MuJoCo model: per body its pos / quat relative to the parent and,
if it has a hinge, the joint anchor and axis in the body frame. 'arm_center' is fixed in the world (fixed base, Lift held), so it is
evaluated once with MuJoCo at the task's Lift value.
IK step: e = [p* - p ; w_r * rotvec(R* R^T)], dq = J^T (J J^T + lam^2 I)^-1 e + (I - J^+ J) k_null (q_nom - q), step clamped, joints kept
inside [lo + margin, hi - margin]. Warm-started from the nominal joints, so a target equal to FK(q_nom) returns q_nom.
"""
import numpy as np, torch, mujoco

ARM_ROOT = 'arm_center'
WRIST = {'L': 'l_wrist', 'R': 'r_wrist'}


def _qmat(q):           # wxyz -> 3x3, batched
    w, x, y, z = q.unbind(-1)
    return torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y), 2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x), 2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1).reshape(*q.shape[:-1], 3, 3)


def _axis_rot(axis, ang):  # Rodrigues, axis (3,) unit, ang (N,) -> (N, 3, 3)
    K = torch.zeros(3, 3, dtype=ang.dtype, device=ang.device); a = axis
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -a[2], a[1], a[2], -a[0], -a[1], a[0]
    s, c = torch.sin(ang)[:, None, None], torch.cos(ang)[:, None, None]
    return torch.eye(3, dtype=ang.dtype, device=ang.device)+s*K+(1-c)*(K@K)


def rotvec_from_mat(R):  # (N, 3, 3) -> (N, 3), robust near 0 and pi
    tr = R[..., 0, 0]+R[..., 1, 1]+R[..., 2, 2]
    w = torch.sqrt((1+tr).clamp_min(1e-12))/2
    v = torch.stack([R[..., 2, 1]-R[..., 1, 2], R[..., 0, 2]-R[..., 2, 0], R[..., 1, 0]-R[..., 0, 1]], -1)/(4*w[..., None].clamp_min(1e-6))
    # near pi, w -> 0: fall back to the largest diagonal element
    near = w < 1e-3
    if near.any():
        Rn = R[near]; d = torch.stack([Rn[:, 0, 0], Rn[:, 1, 1], Rn[:, 2, 2]], -1); i = d.argmax(-1)
        x = torch.sqrt(((1+2*d.gather(1, i[:, None])[:, 0]-(d.sum(-1))).clamp_min(1e-12)))/2
        vv = torch.zeros_like(d); vv[torch.arange(len(i)), i] = x
        for j in range(3):
            o = (i+1+j) % 3
        vfix = torch.stack([(Rn[:, 0, 1]+Rn[:, 1, 0]), (Rn[:, 1, 2]+Rn[:, 2, 1]), (Rn[:, 0, 2]+Rn[:, 2, 0])], -1)
        vv = torch.where(i[:, None] == 0, torch.stack([x, vfix[:, 0]/(4*x), vfix[:, 2]/(4*x)], -1),
             torch.where(i[:, None] == 1, torch.stack([vfix[:, 0]/(4*x), x, vfix[:, 1]/(4*x)], -1), torch.stack([vfix[:, 2]/(4*x), vfix[:, 1]/(4*x), x], -1)))
        v = v.clone(); v[near] = vv/vv.norm(dim=-1, keepdim=True).clamp_min(1e-9); w = w.clone(); w[near] = 0.
    s = v.norm(dim=-1, keepdim=True)
    ang = 2*torch.atan2(s, w[..., None])
    return torch.where(s > 1e-9, v/s.clamp_min(1e-12)*ang, 2*v)


class ArmIK:
    def __init__(self, m: mujoco.MjModel, lift: float, device='cuda:0', dtype=torch.float32):
        self.device, self.dtype = device, dtype
        d = mujoco.MjData(m); d.qpos[:] = m.qpos0; d.qpos[m.jnt_qposadr[m.joint('Lift').id]] = lift; mujoco.mj_kinematics(m, d)
        ac = m.body(ARM_ROOT).id
        T0 = np.eye(4); T0[:3, :3] = d.xmat[ac].reshape(3, 3); T0[:3, 3] = d.xpos[ac]
        self.T_root = torch.as_tensor(T0, dtype=dtype, device=device)
        self.chain, self.qadr, self.lo, self.hi, self.jnames = {}, {}, {}, {}, {}
        for s, wn in WRIST.items():
            b = m.body(wn).id; bodies = []
            while b != ac:
                bodies.append(b); b = int(m.body_parentid[b])
            bodies = bodies[::-1]; ch = []; qa = []; lo = []; hi = []; jn = []
            for b in bodies:
                pos = torch.as_tensor(m.body_pos[b], dtype=dtype, device=device); R = _qmat(torch.as_tensor(m.body_quat[b], dtype=dtype, device=device))
                js = [j for j in range(m.body_jntadr[b], m.body_jntadr[b]+m.body_jntnum[b])] if m.body_jntnum[b] > 0 else []
                if js:
                    assert len(js) == 1 and m.jnt_type[js[0]] == mujoco.mjtJoint.mjJNT_HINGE, m.body(b).name
                    j = js[0]; ch.append((pos, R, torch.as_tensor(m.jnt_pos[j], dtype=dtype, device=device), torch.as_tensor(m.jnt_axis[j], dtype=dtype, device=device), len(qa)))
                    qa.append(int(m.jnt_qposadr[j])); lo.append(m.jnt_range[j, 0]); hi.append(m.jnt_range[j, 1]); jn.append(m.joint(j).name)
                else:
                    ch.append((pos, R, None, None, None))
            assert len(qa) == 7, (s, jn)
            self.chain[s] = ch; self.qadr[s] = qa; self.jnames[s] = jn
            self.lo[s] = torch.as_tensor(lo, dtype=dtype, device=device); self.hi[s] = torch.as_tensor(hi, dtype=dtype, device=device)

    def fk(self, side, q, need_jac=False):
        """q (N, 7) -> wrist T (N, 4, 4) [, J (N, 6, 7) geometric, world frame]"""
        N = q.shape[0]; R = self.T_root[:3, :3].expand(N, 3, 3).clone(); p = self.T_root[:3, 3].expand(N, 3).clone()
        zs, ps = [], []
        for pos, Rb, jp, ja, k in self.chain[side]:
            p = p+(R@pos[:, None])[..., 0]; R = R@Rb
            if jp is not None:
                pj = p+(R@jp[:, None])[..., 0]; z = (R@ja[:, None])[..., 0]
                Rq = _axis_rot(ja, q[:, k]); R = R@Rq; p = pj-(R@jp[:, None])[..., 0]   # rotate about the anchor
                zs.append(z); ps.append(pj)
        T = torch.zeros(N, 4, 4, dtype=q.dtype, device=q.device); T[:, :3, :3] = R; T[:, :3, 3] = p; T[:, 3, 3] = 1
        if not need_jac: return T
        Z = torch.stack(zs, -1); P = torch.stack(ps, -1)                               # (N, 3, 7)
        Jv = torch.cross(Z, p[..., None]-P, dim=1); J = torch.cat([Jv, Z], 1)
        return T, J

    def solve(self, side, T_target, q_seed, q_nom, iters=3, lam=.05, rot_w=.3, k_null=.2, step_max=.15, margin=.02):
        q = q_seed.clone(); lo, hi = self.lo[side]+margin, self.hi[side]-margin
        eye = torch.eye(6, dtype=q.dtype, device=q.device)
        for _ in range(iters):
            T, J = self.fk(side, q, need_jac=True)
            ep = T_target[:, :3, 3]-T[:, :3, 3]; er = rotvec_from_mat(T_target[:, :3, :3]@T[:, :3, :3].transpose(1, 2))
            e = torch.cat([ep, rot_w*er], -1); Jw = torch.cat([J[:, :3], rot_w*J[:, 3:]], 1)
            A = Jw@Jw.transpose(1, 2)+lam**2*eye
            Jp = Jw.transpose(1, 2)@torch.linalg.solve(A, torch.eye(6, dtype=q.dtype, device=q.device).expand_as(A))   # (N, 7, 6)
            dq = (Jp@e[..., None])[..., 0]
            dq = dq+((torch.eye(7, dtype=q.dtype, device=q.device)-Jp@Jw)@(k_null*(q_nom-q))[..., None])[..., 0]
            dq = dq.clamp(-step_max, step_max); q = torch.minimum(torch.maximum(q+dq, lo), hi)
        T = self.fk(side, q)
        res_p = (T_target[:, :3, 3]-T[:, :3, 3]).norm(dim=-1); res_r = rotvec_from_mat(T_target[:, :3, :3]@T[:, :3, :3].transpose(1, 2)).norm(dim=-1)
        return q, res_p, res_r


def selftest(m, lift, n=64, device='cuda:0'):
    """FK vs MuJoCo at random joint angles, and IK recovering a perturbed pose. Returns a dict of max errors."""
    ik = ArmIK(m, lift, device=device, dtype=torch.float64); d = mujoco.MjData(m); rng = np.random.default_rng(0); out = {}
    for s in 'LR':
        lo, hi = ik.lo[s].cpu().numpy(), ik.hi[s].cpu().numpy(); Q = rng.uniform(lo*.8, hi*.8, size=(n, 7))
        T = ik.fk(s, torch.as_tensor(Q, device=device)).cpu().numpy(); err_p, err_r = 0., 0.
        for i in range(n):
            d.qpos[:] = m.qpos0; d.qpos[m.jnt_qposadr[m.joint('Lift').id]] = lift
            for k, a in enumerate(ik.qadr[s]): d.qpos[a] = Q[i, k]
            mujoco.mj_kinematics(m, d); b = m.body(WRIST[s]).id
            err_p = max(err_p, float(np.linalg.norm(d.xpos[b]-T[i, :3, 3]))); err_r = max(err_r, float(np.abs(d.xmat[b].reshape(3, 3)-T[i, :3, :3]).max()))
        q0 = torch.as_tensor(Q, device=device); q1 = (q0+torch.as_tensor(rng.normal(0, .1, size=(n, 7)), device=device)).clamp(ik.lo[s], ik.hi[s])
        Tt = ik.fk(s, q1); qs, rp, rr = ik.solve(s, Tt, q0, q0, iters=30, lam=.01, k_null=0.)
        q_same, rp0, _ = ik.solve(s, ik.fk(s, q0), q0, q0, iters=3)
        out[s] = {'fk_pos_err_m_max': err_p, 'fk_rot_err_max': err_r, 'ik_res_mm_p50_max': [float(rp.median())*1e3, float(rp.max())*1e3],
                  'ik_res_deg_max': float(rr.max())*57.3, 'ik_identity_dq_max': float((q_same-q0).abs().max())}
    return out
