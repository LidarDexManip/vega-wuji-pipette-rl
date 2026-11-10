"""Reward / metric terms for the shared Vega env, as separate batched torch functions (layer: RL ENV LIBRARY).
Each returns a per-world tensor (N,). Positive terms are kernels in [0, 1]; penalties are >= 0 (the env subtracts them).
Jiming Ren's three contact terms follow subagent 3's definitions (h13_work/datagen_rl/DESIGN.md section 7, contact_terms.py).
"""
import torch


def rot_angle(Ra, Rb):                     # angle between two rotations, (N,)
    c = ((Ra.transpose(-1, -2)@Rb).diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2
    return torch.arccos(c.clamp(-1, 1))


def axis_angle(Ra, Rb, axis=2):            # angle between one body axis of two rotations (spin about it is free)
    return torch.arccos((Ra[..., :, axis]*Rb[..., :, axis]).sum(-1).clamp(-1, 1))


# ---------------------------------------------------------------- tracking
def track_pos(p, p_ref, sig):
    return torch.exp(-(p-p_ref).norm(dim=-1)/sig)


def track_rot(R, R_ref, sig, axis_only=False):
    return torch.exp(-(axis_angle(R, R_ref) if axis_only else rot_angle(R, R_ref))/sig)


def track_joints(q, q_ref, sig):
    return torch.exp(-(q-q_ref).abs().mean(-1)/sig)


def plunger(depth, depth_ref, active, sig):
    return torch.where(active, torch.exp(-((depth-depth_ref)/sig)**2), torch.zeros_like(depth))


# ---------------------------------------------------------------- Jiming T1 / T2 / T3
def contact_set(cur, ref):
    """cur, ref (N, L) bool for one side. missed = |ref - cur| / |ref| (0 if ref empty); unintended = |cur - ref|."""
    nref = ref.sum(-1)
    missed = torch.where(nref > 0, (ref & ~cur).sum(-1).float()/nref.clamp_min(1), torch.zeros_like(nref, dtype=torch.float32))
    return missed, (cur & ~ref).sum(-1).float()


def contact_graph(p_cur, has_cur, p_ref, has_ref):
    """Rigid-motion-invariant contact geometry error (m): mean |d_ij - d_ij_ref| over link pairs present in both (object frame points).
    Returns (err, n_pairs); err = 0 where fewer than one common pair."""
    both = has_cur & has_ref                                               # (N, L)
    dc = torch.cdist(p_cur, p_cur); dr = torch.cdist(p_ref, p_ref)        # (N, L, L)
    pair = both[:, :, None] & both[:, None, :]
    pair = pair & torch.triu(torch.ones_like(pair[0]), 1)[None]
    n = pair.sum((1, 2)).float()
    err = ((dc-dr).abs()*pair).sum((1, 2))/n.clamp_min(1)
    return torch.where(n > 0, err, torch.zeros_like(err)), n


def cone_edges(n, pos, com, mu, world, torque_len, edges=8):
    """Unit-normal friction-cone edge wrenches w_cj = [f; (p - com) x f / L] (C, E, 6), f = n + mu (cos t1 + sin t2) (polyhedral cone,
    E edges). n, pos (C, 3) world (n points from the hand into the object), com (N, 3) indexed by world (C,), mu (C,)."""
    t1 = torch.cross(n, torch.tensor([1., 0., 0.], device=n.device).expand_as(n), dim=-1)
    alt = torch.cross(n, torch.tensor([0., 1., 0.], device=n.device).expand_as(n), dim=-1)
    t1 = torch.where((t1.norm(dim=-1, keepdim=True) > .1), t1, alt); t1 = t1/t1.norm(dim=-1, keepdim=True); t2 = torch.cross(n, t1, dim=-1)
    th = torch.arange(edges, device=n.device)*(2*torch.pi/edges)
    f = n[:, None]+mu[:, None, None]*(torch.cos(th)[None, :, None]*t1[:, None]+torch.sin(th)[None, :, None]*t2[:, None])      # (C, E, 3)
    r = (pos-com[world])[:, None].expand_as(f)
    return torch.cat([f, torch.cross(r, f, dim=-1)/torque_len], -1)                                                           # (C, E, 6)


def cone_support(n, pos, com, u, mu, world, N, torque_len, edges=8):
    """Support of a contact set along the required wrench direction u (N, 6): sigma = sum_c max(0, max_j u . w_cj), w_cj the unit-normal
    friction-cone edge wrenches [f; (p - com) x f / L]. n, pos (C, 3) world (n points from the hand into the object), com (N, 3), mu (C,).
    Note: sigma is the support function of the grasp wrench set along u, not a feasibility test -- with a pure-force u (no torque part) the
    torque balance does not enter at all (a finger pushing up 3 cm off the CoM scores like one under it); see wrench_margin."""
    if n.shape[0] == 0: return torch.zeros(N, device=u.device)
    w = cone_edges(n, pos, com, mu, world, torque_len, edges)
    s = (w*u[world][:, None]).sum(-1).max(-1).values.clamp_min(0)
    out = torch.zeros(N, device=u.device); out.index_add_(0, world, s); return out


def _proj_friction_cone(f, n, mu, fmax):
    """Euclidean projection of contact forces f (..., C, 3) onto the truncated circular friction cone {|f_t| <= mu f_n, 0 <= f_n <= fmax}
    (normal n (C, 3), mu (C,)): projection onto the cone; if that exceeds the cap, the cap is active and the answer is the disk
    {f_n = fmax, |f_t| <= mu fmax} (radial clamp of the tangential part)."""
    fn = (f*n).sum(-1, keepdim=True); ft = f-fn*n; s = ft.norm(dim=-1, keepdim=True); m = mu[:, None]
    et = ft/s.clamp_min(1e-12); c = 1./(1.+m*m)
    a = ((m*s+fn)*c).clamp_min(0)                                    # onto the cone surface: normal part a, tangential part mu a
    inside = s <= m*fn
    pn = torch.where(inside, fn, a); pt = torch.where(inside, ft, m*a*et)
    cap = pn > fmax
    pn = torch.where(cap, torch.full_like(pn, fmax), pn); pt = torch.where(cap, ft*(m*fmax/s.clamp_min(1e-12)).clamp(max=1.), pt)
    return pn*n+pt


def wrench_residual(n, pos, com, targets, mu, world, N, torque_len, fmax, iters=100):
    """Distance (N, K) from each target wrench targets (N, K, 6) = [f; tau / L] to the grasp wrench set
    P = {sum_c [f_c; (p_c - com) x f_c / L] : |f_ct| <= mu f_cn, 0 <= f_cn <= fmax}: every contact pushes with normal force <= fmax inside its
    circular (elliptic, as in the scene) friction cone. 0 = the contacts can supply that wrench. Exact convex problem (projection onto P),
    solved batched by FISTA on the contact forces (3 per contact, closed-form cone projection); the iterate is always feasible, so the value
    is an upper bound that converges to the exact distance (checked against an NNLS solve in rl/test_contact_terms.py).
    n, pos (C, 3) world, n from the hand into the object; com (N, 3); mu (C,); world (C,) long."""
    dev = targets.device
    if n.shape[0] == 0: return targets.norm(dim=-1)
    r = (pos-com[world])/torque_len; K = targets.shape[1]
    Rx = torch.zeros(r.shape[0], 3, 3, device=dev)                      # [r]x, so [r]x f = r x f
    Rx[:, 0, 1], Rx[:, 0, 2], Rx[:, 1, 0], Rx[:, 1, 2], Rx[:, 2, 0], Rx[:, 2, 1] = -r[:, 2], r[:, 1], r[:, 2], -r[:, 0], -r[:, 1], r[:, 0]
    Gc = torch.cat([torch.eye(3, device=dev).expand(r.shape[0], 3, 3), Rx], 1)                 # (C, 6, 3): f -> [f; r x f]
    S = torch.zeros(N, 6, 6, device=dev); S.index_add_(0, world, Gc@Gc.transpose(1, 2))
    step = (1./torch.linalg.eigvalsh(S)[:, -1].clamp_min(1e-9))[world][None, :, None]           # 1 / Lipschitz constant of the world
    tg = targets.transpose(0, 1)                                                                # (K, N, 6)

    def resid(f):
        x = torch.zeros(K, N, 6, device=dev); x.index_add_(1, world, torch.cat([f, torch.cross(r.expand_as(f), f, dim=-1)], -1))
        return x-tg
    f = torch.zeros(K, r.shape[0], 3, device=dev); y = f; t = 1.
    for _ in range(iters):
        rc = resid(y)[:, world]                                                                 # (K, C, 6)
        g = rc[..., :3]+torch.cross(rc[..., 3:], r.expand_as(rc[..., 3:]), dim=-1)              # G_c^T residual
        nxt = _proj_friction_cone(y-step*g, n, mu, fmax)
        t2 = (1+(1+4*t*t)**.5)/2; y = nxt+((t-1)/t2)*(nxt-f); f, t = nxt, t2
    return resid(f).norm(dim=-1).T                                                             # (N, K)


def wrench_margin(n, pos, com, w_req, mu, world, N, torque_len, fmax, rho, iters=100):
    """Task-wrench coverage of a grasp: can the contacts supply the required wrench w_req (N, 6) = [f; tau / L] (gravity + inertia), and still
    resist an extra disturbance of size rho along each of the 12 signed wrench axes? Targets w_req and w_req +- rho e_i (13), each projected
    onto the grasp wrench set (wrench_residual). Returns (d_req, d_mean), both (N,) in N:
      d_req  = distance of w_req itself (0: the contacts can hold it, torque balance included -- the 6-D form of "CoM inside the support
               polygon"; > 0: they cannot, e.g. one finger pushing up off the CoM);
      d_mean = mean over the 13 targets (0: the cross-polytope of radius rho around w_req, hence a ball of radius rho / sqrt 6, fits inside the
               grasp wrench set -- the task-oriented margin; > 0 grows smoothly as the grasp loses margin or cannot hold at all)."""
    E6 = torch.eye(6, device=w_req.device)
    targets = torch.cat([w_req[:, None], w_req[:, None]+rho*E6[None], w_req[:, None]-rho*E6[None]], 1)   # (N, 13, 6)
    d = wrench_residual(n, pos, com, targets, mu, world, N, torque_len, fmax, iters)
    return d[:, 0], d.mean(1)


# ---------------------------------------------------------------- reference-free contact stability across consecutive control steps
def contact_lost(prev, still):
    """Fraction of the links in contact at the previous step (prev (N, L) bool) that no longer touch now (still (N, L): contact above the
    lower hysteresis threshold); 0 where nothing was in contact. Counts events: a deliberate release costs one step, chatter costs every flip."""
    n = prev.sum(-1)
    return torch.where(n > 0, (prev & ~still).sum(-1).float()/n.clamp_min(1), torch.zeros(prev.shape[0], device=prev.device))


def contact_slip(p_cur, has_cur, p_prev, has_prev):
    """Mean displacement (m) of each link's contact point ON THE OBJECT (object-frame points) since the previous step, over links in contact at
    both steps; 0 where none. Catches the whole grasp sliding along the object, which the pairwise distances (contact_graph) do not see."""
    both = has_cur & has_prev; n = both.sum(-1)
    d = ((p_cur-p_prev).norm(dim=-1)*both).sum(-1)
    return torch.where(n > 0, d/n.clamp_min(1), torch.zeros_like(d))


# ---------------------------------------------------------------- naturalness (penalties >= 0)
def free_finger_acc(q, q1, q2, mask, dt):
    """Mean squared joint acceleration (rad/s^2)^2 of the fingers that should be still (mask (N, J) float)."""
    acc = (q-2*q1+q2)/dt**2
    return (acc**2*mask).sum(-1)/mask.sum(-1).clamp_min(1)


def posture_prior(q, q_prior, active):
    return torch.where(active, ((q-q_prior)**2).mean(-1), torch.zeros_like(q[:, 0]))


def action_rate(a, a_prev):
    return ((a-a_prev)**2).mean(-1)


def joint_margin(q, lo, hi, margin):
    d = torch.minimum(q-lo, hi-q)
    return (torch.relu(margin-d)**2).sum(-1)


def elbow_above_shoulder(z_elbow, z_shoulder, margin=0.):
    """User rule 11-05: an elbow above its shoulder is an unnatural posture. Penalty = sum over arms of relu(z_elbow - z_shoulder + margin)^2
    (m^2); elbow = origin of {L,R}_arm_l4, shoulder = origin of {L,R}_arm_l1 (V1's definition). z_* (N, 2)."""
    return (torch.relu(z_elbow-z_shoulder+margin)**2).sum(-1)
