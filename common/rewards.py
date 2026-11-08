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


def cone_support(n, pos, com, u, mu, world, N, torque_len, edges=8):
    """Support of a contact set along the required wrench direction u (N, 6): sigma = sum_c max(0, max_j u . w_cj), w_cj the unit-normal
    friction-cone edge wrenches [f; (p - com) x f / L]. n, pos (C, 3) world (n points from the hand into the object), com (N, 3), mu (C,)."""
    if n.shape[0] == 0: return torch.zeros(N, device=u.device)
    t1 = torch.cross(n, torch.tensor([1., 0., 0.], device=n.device).expand_as(n), dim=-1)
    alt = torch.cross(n, torch.tensor([0., 1., 0.], device=n.device).expand_as(n), dim=-1)
    t1 = torch.where((t1.norm(dim=-1, keepdim=True) > .1), t1, alt); t1 = t1/t1.norm(dim=-1, keepdim=True); t2 = torch.cross(n, t1, dim=-1)
    th = torch.arange(edges, device=n.device)*(2*torch.pi/edges)
    f = n[:, None]+mu[:, None, None]*(torch.cos(th)[None, :, None]*t1[:, None]+torch.sin(th)[None, :, None]*t2[:, None])      # (C, E, 3)
    r = (pos-com[world])[:, None].expand_as(f)
    w = torch.cat([f, torch.cross(r, f, dim=-1)/torque_len], -1)                                                              # (C, E, 6)
    s = (w*u[world][:, None]).sum(-1).max(-1).values.clamp_min(0)
    out = torch.zeros(N, device=u.device); out.index_add_(0, world, s); return out


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
