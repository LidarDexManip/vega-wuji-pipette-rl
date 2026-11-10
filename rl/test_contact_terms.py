"""Unit test (CPU, no task meshes): the 11-08 contact terms.
  1. cone_support (T3) is unchanged by the cone_edges refactor, and is blind to torque balance (finger under vs 3 cm off the CoM).
  2. wrench_residual against an exact NNLS projection (64-edge cones): an upper bound, within tolerance after 100 FISTA iterations.
  3. wrench_margin on toy grasps (1 N up required, 3 N per contact, rho 0.3 N): feasibility and margin ordered as expected.
  4. vega_extras.contacts / contact_temporal / reset on a synthetic contact stream: rigid object motion costs nothing, hysteresis absorbs force
     flicker, a sliding finger shows as drift + slip, the whole grasp sliding shows as slip only, pressing the plunger costs nothing.
  python test_contact_terms.py"""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'common'))
import rewards as RW
import vega_extras as X

L_T, MU = .05, 1.5


def old_cone_support(n, pos, com, u, mu, world, N, torque_len, edges=8):   # verbatim copy of the pre-11-08 implementation
    if n.shape[0] == 0: return torch.zeros(N, device=u.device)
    t1 = torch.cross(n, torch.tensor([1., 0., 0.], device=n.device).expand_as(n), dim=-1)
    alt = torch.cross(n, torch.tensor([0., 1., 0.], device=n.device).expand_as(n), dim=-1)
    t1 = torch.where((t1.norm(dim=-1, keepdim=True) > .1), t1, alt); t1 = t1/t1.norm(dim=-1, keepdim=True); t2 = torch.cross(n, t1, dim=-1)
    th = torch.arange(edges, device=n.device)*(2*torch.pi/edges)
    f = n[:, None]+mu[:, None, None]*(torch.cos(th)[None, :, None]*t1[:, None]+torch.sin(th)[None, :, None]*t2[:, None])
    r = (pos-com[world])[:, None].expand_as(f)
    w = torch.cat([f, torch.cross(r, f, dim=-1)/torque_len], -1)
    s = (w*u[world][:, None]).sum(-1).max(-1).values.clamp_min(0)
    out = torch.zeros(N, device=u.device); out.index_add_(0, world, s); return out


def T(x): return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def one(n, p):   # single-world contact tensors
    n = torch.nn.functional.normalize(T(n), dim=-1); return n, T(p), torch.zeros(len(n), dtype=torch.long), torch.full((len(n),), MU)


def test_cone_support():
    g = torch.Generator().manual_seed(1); C, N = 500, 64
    n = torch.nn.functional.normalize(torch.randn(C, 3, generator=g), dim=-1); pos = torch.randn(C, 3, generator=g)*.03
    com = torch.randn(N, 3, generator=g)*.01; u = torch.nn.functional.normalize(torch.randn(N, 6, generator=g), dim=-1)
    w = torch.randint(N, (C,), generator=g); mu = torch.full((C,), MU)
    assert torch.equal(old_cone_support(n, pos, com, u, mu, w, N, L_T), RW.cone_support(n, pos, com, u, mu, w, N, L_T)), 'cone_support changed'
    up = T([[0, 0, 1, 0, 0, 0]])
    s = [float(RW.cone_support(*one([[0, 0, 1]], p)[:2], torch.zeros(1, 3), up, one([[0, 0, 1]], p)[3], torch.zeros(1, dtype=torch.long), 1, L_T)) for p in ([[0, 0, -.01]], [[.03, 0, -.01]])]
    print(f'T3 sigma: finger under the CoM {s[0]:.2f}, 3 cm off the CoM {s[1]:.2f} (same: T3 ignores torque balance for a pure-force need)')
    assert abs(s[0]-s[1]) < 1e-6


def exact_residual(n, p, wq, fmax, E=64):
    """Distance from wq to {sum_c fmax sum_j lam_cj w_cj, lam >= 0, sum_j lam_cj <= 1} by NNLS (slack per contact, heavily weighted sum rows)."""
    from scipy.optimize import nnls
    C = len(n); W = (RW.cone_edges(T(n), T(p), torch.zeros(1, 3), torch.full((C,), MU), torch.zeros(C, dtype=torch.long), L_T, E)*fmax).numpy().astype(np.float64)
    A = np.zeros((6+C, C*E+C)); A[:6, :C*E] = W.reshape(C*E, 6).T; b = np.r_[wq, 1e4*np.ones(C)]
    for c in range(C): A[6+c, c*E:(c+1)*E] = 1e4; A[6+c, C*E+c] = 1e4
    x, _ = nnls(A, b, maxiter=5000); return float(np.linalg.norm(A[:6]@x-wq))


def test_residual_vs_exact():
    rng = np.random.default_rng(0); d = []
    for _ in range(40):
        C = int(rng.integers(1, 7)); ph = rng.uniform(0, 2*np.pi, C)
        p = np.c_[.008*np.cos(ph), .008*np.sin(ph), rng.uniform(-.03, .03, C)]
        n = -np.c_[np.cos(ph), np.sin(ph), np.zeros(C)]+rng.normal(0, .3, (C, 3)); n /= np.linalg.norm(n, axis=1, keepdims=True)
        wq = np.array([*rng.normal(0, .2, 2), rng.uniform(0, 1.2), *rng.normal(0, .2, 3)])
        for tgt in (wq, wq+np.r_[0, 0, 0, .3, 0, 0], wq-np.r_[.3, 0, 0, 0, 0, 0]):
            nt, pt, w, mu = one(n, p)
            ap = float(RW.wrench_residual(nt, pt, torch.zeros(1, 3), T(tgt)[None, None], mu, w, 1, L_T, 3., iters=100)[0, 0])
            d.append((exact_residual(n, p, tgt, 3.), ap))
    d = np.array(d); e = d[:, 1]-d[:, 0]
    print(f'wrench_residual vs exact on {len(d)} targets: approx - exact min {e.min():+.4f}, p95 {np.percentile(e, 95):.4f}, max {e.max():.4f} N '
          f'(exact: {np.mean(d[:, 0] < 1e-4):.0%} feasible, median {np.median(d[:, 0]):.3f} N)')
    assert e.min() > -3e-3, 'not an upper bound (beyond the 64-gon vs circle difference)'
    assert np.percentile(e, 95) < .05 and e.max() < .15, 'FISTA too far from the exact projection after 100 iterations'


def test_margin_toys():
    w_up = T([[0, 0, 1., 0, 0, 0]]); rows = {}
    cases = {'1 finger under the CoM': ([[0, 0, 1]], [[0, 0, -.01]]), '1 finger 3 cm off the CoM': ([[0, 0, 1]], [[.03, 0, -.01]]),
             '2-finger pinch': ([[1, 0, 0], [-1, 0, 0]], [[-.01, 0, 0], [.01, 0, 0]]),
             'pinch + finger under': ([[1, 0, 0], [-1, 0, 0], [0, 0, 1]], [[-.01, 0, 0], [.01, 0, 0], [0, 0, -.01]]),
             '4 fingers around + under': ([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]], [[-.01, 0, .01], [.01, 0, -.01], [0, -.01, .01], [0, .01, -.01], [0, 0, -.02]])}
    for name, (n, p) in cases.items():
        nt, pt, w, mu = one(n, p)
        dr, dm = RW.wrench_margin(nt, pt, torch.zeros(1, 3), w_up, mu, w, 1, L_T, 3., .3, iters=300)
        rows[name] = (float(dr), float(dm)); print(f'  {name:26s} d_req {float(dr):.3f} N   d_mean {float(dm):.3f} N')
    r = rows
    assert r['1 finger under the CoM'][0] < .02 and r['1 finger under the CoM'][1] > .05      # balanced, no margin
    assert r['1 finger 3 cm off the CoM'][0] > .1                                              # cannot hold: torque imbalance
    assert r['2-finger pinch'][0] < .02 and r['2-finger pinch'][1] > .02                       # holds, but not torque about the pinch axis
    assert r['4 fingers around + under'][1] < r['2-finger pinch'][1] and r['4 fingers around + under'][1] < r['1 finger under the CoM'][1]


# ------------------------------------------------------------------ vega_extras glue on a synthetic contact stream
XML = """<mujoco><worldbody>
 <body name="right_object" pos="0 0 .3"><freejoint/><geom name="pip" type="cylinder" size=".01 .05" mass=".08"/>
  <body name="pipette_plunger" pos="0 0 .06"><joint name="pipette_plunger" type="slide" axis="0 0 1" range="-.0134 0"/><geom name="plg" type="cylinder" size=".004 .01" mass=".005"/></body></body>
 <body name="left_object" pos=".3 0 .3"><freejoint/><geom name="tube" type="cylinder" size=".008 .04" mass=".013"/></body>
 <body name="right_l0" pos="0 .1 0"><geom type="sphere" size=".01"/></body><body name="right_l1" pos="0 .2 0"><geom type="sphere" size=".01"/></body>
 <body name="right_l2" pos="0 .3 0"><geom type="sphere" size=".01"/></body><body name="right_thumb" pos="0 .4 0"><geom type="sphere" size=".01"/></body>
 <body name="left_l0" pos="0 .5 0"><geom type="sphere" size=".01"/></body></worldbody></mujoco>"""
LINKS = ['right_l0', 'right_l1', 'right_l2', 'right_thumb', 'left_l0']


def fake_env(N):
    m = mujoco.MjModel.from_xml_string(XML); cfg = X.merge(X.DEFAULT, {}); nb = m.nbody
    gl = np.full(m.ngeom, -1); gs = np.full(m.ngeom, -1)
    for i, nm in enumerate(LINKS): gl[m.body(nm).geomadr[0]] = i
    gs[m.geom('pip').id] = 0; gs[m.geom('plg').id] = 0; gs[m.geom('tube').id] = 1
    env = SimpleNamespace(mjm=m, device='cpu', num_envs=N, xcfg=cfg, x_L=len(LINKS), x_gl=torch.as_tensor(gl), x_gs=torch.as_tensor(gs),
                          x_lside=torch.tensor([0, 0, 0, 0, 1]), x_mass=torch.tensor([.085, .013]),
                          x_objb=torch.tensor([m.body('right_object').id, m.body('left_object').id]),
                          xmat=torch.eye(3).repeat(N, nb, 1, 1), xpos=torch.zeros(N, nb, 3), qpos=torch.zeros(N, m.nq),
                          # reset() needs these
                          x_af=torch.zeros(N, 1), x_prev_arm=torch.zeros(N, 1), ctrl=torch.zeros(N, 1), x_arm=torch.tensor([0]), x_rlast=torch.zeros(N, 1),
                          x_rprev=torch.zeros(N, 1), x_tick=torch.zeros(N, 1), x_tickq=torch.zeros(N, 1), x_q1=None)
    X.temporal_init(env); return env, m


def set_contacts(env, m, rows):
    """rows: (world, link name, object geom name, world position, force). Fills the contact / efc arrays contacts() reads."""
    n = len(rows); env.nacon = torch.tensor([n])
    env.x_cgeom = torch.tensor([[m.body(l).geomadr[0], m.geom(g).id] for _, l, g, _, _ in rows]).int()
    env.x_cworld = torch.tensor([w for w, *_ in rows]).int(); env.x_cefc = torch.arange(n).int()[:, None]
    env.x_cpos = T([p for *_, p, _ in rows]); env.x_cframe = torch.eye(3).repeat(n, 1, 1)
    env.x_force = torch.zeros(env.num_envs, n)
    for i, (w, *_, f) in enumerate(rows): env.x_force[w, i] = f


def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def test_temporal():
    N = 5; env, m = fake_env(N); ob, pb = m.body('right_object').id, m.body('pipette_plunger').id
    thr = .1*.085*9.81                                    # contact threshold on the pipette (N); release below half of it
    grip = np.array([[.01, 0, 0], [-.01, 0, .01], [0, .01, -.01]])   # three fingertip points on the pipette, object frame
    hist = []
    for t in range(4):
        rows = []
        # world 0: the object (and the grasp with it) translates 5 mm and turns 3 deg per step -> nothing to penalise
        R = Rz(np.deg2rad(3*t)); o = np.array([.005*t, 0, .3]); env.xmat[0, ob] = T(R); env.xpos[0, ob] = T(o)
        rows += [(0, f'right_l{i}', 'pip', o+R@grip[i], 1.) for i in range(3)]
        # world 1: link 2's force flickers between 0.8 and 1.2 x threshold (hysteresis keeps it), then drops to 0.3 x at t = 3 (lost)
        env.xpos[1, ob] = T([0, 0, .3])
        f2 = [1.2, .8, 1.2, .3][t]*thr
        rows += [(1, 'right_l0', 'pip', grip[0]+[0, 0, .3], 1.), (1, 'right_l1', 'pip', grip[1]+[0, 0, .3], 1.), (1, 'right_l2', 'pip', grip[2]+[0, 0, .3], f2)]
        # world 2: finger 0 slides 2 mm per step along the pipette axis, the others stay -> drift + slip
        env.xpos[2, ob] = T([0, 0, .3])
        rows += [(2, 'right_l0', 'pip', grip[0]+[0, 0, .3+.002*t], 1.), (2, 'right_l1', 'pip', grip[1]+[0, 0, .3], 1.), (2, 'right_l2', 'pip', grip[2]+[0, 0, .3], 1.)]
        # world 3: the whole grasp slides 2 mm per step down the pipette (pipette slipping through the fingers) -> slip only, no drift
        env.xpos[3, ob] = T([0, 0, .3+.002*t])
        rows += [(3, f'right_l{i}', 'pip', grip[i]+[0, 0, .3], 1.) for i in range(3)]
        # world 4: thumb presses the plunger 1 mm per step (thumb point moves with the plunger), two fingers hold the body -> nothing
        q = -.001*t; env.qpos[4, m.jnt_qposadr[m.joint('pipette_plunger').id]] = q
        env.xpos[4, ob] = T([0, 0, .3]); env.xpos[4, pb] = T([0, 0, .36+q])
        rows += [(4, 'right_l0', 'pip', grip[0]+[0, 0, .3], 1.), (4, 'right_l1', 'pip', grip[1]+[0, 0, .3], 1.), (4, 'right_thumb', 'plg', [0, 0, .37+q], 1.)]
        set_contacts(env, m, rows)
        cur, _, _ = X.contacts(env)
        out = X.contact_temporal(env, cur); hist.append({k: out[k].numpy().round(4) for k in out})
        if t == 2:
            _, praw, _ = X.contacts(env); _, prest, _ = X.contacts(env, rest_plunger=True); th = LINKS.index('right_thumb')
            dz_raw, dz_rest = float(praw[4, th, 2]), float(prest[4, th, 2])
    for t, h in enumerate(hist): print(f'  step {t}: keep {h["contact_keep"]}  drift {h["contact_drift"]}  slip {h["contact_slip"]}')
    h0, h1, h2, h3 = hist
    assert all((h0[k] == 0).all() for k in h0), 'first step after start must cost nothing'
    for h in (h1, h2, h3):
        assert h['contact_keep'][0] == 0 and h['contact_drift'][0] < 1e-3 and h['contact_slip'][0] < 1e-3, 'rigid object motion was penalised'
        assert h['contact_drift'][4] < 1e-3 and h['contact_slip'][4] < 1e-3 and h['contact_keep'][4] == 0, 'plunger stroke was penalised'
        assert h['contact_drift'][3] < 1e-3 and abs(h['contact_slip'][3]-1.) < 1e-3, 'whole-grasp slip: 2 mm per step / 2 hands = 1'
        assert h['contact_drift'][2] > .1 and abs(h['contact_slip'][2]-2/3/2) < 1e-3, 'one finger sliding 2 mm of 3 links / 2 hands = 0.333'
    assert h1['contact_keep'][1] == 0 and h2['contact_keep'][1] == 0, 'force flicker above the release threshold counted as a loss'
    assert abs(h3['contact_keep'][1]-1/3/2) < 1e-3, 'one of three links lost / 2 hands'
    assert abs((dz_raw-dz_rest)-(-.002)) < 1e-6, 'rest_plunger should undo the 2 mm stroke at t = 2'
    X.reset(env, torch.tensor([0, 2])); env.xpos[2, ob] += T([0, 0, .002]); env.xpos[3, ob] += T([0, 0, .002])   # both grasps slip 2 mm more
    out = X.contact_temporal(env, X.contacts(env)[0])
    assert float(out['contact_slip'][2]) == 0 and abs(float(out['contact_slip'][3])-1.) < 1e-3, 'reset must clear the previous step for those worlds only'
    # temporal_off_gates: frames where the (fake) reference gate is on cost nothing, the state still advances
    env.xcfg = X.merge(env.xcfg, {'temporal_off_gates': ['gate_x']}); env.x_tgate = torch.tensor([False, True])
    env.xpos[3, ob] += T([0, 0, .002]); out = X.contact_temporal(env, X.contacts(env)[0], torch.tensor([0, 0, 0, 1, 0]))
    assert float(out['contact_slip'][3]) == 0, 'gated frame was penalised'
    env.xpos[3, ob] += T([0, 0, .002]); out = X.contact_temporal(env, X.contacts(env)[0], torch.zeros(5, dtype=torch.long))
    assert abs(float(out['contact_slip'][3])-1.) < 1e-3, 'state must keep advancing through a gated frame'


def test_margin_glue():
    N = 2; env, m = fake_env(N); ob = m.body('right_object').id
    env.xcfg = X.merge(env.xcfg, {'margin_gate': False}); F = 3
    wq = np.zeros((F, 2, 6)); wq[:, 0, 2] = 1.; wq[:, 1, 2] = 1.; env.x_cref = {'u': T(wq), 'wn': T(np.c_[np.full(F, .83), np.full(F, .13)])}
    env.x_xipos = torch.zeros(N, m.nbody, 3); env.x_xipos[:, ob] = T([0, 0, .3])
    rows = [(0, 'right_l0', 'pip', [-.01, 0, .3], 1.), (0, 'right_l1', 'pip', [.01, 0, .3], 1.), (0, 'right_l2', 'pip', [0, 0, .28], 1.),
            (1, 'right_l0', 'pip', [.03, 0, .29], 1.)]
    set_contacts(env, m, rows); env.x_cframe = torch.stack([T([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), T([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                                                             T([[0, 0, 1], [1, 0, 0], [0, 1, 0]]), T([[0, 0, 1], [1, 0, 0], [0, 1, 0]])])
    _, _, (nrm, pos, cw, cside) = X.contacts(env)
    pen, info = X.margin_term(env, torch.zeros(N, dtype=torch.long), nrm, pos, cw, cside)
    print(f'  margin_term: pinch+under pen {float(pen[0]):.3f} (d_req {float(info["_margin_req_pipette_N"][0]):.3f} N), '
          f'one finger off the CoM pen {float(pen[1]):.3f} (d_req {float(info["_margin_req_pipette_N"][1]):.3f} N)')
    assert float(info['_margin_req_pipette_N'][0]) < .02 and float(info['_margin_req_pipette_N'][1]) > .1 and float(pen[1]) > float(pen[0])


if __name__ == '__main__':
    torch.set_num_threads(4)
    test_cone_support(); test_residual_vs_exact(); print('wrench_margin toys (need 1 N up, <= 3 N per contact, rho 0.3 N):'); test_margin_toys()
    print('contact_temporal on a synthetic contact stream (worlds: rigid motion, flicker, one finger slides, grasp slides, plunger press):'); test_temporal()
    test_margin_glue(); print('PASS')
