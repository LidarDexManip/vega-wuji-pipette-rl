"""(11-09) Wrist (end-effector) path quality of recorded rollouts (record_rollouts.py world_<i>.npz) vs the reference they track.
Separates where back-and-forth motion comes from:
  ref     = the demo wrist path (reference.npz qpos_ref, FK), re-targeted like the env does for the world's rack hole pair
            (left wrist + w_rack(k) x hole offset; rack / stand offsets of the world spec must be 0, which the video worlds are);
  policy  = the rollout wrist path; dev = policy - ref.
Per side, on the wrist position low-passed at LP Hz (zero phase; removes the sub-mm shake so only visible motion counts):
  path_m        path length;  excess = path / ref path - 1
  reversals     direction reversals of the motion: local extrema of each axis with prominence >= PROM mm, summed over x y z
  speed_peaks   local maxima of the speed with prominence >= 2 cm/s (one per smooth point-to-point move)
  dev rms mm    of dev in the bands 0.2-1 Hz (slow wandering), 1-3 Hz (back-and-forth), 3-8 Hz; dev max mm
  MUJOCO_GL=egl python ee_path.py SCENE.xml REFERENCE.npz --run DIR LABEL [...] [--worlds 0 1 2 3] [--plot OUT.png --plot-world 2 (+ OUT.npz traces)] [--json OUT]"""
import argparse, json, re
from pathlib import Path
import numpy as np, mujoco
from scipy.signal import butter, sosfiltfilt, find_peaks

p = argparse.ArgumentParser(); p.add_argument('scene', type=Path); p.add_argument('ref', type=Path)
p.add_argument('--run', nargs=2, action='append', required=True, metavar=('DIR', 'LABEL')); p.add_argument('--worlds', type=int, nargs='+', default=[0, 1, 2, 3])
p.add_argument('--lp', type=float, default=3.); p.add_argument('--prom', type=float, default=5.); p.add_argument('--plot', type=Path); p.add_argument('--plot-world', type=int, default=2)
p.add_argument('--json', type=Path); a = p.parse_args()
m = mujoco.MjModel.from_xml_path(str(a.scene)); d = mujoco.MjData(m); r = np.load(a.ref); Qr = r['qpos_ref']; nf = len(Qr)
WB = {s: m.body(f'{s}_wrist').id for s in ('right', 'left')}; FS = 50.


def wrists(Q):
    out = {s: np.zeros((len(Q), 3)) for s in WB}
    for i, q in enumerate(Q):
        d.qpos[:] = q; mujoco.mj_kinematics(m, d)
        for s, b in WB.items(): out[s][i] = d.xpos[b]
    return out


# ---- rack weight and hole offsets, as vega_extras.holder_init (engage_fix, blend 1.5 s, hole_centres 'scene')
d.qpos[:] = Qr[0]; mujoco.mj_kinematics(m, d); acc = {}
for i in range(m.ngeom):
    mm = re.match(r'furn_rackv_low_(r\d+c\d+)_s\d+$', m.geom(i).name or '')
    if mm: acc.setdefault(mm.group(1), []).append(d.geom_xpos[i, :2].copy())
cen = {k: np.mean(v, 0) for k, v in acc.items()}
gate = r['gate_tube_in_rack'].astype(bool); fr = np.arange(nf); gi = np.flatnonzero(gate); gap = np.flatnonzero(np.diff(gi) > 1)
k_mid = int((gi[gap[0]]+gi[gap[0]+1])//2) if len(gap) else nf//2
oq = int(m.jnt_qposadr[m.joint('left_object_joint').id]); z = Qr[:, oq+2]; xy = Qr[:, oq:oq+2]
segs = [gi[:gap[0]+1], gi[gap[0]+1:]] if len(gap) else [gi]; eng = gate.copy()
for si, sg in enumerate(segs):
    zr = float(np.median(z[sg])); xyr = np.median(xy[sg], 0); part = (fr < k_mid) if si == 0 else (fr >= k_mid)
    eng |= part & (z-zr < .0533+.015) & (np.linalg.norm(xy-xyr, axis=1) < .015)
idx = np.flatnonzero(eng); w_rack = np.clip(1-np.min(np.abs(fr[:, None]-idx[None]), 1)/150., 0, 1)
P_demo = wrists(Qr)

sos_lp = butter(4, a.lp, fs=FS, output='sos')
BANDS = {'0.2-1Hz': (.2, 1.), '1-3Hz': (1., 3.), '3-8Hz': (3., 8.)}
sos_b = {k: butter(2, v, btype='band', fs=FS, output='sos') for k, v in BANDS.items()}


def metrics(P, n):
    P = P[:n]; L = sosfiltfilt(sos_lp, P, axis=0); v = np.gradient(L, 1/FS, axis=0); sp = np.linalg.norm(v, axis=1)
    rev = sum(len(find_peaks(sgn*L[:, j]*1e3, prominence=a.prom)[0]) for j in range(3) for sgn in (1, -1))
    return dict(path_m=float(np.sum(np.linalg.norm(np.diff(L, axis=0), axis=1))), reversals=int(rev),
                speed_peaks=int(len(find_peaks(sp, prominence=.02)[0]))), L, sp


res = {}; plot = {}
for W in a.worlds:
    for dir_, lab in a.run:
        dir_ = Path(dir_); zz = np.load(dir_/f'world_{W}.npz'); info = json.loads((dir_/'summary.json').read_text())['worlds'][W]; sp_ = info['spec']
        assert not any(sp_['rack']) and not any(sp_['stand']), 'only worlds with zero rack / stand offsets (the reference re-targeting here covers the hole pair only)'
        pa, pb = sp_['pair']; off = np.where((fr < k_mid)[:, None], cen[pa]-cen['r0c3'], cen[pb]-cen['r1c3'])*w_rack[:, None]
        ref = {s: P_demo[s].copy() for s in WB}; ref['left'][:, :2] += off
        ref = {s: v[::2] for s, v in ref.items()}   # 100 Hz reference frames -> the 50 Hz control steps
        P = wrists(zz['qpos']); n = len(P['right'])
        if info['first_fail_s'] > 0: n = min(n, int(info['first_fail_s']*FS))   # up to the failure only
        for s in WB:
            mr, Lr, spr = metrics(ref[s], n); mp, Lp, spp = metrics(P[s], n); dev = (P[s][:n]-ref[s][:n])*1e3
            row = {'ref': mr, 'policy': mp, 'excess_path': round(mp['path_m']/mr['path_m']-1, 3),
                   'dev_rms_mm': {k: round(float(np.sqrt(np.mean(np.sum(sosfiltfilt(sb, dev, axis=0)**2, 1)))), 2) for k, sb in sos_b.items()},
                   'dev_max_mm': round(float(np.max(np.linalg.norm(dev, axis=1))), 1), 'steps': n, 'pair': [pa, pb]}
            res.setdefault(lab, {}).setdefault(f'world{W}', {})[s] = row
            if W == a.plot_world: plot.setdefault(s, {'ref': (Lr, spr)})[lab] = (Lp, spp)

# ---- table
print(f'low-pass {a.lp} Hz; reversals = extrema with prominence >= {a.prom} mm (x y z summed); speed peaks prominence >= 2 cm/s')
for s in WB:
    print(f'\n{s} wrist' + ' '*14 + ''.join(f'{"w"+str(W):>34s}' for W in a.worlds))
    hdr = 'path m (excess) | rev ref/pol | speed pk ref/pol | dev 1-3Hz mm'
    for lab in res:
        cells = []
        for W in a.worlds:
            x = res[lab][f'world{W}'][s]
            cells.append(f'{x["policy"]["path_m"]:.2f}({x["excess_path"]:+.0%}) {x["ref"]["reversals"]:>2d}/{x["policy"]["reversals"]:<2d} {x["ref"]["speed_peaks"]:>2d}/{x["policy"]["speed_peaks"]:<2d} {x["dev_rms_mm"]["1-3Hz"]:5.2f}')
        print(f'  {lab:22s}' + ''.join(f'{c:>34s}' for c in cells))
    print('  cells: ' + hdr)
if a.json: a.json.write_text(json.dumps(res, indent=1)+'\n')

if a.plot:   # traces -> OUT.npz; the figure needs matplotlib (not in .venv-rl-head): ee_path_plot.py draws it from the npz with any python
    np.savez(a.plot.with_suffix('.npz'), world=a.plot_world, lp=a.lp, labels=np.array(list(plot['right'])),
             **{f'{s}|{lab}|{k}': v for s in plot for lab, tr in plot[s].items() for k, v in zip(('pos', 'speed'), tr)})
    try:
        import ee_path_plot; ee_path_plot.draw(a.plot.with_suffix('.npz'), a.plot)
    except ImportError: print(f'matplotlib missing here: python3 ee_path_plot.py {a.plot.with_suffix(".npz")} {a.plot}')
