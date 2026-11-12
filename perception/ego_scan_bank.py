"""(11-12) Head-camera SCAN BANK for a learned holder estimate (the "initial depth image only" student).

Renders the start-of-episode head-camera depth frame exactly as ego_holders.characterize does (same scene randomisation: holders +-5 cm /
+-10 deg, a random pick hole, object slack 2 mm / 3 deg; arms at the reference's first frame; ZED-like stereo noise; per-scene head-camera
calibration error) and keeps a fixed crop around both holders, 2x downsampled (int16 mm, 0 = invalid), with the scene's ground truth.

Splits (each its own scenes): train (--noisy noise draws per scene), val, policy (scenes the scan network never sees in training: their
predictions / embeddings become the per-episode measurement bank of the student, so the student trains on the network's real test errors).

  MUJOCO_GL=egl .venv-rl-head/bin/python -I perception/ego_scan_bank.py tasks/task_d02_straight_v3 rl/runs/ego_scan --n-train 40000 --procs 16
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ego_holders as EH

CROP = dict(u0=280, u1=680, v0=280, v1=490, down=2)   # full-res pixel window around both holders (+ DR margin), then down x down


def crop_down(z, c=CROP):
    """crop, then average the valid pixels of each down x down block (a block with < half valid -> 0); returns int16 mm"""
    z = z[c['v0']:c['v1'], c['u0']:c['u1']]; k = c['down']; H, W = z.shape[0]//k, z.shape[1]//k
    b = z[:H*k, :W*k].reshape(H, k, W, k); v = b > 0; n = v.sum((1, 3)); s = np.where(v, b, 0.).sum((1, 3))
    out = np.where(n*2 >= k*k, s/np.maximum(n, 1), 0.); return np.round(out*1e3).astype(np.int16)


def _job(args):
    i, seed, nnoisy = args; rng = np.random.default_rng(seed); M, K = EH._W['M'], EH._W['K']
    s = EH.sample_scene(M, rng); z, T, truth = EH.render_scene(s, noisy=False)
    zc = z[CROP['v0']:CROP['v1'], CROP['u0']:CROP['u1']]   # noise on the crop (same model; the crop keeps its own K principal point irrelevant for fB)
    imgs = []
    for k in range(nnoisy):
        zn = EH.stereo_noise(zc, K, np.random.default_rng(seed*7+k+1)); full = np.zeros_like(z); full[CROP['v0']:CROP['v1'], CROP['u0']:CROP['u1']] = zn
        imgs.append(crop_down(full))
    lab = dict(stand=s['stand'], rack=s['rack'], hole=list(M.holes).index(s['hole']), tube=truth['tube'].tolist(), pipette=truth['pipette'].tolist(),
               pip_slack=s['pip_slack'], tube_slack=s['tube_slack'], cal=s['cal'])
    return i, np.stack(imgs), lab


def build(task, out, n, nnoisy, procs, seed, name):
    import multiprocessing as mp
    out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    H = (CROP['v1']-CROP['v0'])//CROP['down']; W = (CROP['u1']-CROP['u0'])//CROP['down']
    D = np.zeros((n, nnoisy, H, W), np.int16); L = [None]*n
    with mp.get_context('spawn').Pool(procs, initializer=EH._worker_init, initargs=(str(task),)) as pool:
        for k, (i, im, lab) in enumerate(pool.imap_unordered(_job, [(i, seed*1000003+i, nnoisy) for i in range(n)], chunksize=16)):
            D[i] = im; L[i] = lab
            if k % 5000 == 0: print(name, k, '/', n, round(time.time()-t0, 1), 's', flush=True)
    M = EH.Model(task); q0 = M.q0; tube0 = q0[M.adr['left_object_joint']:M.adr['left_object_joint']+3]; pip0 = q0[M.adr['right_object_joint']:M.adr['right_object_joint']+3]
    arr = lambda k: np.array([l[k] for l in L], np.float32)
    np.savez(out/f'{name}.npz', depth=D, stand=arr('stand'), rack=arr('rack'), hole=np.array([l['hole'] for l in L], np.int64), tube=arr('tube'), pipette=arr('pipette'),
             tube0=tube0.astype(np.float32), pipette0=pip0.astype(np.float32), pip_slack=arr('pip_slack'), tube_slack=arr('tube_slack'),
             holes=np.array(list(M.holes)), crop=json.dumps(CROP), cam=json.dumps(EH.CAM), neck=np.array(EH.NECK), noise=json.dumps(EH.NOISE), cal_err=json.dumps(EH.CAL_ERR))
    print(name, 'done', n, 'scenes x', nnoisy, 'in', round(time.time()-t0, 1), 's ->', out/f'{name}.npz', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('task', type=Path); ap.add_argument('out', type=Path)
    ap.add_argument('--n-train', type=int, default=40000); ap.add_argument('--n-val', type=int, default=2000); ap.add_argument('--n-policy', type=int, default=20000)
    ap.add_argument('--noisy', type=int, default=2); ap.add_argument('--procs', type=int, default=16); a = ap.parse_args()
    build(a.task, a.out, a.n_val, 1, a.procs, 1, 'val')
    build(a.task, a.out, a.n_policy, 1, a.procs, 2, 'policy')
    build(a.task, a.out, a.n_train, a.noisy, a.procs, 3, 'train')
