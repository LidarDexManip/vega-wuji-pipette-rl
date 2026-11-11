"""(11-10) Is the object's position ALONG its own long axis observable from the wrist camera? Render-only companion of wristcam_sim.py
(same camera mounts, same recorded rollouts).

1. Image sensitivity: the full scene is rendered with the held object at its true pose and shifted by --shift mm (a) along its long
   axis, (b) sideways (across the axis, in the image plane), (c) along the camera ray. Counted: pixels whose depth changes by more
   than 1 mm. If (a) changes few pixels, the depth image barely constrains the axial position.
2. ICP from a pure offset (--shift mm along the axis, or sideways), noise-free depth, perfect segmentation: how much of the offset the
   wristcam_sim.py ICP removes. Separates "not in the image" from "the fitter does not use it".
  MUJOCO_GL=egl python perception/wristcam_axial.py SCENE.xml OUT --run DIR LABEL --worlds 8 [--mount forearm] [--example WORLD T SIDE ...]
Writes OUT/axial_stats.json and OUT/axial_example_<i>.png.
"""
import argparse, json, math
from pathlib import Path
import numpy as np
import wristcam_sim as WS


def free_adr(m, body):
    j = int(m.body_jntadr[m.body(body).id]); return int(m.jnt_qposadr[j])


def shifted_depth(cam, side, delta):
    """Full-scene depth with the side's object (and, for the pipette, its plunger) moved by delta (world, m)."""
    import mujoco
    m, d = WS._G['m'], WS._G['d']; a = free_adr(m, WS.OBJ[side]); q = d.qpos.copy()
    d.qpos[a:a+3] += delta; mujoco.mj_kinematics(m, d); mujoco.mj_camlight(m, d)
    z = WS.render(cam, {1, 2, 3}, 'depth'); d.qpos[:] = q; mujoco.mj_kinematics(m, d); mujoco.mj_camlight(m, d); return z


def directions(cam, side):
    m, d = WS._G['m'], WS._G['d']; ob = m.body(WS.OBJ[side]).id; ax = d.xmat[ob].reshape(3, 3)[:, 2].copy()
    c = m.camera(cam).id; ray = d.xpos[ob]-d.cam_xpos[c]; ray /= np.linalg.norm(ray)
    side_ = np.cross(ax, ray); side_ /= np.linalg.norm(side_)          # across the axis, in the image plane
    along_ray = ray-(ray@ax)*ax; along_ray /= np.linalg.norm(along_ray)  # across the axis, towards the camera
    return {'along the axis': ax, 'sideways': side_, 'along the camera ray': along_ray}


def sensitivity(cam, side, mm):
    z0 = WS.render(cam, {1, 2, 3}, 'depth'); vis = WS.segmask(WS.render(cam, {1, 2, 3}, 'seg'), WS._G['fitg'][side])
    out = {'object_px': int(vis.sum())}; maps = {}
    for k, u in directions(cam, side).items():
        z1 = shifted_depth(cam, side, mm*1e-3*u); dz = np.abs(np.clip(z1, 0, 2)-np.clip(z0, 0, 2))
        out[k] = int((dz > .001).sum()); maps[k] = dz
    return out, z0, vis, maps


def icp_from(side, cam, offset):
    import mujoco
    m, d = WS._G['m'], WS._G['d']; ob = m.body(WS.OBJ[side]).id; Rg, tg = d.xmat[ob].reshape(3, 3).copy(), d.xpos[ob].copy()
    z = WS.render(cam, {1, 2, 3}, 'depth'); seg = WS.render(cam, {1, 2, 3}, 'seg')
    mask = WS.segmask(seg, WS._G['fitg'][side]) & (z >= WS.ZMIN_FFS)
    P = WS.backproject(z, mask, cam)
    if len(P) < 50: return None
    R, t, _ = WS.icp(P, WS._G['T'][side], WS._G['M'][side], Rg.copy(), tg+offset)
    t2 = WS.axial_search(cam, side, R, t, z, z >= WS.ZMIN_FFS)
    return dict(icp=WS.errors(side, R, t, Rg, tg), icp_render=WS.errors(side, R, t2, Rg, tg))


if __name__ == '__main__':
    import mujoco
    p = argparse.ArgumentParser(); p.add_argument('scene', type=Path); p.add_argument('out', type=Path)
    p.add_argument('--run', nargs=2, required=True); p.add_argument('--worlds', type=int, default=8); p.add_argument('--mount', default='forearm')
    p.add_argument('--shift', type=float, default=3.); p.add_argument('--stride', type=int, default=25)
    p.add_argument('--example', nargs=3, action='append', default=[], metavar=('WORLD', 'T', 'SIDE')); a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    WS.init(a.scene, {a.mount: WS.MOUNTS[a.mount]}, (0., 0.)); m, d = WS._G['m'], WS._G['d']; run = Path(a.run[0])
    HELD = {'right': ('grasp / lift', 'carry', 'tip in tube / press', 'carry back', 'insert into holder'),
            'left': ('grasp / lift', 'carry', 'tip in tube / press', 'carry back', 'insert into holder')}

    def load(w):
        z = np.load(run/f'world_{w}.npz'); info = json.loads((run/'summary.json').read_text())['worlds'][w]; WS.place_holders(info.get('spec'))
        return z['qpos'], z['time'], (info['first_fail_s'] if info['first_fail_s'] >= 0 else 1e9)

    def setq(q):
        d.qpos[:] = q; mujoco.mj_kinematics(m, d); mujoco.mj_camlight(m, d)

    stats = {s: [] for s in ('right', 'left')}
    for w in range(a.worlds):
        Q, T, tf = load(w)
        for k in range(0, len(Q), a.stride):
            if T[k] >= tf: break
            setq(Q[k])
            for side in ('right', 'left'):
                if WS.phase_of(side, float(T[k])) not in HELD[side]: continue
                cam = f'{a.mount}_{side}'; s, *_ = sensitivity(cam, side, a.shift)
                dirs = directions(cam, side)
                s['icp_from_axial'] = icp_from(side, cam, a.shift*1e-3*dirs['along the axis'])
                s['icp_from_sideways'] = icp_from(side, cam, a.shift*1e-3*dirs['sideways'])
                stats[side].append(dict(world=w, t=float(T[k]), **s))
    summ = {}
    for side, L in stats.items():
        e = {'frames': len(L), 'object_px_p50': float(np.median([x['object_px'] for x in L]))}
        for k in ('along the axis', 'sideways', 'along the camera ray'): e[f'px_changed_p50 ({k})'] = float(np.median([x[k] for x in L]))
        for k in ('icp_from_axial', 'icp_from_sideways'):
            ok = [x[k] for x in L if x[k]]
            e[k] = {f'{v} {c}_mm_p50/p90': [round(float(np.median([o[v][c] for o in ok])), 2), round(float(np.percentile([o[v][c] for o in ok], 90)), 2)]
                    for v in ('icp', 'icp_render') for c in ('axial', 'lat')}
        summ[side] = e
    (a.out/'axial_stats.json').write_text(json.dumps(dict(shift_mm=a.shift, mount=a.mount, run=a.run, summary=summ, frames=stats), indent=1))
    print(json.dumps(summ, indent=1))

    # example figures: depth with the object highlighted | |dz| for each shift direction
    from PIL import Image, ImageDraw
    for i, (w, t, side) in enumerate(a.example):
        Q, T, _ = load(int(w)); k = int(np.argmin(np.abs(T-float(t)))); setq(Q[k]); cam = f'{a.mount}_{side}'
        s, z0, vis, maps = sensitivity(cam, side, a.shift)
        zc = np.clip((z0-.05)/(.30-.05), 0, 1); base = np.stack([(255*(1-zc)).astype(np.uint8)]*3, -1); base[vis] = (.4*base[vis]+.6*np.array([60, 200, 60])).astype(np.uint8)
        tiles = [base]
        for kk, dz in maps.items():
            im = (base*.35).astype(np.uint8); hot = dz > .001; c = np.clip(dz/(a.shift*1e-3), 0, 1)
            im[hot] = np.stack([255*np.ones_like(c), 255*(1-c), 40*np.ones_like(c)], -1)[hot].astype(np.uint8); tiles.append(im)
        img = Image.fromarray(np.concatenate(tiles, 1)); dr = ImageDraw.Draw(img)
        dr.text((4, 2), f'{side} ({"pipette" if side == "right" else "tube"}) t={T[k]:.1f}s  {WS.phase_of(side, float(T[k]))}', fill=(255, 255, 0))
        dr.text((4, 14), f'object px {s["object_px"]}', fill=(255, 255, 0))
        for j, kk in enumerate(maps): dr.text((WS.W*(j+1)+4, 2), f'shift {a.shift:g} mm {kk}', fill=(255, 255, 0)); dr.text((WS.W*(j+1)+4, 14), f'depth changed (>1 mm): {s[kk]} px', fill=(255, 255, 0))
        img = img.resize((img.size[0]*2, img.size[1]*2), Image.NEAREST); img.save(a.out/f'axial_example_{i}.png')
