"""TRACKER (11-10): checks of a task built by tracker/build_demo_task.py, before any training. Layer: EVAL (render + zero-residual rollouts).

  1 zero-residual rollouts in the RL env (tracker_env, holders at nominal, no DR), from evenly spaced reference frames, at VOC 1 (objects
    guided, the curriculum's start) and VOC 0: first failure time and reason per start. Pass = no failure at VOC 1 (a failure there means the
    reference itself is not executable: unreachable, too fast, through a holder). Failures at VOC 0 are expected (grip force, press force:
    what RL learns) and are only reported.
  2 montage of the kinematic reference (front and top-down views) at 8 times.
  3 overlay of the scene (rack, stand, objects) on the demo's RGB frames, seen from the demo camera: the ground-truth holder placement
    (first frame) and the mid-air pose (press window).
Writes TASK/check/{check.json, montage.png, overlay_start.png, overlay_press.png}.

  MUJOCO_GL=egl .venv-rl-head/bin/python -I tracker/check_demo_task.py tasks/task_<name> [--episode EPISODE_DIR] [--worlds 32]
"""
import argparse, json, math, os, subprocess, sys
from pathlib import Path
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np, mujoco, torch
from PIL import Image, ImageDraw
HERE = Path(__file__).resolve().parent; V = HERE.parent
sys.path[:0] = [str(HERE), str(V/'rl'), str(V/'common')]
CFG_TEMPLATE = HERE/'configs/kin_tc_s5.json'   # reward / extras of KA (holder randomisation switched off for the check)


def rollouts(task, voc, starts, worlds):
    import tracker_env as E
    C = json.loads(CFG_TEMPLATE.read_text()); ex = json.loads(json.dumps(C['extras'])); tj = json.loads((task/'task.json').read_text())
    h = (tj.get('ref_holes') or ['r0c3', 'r1c3'])[0]
    ex['holder'].update(stand_xy_mm=0, stand_yaw_deg=0, rack_xy_mm=0, rack_yaw_deg=0, meas_xy_mm=0, meas_yaw_deg=0, hole_pairs=[[h, h]])
    r = np.load(task/'reference.npz'); t = r['timestamps']; nfr = len(t); st = np.repeat(starts, worlds); N = len(st); TR = C['tracker']
    env = E.BimanualGpuEnv(task, N, seed=1, nconmax=200, njmax=1600, naconmax_per_world=400, starts=st, dr=None, extras=ex, reward_scale=.01,
                           group_scale=C['group_scale'], voc_scale=voc, voc_kp=TR['voc_kp'], voc_kd=TR['voc_kd'], voc_free_frac=0., terminate=False,
                           auto_reset=False, plunger_weight=C.get('plunger_weight'), thumb_press_scale=C.get('thumb_press_scale'))
    env.reset_envs(torch.ones(N, dtype=torch.bool, device=env.device), frames=torch.as_tensor(st, device=env.device))
    first = np.full(N, -1.); reason = [''] * N
    with torch.inference_mode():
        for _ in range(int(math.ceil((nfr-1-starts.min())/env.frame_step))):
            _, _, _, ex_ = env.step(torch.zeros(N, env.nu, device=env.device))
            rs = ex_['reasons'].cpu().numpy(); k = env.frame.clamp(max=nfr-1).cpu().numpy()
            for i in np.flatnonzero((rs > 0).any(1) & (first < 0)):
                first[i] = t[k[i]]; reason[i] = ','.join(('pipette:' if j == 0 else 'tube:')+E.VR.FAILURES[rs[i, j]-1] for j in (0, 1) if rs[i, j] > 0)
            if (env.frame >= nfr-1).all(): break
    out = {}
    from collections import Counter
    for s in starts:
        sel = np.flatnonzero(st == s); f = first[sel]
        out[f'{t[s]:.2f}'] = {'fail_frac': round(float((f >= 0).mean()), 3), 'first_fail_s_p50': round(float(np.median(f[f >= 0])), 2) if (f >= 0).any() else None,
                              'reasons': Counter([reason[i] for i in sel if first[i] >= 0]).most_common(3)}
    del env; torch.cuda.empty_cache()
    return {'fail_frac_all': round(float((first >= 0).mean()), 3), 'per_start_s': out}


def montage(task, out, times, W=480, H=360):
    m = mujoco.MjModel.from_xml_path(str(task/'scene.xml')); d = mujoco.MjData(m); r = np.load(task/'reference.npz'); t = r['timestamps']
    for i in range(m.ngeom):
        if m.geom_group[i] == 3 and int(m.geom_bodyid[i]) != 0: m.geom_group[i] = 4      # collision proxies on moving bodies
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, W); m.vis.global_.offheight = max(m.vis.global_.offheight, H)
    rd = mujoco.Renderer(m, H, W); opt = mujoco.MjvOption(); opt.geomgroup[:] = [1, 1, 1, 1, 0, 0]
    def cam(look, dist, az, el):
        c = mujoco.MjvCamera(); c.type = mujoco.mjtCamera.mjCAMERA_FREE; c.lookat[:] = look; c.distance = dist; c.azimuth = az; c.elevation = el; return c
    cams = [cam([.35, 0., .95], 1.3, 0, -22), cam([.30, 0., .86], .95, 180, -55)]   # from across the table; from behind the robot, top-down
    tiles = []
    for tk in times:
        k = int(np.argmin(np.abs(t-tk))); d.qpos[:] = r['qpos_ref'][k]; mujoco.mj_forward(m, d); row = []
        for c in cams:
            rd.update_scene(d, camera=c, scene_option=opt); im = Image.fromarray(rd.render()); dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, 200, 14], fill=(0, 0, 0)); dr.text((3, 2), f't={t[k]:.2f}s phase {int(r["phase"][k])}', fill=(255, 255, 255)); row.append(np.asarray(im))
        tiles.append(np.concatenate(row, 1))
    grid = [np.concatenate(tiles[i:i+2], 1) for i in range(0, len(tiles)-1, 2)]
    Image.fromarray(np.concatenate(grid, 0)).save(out)


def overlay(task, ep, t_ref, out):
    """the scene at reference time t_ref, rendered from the demo camera (pinhole, colour K), edges + tint over the demo RGB frame"""
    import build_demo_task as B
    from scipy.ndimage import binary_erosion
    tj = json.loads((task/'task.json').read_text()); a = tj['build']['args']
    TF = B.table_frame(ep); T_tc = B.T_(TF['R_tc'], -TF['R_tc']@TF['foot'])                 # camera pose in the table frame
    sp = mujoco.MjSpec.from_file(str(task/'scene.xml')); m0 = sp.compile()
    z_table = max(m0.geom_pos[g][2]+m0.geom_size[g][2] for g in range(m0.ngeom) if (m0.geom(g).name or '').startswith('furn_table'))
    T_wc = B.T_(B.rotz(math.pi), [a['base_x']-a['fwd'], a['base_y']-a['lat'], z_table])@T_tc
    K = TF['K']; W, H = 640, 480; R_mj = T_wc[:3, :3]@np.diag([1, -1, -1.]); q = np.zeros(4); mujoco.mju_mat2Quat(q, R_mj.ravel())
    sp.worldbody.add_camera(name='democam', pos=T_wc[:3, 3], quat=q, fovy=math.degrees(2*math.atan(H/2/K[1, 1])))
    m = sp.compile(); m.vis.global_.offwidth = max(m.vis.global_.offwidth, W); m.vis.global_.offheight = max(m.vis.global_.offheight, H)
    for i in range(m.ngeom):
        if m.geom_group[i] == 3 and int(m.geom_bodyid[i]) != 0: m.geom_group[i] = 4
    d = mujoco.MjData(m); r = np.load(task/'reference.npz'); k = int(np.argmin(np.abs(r['timestamps']-t_ref))); d.qpos[:] = r['qpos_ref'][k]; mujoco.mj_forward(m, d)
    rd = mujoco.Renderer(m, H, W); opt = mujoco.MjvOption(); opt.geomgroup[:] = [1, 1, 1, 1, 0, 0]
    rd.update_scene(d, camera='democam', scene_option=opt); rd.enable_segmentation_rendering(); seg = rd.render()[..., 0].copy()
    fr = int(round((r['demo_time'][k] if 'demo_time' in r.files else r['timestamps'][k])*30))   # (11-12) demo clock (retimed references)
    raw = subprocess.run(['ffmpeg', '-loglevel', 'error', '-ss', f'{fr/30:.3f}', '-i', str(ep/'reconstruction/rgb.mp4'), '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                         capture_output=True).stdout
    img = np.frombuffer(raw, np.uint8).reshape(H, W, 3).astype(float)
    dx, dy = int(round(K[0, 2]-W/2)), int(round(K[1, 2]-H/2))                                   # principal point offset of the real camera
    def mask(prefix=None, bodies=()):
        ids = [g for g in range(m.ngeom) if (prefix and (m.geom(g).name or '').startswith(prefix)) or m.body(int(m.geom_bodyid[g])).name in bodies]
        mk = np.isin(seg, ids); sh = np.zeros_like(mk)                                          # shift without wrap-around
        ys, xs = slice(max(dy, 0), H+min(dy, 0)), slice(max(dx, 0), W+min(dx, 0)); ys0, xs0 = slice(max(-dy, 0), H-max(dy, 0)), slice(max(-dx, 0), W-max(dx, 0))
        sh[ys, xs] = mk[ys0, xs0]; return sh
    for mk, col in ((mask(('furn_rackv_', 'furn_rack_plate_')), (0, 255, 255)), (mask(('furn_stand_',)), (255, 0, 255)),
                    (mask(None, ('right_object', 'pipette_plunger')), (255, 255, 0)), (mask(None, ('left_object',)), (0, 255, 0))):
        img[mk] = .75*img[mk]+.25*np.array(col); img[mk & ~binary_erosion(mk, iterations=1)] = col
    im = Image.fromarray(img.astype(np.uint8)); dr = ImageDraw.Draw(im); dr.rectangle([0, 0, 470, 14], fill=(0, 0, 0))
    dr.text((3, 2), f'demo frame {fr} (t={fr/30:.2f}s); cyan rack, magenta stand, yellow pipette, green tube (sim)', fill=(255, 255, 255)); im.save(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('task', type=Path); ap.add_argument('--episode', type=Path); ap.add_argument('--worlds', type=int, default=32)
    ap.add_argument('--no-rollouts', action='store_true'); a = ap.parse_args()
    task = a.task.resolve(); outd = task/'check'; outd.mkdir(exist_ok=True); r = np.load(task/'reference.npz'); t = r['timestamps']; tj = json.loads((task/'task.json').read_text())
    rep = {'task': str(task)}
    if not a.no_rollouts:
        starts = np.unique(np.linspace(0, len(t)-int(1.5*100), 9).astype(int))
        rep['zero_residual_voc1'] = rollouts(task, 1., starts, a.worlds); rep['zero_residual_voc0'] = rollouts(task, 0., starts, a.worlds)
        rep['pass_voc1'] = rep['zero_residual_voc1']['fail_frac_all'] == 0.
    ph = tj.get('phases', {}); ev = tj.get('build', {}).get('events_ref_s') or tj.get('build', {}).get('events_demo_s', {})   # (11-12) reference clock
    times = sorted({float(t[0])+.5, ev.get('pipette_lift', t[0])-.6, ev.get('pipette_lift', t[0])+1., ev.get('tube_lift', t[0])+.6,
                    (ph['press_on']+ph['press_off'])/2 if ph.get('press_on') else float(t[len(t)//2]), ev.get('pipette_seat', t[-1])+.2,
                    ev.get('tube_seat', t[-1])+.2, float(t[-1])-.2})
    montage(task, outd/'montage.png', times[:8])
    if a.episode:
        overlay(task, a.episode, float(t[0]), outd/'overlay_start.png')
        if ph.get('press_on'): overlay(task, a.episode, (ph['press_on']+ph['press_off'])/2, outd/'overlay_press.png')
    b = tj.get('build', {}); rep['build_summary'] = {k: b.get(k) for k in ('midair_search', 'ik', 'reference_penetrations_gt3mm', 'tip_in_tube', 'thumb_open_clearance_mm')}
    (outd/'check.json').write_text(json.dumps(rep, indent=1, default=float)+'\n')
    print(json.dumps({k: rep.get(k) for k in ('pass_voc1',)} | ({'voc1_fail': rep['zero_residual_voc1']['fail_frac_all'], 'voc0_fail': rep['zero_residual_voc0']['fail_frac_all']}
                                                                if 'zero_residual_voc1' in rep else {})), flush=True)
    print('written', outd)


if __name__ == '__main__':
    main()
