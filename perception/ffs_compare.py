"""Live comparison: the D405's on-chip depth vs Fast-FoundationStereo (NVlabs, CVPR 2026) run on the D405's own left / right images.

  <ffs-dir>/.venv/bin/python perception/ffs_compare.py [--ckpt 23-36-37] [--iters 8] [--preset Default] [--filters]

Runs in the Fast-FoundationStereo venv (torch 2.6 cu124, its own deps; the RL venv is untouched). Checkpoints (research licence, non-commercial):
<ffs-dir>/weights/weights/{23-36-37, 20-26-39, 20-30-48, 15-44-51}; 23-36-37 is the most accurate, 20-30-48 the fastest.
RTX 4080, 640x480, PyTorch fp16 + Triton cost volume (11-09): 23-36-37 36 ms (8 iters) / 29 ms (4); 20-30-48 25 ms (8) / 21 ms (4).
--net-size runs the network on a downscaled pair (the camera stays at --res; depth is upsampled back, nearest). Jetson-sized setting:
  ... ffs_compare.py --ckpt 20-30-48 --iters 4 --net-size 256x192 --max-disp 64     (near limit 4.4 cm, 24 GFLOP vs 253 at 640x480;
on the saved hand / rack snapshots median 1.2-1.6 mm from the 640x480 result, 11-09). max_disp is in network-input pixels.

Window (2x2): colour | D405 on-chip depth
              Fast-FoundationStereo depth | |difference| (mm) where both have depth inside the colour range
Neither is ground truth: the difference panel shows where they disagree, not which one is right.
Keys   q / Esc quit | s snapshot (view, colour, left / right, both depths .npy in m) | f toggle librealsense spatial + temporal filters on the
       D405 depth | [ ] colour range max -/+ 1 cm (D405) / 5 cm | ; ' min -/+ 1 cm
Mouse  hover: both depths under the cursor and their temporal noise (std over the last 15 frames, 7x7 patch).
Inputs: the D405's rectified left / right imagers (Infrared 1 / 2, y8, tiled to 3 channels), depth = fx B / disparity with fx, B from the
camera's calibration; pixels whose match would fall outside the right image are dropped. The on-chip depth uses the same frame
(synchronised frameset) and is in the left imager's pixel grid, as is the colour image, so all panels are pixel-aligned.
Near limits at 640x480 (fx 390, B 18 mm): on-chip >= 5.6 cm (126 px search, --shift 0); Fast-FoundationStereo >= 3.7 cm (max_disp 192).
D435 / D435i (11-09): same, from its two IR imagers (B 50 mm); its colour camera is a separate sensor (not pixel-aligned), so the first
panel shows the left IR image instead and colour is not streamed. --emitter on|off switches the IR dot projector (on helps the on-chip
matcher; the dots are also in the images FFS sees).
Defaults follow the working ranges (11-10): D405 on the wrist 7-15 cm -> colour range 5-18 cm, difference 0-5 mm, max_disp 192 (near
limit 3.7 cm); D435i 40-80 cm -> 35-85 cm, 0-10 mm, max_disp 128 (near limit 15 cm; same result as 192, 6 ms faster). Model 23-36-37 x8
(29 ms per frame on the D435i at 640x480); offline best quality into a ROS 2 bag: ffs_bag.py.
"""
import argparse, collections, os, sys, time
from pathlib import Path
import numpy as np, cv2, torch, pyrealsense2 as rs


def colorize(z, lo, hi):
    v = z > 0; u8 = (np.clip((z-lo)/max(hi-lo, 1e-3), 0, 1)*255).astype(np.uint8)
    vis = cv2.applyColorMap(255-u8, cv2.COLORMAP_TURBO); vis[v & (z > hi)] = (70, 70, 70); vis[~v] = 0
    return vis


def text(img, s, org, scale=.5):
    """White text with a black outline. The outline is the same thin text drawn at 1 px offsets: a thicker stroke would also widen
    OpenCV's glyph advance, so the outline drifts right of the text along the line (the ghosting seen 11-09)."""
    x, y = org
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
        cv2.putText(img, s, (x+dx, y+dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def patch_noise(hist, x, y, r=3):
    if not hist: return None, 0.
    P = np.stack([h[max(0, y-r):y+r+1, max(0, x-r):x+r+1] for h in hist]); ok = (P > 0).all(0)
    return (float(np.median(P.std(0)[ok]))*1e3 if ok.any() else None), float((P > 0).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ffs-dir', type=Path, default=Path.home()/'Projects/Fast-FoundationStereo')
    ap.add_argument('--ckpt', default='23-36-37'); ap.add_argument('--iters', type=int, default=8); ap.add_argument('--max-disp', type=int, default=None, help='D405 192, others 128')
    ap.add_argument('--res', default='640x480', help='camera resolution'); ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--net-size', default=None, help='network input WxH (default: --res), e.g. 256x192; padded to multiples of 32')
    ap.add_argument('--preset', default='Default'); ap.add_argument('--shift', type=int, default=0, help='on-chip disparity shift (written before the start)')
    ap.add_argument('--filters', action='store_true', help='librealsense spatial + temporal filters on the on-chip depth')
    ap.add_argument('--min', type=float, default=None, help='near end of the colour range, m (D405 0.05, others 0.35)'); ap.add_argument('--max', type=float, default=None, help='far end, m (D405 0.18, others 0.85)')
    ap.add_argument('--diff-max-mm', type=float, default=None, help='difference colour scale (D405 10, others 30)')
    ap.add_argument('--emitter', choices=['on', 'off'], default='on', help='IR dot projector (D435 / D435i)')
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parent/'snapshots')
    ap.add_argument('--snapshot-after', type=int, default=0); ap.add_argument('--snapshot-path', type=Path, default=Path('ffs_compare.png'))
    ap.add_argument('--frames', type=int, default=0, help='exit after N frames (0 = until q)')
    a = ap.parse_args(); W, H = map(int, a.res.lower().split('x')); w, h = map(int, (a.net_size or a.res).lower().split('x'))
    devs0 = rs.context().query_devices()
    if len(devs0) == 0: raise SystemExit('no RealSense camera found')
    model_name = devs0[0].get_info(rs.camera_info.name).replace('Intel RealSense ', ''); use_color = 'D405' in model_name   # D405: colour = left imager
    if a.min is None: a.min = .05 if use_color else .35                 # working ranges (11-10): D405 on the wrist 7-15 cm, D435i 40-80 cm
    if a.max is None: a.max = .18 if use_color else .85
    if a.diff_max_mm is None: a.diff_max_mm = 5. if use_color else 10.
    if a.max_disp is None: a.max_disp = 192 if use_color else 128      # near limit fx B / max_disp at 640x480: D405 3.7 cm, D435i 15 cm
    step = .01 if use_color else .05                                    # [ ] step of the colour range's far end
    st = {'filters': a.filters, 'lo': a.min, 'hi': a.max}

    # ------------------------------------------------------------------ network
    sys.path.insert(0, str(a.ffs_dir))
    from core.utils.utils import InputPadder                         # noqa: E402 (Fast-FoundationStereo code)
    ck = next((p for p in (a.ffs_dir/'weights/weights'/a.ckpt, a.ffs_dir/'weights'/a.ckpt) if (p/'model_best_bp2_serialize.pth').exists()), None)
    if ck is None: raise SystemExit(f'no checkpoint {a.ckpt} under {a.ffs_dir}/weights')
    torch.backends.cudnn.benchmark = True; torch.autograd.set_grad_enabled(False)
    model = torch.load(ck/'model_best_bp2_serialize.pth', map_location='cpu', weights_only=False)   # NVlabs' serialized model (full pickle)
    model.args.valid_iters = a.iters; model.args.max_disp = a.max_disp; model.cuda().eval()
    padder = InputPadder((1, 3, h, w), divis_by=32, force_square=False)

    def ffs(L, R):
        """Disparity (h, w) in network-input pixels."""
        if (w, h) != (W, H): L, R = (cv2.resize(I, (w, h), interpolation=cv2.INTER_AREA) for I in (L, R))
        t = torch.from_numpy(np.stack([L, R])).cuda().float()        # (2, h, w) 0..255; the model normalises inside
        i0, i1 = padder.pad(t[0][None, None].repeat(1, 3, 1, 1), t[1][None, None].repeat(1, 3, 1, 1))
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            d = model.forward(i0, i1, iters=a.iters, test_mode=True, optimize_build_volume='triton')
        return padder.unpad(d.float())[0, 0]

    print(f'Fast-FoundationStereo {a.ckpt}, {a.iters} iters, max_disp {a.max_disp}, network input {w}x{h}: warming up (Triton autotune) ...', flush=True)
    dummy = np.random.randint(0, 255, (H, W), np.uint8)
    for _ in range(5): ffs(dummy, dummy)
    torch.cuda.synchronize()

    # ------------------------------------------------------------------ camera
    cfg = rs.config()
    for i in (1, 2): cfg.enable_stream(rs.stream.infrared, i, W, H, rs.format.y8, a.fps)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, a.fps)
    if use_color: cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, a.fps)

    def start():
        """Preset and disparity shift go to the idle camera (with colour on, the D405 ignores depth-table writes while streaming)."""
        devs = rs.context().query_devices()
        if len(devs) == 0: raise RuntimeError('no RealSense device found')
        s = devs[0].first_depth_sensor(); r = s.get_option_range(rs.option.visual_preset)
        names = {s.get_option_value_description(rs.option.visual_preset, float(i)): i for i in range(int(r.min), int(r.max)+1)}
        if a.preset in names: s.set_option(rs.option.visual_preset, names[a.preset])
        am = rs.rs400_advanced_mode(devs[0])
        if am.is_enabled(): t = am.get_depth_table(); t.disparityShift = a.shift; am.set_depth_table(t)
        if s.supports(rs.option.emitter_enabled): s.set_option(rs.option.emitter_enabled, 1. if a.emitter == 'on' else 0.)
        p = rs.pipeline(); prof = p.start(cfg)
        k = prof.get_stream(rs.stream.infrared, 1).as_video_stream_profile().get_intrinsics()
        B = abs(prof.get_stream(rs.stream.infrared, 1).get_extrinsics_to(prof.get_stream(rs.stream.infrared, 2)).translation[0])
        msg = ''
        if use_color: msg = f', colour vs left imager: {np.linalg.norm(prof.get_stream(rs.stream.infrared, 1).get_extrinsics_to(prof.get_stream(rs.stream.color)).translation)*1e3:.2f} mm offset'
        emi = f', emitter {a.emitter}' if s.supports(rs.option.emitter_enabled) else ''
        print(f'{model_name}: {W}x{H}@{a.fps}, preset {a.preset}, shift {a.shift}{emi}, fx {k.fx:.1f}, baseline {B*1e3:.2f} mm{msg}', flush=True)
        return p, s.get_depth_scale(), k.fx*B

    def make_filters():
        sp = rs.spatial_filter(); sp.set_option(rs.option.filter_magnitude, 2); sp.set_option(rs.option.filter_smooth_alpha, .5); sp.set_option(rs.option.filter_smooth_delta, 20)
        tp = rs.temporal_filter(); tp.set_option(rs.option.filter_smooth_alpha, .4); tp.set_option(rs.option.filter_smooth_delta, 20)
        return [rs.disparity_transform(True), sp, tp, rs.disparity_transform(False)]

    pipe, scale, fxB = start(); filters = make_filters()
    win = f'{model_name} on-chip vs Fast-FoundationStereo  (q quit, s snapshot, f filters, [ ] ; \' range)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL); cv2.resizeWindow(win, min(1400, 2*W), int(min(1400, 2*W)*H/W))
    mouse = [W//2, H//2]

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_MOUSEMOVE: mouse[0], mouse[1] = min(max(x % W, 0), W-1), min(max(y % H, 0), H-1)
    cv2.setMouseCallback(win, on_mouse)
    xx = np.arange(w, dtype=np.float32)[None]
    h_rs, h_ff = collections.deque(maxlen=15), collections.deque(maxlen=15)
    n, fps, t_prev, ms_avg = 0, 0., time.time(), 0.
    try:
        while True:
            try: fs = pipe.wait_for_frames(3000)
            except RuntimeError as e:
                print(f'{time.strftime("%H:%M:%S")} no frames ({e}); restarting the camera', flush=True)
                try: pipe.stop()
                except RuntimeError: pass
                time.sleep(1.)
                try: pipe, scale, fxB = start(); filters = make_filters()
                except RuntimeError as e2: print('  still waiting:', str(e2)[:120], flush=True)
                if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27): break
                continue
            L = np.asanyarray(fs.get_infrared_frame(1).get_data()).copy(); R = np.asanyarray(fs.get_infrared_frame(2).get_data()).copy()
            col = np.asanyarray(fs.get_color_frame().get_data()).copy() if use_color else cv2.cvtColor(L, cv2.COLOR_GRAY2BGR); d = fs.get_depth_frame()
            if st['filters']:
                for f in filters: d = f.process(d)
            z_rs = np.asanyarray(d.get_data()).astype(np.float32)*scale

            torch.cuda.synchronize(); t0 = time.perf_counter()
            disp = ffs(L, R).cpu().numpy(); ms = (time.perf_counter()-t0)*1e3   # .cpu() synchronises
            ms_avg = ms if n == 0 else .9*ms_avg+.1*ms
            ok = (disp > .5) & (xx-disp >= 0)                                   # match inside the right image, finite depth
            z_ff = np.where(ok, fxB*(w/W)/np.maximum(disp, 1e-3), 0.).astype(np.float32)   # fx scales with the input width
            if (w, h) != (W, H): z_ff = cv2.resize(z_ff, (W, H), interpolation=cv2.INTER_NEAREST)
            h_rs.append(z_rs); h_ff.append(z_ff)

            lo, hi = st['lo'], st['hi']
            both = (z_rs > 0) & (z_ff > 0) & (z_rs <= hi) & (z_ff <= hi)
            dz = np.abs(z_rs-z_ff)*1e3
            dvis = cv2.applyColorMap((np.clip(dz/a.diff_max_mm, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_INFERNO); dvis[~both] = 0
            p_rs, p_ff = colorize(z_rs, lo, hi), colorize(z_ff, lo, hi)
            now = time.time(); fps = .9*fps+.1/max(now-t_prev, 1e-6) if n else 0.; t_prev = now
            x, y = mouse
            nr, vr = patch_noise(h_rs, x, y); nf, vf = patch_noise(h_ff, x, y)
            def zs(z): return f'{z*100:.1f} cm' if z > 0 else 'no depth'
            def ns(nz, vv): return f'noise {nz:.2f} mm' if nz is not None else f'noise n/a ({vv:.0%} valid)'
            text(col, f'{"colour" if use_color else "left IR"}   {fps:4.1f} fps loop   range {lo*100:.0f}-{hi*100:.0f} cm', (6, 18))
            text(p_rs, f'{model_name} on-chip ({a.preset}{", filters" if st["filters"] else ""})   valid {(z_rs > 0).mean():.0%}', (6, 18))
            text(p_rs, f'cursor {zs(z_rs[y, x])}   {ns(nr, vr)}', (6, 38))
            text(p_ff, f'FFS {a.ckpt} x{a.iters} @{w}x{h} disp{a.max_disp}   {ms_avg:.0f} ms   valid {(z_ff > 0).mean():.0%}', (6, 18))
            text(p_ff, f'cursor {zs(z_ff[y, x])}   {ns(nf, vf)}', (6, 38))
            if both.any():
                text(dvis, f'|on-chip - FFS|, 0-{a.diff_max_mm:.0f} mm   median {np.median(dz[both]):.1f} mm   <2 mm {np.mean(dz[both] < 2):.0%}', (6, 18))
                text(dvis, f'both valid in range: {both.mean():.0%} of pixels', (6, 38))
            else: text(dvis, 'no pixel with depth from both inside the range', (6, 18))
            for im in (col, p_rs, p_ff, dvis): cv2.rectangle(im, (x-3, y-3), (x+3, y+3), (255, 255, 255), 1)
            view = np.vstack([np.hstack([col, p_rs]), np.hstack([p_ff, dvis])])
            cv2.imshow(win, view); n += 1
            if a.snapshot_after and n == a.snapshot_after: cv2.imwrite(str(a.snapshot_path), view); print('wrote', a.snapshot_path, f'(FFS {ms_avg:.1f} ms)', flush=True)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27) or (a.frames and n >= a.frames): break
            elif k == ord('f'): st['filters'] = not st['filters']; filters = make_filters(); h_rs.clear()
            elif k == ord(']'): st['hi'] = hi+step
            elif k == ord('['): st['hi'] = max(lo+.02, hi-step)
            elif k == ord("'"): st['lo'] = min(hi-.02, lo+.01)
            elif k == ord(';'): st['lo'] = max(0., lo-.01)
            elif k in (ord('s'), ord('S')):   # caps lock on still saves
                a.out.mkdir(parents=True, exist_ok=True); ts = time.strftime('%Y%m%d_%H%M%S'); b = a.out/f'{ts}_ffs'
                cv2.imwrite(f'{b}_view.png', view); cv2.imwrite(f'{b}_color.png', col); cv2.imwrite(f'{b}_left.png', L); cv2.imwrite(f'{b}_right.png', R)
                np.save(f'{b}_depth_onchip_m.npy', z_rs); np.save(f'{b}_depth_ffs_m.npy', z_ff); print('saved', f'{b}_*', flush=True)
            try:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1: break
            except cv2.error: break
    finally:
        try: pipe.stop()
        except RuntimeError: pass
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
