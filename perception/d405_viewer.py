"""Live RealSense D405 depth viewer (OpenCV window): colourised depth next to the D405's own colour image (pixel-aligned).

  .venv-rl-head/bin/python perception/d405_viewer.py [--res 1280x720] [--fps 30] [--preset "High Accuracy"] [--min-z 0.07 | --shift N]
                                                    [--min 0.07] [--max 0.5] [--filters] [--despeckle] [--fill-holes] [--no-color]

Keys   q / Esc quit | p cycle the camera's visual preset | f toggle spatial + temporal filters (disparity domain)
       k toggle speckle removal | h toggle small-hole filling | - = disparity shift -/+ 5 (nearer / farther depth window)
       a toggle auto range (p2-p98) | [ ] max range -/+ 5 cm | ; ' min range -/+ 1 cm | s save a snapshot (view png, colour png, depth .npy in m)
Near range (11-09): the stereo search covers disparities shift .. shift+126 px, so depth exists only in fx B / (shift+126) .. fx B / shift
(B = 18 mm). With no shift that is >= 9.4 cm at 1280x720 (fx 652), 6.2 cm at 848x480, 4.7 cm at 640x360. --min-z (default 7 cm) picks the
smallest shift that reaches it: shift 43 at 1280x720 = depth window 7.0-27 cm; none needed at 848x480. Caveat: a scene farther than the
window does not just vanish, textured background can alias into false blobs inside it (seen 11-09 on a room scene); 848x480 with no shift
reaches ~6 cm without a far limit. Preset and shift are written before the stream starts (with colour on, the D405 ignores depth-table
writes while streaming), so p / - / = restart the stream (~1 s).
Presets: High Accuracy keeps only confident matches and drops most of a bare hand (low texture); Default / High Density keep far more.
Mouse  hover: depth under the cursor, its temporal noise (std over the last 15 frames, 7x7 patch) and the pixel footprint (mm per pixel).
Colours: red = near, blue = far, grey = beyond the max range, black = no depth (closer than ~7 cm, no stereo match, or rejected by the preset).
The D405 is passive stereo (no projector, 18 mm baseline, best at 7-50 cm): depth noise grows ~ with distance squared, and textureless,
transparent or shiny surfaces give holes or false matches. If the camera drops off USB, the window says so and reconnects by itself.
"""
import argparse, collections, math, time
from pathlib import Path
import numpy as np, cv2, pyrealsense2 as rs

PRESETS = ['High Accuracy', 'Default', 'High Density', 'Medium Density', 'Hand', 'Edge Enhancement']
NDISP = 126   # disparity search range of the D4 stereo ASIC (px); measured on this D405: shift 0 -> 9.4 cm, shift 42 -> 7.0 cm at fx 651.8
K3 = np.ones((3, 3), np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--res', default='1280x720'); ap.add_argument('--fps', type=int, default=30); ap.add_argument('--preset', default='High Accuracy', choices=PRESETS)
    ap.add_argument('--min', type=float, default=.07, help='near end of the colour range, m'); ap.add_argument('--max', type=float, default=.5, help='far end, m')
    ap.add_argument('--no-color', action='store_true'); ap.add_argument('--filters', action='store_true'); ap.add_argument('--auto', action='store_true')
    ap.add_argument('--min-z', type=float, default=.07, help='nearest depth to measure, m (sets the smallest disparity shift reaching it; 0 = no shift)')
    ap.add_argument('--shift', type=int, default=None, help='disparity shift in px (overrides --min-z)')
    ap.add_argument('--despeckle', action='store_true', help='drop small isolated depth blobs (false stereo matches)')
    ap.add_argument('--fill-holes', action='store_true', help='fill small holes from the farther neighbouring depth')
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parent/'snapshots', help='where s saves')
    ap.add_argument('--snapshot-after', type=int, default=0, help='also write the view to --snapshot-path after N frames (check without looking)')
    ap.add_argument('--snapshot-path', type=Path, default=Path('d405_view.png')); ap.add_argument('--frames', type=int, default=0, help='exit after N frames (0 = until q)')
    a = ap.parse_args(); W, H = map(int, a.res.lower().split('x')); fs_ = W/848   # text scale
    st = {'preset': a.preset, 'filters': a.filters, 'auto': a.auto, 'lo': a.min, 'hi': a.max, 'minz': a.min_z, 'shift': a.shift,
          'speck': a.despeckle, 'fill': a.fill_holes, 'sh': 0, 'zwin': (0., math.inf)}
    cam = {}
    speck_px, hole_px = int(W*H*2e-4), int(W*H*1e-3)   # 184 / 922 px at 1280x720

    def connect():
        """(Re)start the stream. The preset and the disparity shift are written to the idle camera BEFORE the start: with the colour stream
        on, the D405 ignores depth-table writes made while streaming (measured 11-09: the write only took effect at the next start), so
        changing either restarts the stream. Also the retry path when the D405 drops off USB and re-enumerates."""
        devs = rs.context().query_devices()
        if len(devs) == 0: raise RuntimeError('no RealSense device found')
        dev = devs[0]; s = dev.first_depth_sensor()
        r = s.get_option_range(rs.option.visual_preset)
        names = {s.get_option_value_description(rs.option.visual_preset, float(i)): i for i in range(int(r.min), int(r.max)+1)}
        if st['preset'] in names: s.set_option(rs.option.visual_preset, names[st['preset']])
        intr = None
        for p_ in s.get_stream_profiles():
            v_ = p_.as_video_stream_profile()
            if p_.stream_type() == rs.stream.depth and p_.format() == rs.format.z16 and (v_.width(), v_.height()) == (W, H): intr = v_.get_intrinsics(); break
        if intr is None: raise SystemExit(f'the D405 has no {W}x{H} depth mode')
        B = s.get_option(rs.option.stereo_baseline)*1e-3 if s.supports(rs.option.stereo_baseline) else .018
        apply_shift(dev, intr.fx*B)                                    # after the preset: a preset load rewrites the depth table
        cfg = rs.config(); cfg.enable_device(dev.get_info(rs.camera_info.serial_number)); cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, a.fps)
        if not a.no_color: cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, a.fps)
        p = rs.pipeline(); p.start(cfg)
        print(f'D405 {dev.get_info(rs.camera_info.serial_number)}: depth {W}x{H}@{a.fps}, preset {st["preset"]}, scale {s.get_depth_scale()*1e3:.2f} mm/unit, '
              f'fx {intr.fx:.1f} fy {intr.fy:.1f} cx {intr.ppx:.1f} cy {intr.ppy:.1f}, baseline {B*1e3:.2f} mm, disparity shift {st["sh"]}: depth {zwin_str()}', flush=True)
        return p, s, names, s.get_depth_scale(), intr.fx

    def apply_shift(dev, fxB):
        """Disparity shift in the advanced-mode depth table (volatile: gone after a power cycle). The stereo search covers disparities
        shift .. shift+NDISP, i.e. depth fx B / (shift+NDISP) .. fx B / shift."""
        sh = st['shift'] if st['shift'] is not None else (max(0, math.ceil(fxB/st['minz']-NDISP)) if st['minz'] > 0 else 0)
        try:
            am = rs.rs400_advanced_mode(dev)
            if not am.is_enabled(): print('advanced mode is off: disparity shift not applied (rs400_advanced_mode(dev).toggle_advanced_mode(True) enables it)', flush=True); sh = 0
            else: t = am.get_depth_table(); t.disparityShift = int(sh); am.set_depth_table(t); sh = am.get_depth_table().disparityShift
        except RuntimeError as e:
            print('disparity shift not applied:', e, flush=True); sh = 0
        st['sh'] = sh; st['zwin'] = (fxB/(sh+NDISP), fxB/sh if sh else math.inf)

    def zwin_str():
        lo_, hi_ = st['zwin']; return f'{lo_*100:.1f}-{"inf" if math.isinf(hi_) else f"{hi_*100:.0f}"} cm'

    def despeckle(z):
        """Remove connected blobs smaller than speck_px (neighbours within 3 mm count as one blob): the isolated false matches."""
        mm = np.clip(z*1e3, 0, 32767).astype(np.int16); cv2.filterSpeckles(mm, 0, speck_px, 3)
        z = z.copy(); z[(mm == 0) & (z > 0)] = 0; return z

    def fill_holes(z):
        """Fill holes of at most hole_px pixels, growing in from their border with the FARTHER neighbouring depth (3x3 max), so a gap
        between a finger and the background takes the background depth and no flying pixels appear. Larger holes stay black."""
        hole = (z == 0).astype(np.uint8); _, lab, stats, _ = cv2.connectedComponentsWithStats(hole, connectivity=4)
        small = (stats[lab, cv2.CC_STAT_AREA] <= hole_px) & (hole > 0)
        for _ in range(16):
            todo = small & (z == 0)
            if not todo.any(): break
            z = np.where(todo, cv2.dilate(z, K3), z)
        return z

    def make_filters():
        sp = rs.spatial_filter(); sp.set_option(rs.option.filter_magnitude, 2); sp.set_option(rs.option.filter_smooth_alpha, .5); sp.set_option(rs.option.filter_smooth_delta, 20)
        tp = rs.temporal_filter(); tp.set_option(rs.option.filter_smooth_alpha, .4); tp.set_option(rs.option.filter_smooth_delta, 20)
        return [rs.disparity_transform(True), sp, tp, rs.disparity_transform(False)]   # librealsense's recommended order: smooth in disparity

    def text(img, s, org, scale=.55):
        scale *= fs_; th = max(1, int(round(fs_))); x, y = org   # outline = the same text at 1 px offsets (a thicker stroke widens the glyph advance -> ghosting)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
            cv2.putText(img, s, (x+dx, y+dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th, cv2.LINE_AA)
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), th, cv2.LINE_AA)

    pipe, sen, names, scale, fx = connect(); filters = make_filters()
    mouse = [W//2, H//2]; win = 'D405  (q quit, p preset, f filters, k despeckle, h fill holes, - = shift, a auto range, s snapshot)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL); cv2.resizeWindow(win, min(1700, W*(1 if a.no_color else 2)), int(min(1700, W*(1 if a.no_color else 2))*H/(W*(1 if a.no_color else 2))))

    def closed():   # window closed with the mouse (the Qt backend raises instead of returning 0 once the window is gone)
        try: return cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error: return True

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_MOUSEMOVE: mouse[0], mouse[1] = min(max(x % W, 0), W-1), min(max(y, 0), H-1)
    cv2.setMouseCallback(win, on_mouse)
    bar = cv2.applyColorMap((255-np.linspace(0, 255, W)).astype(np.uint8)[None].repeat(int(10*fs_), 0), cv2.COLORMAP_TURBO)
    hist = collections.deque(maxlen=15); n, fps, t_prev = 0, 0., time.time()
    try:
        while True:
            try:
                fs = pipe.wait_for_frames(2000)
            except RuntimeError as e:   # stream stalled or the camera re-enumerated: show it and keep retrying until q
                print(f'{time.strftime("%H:%M:%S")} no frames ({e}); reconnecting', flush=True)
                try: pipe.stop()
                except RuntimeError: pass
                while True:
                    msg = np.zeros((H, W*(1 if a.no_color else 2), 3), np.uint8); text(msg, 'camera not streaming - reconnecting (check the USB cable) ...', (20, H//2), .7)
                    cv2.imshow(win, msg)
                    if (cv2.waitKey(1000) & 0xFF) in (ord('q'), 27) or closed(): return
                    try: pipe, sen, names, scale, fx = connect(); break
                    except RuntimeError as e2: print('  still waiting:', str(e2)[:120], flush=True)
                filters = make_filters(); hist.clear(); continue
            d = fs.get_depth_frame()
            if st['filters']:
                for f in filters: d = f.process(d)
            z = np.asanyarray(d.get_data()).astype(np.float32)*scale
            if st['speck']: z = despeckle(z)
            if st['fill']: z = fill_holes(z)
            valid = z > 0; hist.append(z)
            lo, hi = st['lo'], st['hi']
            if st['auto'] and valid.sum() > 100: lo, hi = (float(x) for x in np.percentile(z[valid], [2, 98])); hi = max(hi, lo+.01)
            u8 = (np.clip((z-lo)/max(hi-lo, 1e-3), 0, 1)*255).astype(np.uint8)
            vis = cv2.applyColorMap(255-u8, cv2.COLORMAP_TURBO); vis[valid & (z > hi)] = (70, 70, 70); vis[~valid] = 0
            vis[H-bar.shape[0]:] = bar; text(vis, f'{lo*100:.0f} cm', (4, H-bar.shape[0]-6), .45); text(vis, f'{hi*100:.0f} cm', (W-int(70*fs_), H-bar.shape[0]-6), .45)
            panels = [vis]
            if not a.no_color: panels.append(np.asanyarray(fs.get_color_frame().get_data()).copy())
            view = np.hstack(panels) if len(panels) > 1 else vis
            x, y = mouse; zc = z[y, x]; r_ = 3
            for i in range(len(panels)): cv2.rectangle(view, (x-r_+i*W, y-r_), (x+r_+i*W, y+r_), (255, 255, 255), 1)
            P = np.stack([h[max(0, y-r_):y+r_+1, max(0, x-r_):x+r_+1] for h in hist]); ok = (P > 0).all(0)
            noise = f'noise {np.median(P.std(0)[ok])*1e3:.2f} mm (std, {len(hist)} frames)' if ok.any() else f'noise: no stable depth here ({(P > 0).mean():.0%} of samples valid)'
            now = time.time(); fps = .9*fps+.1/max(now-t_prev, 1e-6) if n else 0.; t_prev = now
            lines = [f'{fps:4.1f} fps   {W}x{H}   preset {st["preset"]}   filters {"on" if st["filters"] else "off"}{"  +despeckle" if st["speck"] else ""}{"  +fill" if st["fill"] else ""}'
                     f'   shift {st["sh"]} (depth {zwin_str()})   colours {lo*100:.0f}-{hi*100:.0f} cm{" (auto)" if st["auto"] else ""}',
                     (f'cursor: {zc*100:.1f} cm   1 px = {zc/fx*1e3:.2f} mm   ' if zc > 0 else 'cursor: no depth   ')+noise,
                     (f'valid {valid.mean():.0%}   nearest (p1) {np.percentile(z[valid], 1)*100:.1f} cm   median {np.median(z[valid])*100:.1f} cm   far (p99) {np.percentile(z[valid], 99)*100:.1f} cm') if valid.any() else 'no valid depth']
            for i, s in enumerate(lines): text(view, s, (8, int((22+22*i)*fs_)))
            cv2.imshow(win, view); n += 1
            if a.snapshot_after and n == a.snapshot_after: cv2.imwrite(str(a.snapshot_path), view); print('wrote', a.snapshot_path, flush=True)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27) or (a.frames and n >= a.frames): break
            restart = False
            if k == ord('p'):
                avail = [p for p in PRESETS if p in names]; st['preset'] = avail[(avail.index(st['preset'])+1) % len(avail)] if st['preset'] in avail else avail[0]
                restart = True
            elif k == ord('f'): st['filters'] = not st['filters']; filters = make_filters(); hist.clear()
            elif k == ord('k'): st['speck'] = not st['speck']; hist.clear()
            elif k == ord('h'): st['fill'] = not st['fill']; hist.clear()
            elif k in (ord('-'), ord('=')): st['shift'] = max(0, st['sh']+(5 if k == ord('=') else -5)); restart = True
            elif k == ord('a'): st['auto'] = not st['auto']
            elif k == ord(']'): st['auto'] = False; st['hi'] = hi+.05; st['lo'] = lo
            elif k == ord('['): st['auto'] = False; st['hi'] = max(lo+.02, hi-.05); st['lo'] = lo
            elif k == ord("'"): st['auto'] = False; st['lo'] = min(hi-.02, lo+.01); st['hi'] = hi
            elif k == ord(';'): st['auto'] = False; st['lo'] = max(0., lo-.01); st['hi'] = hi
            elif k in (ord('s'), ord('S')):   # caps lock on still saves
                a.out.mkdir(parents=True, exist_ok=True); ts = time.strftime('%Y%m%d_%H%M%S')
                cv2.imwrite(str(a.out/f'{ts}_view.png'), view); np.save(a.out/f'{ts}_depth_m.npy', z)
                if not a.no_color: cv2.imwrite(str(a.out/f'{ts}_color.png'), panels[1])
                print('saved', a.out/f'{ts}_*', flush=True)
            if restart:   # preset / shift only take effect on a stream start (see connect)
                try: pipe.stop()
                except RuntimeError: pass
                try: pipe, sen, names, scale, fx = connect()
                except RuntimeError as e: print('restart failed:', str(e)[:120], flush=True)   # the next wait_for_frames raises -> reconnect loop
                filters = make_filters(); hist.clear()
            if closed(): break
    finally:
        try: pipe.stop()
        except RuntimeError: pass   # already stopped (exit during a reconnect)
        if st['sh']:   # the depth table lives on the camera until a power cycle: leave shift 0 for the next program (e.g. deployment)
            try:
                am = rs.rs400_advanced_mode(rs.context().query_devices()[0]); t = am.get_depth_table(); t.disparityShift = 0; am.set_depth_table(t)
                print('disparity shift reset to 0', flush=True)
            except (RuntimeError, IndexError) as e: print('could not reset the disparity shift:', e, flush=True)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
