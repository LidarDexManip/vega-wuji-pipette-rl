"""Record a RealSense camera (D435i, D405, ...): colour video, and with --depth the raw metric depth of every frame, with a live preview.

  .venv-rl-head/bin/python perception/record_realsense.py [--res 640x480] [--fps 30] [--depth [--align]] [--bag] [--seconds N] [--serial S]
  .venv-rl-head/bin/python perception/record_realsense.py --rosbag [--align] [--zstd] ...       colour + depth into one ROS 2 bag only

Recording starts at launch; q / Esc (or closing the window, or --seconds) stops it and finalises the files <out>/<time>_<camera>_*:
  _color.mp4          colour video (mp4v, real time: camera frames the loop missed are filled with the previous frame)
  with --depth:
  _depth/NNNNNN.png   raw depth of every received frame, 16-bit PNG (lossless) in camera units; metres = value * depth_scale
                      (meta.json; 0.001 on the D435i, i.e. the value is in mm); 0 = no depth
  _depth_vis.mp4      colourised depth for viewing only (near = red, far = blue, black = no depth; not for measurement)
  _frames.csv         one row per received frame: video_frame (index in the mp4s), depth_png, colour / depth frame numbers and
                      timestamps (ms, camera clock)
  _meta.json          camera, serial, firmware, resolution, depth_scale, colour / depth intrinsics, depth -> colour extrinsics
  --align             depth reprojected into the colour camera (same pixel grid and intrinsics as the colour image), so pixel (u, v)
                      of the PNG is pixel (u, v) of the video; without it the depth stays in the depth (left IR) camera
  --bag               every enabled stream raw, librealsense's own .bag (ROS 1 format: replays with librealsense / realsense-viewer,
                      not with ros2 bag)
--rosbag replaces all of the above with one ROS 2 bag, the folder <out>/<time>_<camera>/ (metadata.yaml + .db3, sqlite3 storage;
opens with Humble's ros2 bag info / play, rosbag2_py, or the rosbags Python package without ROS):
  /camera/color/image_raw (sensor_msgs/Image, bgr8) and /camera/color/camera_info
  /camera/depth/image_rect_raw and /camera/depth/camera_info, depth camera; with --align instead
  /camera/aligned_depth_to_color/image_raw and .../camera_info, same pixel grid as the colour image
      depth 16UC1 in mm, 0 = no depth (REP 118); on a camera whose depth unit is not 1 mm (D405: 0.1 mm) 32FC1 in m, NaN = no depth
  /tf_static (without --align): camera_color_optical_frame -> camera_depth_optical_frame, the depth -> colour extrinsics
  Every received frame once, stamped with the camera's frame time on the host clock; no fill-in for missed frames. Camera serial,
  firmware and depth scale are in metadata.yaml (custom_data). Size at 640x480@30 (D435i, 11-09): ~2.6 GB per minute; --zstd
  compresses each message, lossless, to ~1.0 GB per minute (Humble's ros2 bag play decompresses it). Needs the rosbags package.
  The first ~1 s after start is skipped (both streams must run gap-free first; the D435i stalls depth for ~1 s after start-up).
  Python:  from rosbags.highlevel import AnyReader;  from rosbags.typesys import Stores, get_typestore
           with AnyReader([Path(bag)], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as r:
               for c, t, raw in r.messages([c for c in r.connections if c.topic.endswith('image_raw') ...]):
                   m = r.deserialize(raw, c.msgtype); z_mm = m.data.view(np.uint16).reshape(m.height, m.width)
Load a depth PNG:  z_m = cv2.imread('.../_depth/000123.png', cv2.IMREAD_UNCHANGED) * meta['depth_scale']
With several cameras connected, --serial picks one (the serials are printed at start).
D435i on this PC (11-09): needs a stable USB 3 link (a USB-C to USB-C cable in the 20 Gbps Type-C port kept dropping every ~6 s or
fell back to USB 2 with no frames); with the link it got, 640x480 colour + depth ran at the full 30 fps but 1280x720 lost ~35 % of
the frames in USB transport (19 fps delivered), hence the 640x480 default.
"""
import argparse, csv, json, queue, threading, time
from pathlib import Path
import numpy as np, cv2, pyrealsense2 as rs


class Worker:
    """Background thread draining a queue with fn(item) (OpenCV releases the GIL while encoding, so the capture loop keeps up).
    start / done run on the thread before the first / after the last item. An exception is kept in .err and the queue is still
    drained, so the capture loop never blocks on a dead writer (it checks .err and stops)."""
    def __init__(self, fn, done=None, start=None):
        self.q, self.fn, self.done, self.start, self.err = queue.Queue(maxsize=240), fn, done, start, None
        self.t = threading.Thread(target=self.run, daemon=True); self.t.start()

    def run(self):
        try:
            if self.start: self.start()
            while (item := self.q.get()) is not None: self.fn(item)
            if self.done: self.done()
        except Exception as ex:
            self.err = ex; import traceback; traceback.print_exc()
            while self.q.get() is not None: pass

    def close(self):
        self.q.put(None); self.t.join()


def video_worker(path, fps, size, convert=None):
    """mp4 writer: items (image, gap) -> writes the previous image gap times (missed camera frames, keeps real time), then this one."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    if not vw.isOpened(): raise SystemExit(f'could not open the mp4 writer for {path}')
    st = {'prev': None}

    def write(item):
        img, gap = item; img = convert(img) if convert else img
        for _ in range(gap if st['prev'] is not None else 0): vw.write(st['prev'])
        vw.write(img); st['prev'] = img
    return Worker(write, vw.release)


def quat(R):
    """3x3 rotation -> quaternion (x, y, z, w)."""
    w, x, y, z = (np.sqrt(max(0., 1+s0*R[0, 0]+s1*R[1, 1]+s2*R[2, 2]))/2 for s0, s1, s2 in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)))
    return float(np.copysign(x, R[2, 1]-R[1, 2])), float(np.copysign(y, R[0, 2]-R[2, 0])), float(np.copysign(z, R[1, 0]-R[0, 1])), float(w)


class RosbagWriter:
    """ROS 2 bag (rosbag2 v8, sqlite3) of colour + depth images with their camera_info, written on a background thread.
    Items: ('color' | 'depth', stamp_ns, image) with the bgr8 colour or the raw z16 depth image."""
    def __init__(self, path, cintr, dintr, aligned, scale, extr, zstd, info):
        from rosbags.interfaces import Qos, QosDurability, QosHistory, QosLiveliness, QosReliability, QosTime
        from rosbags.rosbag2 import CompressionFormat, CompressionMode, Writer
        from rosbags.typesys import Stores, get_typestore
        self.ts = get_typestore(Stores.ROS2_HUMBLE); self.T = self.ts.types
        self.w = Writer(path, version=8)                                  # metadata v8: read by Humble's ros2 bag (checked 11-09)
        if zstd: self.w.set_compression(CompressionMode.MESSAGE, CompressionFormat.ZSTD)
        for k, v in info.items(): self.w.set_custom_data(k, str(v))
        dtop = '/camera/aligned_depth_to_color' if aligned else '/camera/depth'
        self.cams = {'color': ('/camera/color/image_raw', '/camera/color/camera_info', 'camera_color_optical_frame', cintr),
                     'depth': (dtop+('/image_raw' if aligned else '/image_rect_raw'), dtop+'/camera_info',
                               'camera_color_optical_frame' if aligned else 'camera_depth_optical_frame', dintr)}
        self.mm = abs(scale-1e-3) < 1e-9; self.scale = np.float32(scale)
        self.tf = None if aligned else extr
        inf = QosTime(2147483647, 4294967295)                             # /tf_static latched like a tf_static publisher, so ros2 bag play serves late subscribers
        self.latched = Qos(QosHistory.KEEP_LAST, 1, QosReliability.RELIABLE, QosDurability.TRANSIENT_LOCAL, inf, inf, QosLiveliness.AUTOMATIC, inf, False)
        self.n = {'color': 0, 'depth': 0}; self.last_commit = 0
        self.worker = Worker(self.write, self.w.close, self.open)         # sqlite objects only work on the thread that made them

    def open(self):
        self.w.open(); self.conn = {}
        for kind, (timg, tinfo, _, _) in self.cams.items():
            self.conn[kind] = self.w.add_connection(timg, 'sensor_msgs/msg/Image', typestore=self.ts)
            self.conn[kind+'_info'] = self.w.add_connection(tinfo, 'sensor_msgs/msg/CameraInfo', typestore=self.ts)
        if self.tf is not None:
            self.conn['tf'] = self.w.add_connection('/tf_static', 'tf2_msgs/msg/TFMessage', typestore=self.ts, offered_qos_profiles=[self.latched])

    def header(self, t, frame):
        T = self.T
        return T['std_msgs/msg/Header'](stamp=T['builtin_interfaces/msg/Time'](sec=t//10**9, nanosec=t % 10**9), frame_id=frame)

    def put(self, conn, t, msg):
        self.w.write(self.conn[conn], t, self.ts.serialize_cdr(msg, msg.__msgtype__))

    def write(self, item):
        kind, t, img = item; T = self.T
        if self.tf is not None:                                           # p_colour = R p_depth + t: the depth frame's pose in the colour frame
            R = np.array(self.tf.rotation, np.float64).reshape(3, 3).T; tr = self.tf.translation; qx, qy, qz, qw = quat(R)
            tf = T['geometry_msgs/msg/TransformStamped'](
                header=self.header(t, 'camera_color_optical_frame'), child_frame_id='camera_depth_optical_frame',
                transform=T['geometry_msgs/msg/Transform'](translation=T['geometry_msgs/msg/Vector3'](x=tr[0], y=tr[1], z=tr[2]),
                                                           rotation=T['geometry_msgs/msg/Quaternion'](x=qx, y=qy, z=qz, w=qw)))
            self.put('tf', t, T['tf2_msgs/msg/TFMessage'](transforms=[tf])); self.tf = None
        _, _, frame, i = self.cams[kind]; hdr = self.header(t, frame)
        if kind == 'color': enc = 'bgr8'
        elif self.mm: enc = '16UC1'
        else: img, enc = np.where(img > 0, img*self.scale, np.float32(np.nan)).astype(np.float32), '32FC1'
        img = np.ascontiguousarray(img)
        self.put(kind, t, T['sensor_msgs/msg/Image'](header=hdr, height=img.shape[0], width=img.shape[1], encoding=enc, is_bigendian=0,
                                                       step=img.strides[0], data=img.reshape(-1).view(np.uint8)))
        self.put(kind+'_info', t, T['sensor_msgs/msg/CameraInfo'](
            header=hdr, height=i.height, width=i.width, distortion_model='plumb_bob', d=np.array(i.coeffs, np.float64),
            k=np.array([i.fx, 0, i.ppx, 0, i.fy, i.ppy, 0, 0, 1.]), r=np.eye(3).reshape(-1),
            p=np.array([i.fx, 0, i.ppx, 0, 0, i.fy, i.ppy, 0, 0, 0, 1., 0]), binning_x=0, binning_y=0,
            roi=T['sensor_msgs/msg/RegionOfInterest'](x_offset=0, y_offset=0, height=0, width=0, do_rectify=False)))
        self.n[kind] += 1
        if t-self.last_commit > 2e9 and hasattr(getattr(self.w, 'storage', None), 'conn'):   # commit every 2 s: a crash keeps what was
            self.w.storage.conn.commit(); self.last_commit = t                               # written (recover with ros2 bag reindex)


def stamp_ns(f):
    """Frame time in ns: the camera's timestamp when it is on the host clock (global time, the D400 default), else the time of receipt."""
    return int(f.get_timestamp()*1e6) if f.get_frame_timestamp_domain() == rs.timestamp_domain.global_time else time.time_ns()


def intr_dict(i):
    return {'width': i.width, 'height': i.height, 'fx': i.fx, 'fy': i.fy, 'ppx': i.ppx, 'ppy': i.ppy, 'model': str(i.model), 'coeffs': list(i.coeffs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--res', default='640x480'); ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--depth', action='store_true', help='also record raw depth (16-bit PNG per frame) + a colourised depth video')
    ap.add_argument('--align', action='store_true', help='reproject the depth into the colour camera (pixel-aligned with the video)')
    ap.add_argument('--rosbag', action='store_true', help='record only colour + depth, into one ROS 2 bag (no mp4 / PNG)')
    ap.add_argument('--zstd', action='store_true', help='zstd-compress the --rosbag messages')
    ap.add_argument('--depth-max', type=float, default=2., help='far end of the depth colours in the preview / _depth_vis.mp4, m')
    ap.add_argument('--bag', action='store_true', help="also record the raw streams to librealsense's .bag (ROS 1 format)")
    ap.add_argument('--seconds', type=float, default=0, help='stop after N s (0 = until q)')
    ap.add_argument('--serial', default=None); ap.add_argument('--no-preview', action='store_true')
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parent/'recordings')
    a = ap.parse_args(); W, H = map(int, a.res.lower().split('x'))
    if a.rosbag: a.depth = True
    if a.align and not a.depth: raise SystemExit('--align needs --depth (or --rosbag)')
    if a.zstd and not a.rosbag: raise SystemExit('--zstd is for --rosbag')

    devs = list(rs.context().query_devices())
    if not devs: raise SystemExit('no RealSense camera found (check the USB-C plug and use a USB 3 port)')
    for d in devs: print(f'found {d.get_info(rs.camera_info.name)}  serial {d.get_info(rs.camera_info.serial_number)}  USB {d.get_info(rs.camera_info.usb_type_descriptor)}')
    dev = next((d for d in devs if a.serial in (None, d.get_info(rs.camera_info.serial_number))), None)
    if dev is None: raise SystemExit(f'no camera with serial {a.serial}')
    name = dev.get_info(rs.camera_info.name).replace('Intel RealSense ', '').replace(' ', ''); serial = dev.get_info(rs.camera_info.serial_number)
    if dev.get_info(rs.camera_info.usb_type_descriptor).startswith('2'): print('warning: connected at USB 2 -- high resolutions / rates may fail; use a USB 3 port / cable')

    cfg = rs.config(); cfg.enable_device(serial); cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, a.fps)
    if a.depth: cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, a.fps)
    a.out.mkdir(parents=True, exist_ok=True); stem = a.out/f'{time.strftime("%Y%m%d_%H%M%S")}_{name}'
    if a.bag: cfg.enable_record_to_file(f'{stem}.bag')
    pipe = rs.pipeline(); prof = pipe.start(cfg)
    cprof = prof.get_stream(rs.stream.color).as_video_stream_profile()
    meta = {'camera': dev.get_info(rs.camera_info.name), 'serial': serial, 'firmware': dev.get_info(rs.camera_info.firmware_version),
            'usb': dev.get_info(rs.camera_info.usb_type_descriptor), 'resolution': [W, H], 'fps': a.fps, 'started': time.strftime('%Y-%m-%d %H:%M:%S'),
            'color_intrinsics': intr_dict(cprof.get_intrinsics())}
    align = rs.align(rs.stream.color) if a.align else None
    scale = 1.
    if a.depth:
        scale = prof.get_device().first_depth_sensor().get_depth_scale(); dprof = prof.get_stream(rs.stream.depth).as_video_stream_profile()
        e = dprof.get_extrinsics_to(cprof)
        meta.update({'depth_scale': scale, 'depth_aligned_to_color': bool(a.align),
                     'depth_intrinsics': intr_dict(cprof.get_intrinsics() if a.align else dprof.get_intrinsics()),
                     'depth_to_color_extrinsics': {'rotation_colmajor': list(e.rotation), 'translation_m': list(e.translation)},
                     'depth_png': 'uint16, metres = value * depth_scale, 0 = no depth'})

    def colour_depth(z16):
        dv = cv2.applyColorMap(255-(np.clip(z16*(scale/a.depth_max), 0, 1)*255).astype(np.uint8), cv2.COLORMAP_TURBO); dv[z16 == 0] = 0; return dv

    if a.rosbag:
        rb = RosbagWriter(stem, cprof.get_intrinsics(), cprof.get_intrinsics() if a.align else dprof.get_intrinsics(), a.align, scale, e, a.zstd,
                          {k: meta[k] for k in ('camera', 'serial', 'firmware', 'usb', 'depth_scale', 'depth_aligned_to_color')})
        workers = [rb.worker]
    else:
        (Path(f'{stem}_meta.json')).write_text(json.dumps(meta, indent=1))
        workers = [video_worker(f'{stem}_color.mp4', a.fps, (W, H))]
        if a.depth:
            ddir = Path(f'{stem}_depth'); ddir.mkdir()
            workers += [video_worker(f'{stem}_depth_vis.mp4', a.fps, (W, H), colour_depth),
                        Worker(lambda it: cv2.imwrite(str(ddir/f'{it[0]:06d}.png'), it[1], [cv2.IMWRITE_PNG_COMPRESSION, 1]))]
            fcsv = open(f'{stem}_frames.csv', 'w', newline=''); cw = csv.writer(fcsv)
            cw.writerow(['video_frame', 'depth_png', 'color_frame_number', 'color_timestamp_ms', 'depth_frame_number', 'depth_timestamp_ms'])
    what = (f'ROS 2 bag{" (zstd)" if a.zstd else ""}: colour + depth{" aligned to colour" if a.align else ""}' if a.rosbag else
            f'{"colour + raw depth" if a.depth else "colour"}{" (aligned to colour)" if a.align else ""}')
    print(f'recording {name} {serial}: {W}x{H}@{a.fps} {what}{" + bag" if a.bag else ""} -> {stem}{"/" if a.rosbag else "_*"}  (q to stop)', flush=True)
    win = f'{name} recording (q to stop)'; PW = min(W, 960)
    if not a.no_preview: cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    n, k, t0, last_no, dropped, last_bag = 0, 0, None, None, 0, [None, None]
    warm, prev, t_start = 0, None, time.time()
    try:
        while True:
            fs = pipe.wait_for_frames(5000)
            if align is not None: fs = align.process(fs)
            cf = fs.get_color_frame(); fno = cf.get_frame_number()
            if warm < 15:   # start-up: the D435i gives one frame, then nothing for ~1 s on depth (and colour with --align); record from
                dno = fs.get_depth_frame().get_frame_number() if a.depth else -1      # 15 frame sets in a row where every stream advanced by 1
                warm = warm+1 if prev is not None and fno == prev[0]+1 and dno in (-1, prev[1]+1) else 0; prev = (fno, dno)
                if time.time()-t_start > 5: print('warning: frames still irregular after 5 s, recording anyway', flush=True); warm = 15
                continue
            if t0 is None: t0 = time.time()                                   # time from the first recorded frame (not the camera start-up)
            col = np.asanyarray(cf.get_data()).copy()
            gap = 0 if last_no is None else int(min(max(fno-last_no-1, 0), a.fps)); last_no = fno; dropped += gap
            if a.rosbag:                                                      # the syncer sometimes repeats a frame in the next set: bag each frame once
                if fno != last_bag[0]: rb.worker.q.put(('color', stamp_ns(cf), col)); last_bag[0] = fno
            else: n += gap; workers[0].q.put((col, gap))
            if a.depth:
                df = fs.get_depth_frame(); z16 = np.asanyarray(df.get_data()).copy()
                if a.rosbag:
                    if df.get_frame_number() != last_bag[1]: rb.worker.q.put(('depth', stamp_ns(df), z16)); last_bag[1] = df.get_frame_number()
                else:
                    workers[1].q.put((z16, gap)); workers[2].q.put((k, z16))
                    cw.writerow([n, f'{k:06d}.png', fno, f'{cf.get_timestamp():.3f}', df.get_frame_number(), f'{df.get_timestamp():.3f}'])
                k += 1
            n += 1; el = time.time()-t0
            if a.seconds and el >= a.seconds: break
            if any(w.err for w in workers): print('a writer failed (traceback above), stopping', flush=True); break
            if not a.no_preview:
                pv = cv2.resize(col, (PW, PW*H//W), interpolation=cv2.INTER_AREA)
                if a.depth: pv = np.hstack([pv, cv2.resize(colour_depth(z16[::2, ::2]), (PW, PW*H//W), interpolation=cv2.INTER_NEAREST)])
                cv2.circle(pv, (22, 22), 9, (0, 0, 255), -1)
                for dx, dy, c_ in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (1, 1, 0), (-1, -1, 0), (0, 0, 255)):   # outline at 1 px offsets, then white
                    cv2.putText(pv, f'REC {int(el)//60:02d}:{int(el)%60:02d}  {n} frames', (40+dx, 30+dy), cv2.FONT_HERSHEY_SIMPLEX, .7, (c_, c_, c_), 1, cv2.LINE_AA)
                cv2.imshow(win, pv)
                if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27): break
                try:
                    if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1: break
                except cv2.error: break
    finally:
        el = time.time()-t0 if t0 else 0.; pipe.stop(); cv2.destroyAllWindows()
        print('finishing the files ...', flush=True)
        for w in workers: w.close()
        if a.depth and not a.rosbag: fcsv.close()
        if a.rosbag:
            print(f'stopped: {n} frames for {el:.1f} s ({n/max(el, 1e-6):.1f} fps); {dropped} camera frames missed (not in the bag); '
                  f'bag has {rb.n["color"]} colour + {rb.n["depth"]} depth images')
        else:
            print(f'stopped: {n} video frames for {el:.1f} s ({n/max(el, 1e-6):.1f} fps); {dropped} ({dropped/max(n, 1):.0%}) repeat the previous '
                  f'frame because the camera frame was missed' + (f'; {k} depth frames saved' if a.depth else ''))
        for p in sorted(stem.parent.glob(stem.name+'*')):
            sz = sum(f.stat().st_size for f in p.iterdir()) if p.is_dir() else p.stat().st_size
            print(f'  {p}{"/  (" + str(len(list(p.iterdir()))) + " files)" if p.is_dir() else ""}  {sz/2**20:.1f} MB')


if __name__ == '__main__':
    main()
