"""Export a ROS 2 bag from record_realsense.py --rosbag (or its ffs_bag.py output) to plain files: colour video + greyscale depth.

  .venv-rl-head/bin/python perception/bag_export.py <bag folder> [--depth-topic auto] [--min 0.2 --max 0.8] [--far-bright] [--no-png]

Writes <bag folder>_export/:
  color.mp4             the colour stream
  depth_gray.mp4        depth as greyscale video
  depth_gray/NNNNNN.png the same, 8-bit greyscale, one PNG per depth frame
  depth_mm/NNNNNN.png   metric depth, 16-bit greyscale PNG, value = depth in mm (0 = no depth); lossless up to the 1 mm step
  frames.csv            png index -> depth stamp (ns), and the nearest colour frame (index in color.mp4, stamp)
  export_info.json      depth topic, grey range and the formula back to metres
Greyscale mapping (8-bit): linear in depth over [min, max] m, nearer = brighter (gray 255 at min, 1 at max, clipped outside),
0 (black) = no depth; --far-bright flips it (1 at min, 255 at max). One range for the whole export: --min / --max, by default the
1st-99th percentile of the recording's depth (rounded to cm), so near objects are not clipped into black.
Back to metres: z = min + (255 - gray) / 254 * (max - min)   (--far-bright: z = min + (gray - 1) / 254 * (max - min)).
Depth topic (auto): the Fast-FoundationStereo depth aligned to colour if the bag has it, else the on-chip depth (aligned if recorded so).
"""
import argparse, csv, json
from pathlib import Path
import numpy as np, cv2
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

AUTO = ['/camera/ffs/aligned_depth_to_color/image_raw', '/camera/aligned_depth_to_color/image_raw', '/camera/ffs/depth/image_rect_raw',
        '/camera/depth/image_rect_raw']


def to_m(m):
    """sensor_msgs/Image depth -> float32 metres, 0 = no depth."""
    if m.encoding == '16UC1': return m.data.view(np.uint16).reshape(m.height, m.width).astype(np.float32)*1e-3
    if m.encoding == '32FC1': return np.nan_to_num(m.data.view(np.float32).reshape(m.height, m.width), nan=0.)
    raise SystemExit(f'unexpected depth encoding {m.encoding}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag', type=Path)
    ap.add_argument('--depth-topic', default='auto', help='auto, or a topic name, e.g. /camera/depth/image_rect_raw')
    ap.add_argument('--min', type=float, default=None, help='near end of the grey range, m (default: 1st percentile of the depth)')
    ap.add_argument('--max', type=float, default=None, help='far end, m (default: 99th percentile)')
    ap.add_argument('--far-bright', action='store_true', help='farther = brighter (default: nearer = brighter)')
    ap.add_argument('--no-png', action='store_true', help='videos only')
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args(); out = a.out or a.bag.parent/(a.bag.name.rstrip('/')+'_export')
    ts = get_typestore(Stores.ROS2_HUMBLE)

    with Reader(a.bag) as rd:
        by = {c.topic: c for c in rd.connections}
        dtop = next((t for t in AUTO if t in by), None) if a.depth_topic == 'auto' else a.depth_topic
        if dtop not in by: raise SystemExit(f'no depth topic {a.depth_topic} in {a.bag}; it has {sorted(by)}')
        if '/camera/color/image_raw' not in by: raise SystemExit('no /camera/color/image_raw in the bag')
        if a.min is None or a.max is None:                                 # range from ~30 depth frames spread over the recording
            k = max(1, by[dtop].msgcount//30); smp = []
            for i, (c, t, raw) in enumerate(rd.messages(connections=[by[dtop]])):
                if i % k == 0: z = to_m(ts.deserialize_cdr(raw, c.msgtype)); smp.append(z[z > 0][::7])
            v = np.concatenate(smp) if smp else np.array([.2, 1.])
        lo = a.min if a.min is not None else np.floor(np.percentile(v, 1)*100)/100
        hi = a.max if a.max is not None else np.ceil(np.percentile(v, 99)*100)/100
        conns = [by['/camera/color/image_raw'], by[dtop]]
        stamps = []                                                       # fps from the colour stamps
        for c, t, raw in rd.messages(connections=[by['/camera/color/image_raw']]):
            stamps.append(t)
            if len(stamps) > 60: break
        fps = 1e9/float(np.median(np.diff(stamps))) if len(stamps) > 1 else 30.
        out.mkdir(parents=True, exist_ok=False)
        if not a.no_png: (out/'depth_gray').mkdir(); (out/'depth_mm').mkdir()
        print(f'{a.bag.name}: colour + {dtop}, grey range {lo:.2f}-{hi:.2f} m ({"farther" if a.far_bright else "nearer"} = brighter), '
              f'{fps:.1f} fps -> {out}/', flush=True)
        vc = vd = None; nc = nd = 0; col_t = []; rows = []
        for c, t, raw in rd.messages(connections=conns):
            m = ts.deserialize_cdr(raw, c.msgtype); st = m.header.stamp.sec*10**9+m.header.stamp.nanosec
            if c.topic == '/camera/color/image_raw':
                img = m.data.reshape(m.height, m.width, 3)
                if m.encoding == 'rgb8': img = img[..., ::-1]
                if vc is None: vc = cv2.VideoWriter(str(out/'color.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (m.width, m.height))
                vc.write(np.ascontiguousarray(img)); col_t.append(st); nc += 1
            else:
                z = to_m(m); ok = z > 0
                g = 1+np.clip((z-lo)/(hi-lo), 0, 1)*254
                if not a.far_bright: g = 256-g
                g = np.where(ok, np.round(g), 0).astype(np.uint8)
                if vd is None: vd = cv2.VideoWriter(str(out/'depth_gray.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (m.width, m.height), isColor=False)
                vd.write(g)
                if not a.no_png:
                    cv2.imwrite(str(out/f'depth_gray/{nd:06d}.png'), g)
                    cv2.imwrite(str(out/f'depth_mm/{nd:06d}.png'), np.clip(np.round(z*1e3), 0, 65535).astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 1])
                rows.append((nd, st)); nd += 1
        for v in (vc, vd):
            if v is not None: v.release()
    ct = np.array(col_t)
    with open(out/'frames.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['png', 'depth_stamp_ns', 'color_frame', 'color_stamp_ns'])
        for i, st in rows:
            j = int(np.argmin(np.abs(ct-st))) if len(ct) else -1
            w.writerow([f'{i:06d}.png', st, j, ct[j] if j >= 0 else ''])
    (out/'export_info.json').write_text(json.dumps({
        'bag': str(a.bag), 'depth_topic': dtop, 'gray_min_m': round(float(lo), 4), 'gray_max_m': round(float(hi), 4), 'nearer_brighter': not a.far_bright,
        'gray_to_m': f'z = {lo:.2f} + ({"gray - 1" if a.far_bright else "255 - gray"}) / 254 * {hi-lo:.2f}; gray 0 = no depth',
        'depth_mm_png': 'uint16, depth in mm, 0 = no depth'}, indent=1))
    sz = sum(f.stat().st_size for f in out.rglob('*') if f.is_file())/2**20
    print(f'wrote {nc} colour frames, {nd} depth frames ({sz:.0f} MB)')


if __name__ == '__main__':
    main()
