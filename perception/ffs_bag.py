"""Add Fast-FoundationStereo depth to a ROS 2 bag recorded with record_realsense.py --rosbag --ir (offline, every frame).

  ~/Projects/Fast-FoundationStereo/.venv/bin/python perception/ffs_bag.py <bag folder> [--iters 16] [--frame color|depth|both] [--keep-ir]

Writes <bag folder>_ffs/, a new ROS 2 bag (the input is not changed), with every topic of the input (the IR pair only with
--keep-ir) plus
  /camera/ffs/aligned_depth_to_color/image_raw + camera_info   depth in the colour camera: pixel (u, v) is colour pixel (u, v)
  /camera/ffs/depth/image_rect_raw + camera_info               depth in the left IR camera (= the on-chip depth's camera)   --frame depth / both
32FC1 in m, NaN = no depth (sub-mm precision, so not 16UC1 mm); stamped like the IR pair it came from. zstd if the input is.
The model sees the rectified left / right IR images (y8 tiled to 3 channels, emitter dots included); depth = fx B / disparity with
fx, B from the bag's camera_info. Dropped: pixels whose match falls outside the right image, and disparities at max_disp (closer than
fx B / max_disp, where the network saturates instead of reporting nothing).
Default: checkpoint 23-36-37 (the most accurate of the four), 16 refinement iterations, max_disp 192: at 640x480 the near limit is
10 cm on the D435i (working range 40-80 cm) and 3.7 cm on the D405 (7-15 cm on the wrist).
D435i, 640x480, desk + pipette at 36-81 cm, RTX 4080 (11-10), median / p95 change against 32 iterations: 4 iters 0.11 / 0.47 mm,
8 iters 0.08 / 0.36 mm (36 ms per frame), 16 iters 0.05 / 0.23 mm (48 ms), so 16 is converged; the fast 20-30-48 is 0.28 / 1.1 mm
off. On-chip depth vs this: median 1.9 mm apart, temporal noise 0.49 mm vs 0.05 mm.
Colour alignment reproduces librealsense's rs.align (each depth pixel's footprint, corners at +-0.5 px, projected into the colour
image and filled; the nearest depth wins); lens distortion is not modelled (the D435i reports none for both cameras).
Runs in the Fast-FoundationStereo venv (torch, research-licence weights) with the rosbags package.
"""
import argparse, dataclasses, sys, time
from pathlib import Path
import numpy as np, torch
from rosbags.rosbag2 import CompressionFormat, CompressionMode, Reader, Writer
from rosbags.typesys import Stores, get_typestore


def quat_to_R(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def make_aligner(Kd, Wd, Hd, Kc, Wc, Hc, R, t, dev='cuda'):
    """Depth (Hd, Wd) in m (0 = none) in the depth camera -> (Hc, Wc) in the colour camera, as librealsense's align. K = (fx, fy, cx, cy);
    R, t: depth -> colour (p_c = R p_d + t)."""
    v, u = torch.meshgrid(torch.arange(Hd, device=dev, dtype=torch.float32), torch.arange(Wd, device=dev, dtype=torch.float32), indexing='ij')
    R = torch.tensor(R, dtype=torch.float32, device=dev); t = torch.tensor(t, dtype=torch.float32, device=dev)

    def corner(z, o):
        X, Y = (u+o-Kd[2])/Kd[0]*z, (v+o-Kd[3])/Kd[1]*z
        P = [R[i, 0]*X+R[i, 1]*Y+R[i, 2]*z+t[i] for i in range(3)]
        return torch.trunc(Kc[0]*P[0]/P[2]+Kc[2]+.5).long(), torch.trunc(Kc[1]*P[1]/P[2]+Kc[3]+.5).long()   # (int)(x + 0.5f) as in align.cpp

    def align(z):
        x0, y0 = corner(z, -.5); x1, y1 = corner(z, .5)
        ok = (z > 0) & (x0 >= 0) & (y0 >= 0) & (x1 < Wc) & (y1 < Hc)
        out = torch.full((Hc*Wc,), float('inf'), device=dev)
        if ok.any():
            span = int(min(max((x1-x0)[ok].max().item(), (y1-y0)[ok].max().item()), 8))
            for dy in range(span+1):
                for dx in range(span+1):
                    m = ok & (x0+dx <= x1) & (y0+dy <= y1)
                    out.scatter_reduce_(0, ((y0+dy)*Wc+x0+dx)[m], z[m], 'amin')
        return torch.where(torch.isinf(out), 0., out).view(Hc, Wc)
    return align


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag', type=Path, help='ROS 2 bag folder from record_realsense.py --rosbag --ir')
    ap.add_argument('--ffs-dir', type=Path, default=Path.home()/'Projects/Fast-FoundationStereo')
    ap.add_argument('--ckpt', default='23-36-37'); ap.add_argument('--iters', type=int, default=16); ap.add_argument('--max-disp', type=int, default=192)
    ap.add_argument('--frame', choices=['color', 'depth', 'both'], default='color', help='which camera the FFS depth is written in')
    ap.add_argument('--keep-ir', action='store_true', help='also copy the IR pair into the output bag')
    ap.add_argument('--zstd', action='store_true', help='compress the output even if the input is not')
    ap.add_argument('--out', type=Path, default=None, help='output bag folder (default <bag>_ffs)')
    a = ap.parse_args(); out = a.out or a.bag.parent/(a.bag.name.rstrip('/')+'_ffs')
    if out.exists(): raise SystemExit(f'{out} exists; remove it or pass --out')
    ts = get_typestore(Stores.ROS2_HUMBLE); T = ts.types; des = ts.deserialize_cdr

    with Reader(a.bag) as rd:
        by = {c.topic: c for c in rd.connections}
        need = ['/camera/infra1/image_rect_raw', '/camera/infra2/image_rect_raw', '/camera/infra1/camera_info', '/camera/infra2/camera_info', '/camera/color/camera_info']
        if missing := [x for x in need if x not in by]: raise SystemExit(f'{a.bag} lacks {missing}: record with record_realsense.py --rosbag --ir')
        first = {}
        for c, _, raw in rd.messages(connections=[by[x] for x in need[2:]+['/tf_static'] if x in by]):
            first.setdefault(c.topic, des(raw, c.msgtype))
            if len(first) == len(need)-2+('/tf_static' in by): break
        i1, i2, ci = first['/camera/infra1/camera_info'], first['/camera/infra2/camera_info'], first['/camera/color/camera_info']
        W, H, fx, B = i1.width, i1.height, i1.k[0], -i2.p[3]/i2.k[0]
        if B <= 0: raise SystemExit('no stereo baseline in /camera/infra2/camera_info (P[3])')
        tf = {(x.header.frame_id, x.child_frame_id): x.transform for x in first['/tf_static'].transforms} if '/tf_static' in first else {}

        def pose(parent, child):
            if parent == child: return np.eye(3), np.zeros(3)
            tr = tf[(parent, child)]; return quat_to_R(tr.rotation), np.array([tr.translation.x, tr.translation.y, tr.translation.z])
        try:                                                              # infra1 -> colour, through the depth frame
            R_cd, t_cd = pose(ci.header.frame_id, 'camera_depth_optical_frame'); R_d1, t_d1 = pose('camera_depth_optical_frame', i1.header.frame_id)
        except KeyError as ex: raise SystemExit(f'/tf_static lacks {ex}: needed to place the IR camera relative to the colour camera')
        R_c1, t_c1 = R_cd@R_d1, R_cd@t_d1+t_cd
        if np.any(np.abs(ci.d) > 1e-6) or np.any(np.abs(i1.d) > 1e-6): print('warning: the bag reports lens distortion, which the colour alignment ignores')
        print(f'{a.bag.name}: IR {W}x{H} fx {fx:.1f} B {B*1e3:.2f} mm (near limit {fx*B/a.max_disp*100:.1f} cm), colour {ci.width}x{ci.height}, '
              f'IR -> colour {np.linalg.norm(t_c1)*1e3:.1f} mm; {by["/camera/infra1/image_rect_raw"].msgcount} IR pairs', flush=True)

        # ------------------------------------------------------------------ network
        sys.path.insert(0, str(a.ffs_dir))
        from core.utils.utils import InputPadder                     # noqa: E402 (Fast-FoundationStereo code)
        ck = next((p for p in (a.ffs_dir/'weights/weights'/a.ckpt, a.ffs_dir/'weights'/a.ckpt) if (p/'model_best_bp2_serialize.pth').exists()), None)
        if ck is None: raise SystemExit(f'no checkpoint {a.ckpt} under {a.ffs_dir}/weights')
        torch.backends.cudnn.benchmark = True; torch.autograd.set_grad_enabled(False)
        model = torch.load(ck/'model_best_bp2_serialize.pth', map_location='cpu', weights_only=False)   # NVlabs' serialized model (full pickle)
        model.args.valid_iters = a.iters; model.args.max_disp = a.max_disp; model.cuda().eval()
        padder = InputPadder((1, 3, H, W), divis_by=32, force_square=False); xx = torch.arange(W, device='cuda')[None].float()

        def ffs_depth(L, Rt):
            t = torch.from_numpy(np.stack([L, Rt])).cuda().float()    # (2, H, W) 0..255; the model normalises inside
            i0, i1_ = padder.pad(t[0][None, None].repeat(1, 3, 1, 1), t[1][None, None].repeat(1, 3, 1, 1))
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
                d = model.forward(i0, i1_, iters=a.iters, test_mode=True, optimize_build_volume='triton')
            d = padder.unpad(d.float())[0, 0]
            ok = (d > .5) & (xx-d >= 0) & (d < a.max_disp-1)
            return torch.where(ok, fx*B/d.clamp(min=1e-3), 0.)
        align = make_aligner((i1.k[0], i1.k[4], i1.k[2], i1.k[5]), W, H, (ci.k[0], ci.k[4], ci.k[2], ci.k[5]), ci.width, ci.height, R_c1, t_c1)

        # ------------------------------------------------------------------ output bag
        w = Writer(out, version=8)
        if a.zstd or rd.compression_mode: w.set_compression(CompressionMode.MESSAGE, CompressionFormat.ZSTD)
        for k, v in (rd.storage.metadata.custom or {}).items(): w.set_custom_data(k, str(v))
        w.set_custom_data('ffs', f'Fast-FoundationStereo {a.ckpt}, {a.iters} iters, max_disp {a.max_disp}, from {a.bag.name}')
        w.open()
        skip = set() if a.keep_ir else {x for x in by if x.startswith(('/camera/infra1/', '/camera/infra2/'))}
        oc = {c.topic: w.add_connection(c.topic, c.msgtype, typestore=ts, offered_qos_profiles=c.ext.offered_qos_profiles)
              for c in rd.connections if c.topic not in skip}
        frames = {'color': ('/camera/ffs/aligned_depth_to_color', ci), 'depth': ('/camera/ffs/depth', i1)}
        frames = {k: v for k, v in frames.items() if a.frame in (k, 'both')}
        for k, (top, _) in frames.items():
            oc[top+'/image'] = w.add_connection(top+('/image_raw' if k == 'color' else '/image_rect_raw'), 'sensor_msgs/msg/Image', typestore=ts)
            oc[top+'/info'] = w.add_connection(top+'/camera_info', 'sensor_msgs/msg/CameraInfo', typestore=ts)

        def put(key, t, msg): w.write(oc[key], t, ts.serialize_cdr(msg, msg.__msgtype__))
        pend, n, t0, n_all = {}, 0, time.time(), by['/camera/infra1/image_rect_raw'].msgcount
        try:
            for c, t, raw in rd.messages():
                if c.topic in oc: w.write(oc[c.topic], t, raw)
                if c.topic not in need[:2]: continue
                m = des(raw, c.msgtype); st = (m.header.stamp.sec, m.header.stamp.nanosec)
                pend.setdefault(st, {})[c.topic] = (t, m.data.reshape(m.height, m.width))
                if len(pend[st]) < 2: continue
                p = pend.pop(st); tb = p[need[0]][0]
                z = ffs_depth(p[need[0]][1], p[need[1]][1])
                for k, (top, info) in frames.items():
                    zz = (align(z) if k == 'color' else z).cpu().numpy().astype(np.float32); zz[zz <= 0] = np.nan
                    hdr = T['std_msgs/msg/Header'](stamp=m.header.stamp, frame_id=info.header.frame_id)
                    put(top+'/image', tb, T['sensor_msgs/msg/Image'](header=hdr, height=zz.shape[0], width=zz.shape[1], encoding='32FC1', is_bigendian=0,
                                                                    step=zz.strides[0], data=zz.reshape(-1).view(np.uint8)))
                    put(top+'/info', tb, dataclasses.replace(info, header=hdr))
                n += 1
                if n % 100 == 0 or n == n_all:
                    el = time.time()-t0; print(f'  {n}/{n_all} frames, {n/el:.1f} fps, {(n_all-n)/max(n/el, 1e-6):.0f} s left', flush=True)
            if pend: print(f'warning: {len(pend)} IR images without their pair (skipped)')
        finally:
            w.close()
    sz = sum(f.stat().st_size for f in out.iterdir())/2**20
    print(f'wrote {out}/ ({sz:.0f} MB): {n} FFS depth frames in {", ".join(t for t, _ in frames.values())}, {time.time()-t0:.0f} s')


if __name__ == '__main__':
    main()
