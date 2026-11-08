"""Tube seat misses and arm joint margins from eval_rr per_world_detail (layer: RL env rollouts; CPU analysis).
Seat rule (eval_rr): episode reaches the end AND tube within 5 mm / 3 deg of its holder-adjusted seated reference.
For every world that reaches the end but misses the seat: lateral offset vs the hole axis (mm), height vs seated (mm; > ~10 mm = resting on the rim /
not down), axis tilt (deg), and the tube state at the planned release frame (lateral / height vs the reference there: still high above the
seat at release = left the hand before entering the funnel). Breakdown by hole pair and by placement offset. Margins: commanded arm targets vs
executed q (deg; real-robot rule >= 5.73 deg on the commanded targets).
  python seat_miss_analysis.py EVAL.json [EVAL.json ...]"""
import json, sys, collections
import numpy as np
for f in sys.argv[1:]:
    d = json.load(open(f)); W, D = d['per_world'], d['per_world_detail']; n = len(W['end_ok'])
    end = np.array(W['end_ok']); seat = np.array(W['tube_seated']); miss = end & ~seat
    lat, dz, tl = np.array(D['end_tube_lat_mm']), np.array(D['end_tube_dz_mm']), np.array(D['end_tube_tilt_deg'])
    rl, rz = np.array(D['release_lat_mm'], dtype=float), np.array(D['release_dz_mm'], dtype=float)
    print(f'== {f}\n   worlds {n}, reach end {int(end.sum())}, tube seated {int(seat.sum())}, misses {int(miss.sum())}')
    if miss.any():
        cls = []
        for i in np.flatnonzero(miss):
            c = 'tilt' if tl[i] > 3 and lat[i] <= 5 and dz[i] <= 10 else ('on rim / not down' if dz[i] > 10 else ('lateral' if lat[i] > 5 else 'tilt'))
            pre = 'left hand above funnel (release dz %.0f mm)' % rz[i] if np.isfinite(rz[i]) and rz[i] > 25 else ('released inside / at funnel' if np.isfinite(rz[i]) else 'release frame not reached')
            cls.append((c, pre))
        print('   misses by class:', dict(collections.Counter(c for c, _ in cls)), '| release:', dict(collections.Counter(p.split(' (')[0] for _, p in cls)))
        q = lambda x: (round(float(np.median(x[miss])), 1), round(float(np.percentile(x[miss], 90)), 1))
        print('   misses: lateral mm p50/p90', q(lat), '| height mm p50/p90', q(dz), '| tilt deg p50/p90', q(tl))
        q2 = lambda x: (round(float(np.median(x[seat])), 1), round(float(np.percentile(x[seat], 90)), 1)) if seat.any() else None
        print('   seated (for scale): lateral', q2(lat), 'height', q2(dz), 'tilt', q2(tl))
        if D.get('hole_pair'):
            hp = np.array(D['hole_pair']); place = np.array([h.split('-')[1] for h in hp]); pick = np.array([h.split('-')[0] for h in hp])
            for lab, arr in (('place hole', place), ('pick hole', pick)):
                rows = {h: f'{int((miss & (arr == h)).sum())}/{int((end & (arr == h)).sum())}' for h in sorted(set(arr[end]))}
                print(f'   miss / reach-end by {lab}:', rows)
        if D.get('rack_dxy_mm'):
            rd = np.linalg.norm(np.array(D['rack_dxy_mm']), axis=1); ry = np.abs(np.array(D['rack_dyaw_deg']))
            for lab, x, edges in (('rack |dxy| mm', rd, [0, 20, 35, 50, 1e9]), ('rack |dyaw| deg', ry, [0, 4, 7, 10, 1e9])):
                b = np.digitize(x, edges[1:-1]); print(f'   miss / reach-end by {lab}:', {f'{edges[k]}-{edges[k+1] if edges[k+1] < 1e8 else "max"}': f'{int((miss & (b == k)).sum())}/{int((end & (b == k)).sum())}' for k in range(len(edges)-1)})
    mc, mq = np.array(D['arm_margin_cmd_deg']), np.array(D['arm_margin_q_deg'])
    print(f'   arm margin min over episodes: commanded {mc.min():.2f} deg (p5 {np.percentile(mc, 5):.2f}), executed q {mq.min():.2f} deg (p5 {np.percentile(mq, 5):.2f}); '
          f'episodes with commanded < 5.73: {int((mc < 5.73).sum())}/{n}, executed < 5.73: {int((mq < 5.73).sum())}/{n}')
