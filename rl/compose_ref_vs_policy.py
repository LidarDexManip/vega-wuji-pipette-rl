"""Hand close-up grid from rendered 50 fps videos (render_rollout / render_vega_visual, 1280x960, 2x2 panels: bottom-left = left hand (tube),
bottom-right = right hand (pipette)). Columns = the input videos (titles), top row = right hand, bottom row = left hand. Each row carries the
ACTIVE REFERENCE finger event of that hand (finger_event_labels.json; kinematic reference of the RL task) + the reference finger speed.
Slow-down: every frame written SLOW times at 50 fps (2 = 0.5x).
  python compose_ref_vs_policy.py LABELS.json OUT.mp4 --col VIDEO "TITLE" [--col ...] [--slow 2]"""
import argparse, json, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
p = argparse.ArgumentParser(); p.add_argument('labels'); p.add_argument('out'); p.add_argument('--col', nargs=2, action='append', required=True)
p.add_argument('--slow', type=int, default=2); a = p.parse_args()
L = json.load(open(a.labels)); T = L['time']; W, H, w, h, HDR = 1280, 960, 640, 480, 44
F = lambda s, b=False: ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf' % ('-Bold' if b else ''), s)
f_t, f_e, f_s, f_h = F(17, True), F(19, True), F(14), F(18, True)
rd = [subprocess.Popen(['ffmpeg', '-v', 'error', '-i', v, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE) for v, _ in a.col]
nc = len(a.col); OW, OH = nc*w, 2*h+HDR
wr = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{OW}x{OH}', '-r', '50', '-i', '-',
                       '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', a.out], stdin=subprocess.PIPE)
EVC = {'IDLE': (120, 255, 120), 'HOLD (reference ~still': (160, 230, 160)}
k = 0
while True:
    fr = [r.stdout.read(W*H*3) for r in rd]
    if any(len(x) < W*H*3 for x in fr): break
    fr = [np.frombuffer(x, np.uint8).reshape(H, W, 3) for x in fr]; kk = min(k, len(T)-1)
    out = Image.new('RGB', (OW, OH), (0, 0, 0)); d = ImageDraw.Draw(out)
    d.text((10, 10), 't = %.2f s (RL reference clock)   %s   frame %d   |   REF EVENT = active finger event of the hold-still base reference (RL task)' % (T[kk], '%.1fx speed' % (1/a.slow), k), font=f_h if nc > 2 else f_s, fill=(255, 255, 255))
    for row, (side, y0, x0, nm) in enumerate((('right', 480, 640, 'right hand (pipette)'), ('left', 480, 0, 'left hand (tube)'))):
        ev = L[side]['labels'][kk]; sub = L[side]['sub'][kk]; col = next((c for key, c in EVC.items() if ev.startswith(key)), (255, 210, 80))
        for c in range(nc):
            tile = Image.fromarray(fr[c][y0:y0+h, x0:x0+w]); out.paste(tile, (c*w, HDR+row*h)); X, Y = c*w, HDR+row*h
            d.rectangle([X, Y, X+w-1, Y+26], fill=(0, 0, 0)); d.text((X+6, Y+4), a.col[c][1], font=f_t, fill=(255, 255, 255))
            d.rectangle([X, Y+h-58, X+w-1, Y+h-1], fill=(0, 0, 0)); d.text((X+6, Y+h-56), f'REF EVENT: {ev}', font=f_e, fill=col)
            d.text((X+6, Y+h-26), f'{nm} | {sub}', font=f_s, fill=(220, 220, 220))
            if c: d.line([X, Y, X, Y+h], fill=(255, 255, 255), width=2)
        d.line([0, HDR+h, OW, HDR+h], fill=(255, 255, 255), width=2)
    b = np.asarray(out).tobytes()
    for _ in range(a.slow): wr.stdin.write(b)
    k += 1
wr.stdin.close(); wr.wait(); print('frames in', k, 'labels', len(T), 'out frames', k*a.slow)
