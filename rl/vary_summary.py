"""(11-10) Variation study of a policy (rl/record_rollouts.py runs; KA in results/kin_tc_s5/vary/): outcome per world, failure cause by object
and task phase (times from the kinematic reference timeline), success by pick / place hole and by stand yaw.
  python3 rl/vary_summary.py results/kin_tc_s5/vary          (reads <name>_summary.json, or <name>/summary.json as record_rollouts.py writes it)
Full success = tube seated + pipette seated + press ok (eval_rr.py press rule) at the end of the clip."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

V = Path(sys.argv[1])
TRAIN = {tuple(p) for p in json.load(open(Path(__file__).resolve().parents[1]/'tracker/configs/kin_tc_s5.json'))['extras']['holder']['hole_pairs']}


def cause(w):
    if w['tube_seated'] and w['pipette_seated'] and w['press_ok']: return 'success'
    t = w['first_fail_s']; pr, tr = w.get('fail_reason') or [None, None]
    if t < 0:
        if not w['tube_seated']: return 'end: tube not seated'
        if not w['pipette_seated']: return 'end: pipette not seated'
        return 'end: press not done'
    if pr:
        if t < 5.3: return 'pipette: extraction from the stand'
        if t < 23.1: return 'pipette: carry / press'
        if t < 25.4: return 'pipette: return into the stand'
        return 'pipette: after the release'
    if t < 8.3: return 'tube: grasp / lift from the rack'
    if t < 15.1: return 'tube: carry / press'
    if t < 20.4: return 'tube: carry back'
    if t < 21.8: return 'tube: entering the rack'
    return 'tube: after the release'


def load(name):
    p = V/f'{name}_summary.json'
    if not p.exists(): p = V/name/'summary.json'
    return json.loads(p.read_text())['worlds'] if p.exists() else None


def ok(w): return cause(w) == 'success'


for name in ('rand64', 'holes', 'standyaw'):
    W = load(name)
    if W is None: continue
    n = len(W); c = Counter(cause(w) for w in W)
    print(f'\n=== {name}: {n} worlds | reach end {sum(w["reach_end"] for w in W)} | tube seated {sum(w["tube_seated"] for w in W)} | '
          f'pipette seated {sum(w["pipette_seated"] for w in W)} | press ok {sum(w["press_ok"] for w in W)} | full success {c["success"]}')
    for k, v in c.most_common(): print(f'   {v:3d}  {k}')
    if name == 'rand64':
        for i, w in enumerate(W):
            if not ok(w): print(f'   world {i:2d}: {cause(w):36s} t={w["first_fail_s"]:6.2f} {w.get("fail_reason")} spec {w["spec"]}')
    if name == 'holes':
        for key, lab in ((0, 'pick hole'), (1, 'place hole')):
            g = defaultdict(list)
            for w in W: g[w['spec']['pair'][key]].append(w)
            print(f'   by {lab}: ' + '  '.join(f'{h} {sum(map(ok, L))}/{len(L)}' for h, L in sorted(g.items())))
        ut = [w for w in W if tuple(w['spec']['pair']) not in TRAIN]; tr = [w for w in W if tuple(w['spec']['pair']) in TRAIN]
        print(f'   trained pairs {sum(map(ok, tr))}/{len(tr)}   never-trained pairs {sum(map(ok, ut))}/{len(ut)}')
        print('   causes, never-trained pairs:', dict(Counter(cause(w) for w in ut)))
    if name == 'standyaw':
        g = defaultdict(list)
        for w in W: g[w['spec']['stand'][2]].append(w)
        for y, L in sorted(g.items()):
            cc = Counter(cause(w) for w in L); cc.pop('success', None)
            print(f'   stand yaw {y:+4.0f} deg: success {sum(map(ok, L)):2d}/{len(L)}   pipette seated {sum(w["pipette_seated"] for w in L):2d}  press {sum(w["press_ok"] for w in L):2d}   {dict(cc)}')
