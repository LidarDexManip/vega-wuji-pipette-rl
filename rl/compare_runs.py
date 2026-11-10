"""(11-09) One table over runs evaluated by node_eval.sh: label=DIR with fromstart / train_starts / holdout / holdplace (eval_rr.py),
contact (eval_contact_stability.py), rb (reward_breakdown.py) .json; missing files print '-'.
  python3 compare_runs.py R3B=runs/local_eval/rr_T1k_r3B_s5 tc=runs/local_eval/rr_T1k_tc_s5 ... [--json out.json]"""
import json, sys
from pathlib import Path

args = [a for a in sys.argv[1:] if '=' in a]; out_json = sys.argv[sys.argv.index('--json')+1] if '--json' in sys.argv else None
runs = [(a.split('=', 1)[0], Path(a.split('=', 1)[1])) for a in args]


def load(d, name):
    f = d/f'{name}.json'
    return json.loads(f.read_text()) if f.exists() else None


def nat(d, key_start):
    n = (load(d, 'fromstart') or {}).get('natural', {})
    return next((v for k, v in n.items() if k.startswith(key_start)), None)


rows = {}
def put(metric, label, v): rows.setdefault(metric, {})[label] = v

for lab, d in runs:
    for s in ('fromstart', 'train_starts', 'holdout', 'holdplace'):
        e = load(d, s)
        if e: put(f'{s}: reach / tube seated / press (n={e["n"]})', lab, f'{e["reach_end"]} / {e["tube_seated"]} / {e["press_ok_of_press_starts"].split("/")[0]}')
    sh = nat(d, 'shake')
    if sh:
        w = sh['wrist_dev_band_mm [R, L]']
        for ph in ('carry', 'press', 'post_release'):
            put(f'wrist 8-24 Hz {ph} R/L mm (<= 0.1)', lab, '/'.join(f'{x:.3f}' for x in w[ph]['8-24Hz']))
        put('wrist 3-8 Hz carry R/L mm (<= 0.3)', lab, '/'.join(f'{x:.3f}' for x in w['carry']['3-8Hz']))
    io = nat(d, 'idle_fingers_over'); il = nat(d, 'idle_fingers [')
    if io is not None and il is not None: put('idle fingers over limits', lab, f'{len(io)}/{len(il)}')
    ff = nat(d, 'free_finger_hp_rms'); ar = nat(d, 'action_rate_mean_sq')
    if ff is not None: put('free-finger high-pass RMS deg p50/p95', lab, f'{ff[0]:.2f}/{ff[1]:.2f}')
    if ar is not None: put('action rate mean sq (eval_rr)', lab, f'{ar:.2e}')
    c = load(d, 'contact')
    if c:
        put('contacts lost per step', lab, f'{c["lost_fraction_per_step"]["mean"]:.4f}')
        put('contact drift / slip mm per step', lab, f'{c["drift_mm_per_step (penalty units, side-averaged, capped 5)"]["mean"]:.3f} / {c["slip_mm_per_step (same)"]["mean"]:.3f}')
        h = c['holds_required_wrench_share (ref-held steps)']; put('holds required wrench pipette / tube', lab, f'{h["pipette"]:.3f} / {h["tube"]:.3f}')
    r = load(d, 'rb')
    if r:
        raw = r.get('raw', {})
        for k in ('p_torque_rate_arm', 'p_torque_rate_finger'):
            if k in raw: put(f'{k[2:]} raw mean (1 = 1 % of limit per step)', lab, f'{raw[k]["mean"]:.3f}')
        put('reward per step (own config)', lab, f'{r["total_mean"]:.2f}')

labs = [l for l, _ in runs]; w0 = max(len(m) for m in rows) + 2; w = max(14, *(len(l)+2 for l in labs))
print(f'{"":{w0}s}' + ''.join(f'{l:>{w}s}' for l in labs))
for m, v in rows.items(): print(f'{m:{w0}s}' + ''.join(f'{str(v.get(l, "-")):>{w}s}' for l in labs))
if out_json: Path(out_json).write_text(json.dumps(rows, indent=1)+'\n')
