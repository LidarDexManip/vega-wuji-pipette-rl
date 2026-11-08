"""Data gate before RL training (hard rule; real-robot line). Reads what the pipeline already wrote and prints PASS / FAIL per item:
  1. demo seed: status_T1.json has no failing item other than the deferred ones (layer: physics demo, gate rrline-t1-v2);
  2. exported reference: zero-residual replay reaches the end from every start with no DR (layer: RL env; replay_check.py output);
  3. contact labels sane (layer: reference labels): every grasp finger of the demo has labelled contact, left ring / pinky free;
  4. valid-start audit keeps >= 95 % of the candidates (layer: RL env);
  5. smoothed RL reference: grasp (latched) fingers change <= 1 deg vs the executed demo (ctrl and qpos) (layer: export).
  python data_gate.py RUN_DIR TASK_DIR REPLAY_JSON [--out J]"""
import argparse, json
from pathlib import Path
p = argparse.ArgumentParser(); p.add_argument('run', type=Path); p.add_argument('task', type=Path); p.add_argument('replay', type=Path); p.add_argument('--out', type=Path); a = p.parse_args()
st = json.loads((a.run/'status_T1.json').read_text()); deferred = set(st.get('deferred_but_failing', []))
nd = [f for f in st['failing'] if f not in deferred and not f.startswith('[deferred]')]
items = []
items.append(('1 demo seed: no non-deferred failing item', not nd, {'non_deferred_failing': nd, 'deferred_printed_failing': sorted(deferred)}))
rp = json.loads(a.replay.read_text()); items.append(('2 zero-residual replay, no DR, reaches the end from every start', rp['dr'] is None and rp['reach_end_without_failure'] == rp['n'],
                                                     {'reach_end': f"{rp['reach_end_without_failure']}/{rp['n']}", 'dr': rp['dr'], 'fails': [r for r in rp['per_start'] if not r['ok']][:6]}))
cr = json.loads((a.task/'contact_ref.json').read_text()); cf = cr['contact_fraction_per_finger']
need = ['r_index', 'r_middle', 'r_ring', 'r_pinky', 'l_thumb', 'l_index']
ok3 = all(cf.get(f, 0.) > 0.05 for f in need) and cr.get('free_fraction_left_ring_pinky', 0.) >= .95
items.append(('3 contact labels: grasp fingers labelled (>5 % of frames), left ring / pinky free (>= 95 %)', ok3, {'contact_fraction_per_finger': cf, 'free_fraction_left_ring_pinky': cr.get('free_fraction_left_ring_pinky')}))
au = json.loads((a.task/'valid_starts_audit.json').read_text()); frac = au['kept']/au['candidates']
items.append(('4 valid-start audit keeps >= 95 %', frac >= .95, {'kept': au['kept'], 'candidates': au['candidates'], 'fraction': round(frac, 4)}))
tj = json.loads((a.task/'task.json').read_text()); fs = tj.get('finger_smoothing') or {}
lat = {j: (v['ctrl_max_change_deg_latched_free'][0], v['qpos_max_change_deg_latched']) for j, v in fs.items() if isinstance(v, dict)} if isinstance(fs, dict) else {}
worst = max([max(x) for x in lat.values()] or [None]) if lat else None
items.append(('5 smoothed reference: latched grasp fingers change <= 1 deg (ctrl and qpos)', bool(lat) and worst <= 1.0, {'worst_deg': worst, 'joints_over_1deg': {j: v for j, v in lat.items() if max(v) > 1.0}}))
ok = all(x[1] for x in items)
for name, passed, info in items: print(('PASS  ' if passed else 'FAIL  ')+name+'  '+json.dumps(info)[:600])
print('DATA GATE:', 'PASS' if ok else 'FAIL')
if a.out: a.out.write_text(json.dumps({'gate': 'rrline data gate (hard rule)', 'pass': ok, 'items': [{'item': n, 'pass': p_, 'info': i} for n, p_, i in items]}, indent=1))
