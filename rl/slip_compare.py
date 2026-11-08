"""Slip / tilt of the tube in the left hand, policy vs zero residual on the SAME perturbations (eval_rr per_world rows; layer: RL env).
Only FULL-WINDOW episodes count (start at or before the slip-window start, so the whole [dwell end, held end] window is inside the episode;
later starts see a partial or empty window and would add zeros). (a) all full-window episodes, (b) full-window worlds where BOTH reach the
end, (c) per phase over (b): carry (dwell end -> press), meeting / press, after the press (carry back + place). Median, p90, n.
  python slip_compare.py POLICY.json ZERO.json [--t0 3.0333]"""
import json, sys
import numpy as np
a = sys.argv[1:]; t0 = float(a[a.index('--t0')+1]) if '--t0' in a else 3.0333333
JP, JZ = (json.load(open(f)) for f in a[:2]); P, Z = JP['per_world'], JZ['per_world']; ws0 = JP['tube_slip_checker_def']['window_s'][0]
full = t0+np.array(P['start_frame'])*.01 <= ws0+1e-6
eP, eZ = np.array(P['end_ok']), np.array(Z['end_ok']); both = eP & eZ & full
def st(x, sel): x = np.array(x)[sel]; return (round(float(np.median(x)), 2), round(float(np.percentile(x, 90)), 2), int(sel.sum())) if sel.any() else None
out = {'full_window_episodes': int(full.sum()), 'reach_end_full_window': {'policy': int((eP & full).sum()), 'zero': int((eZ & full).sum()), 'both': int(both.sum())}}
for lab, sel in (('a_all_full_window', full), ('b_both_reach_end', both)):
    out[lab] = {'slip_mm_p50_p90_n': {'policy': st(P['slip_mm'], sel), 'zero': st(Z['slip_mm'], sel)}, 'tilt_deg_p50_p90_n': {'policy': st(P['tilt_deg'], sel), 'zero': st(Z['tilt_deg'], sel)}}
out['c_per_phase_both_reach_end'] = {ph: {'slip_mm': {'policy': st(P['slip_phase_mm'][ph], both), 'zero': st(Z['slip_phase_mm'][ph], both)},
                                          'tilt_deg': {'policy': st(P['tilt_phase_deg'][ph], both), 'zero': st(Z['tilt_phase_deg'][ph], both)}} for ph in ('carry', 'press', 'place')}
print(json.dumps(out, indent=1))
