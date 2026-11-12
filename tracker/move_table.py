"""(11-12) Desk placement relative to the robot, matched to the real robot (user: keep the sim robot's position relative to the desk
close to reality). Copies a task and moves the NEAR (robot-side) edge of the table (furn_table_0 top + the two near legs) by --near-dx
along world +x (towards the robot; the robot base stands at +x looking along -x). The far edge, the holders, the objects, the robot and the
reference stay where they are, so the task geometry relative to the robot is unchanged; only the desk edge / the desk under the arms moves.
Table HEIGHT is not changed here: the holders and objects sit on the table, so a height change moves the whole task (rebuild + retrain).

Measured 11-12 from the real head ZED (screenshot, depth recovered from the viewer's colour bar; same table-plane fit validated on the sim
render to 0.05 cm): the real near edge is 33.3 cm ahead of the head camera (horizontal), the sim's 46.65 cm -> --near-dx 0.133.
  .venv-rl-head/bin/python -I tracker/move_table.py tasks/task_d02_straight_v3 tasks/task_d02_straight_v3_desk --near-dx 0.133
"""
import argparse, json, shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('src', type=Path); ap.add_argument('dst', type=Path)
ap.add_argument('--near-dx', type=float, required=True, help='move the robot-side table edge this far towards the robot (m; negative = away)')
ap.add_argument('--force', action='store_true'); a = ap.parse_args()
if a.dst.exists():
    assert a.force, f'{a.dst} exists (--force to overwrite)'; shutil.rmtree(a.dst)
shutil.copytree(a.src, a.dst)   # ego/ (head-camera estimator error model) is kept: characterised with the old desk edge, which lies outside its search windows

f = lambda s: np.array([float(x) for x in s.split()])
tree = ET.parse(a.dst/'scene.xml'); G = {g.get('name'): g for g in tree.getroot().iter('geom') if (g.get('name') or '').startswith('furn_table_')}
top = G['furn_table_0']; p, s = f(top.get('pos')), f(top.get('size')); assert (top.get('quat') or '1 0 0 0').split()[0] in ('1', '1.0'), 'rotated table'
near0, far0 = p[0]+s[0], p[0]-s[0]; near1 = near0+a.near_dx
p[0], s[0] = (near1+far0)/2, (near1-far0)/2
top.set('pos', ' '.join(f'{x:.9f}' for x in p)); top.set('size', ' '.join(f'{x:.9f}' for x in s))
moved = []
for n, g in G.items():
    if n == 'furn_table_0': continue
    q = f(g.get('pos'))
    if q[0] > (near0+far0)/2: q[0] += a.near_dx; g.set('pos', ' '.join(f'{x:.9f}' for x in q)); moved.append(n)
tree.write(a.dst/'scene.xml', encoding='unicode')

tj = json.loads((a.dst/'task.json').read_text())
tj['table_edit'] = {'src': str(a.src), 'near_dx_m': a.near_dx, 'near_edge_x': [round(near0, 4), round(near1, 4)], 'far_edge_x': round(far0, 4),
                    'moved_legs': moved, 'note': 'tracker/move_table.py (11-12): desk edge matched to the real robot; holders / objects / reference unchanged'}
(a.dst/'task.json').write_text(json.dumps(tj, indent=1)+'\n')
print(json.dumps(tj['table_edit'], indent=1))
