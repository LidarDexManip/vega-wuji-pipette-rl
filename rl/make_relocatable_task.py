"""Relocatable copy of an RL task (node readiness, 11-06): every absolute file="..." in scene.xml (meshes from g1-pipette-sim,
g1-wuji-retargeting, RFT) is copied into OUT/assets/ and the XML points at it relatively; the copy is checked to compile to the same model
(sizes, mesh vertex / face counts, geom sizes and positions).   python make_relocatable_task.py TASK_DIR OUT_DIR"""
import hashlib, re, shutil, sys
from pathlib import Path
import numpy as np, mujoco


def main():
    src, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    if out.exists(): raise SystemExit(f'{out} exists')
    shutil.copytree(src, out); (out/'assets').mkdir(); txt = (out/'scene.xml').read_text(); seen = {}
    def rep(mt):
        p = Path(mt.group(1))
        if not p.is_absolute(): return mt.group(0)
        if str(p) not in seen:
            nm = f"{hashlib.sha1(str(p.parent).encode()).hexdigest()[:8]}_{p.name}"; shutil.copy2(p, out/'assets'/nm); seen[str(p)] = nm
        return f'file="assets/{seen[str(p)]}"'
    txt2 = re.sub(r'file="([^"]+)"', rep, txt); (out/'scene.xml').write_text(txt2)
    a, b = mujoco.MjModel.from_xml_path(str(src/'scene.xml')), mujoco.MjModel.from_xml_path(str(out/'scene.xml'))
    same = all(getattr(a, k) == getattr(b, k) for k in ('nq', 'nv', 'nu', 'nbody', 'ngeom', 'nmesh', 'npair', 'nmeshvert', 'nmeshface'))
    same &= np.allclose(a.geom_size, b.geom_size) and np.allclose(a.geom_pos, b.geom_pos) and np.allclose(a.mesh_vert, b.mesh_vert)
    if not same: raise SystemExit('relocated scene compiles differently')
    left = re.findall(r'file="(/[^"]+)"', txt2)
    print({'files_copied': len(seen), 'absolute_paths_left': len(left), 'identical_model': bool(same), 'out': str(out)})


if __name__ == '__main__':
    main()
