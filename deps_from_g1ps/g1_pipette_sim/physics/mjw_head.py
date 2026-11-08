"""W28 (11-04): helpers for the DAID physics worker on upstream mujoco_warp (--mjw-head). Used only with that flag or from the head env; nothing on the pinned stack imports this.

Stack: mujoco 3.17.0, warp-lang 1.19.0, mujoco_warp 4911167e3f2d, DAID retargeting package = 824591b plus the working-tree diff in <projects>/do-as-i-do-mjwhead
(h13_work/daid_port/daid_mjwhead.patch). mjw_head_manifest.json holds sha256 of every file of the mujoco_warp package and the port, written by tools/daid_physics_worker.py's
sibling build_manifest below, so a run cannot silently use something else.

Overflow bits: Data.overflow is a per-world sticky bitmask (never reset by step); the worker's W6 folds it into overflow.json.
  hard (contacts or rows were dropped): NEFC, NJMAX_NNZ, BROADPHASE, NARROWPHASE, CCD, HFIELD, NVMAX. Same meaning as ncollision / nacon / nefc > capacity on the pin, plus the new capacities.
  contact-affecting, reported as a count, not fatal: EPA_HORIZON. EPA gave up on that pair for that step. The pin has the same limit with a print only, no flag.
  info: ITERATIONS, LS_ITERATIONS (solver stopped at its limit: the pin does this silently), CONTACT_MATCH, TACTILE (no such sensors here).
"""
from __future__ import annotations

import hashlib
import json
from importlib import metadata
from pathlib import Path

MANIFEST = Path(__file__).with_name('mjw_head_manifest.json')
VERSIONS = {'mujoco': '3.17.0', 'warp-lang': '1.19.0', 'mujoco-warp': '3.17.0'}
MJW_COMMIT = '4911167e3f2deebe5843fab28a01dec733343803'
DAID_BASE_COMMIT = '824591b808c342b20079c3b4198a8c2bdf88c74e'
DEFAULT_PORT_ROOT = Path('do-as-i-do-mjwhead/retargeting')   # (public copy) DAID port checkout; unused by the RL code
BITS = ('NEFC', 'NJMAX_NNZ', 'BROADPHASE', 'NARROWPHASE', 'CCD', 'HFIELD', 'CONTACT_MATCH', 'NVMAX', 'EPA_HORIZON', 'ITERATIONS', 'LS_ITERATIONS', 'TACTILE')
HARD = ('NEFC', 'NJMAX_NNZ', 'BROADPHASE', 'NARROWPHASE', 'CCD', 'HFIELD', 'NVMAX')


def _hash_tree(root, patterns):
    out = {}
    for pat in patterns:
        for f in sorted(Path(root).glob(pat)):
            if f.is_file() and '__pycache__' not in f.parts: out[str(f.relative_to(root))] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def _mjw_dir():
    import importlib.util
    return Path(importlib.util.find_spec('mujoco_warp').origin).parent


def build_manifest(port_root=DEFAULT_PORT_ROOT):
    man = {'mjw_commit': MJW_COMMIT, 'daid_base_commit': DAID_BASE_COMMIT, 'versions': VERSIONS, 'mujoco_warp': _hash_tree(_mjw_dir(), ['**/*.py']),
           'port': _hash_tree(port_root, ['retargeting/**/*.py', 'config/**/*.yaml'])}
    MANIFEST.write_text(json.dumps(man, indent=0)); return man


def verify_stack(port_root=DEFAULT_PORT_ROOT):
    """Exact versions, every mujoco_warp and port file as frozen. Raises otherwise. Call before mujoco_warp is imported."""
    bad = {k: metadata.version(k) for k, v in VERSIONS.items() if metadata.version(k) != v}
    if bad: raise RuntimeError(f'--mjw-head expects {VERSIONS}, this env has {bad}')
    man = json.loads(MANIFEST.read_text())
    h = _hash_tree(_mjw_dir(), ['**/*.py']); d = sorted(k for k in set(h) | set(man['mujoco_warp']) if h.get(k) != man['mujoco_warp'].get(k))
    if d: raise RuntimeError(f'mujoco_warp differs from the manifest: {d[:5]}')
    h = _hash_tree(port_root, ['retargeting/**/*.py', 'config/**/*.yaml']); d = sorted(k for k in set(h) | set(man['port']) if h.get(k) != man['port'].get(k))
    if d: raise RuntimeError(f'DAID port {port_root} differs from the manifest: {d[:5]}')
    return {'commit': f"{man['daid_base_commit']}+port", 'mjw_commit': man['mjw_commit'], 'versions': man['versions']}


def overflow_report(d, wp):
    """Worlds with each Data.overflow bit set so far, the new capacities, and the hard verdict. Host read (only at W6's checks)."""
    import numpy as np
    from mujoco_warp._src.types import OverflowType
    v = wp.to_torch(d.overflow).cpu().numpy()
    worlds = {b: int(((v & int(OverflowType[b])) != 0).sum()) for b in BITS}
    return {'overflow_bits_worlds': worlds, 'overflow_hard_bits': [b for b in HARD if worlds[b]], 'naccdmax': int(d.naccdmax), 'njmax_nnz': int(d.njmax_nnz), 'nvmax': int(d.nvmax),
            'overflow_worlds_any_hard': int(np.count_nonzero(v & sum(int(OverflowType[b]) for b in HARD)))}


if __name__ == '__main__':   # refresh the manifest after a deliberate change of the port
    print({k: (len(v) if isinstance(v, dict) else v) for k, v in build_manifest().items()})
