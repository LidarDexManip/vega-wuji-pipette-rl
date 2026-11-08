# AUDIT (2026-11-07, before the first commit)

All checks ran locally on CPU. Nothing was pushed.

## 1. Secrets and infra

**Text grep.** The pattern `hding|100\.121|ssh |@[a-z0-9-]+\.|/home/|token|api[_-]?key|password|gatech|@gmail` was run case-insensitively over every file except `.git/` and this file, which contains the pattern itself.

- 18 hits. All 18 are the Python matrix-multiply operator followed by an attribute, e.g. `rinv(W)@self.W_ref[s]`, `@torch.no_grad()`, `Jw@Jw.transpose`, `@wp.func` in a docstring.
- None is an email, host or path.
- With those excluded: **0 hits**.

**Extra infra grep.** Pattern: `/data/`, `/tmp/claude`, `miniconda`, personal email names, `.venv-rl-head/bin`.

- The only hits are the relative venv name `.venv-rl-head/bin/python`, in docstring usage lines and in `requirements_rl_head.txt`. This is harmless.

**Binary scan.** All 30 binary files (`*.pt`, `*.npz`, `*.npy`, `*.mp4`) were scanned:

- the full content of every zip member;
- npz key names.

Result: **0 hits**. The only matches were in torch's internal zip member names, `final/data/<n>`, which are not content.

**Video metadata.** Container tags are only `major_brand`, `minor_version`, `compatible_brands` and `encoder=Lavf60.16.100`. One frame of each video was checked: the burned-in labels show the layer, round and timing only.

**What was scrubbed** (per file in `MANIFEST.md`):

| Original | Replaced by | Where |
|---|---|---|
| node user@host | `<user>@<node>` | — |
| `/data/<user>/ckpts/rrline/` | `<node_ckpts>/` | `ckpt` fields in eval jsons |
| node code root | `<node_code>/` | `state.json` |
| local work-dir prefix | `<work>/` | `task.json` `stage` / `scene_source`, docs |
| g1-pipette-sim checkout | `<g1-pipette-sim>` | — |
| conda path | `<conda>` | — |
| home dir | `<home>` / `<projects>` | — |

**Code path defaults.** `G1PS_ROOT` now defaults to `deps_from_g1ps/` instead of `/home/...`. In `render_rollout.py`, `PY` is `sys.executable`. In `mjw_head.py`, `DEFAULT_PORT_ROOT` is a relative placeholder.

**Excluded entirely:**

- all `logs_*.txt` and `*.log` files;
- `node_pkg/` (launch scripts with node host and paths, plus tarballs);
- local `run_*.sh` launchers;
- `fetch_and_eval.sh` (ssh/rsync);
- `done.marker` and `*_DONE` files.

## 2. Sizes

- Total: 220,502,969 bytes (about 210 MB) in 264 files, excluding `.git`.
- No file is over 95 MB.

Files over 10 MB:

| File | Size |
|---|---|
| `results/rr_T1k_full_lp_s0/final.pt` | 20.3 MB |
| `results/rr_T1k_r2_s2/final.pt` | 20.3 MB |
| `results/rr_T1k_r2w_s3/final.pt` | 20.3 MB |
| `results/rr_T1k_r3A_s4/final.pt` | 20.3 MB |
| `results/rr_T1k_r3B_s5/final.pt` | 20.3 MB |
| `tracker/evals/ckpt_75M.pt` | 20.3 MB |
| `videos/pipette_vega_wuji_R3_reference_vs_R3B_hands_closeup_with_MPC_50fps.mp4` | 14.8 MB |

Only `final.pt` is kept per round, plus the tracker's `ckpt_75M.pt`. No `model_*.pt` intermediates, `__pycache__`, `MUJOCO_LOG.TXT`, tarballs or `*_render_scene*.xml`.

## 3. Code

**Compile.** `python -m py_compile` on all 43 `.py` files: **OK**. Stack: mujoco 3.17, warp 1.19, torch 2.12, rsl_rl 5.5.2.

**Unit test.** `rl/test_arm_clamp.py`, CPU, no MuJoCo: **PASS**. The commanded-target margin is −0.075 rad with `clamp_after_ff` off and 0.100 rad with it on.

**Import smoke test** (CPU, `CUDA_VISIBLE_DEVICES=""`, 1.1 s):

- These import from inside the repo: `vega_extras`, `vega_res_env`, `distill` (rl/), `gpu_train` (deps_from_g1ps/tools/rl), `tracker_env`, `rewards`, `arm_ik`, `g1_pipette_sim.retarget.gpu.core`, `g1_pipette_sim.physics.mjw_head`.
- All 6 task packages load (`reference.npz`, `valid_starts.npy`).
- Every `rl/configs/rr_T1k_*.json` and `tracker/configs/stage0*.json` resolves its `task` directory.

No GPU env was built, because a training run was live on the GPU.

**Scene check.**

- The rewritten `scene.xml` (mesh paths → `../../assets/...`) was compiled against a temporary tree of symlinks to the original meshes.
- It gives an identical model: nq / nv / nu / nbody / ngeom / nmesh / npair / mesh vertices / geom sizes and positions / body masses all equal.
- All 6 task `scene.xml` files are byte-identical.

## 4. Mesh files

`find` for `*.stl`, `*.obj`, `*.dae`, `*.ply`, `*.glb` and `*.usd*` (case-insensitive) returns **0 files**.

The 112 required meshes are listed, as paths only, in `assets/asset_list.tsv`. `.gitignore` ignores `assets/*` except that list.

## 5. MANIFEST

`MANIFEST.md` lists every file with its source path and size. It was cross-checked against `find`: no file is missing or extra.
