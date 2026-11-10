# MANIFEST

Every file in this repo, its source and its size.

Source roots:
- `<work>/` = `g1-pipette-sim/outputs/retarget_demo/pipette_g1_run01_cam2_gpu/h13_work/`, a git-ignored local work dir. Everything under it is ours.
- `<g1-pipette-sim>/` = the lab repo checkout. Only files that are untracked there, i.e. our own, were copied.
- `<node_code>/` = this repo's checkout on the GPU node (runs and evals of the second commit). `<repo>/` = a local checkout of this repo.

Notes column:
- "path fix": an import or path default was changed so the code finds its modules inside this repo. Behaviour is unchanged.
- "scrubbed": local, node or user paths were replaced by placeholders (`<work>`, `<node_ckpts>`, `<user>@<node>`, ...).

Sizes for README.md, MANIFEST.md and AUDIT.md are as of writing. Updated 2026-11-10 (second commit): new files are marked `(new...)`; files changed since the first commit are marked "updated 11-08..11-10". Updated 2026-11-10 (second commit): new files are marked `(new...)`; files changed since the first commit are marked "updated 11-08..11-10".

| File | Source | Size | Notes |
|---|---|---|---|
| `.gitignore` | `(new)` | 640 B | new; 11-10: Blender scenes, raw camera captures, superseded / duplicate videos |
| `AUDIT.md` | `(new)` | 7.4 KB | new |
| `MANIFEST.md` | `(new)` | 51.7 KB | new |
| `README.md` | `(new)` | 37.1 KB | new |
| `assets/asset_list.tsv` | `(generated from scene.xml)` | 17.5 KB | 112 mesh files expected |
| `common/arm_ik.py` | `<work>/vega/common/arm_ik.py` | 8.3 KB |  |
| `common/rack_visual/profile.json` | `<work>/rack_visual/profile.json` | 10.0 KB | rack hole-centre fits (numbers only) |
| `common/rewards.py` | `<work>/vega/common/rewards.py` | 10.8 KB | updated 11-08..11-10; updated 11-08..11-10 |
| `deps_from_g1ps/g1_pipette_sim/__init__.py` | `(new; the upstream __init__.py is a tracked file of another author and is not copied)` | 108 B | new |
| `deps_from_g1ps/g1_pipette_sim/physics/__init__.py` | `<g1-pipette-sim>/g1_pipette_sim/physics/__init__.py` | 542 B |  |
| `deps_from_g1ps/g1_pipette_sim/physics/mjw_head.py` | `<g1-pipette-sim>/g1_pipette_sim/physics/mjw_head.py` | 4.4 KB | path fix; scrubbed 1 local/node path(s) |
| `deps_from_g1ps/g1_pipette_sim/retarget/__init__.py` | `<g1-pipette-sim>/g1_pipette_sim/retarget/__init__.py` | 89 B |  |
| `deps_from_g1ps/g1_pipette_sim/retarget/gpu/__init__.py` | `<g1-pipette-sim>/g1_pipette_sim/retarget/gpu/__init__.py` | 564 B |  |
| `deps_from_g1ps/g1_pipette_sim/retarget/gpu/core.py` | `<g1-pipette-sim>/g1_pipette_sim/retarget/gpu/core.py` | 13.3 KB |  |
| `deps_from_g1ps/tools/rl/gpu_env.py` | `<g1-pipette-sim>/tools/rl/gpu_env.py` | 38.7 KB |  |
| `deps_from_g1ps/tools/rl/gpu_replay.py` | `<g1-pipette-sim>/tools/rl/gpu_replay.py` | 8.2 KB |  |
| `deps_from_g1ps/tools/rl/gpu_train.py` | `<g1-pipette-sim>/tools/rl/gpu_train.py` | 14.8 KB |  |
| `deps_from_g1ps/tools/rl/guarded.sh` | `<g1-pipette-sim>/tools/rl/guarded.sh` | 683 B |  |
| `docs/NODE_PLAN_R3.md` | `<work>/vega/rrline/NODE_PLAN_R3.md` | 14.9 KB | scrubbed 1 local/node path(s) |
| `docs/RL_PLAN.md` | `<work>/vega/rrline/RL_PLAN.md` | 11.9 KB | scrubbed 3 local/node path(s) |
| `node_eval.sh` | `(new)` | 1.7 KB | new |
| `node_launch.sh` | `(new)` | 810 B | new |
| `node_setup.sh` | `(new)` | 959 B | new |
| `perception/d405_viewer.py` | `(new)` | 16.5 KB | new |
| `perception/ffs_compare.py` | `(new)` | 15.3 KB | new |
| `perception/record_realsense.py` | `(new)` | 20.8 KB | new |
| `render/blender_replay.py` | `(new)` | 13.4 KB | new |
| `render/mj_to_blender.py` | `(new)` | 5.6 KB | new |
| `requirements_rl_head.txt` | `<work>/vega/rrline/requirements_rl_head.txt` | 2.3 KB | scrubbed 1 local/node path(s) |
| `results/compare_kin.json` | `(new: rl/compare_runs.py over the node evals)` | 2.3 KB |  |
| `results/compare_tc_rounds.json` | `(new: rl/compare_runs.py over the node evals)` | 3.1 KB |  |
| `results/final_table.json` | `<work>/vega/rrline/node_results/final_table.json` | 9.5 KB |  |
| `results/kin_tc_s5/eval/contact.json` | `<node_code>/rl/runs/local_eval/kin_tc_s5/contact.json` (node_eval.sh) | 999 B | scrubbed 1 local/node path(s) |
| `results/kin_tc_s5/eval/fromstart.json` | `<node_code>/rl/runs/local_eval/kin_tc_s5/fromstart.json` (node_eval.sh) | 53.3 KB |  |
| `results/kin_tc_s5/eval/holdout.json` | `<node_code>/rl/runs/local_eval/kin_tc_s5/holdout.json` (node_eval.sh) | 66.0 KB |  |
| `results/kin_tc_s5/eval/holdplace.json` | `<node_code>/rl/runs/local_eval/kin_tc_s5/holdplace.json` (node_eval.sh) | 60.9 KB |  |
| `results/kin_tc_s5/eval/rb.json` | `<node_code>/rl/runs/local_eval/kin_tc_s5/rb.json` (node_eval.sh) | 4.3 KB |  |
| `results/kin_tc_s5/eval/train_starts.json` | `<node_code>/rl/runs/local_eval/kin_tc_s5/train_starts.json` (node_eval.sh) | 66.2 KB |  |
| `results/kin_tc_s5/final.pt` | `<node_code>/tracker/runs/kin_tc_s5/final.pt` | 20.3 MB |  |
| `results/kin_tc_s5/progress.jsonl` | `<node_code>/tracker/runs/kin_tc_s5/progress.jsonl` | 2.0 MB |  |
| `results/kin_tc_s5/state.json` | `<node_code>/tracker/runs/kin_tc_s5/state.json` | 15.3 KB | scrubbed 1 local/node path(s) |
| `results/kin_tc_s5/vary/holes_summary.json` | `(new: rl/record_rollouts.py runs of KA final, local)` | 110.0 KB | scrubbed 1 local/node path(s) |
| `results/kin_tc_s5/vary/rand64_summary.json` | `(new: rl/record_rollouts.py runs of KA final, local)` | 36.7 KB | scrubbed 1 local/node path(s) |
| `results/kin_tc_s5/vary/specs_holes.json` | `(new: rl/record_rollouts.py runs of KA final, local)` | 13.1 KB |  |
| `results/kin_tc_s5/vary/specs_rand64.json` | `(new: rl/record_rollouts.py runs of KA final, local)` | 5.3 KB |  |
| `results/kin_tc_s5/vary/specs_standyaw.json` | `(new: rl/record_rollouts.py runs of KA final, local)` | 11.8 KB |  |
| `results/kin_tc_s5/vary/standyaw_summary.json` | `(new: rl/record_rollouts.py runs of KA final, local)` | 98.1 KB | scrubbed 1 local/node path(s) |
| `results/kin_tcw_s5/eval/contact.json` | `<node_code>/rl/runs/local_eval/kin_tcw_s5/contact.json` (node_eval.sh) | 1000 B | scrubbed 1 local/node path(s) |
| `results/kin_tcw_s5/eval/fromstart.json` | `<node_code>/rl/runs/local_eval/kin_tcw_s5/fromstart.json` (node_eval.sh) | 52.9 KB |  |
| `results/kin_tcw_s5/eval/holdout.json` | `<node_code>/rl/runs/local_eval/kin_tcw_s5/holdout.json` (node_eval.sh) | 66.8 KB |  |
| `results/kin_tcw_s5/eval/holdplace.json` | `<node_code>/rl/runs/local_eval/kin_tcw_s5/holdplace.json` (node_eval.sh) | 61.4 KB |  |
| `results/kin_tcw_s5/eval/rb.json` | `<node_code>/rl/runs/local_eval/kin_tcw_s5/rb.json` (node_eval.sh) | 4.3 KB |  |
| `results/kin_tcw_s5/eval/train_starts.json` | `<node_code>/rl/runs/local_eval/kin_tcw_s5/train_starts.json` (node_eval.sh) | 66.7 KB |  |
| `results/kin_tcw_s5/final.pt` | `<node_code>/tracker/runs/kin_tcw_s5/final.pt` | 20.3 MB |  |
| `results/kin_tcw_s5/progress.jsonl` | `<node_code>/tracker/runs/kin_tcw_s5/progress.jsonl` | 2.0 MB |  |
| `results/kin_tcw_s5/state.json` | `<node_code>/tracker/runs/kin_tcw_s5/state.json` | 15.6 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_full_lp_s0/eval/detail_fromstart_final.json` | `<work>/vega/rrline/node_results/rr_T1k_full_lp_s0/eval/detail_fromstart_final.json` | 33.8 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_full_lp_s0/eval/detail_train_final.json` | `<work>/vega/rrline/node_results/rr_T1k_full_lp_s0/eval/detail_train_final.json` | 45.0 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_full_lp_s0/eval/rr_T1k_eval_holdout_lp__final.json` | `<work>/vega/rrline/node_results/rr_T1k_full_lp_s0/eval/rr_T1k_eval_holdout_lp__final.json` | 17.6 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_full_lp_s0/eval/rr_T1k_eval_holdout_lp__final.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_full_lp_s0/eval/rr_T1k_eval_holdout_lp__final.slip.json` | 1.5 KB |  |
| `results/rr_T1k_full_lp_s0/eval/rr_T1k_eval_holdout_lp__model_100073472.json` | `<work>/vega/rrline/node_results/rr_T1k_full_lp_s0/eval/rr_T1k_eval_holdout_lp__model_100073472.json` | 17.7 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_full_lp_s0/eval/rr_T1k_eval_holdout_lp__model_100073472.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_full_lp_s0/eval/rr_T1k_eval_holdout_lp__model_100073472.slip.json` | 1.5 KB |  |
| `results/rr_T1k_full_lp_s0/eval/rr_T1k_eval_holdout_lp__model_75005952.json` | `<work>/vega/rrline/node_results/rr_T1k_full_lp_s0/eval/rr_T1k_eval_holdout_lp__model_75005952.json` | 17.7 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_full_lp_s0/eval/rr_T1k_eval_holdout_lp__model_75005952.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_full_lp_s0/eval/rr_T1k_eval_holdout_lp__model_75005952.slip.json` | 1.5 KB |  |
| `results/rr_T1k_full_lp_s0/eval/rr_T1k_eval_holdout_lp__zero.json` | `<work>/vega/rrline/node_results/rr_T1k_full_lp_s0/eval/rr_T1k_eval_holdout_lp__zero.json` | 16.7 KB |  |
| `results/rr_T1k_full_lp_s0/eval/rr_T1k_eval_holdplace_lp__final.json` | `<work>/vega/rrline/node_results/rr_T1k_full_lp_s0/eval/rr_T1k_eval_holdplace_lp__final.json` | 17.2 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_full_lp_s0/eval/rr_T1k_eval_holdplace_lp__final.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_full_lp_s0/eval/rr_T1k_eval_holdplace_lp__final.slip.json` | 1.5 KB |  |
| `results/rr_T1k_full_lp_s0/eval/rr_T1k_eval_holdplace_lp__model_100073472.json` | `<work>/vega/rrline/node_results/rr_T1k_full_lp_s0/eval/rr_T1k_eval_holdplace_lp__model_100073472.json` | 17.1 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_full_lp_s0/eval/rr_T1k_eval_holdplace_lp__model_100073472.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_full_lp_s0/eval/rr_T1k_eval_holdplace_lp__model_100073472.slip.json` | 1.5 KB |  |
| `results/rr_T1k_full_lp_s0/eval/rr_T1k_eval_holdplace_lp__model_75005952.json` | `<work>/vega/rrline/node_results/rr_T1k_full_lp_s0/eval/rr_T1k_eval_holdplace_lp__model_75005952.json` | 17.3 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_full_lp_s0/eval/rr_T1k_eval_holdplace_lp__model_75005952.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_full_lp_s0/eval/rr_T1k_eval_holdplace_lp__model_75005952.slip.json` | 1.5 KB |  |
| `results/rr_T1k_full_lp_s0/eval/rr_T1k_eval_holdplace_lp__zero.json` | `<work>/vega/rrline/node_results/rr_T1k_full_lp_s0/eval/rr_T1k_eval_holdplace_lp__zero.json` | 16.5 KB |  |
| `results/rr_T1k_full_lp_s0/eval/rr_T1k_full_lp_s0__final.json` | `<work>/vega/rrline/node_results/rr_T1k_full_lp_s0/eval/rr_T1k_full_lp_s0__final.json` | 17.4 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_full_lp_s0/eval/rr_T1k_full_lp_s0__final.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_full_lp_s0/eval/rr_T1k_full_lp_s0__final.slip.json` | 1.5 KB |  |
| `results/rr_T1k_full_lp_s0/eval/rr_T1k_full_lp_s0__fromstart__final.json` | `<work>/vega/rrline/node_results/rr_T1k_full_lp_s0/eval/rr_T1k_full_lp_s0__fromstart__final.json` | 14.1 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_full_lp_s0/eval/rr_T1k_full_lp_s0__fromstart__final.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_full_lp_s0/eval/rr_T1k_full_lp_s0__fromstart__final.slip.json` | 1.5 KB |  |
| `results/rr_T1k_full_lp_s0/eval/rr_T1k_full_lp_s0__fromstart__model_100073472.json` | `<work>/vega/rrline/node_results/rr_T1k_full_lp_s0/eval/rr_T1k_full_lp_s0__fromstart__model_100073472.json` | 35.8 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_full_lp_s0/eval/rr_T1k_full_lp_s0__fromstart__model_100073472.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_full_lp_s0/eval/rr_T1k_full_lp_s0__fromstart__model_100073472.slip.json` | 1.5 KB |  |
| `results/rr_T1k_full_lp_s0/eval/rr_T1k_full_lp_s0__fromstart__zero.json` | `<work>/vega/rrline/node_results/rr_T1k_full_lp_s0/eval/rr_T1k_full_lp_s0__fromstart__zero.json` | 11.4 KB |  |
| `results/rr_T1k_full_lp_s0/eval/rr_T1k_full_lp_s0__model_100073472.json` | `<work>/vega/rrline/node_results/rr_T1k_full_lp_s0/eval/rr_T1k_full_lp_s0__model_100073472.json` | 17.6 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_full_lp_s0/eval/rr_T1k_full_lp_s0__model_100073472.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_full_lp_s0/eval/rr_T1k_full_lp_s0__model_100073472.slip.json` | 1.5 KB |  |
| `results/rr_T1k_full_lp_s0/eval/rr_T1k_full_lp_s0__model_75005952.json` | `<work>/vega/rrline/node_results/rr_T1k_full_lp_s0/eval/rr_T1k_full_lp_s0__model_75005952.json` | 17.7 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_full_lp_s0/eval/rr_T1k_full_lp_s0__model_75005952.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_full_lp_s0/eval/rr_T1k_full_lp_s0__model_75005952.slip.json` | 1.5 KB |  |
| `results/rr_T1k_full_lp_s0/eval/rr_T1k_full_lp_s0__zero.json` | `<work>/vega/rrline/node_results/rr_T1k_full_lp_s0/eval/rr_T1k_full_lp_s0__zero.json` | 16.9 KB |  |
| `results/rr_T1k_full_lp_s0/final.pt` | `<work>/vega/rrline/node_results/rr_T1k_full_lp_s0/final.pt` | 20.3 MB |  |
| `results/rr_T1k_full_lp_s0/progress.jsonl` | `<work>/vega/rrline/node_results/rr_T1k_full_lp_s0/progress.jsonl` | 737.4 KB |  |
| `results/rr_T1k_r2_s2/eval/rr_T1k_r2_eval_holdout__final.json` | `<work>/vega/rrline/node_results/rr_T1k_r2_s2/eval/rr_T1k_r2_eval_holdout__final.json` | 65.9 KB |  |
| `results/rr_T1k_r2_s2/eval/rr_T1k_r2_eval_holdout__final.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_r2_s2/eval/rr_T1k_r2_eval_holdout__final.slip.json` | 1.5 KB |  |
| `results/rr_T1k_r2_s2/eval/rr_T1k_r2_eval_holdout__zero.json` | `<work>/vega/rrline/node_results/rr_T1k_r2_s2/eval/rr_T1k_r2_eval_holdout__zero.json` | 68.5 KB |  |
| `results/rr_T1k_r2_s2/eval/rr_T1k_r2_eval_holdplace__final.json` | `<work>/vega/rrline/node_results/rr_T1k_r2_s2/eval/rr_T1k_r2_eval_holdplace__final.json` | 61.0 KB |  |
| `results/rr_T1k_r2_s2/eval/rr_T1k_r2_eval_holdplace__final.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_r2_s2/eval/rr_T1k_r2_eval_holdplace__final.slip.json` | 1.5 KB |  |
| `results/rr_T1k_r2_s2/eval/rr_T1k_r2_eval_holdplace__zero.json` | `<work>/vega/rrline/node_results/rr_T1k_r2_s2/eval/rr_T1k_r2_eval_holdplace__zero.json` | 61.8 KB |  |
| `results/rr_T1k_r2_s2/eval/rr_T1k_r2_s2__final.json` | `<work>/vega/rrline/node_results/rr_T1k_r2_s2/eval/rr_T1k_r2_s2__final.json` | 65.5 KB |  |
| `results/rr_T1k_r2_s2/eval/rr_T1k_r2_s2__final.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_r2_s2/eval/rr_T1k_r2_s2__final.slip.json` | 1.5 KB |  |
| `results/rr_T1k_r2_s2/eval/rr_T1k_r2_s2__fromstart__final.json` | `<work>/vega/rrline/node_results/rr_T1k_r2_s2/eval/rr_T1k_r2_s2__fromstart__final.json` | 98.0 KB |  |
| `results/rr_T1k_r2_s2/eval/rr_T1k_r2_s2__fromstart__final.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_r2_s2/eval/rr_T1k_r2_s2__fromstart__final.slip.json` | 1.5 KB |  |
| `results/rr_T1k_r2_s2/eval/rr_T1k_r2_s2__fromstart__zero.json` | `<work>/vega/rrline/node_results/rr_T1k_r2_s2/eval/rr_T1k_r2_s2__fromstart__zero.json` | 80.5 KB |  |
| `results/rr_T1k_r2_s2/eval/rr_T1k_r2_s2__zero.json` | `<work>/vega/rrline/node_results/rr_T1k_r2_s2/eval/rr_T1k_r2_s2__zero.json` | 68.2 KB |  |
| `results/rr_T1k_r2_s2/eval/rr_T1k_r2_s2_evalzeta__fromstart__final.json` | `<work>/vega/rrline/node_results/rr_T1k_r2_s2/eval/rr_T1k_r2_s2_evalzeta__fromstart__final.json` | 100.5 KB |  |
| `results/rr_T1k_r2_s2/eval_node_1106/rr_T1k_r2_eval_holdout__final.json` | `<work>/vega/rrline/node_results/rr_T1k_r2_s2/eval_node_1106/rr_T1k_r2_eval_holdout__final.json` | 46.5 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_r2_s2/eval_node_1106/rr_T1k_r2_eval_holdout__final.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_r2_s2/eval_node_1106/rr_T1k_r2_eval_holdout__final.slip.json` | 1.5 KB |  |
| `results/rr_T1k_r2_s2/eval_node_1106/rr_T1k_r2_eval_holdout__zero.json` | `<work>/vega/rrline/node_results/rr_T1k_r2_s2/eval_node_1106/rr_T1k_r2_eval_holdout__zero.json` | 16.7 KB |  |
| `results/rr_T1k_r2_s2/eval_node_1106/rr_T1k_r2_eval_holdplace__final.json` | `<work>/vega/rrline/node_results/rr_T1k_r2_s2/eval_node_1106/rr_T1k_r2_eval_holdplace__final.json` | 40.2 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_r2_s2/eval_node_1106/rr_T1k_r2_eval_holdplace__final.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_r2_s2/eval_node_1106/rr_T1k_r2_eval_holdplace__final.slip.json` | 1.5 KB |  |
| `results/rr_T1k_r2_s2/eval_node_1106/rr_T1k_r2_eval_holdplace__zero.json` | `<work>/vega/rrline/node_results/rr_T1k_r2_s2/eval_node_1106/rr_T1k_r2_eval_holdplace__zero.json` | 16.2 KB |  |
| `results/rr_T1k_r2_s2/eval_node_1106/rr_T1k_r2_s2__final.json` | `<work>/vega/rrline/node_results/rr_T1k_r2_s2/eval_node_1106/rr_T1k_r2_s2__final.json` | 46.4 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_r2_s2/eval_node_1106/rr_T1k_r2_s2__final.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_r2_s2/eval_node_1106/rr_T1k_r2_s2__final.slip.json` | 1.5 KB |  |
| `results/rr_T1k_r2_s2/eval_node_1106/rr_T1k_r2_s2__fromstart__final.json` | `<work>/vega/rrline/node_results/rr_T1k_r2_s2/eval_node_1106/rr_T1k_r2_s2__fromstart__final.json` | 35.7 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_r2_s2/eval_node_1106/rr_T1k_r2_s2__fromstart__final.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_r2_s2/eval_node_1106/rr_T1k_r2_s2__fromstart__final.slip.json` | 1.5 KB |  |
| `results/rr_T1k_r2_s2/eval_node_1106/rr_T1k_r2_s2__fromstart__zero.json` | `<work>/vega/rrline/node_results/rr_T1k_r2_s2/eval_node_1106/rr_T1k_r2_s2__fromstart__zero.json` | 11.1 KB |  |
| `results/rr_T1k_r2_s2/eval_node_1106/rr_T1k_r2_s2__zero.json` | `<work>/vega/rrline/node_results/rr_T1k_r2_s2/eval_node_1106/rr_T1k_r2_s2__zero.json` | 16.6 KB |  |
| `results/rr_T1k_r2_s2/final.pt` | `<work>/vega/rrline/node_results/rr_T1k_r2_s2/final.pt` | 20.3 MB |  |
| `results/rr_T1k_r2_s2/progress.jsonl` | `<work>/vega/rrline/node_results/rr_T1k_r2_s2/progress.jsonl` | 901.4 KB |  |
| `results/rr_T1k_r2w_s3/eval/rr_T1k_r2_eval_holdout__final.json` | `<work>/vega/rrline/node_results/rr_T1k_r2w_s3/eval/rr_T1k_r2_eval_holdout__final.json` | 46.9 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_r2w_s3/eval/rr_T1k_r2_eval_holdout__final.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_r2w_s3/eval/rr_T1k_r2_eval_holdout__final.slip.json` | 1.5 KB |  |
| `results/rr_T1k_r2w_s3/eval/rr_T1k_r2_eval_holdout__zero.json` | `<work>/vega/rrline/node_results/rr_T1k_r2w_s3/eval/rr_T1k_r2_eval_holdout__zero.json` | 45.8 KB |  |
| `results/rr_T1k_r2w_s3/eval/rr_T1k_r2_eval_holdplace__final.json` | `<work>/vega/rrline/node_results/rr_T1k_r2w_s3/eval/rr_T1k_r2_eval_holdplace__final.json` | 40.2 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_r2w_s3/eval/rr_T1k_r2_eval_holdplace__final.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_r2w_s3/eval/rr_T1k_r2_eval_holdplace__final.slip.json` | 1.5 KB |  |
| `results/rr_T1k_r2w_s3/eval/rr_T1k_r2_eval_holdplace__zero.json` | `<work>/vega/rrline/node_results/rr_T1k_r2w_s3/eval/rr_T1k_r2_eval_holdplace__zero.json` | 38.7 KB |  |
| `results/rr_T1k_r2w_s3/eval/rr_T1k_r2w_s3__final.json` | `<work>/vega/rrline/node_results/rr_T1k_r2w_s3/eval/rr_T1k_r2w_s3__final.json` | 46.5 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_r2w_s3/eval/rr_T1k_r2w_s3__final.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_r2w_s3/eval/rr_T1k_r2w_s3__final.slip.json` | 1.4 KB |  |
| `results/rr_T1k_r2w_s3/eval/rr_T1k_r2w_s3__fromstart__final.json` | `<work>/vega/rrline/node_results/rr_T1k_r2w_s3/eval/rr_T1k_r2w_s3__fromstart__final.json` | 35.7 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_r2w_s3/eval/rr_T1k_r2w_s3__fromstart__final.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_r2w_s3/eval/rr_T1k_r2w_s3__fromstart__final.slip.json` | 1.5 KB |  |
| `results/rr_T1k_r2w_s3/eval/rr_T1k_r2w_s3__fromstart__zero.json` | `<work>/vega/rrline/node_results/rr_T1k_r2w_s3/eval/rr_T1k_r2w_s3__fromstart__zero.json` | 31.2 KB |  |
| `results/rr_T1k_r2w_s3/eval/rr_T1k_r2w_s3__zero.json` | `<work>/vega/rrline/node_results/rr_T1k_r2w_s3/eval/rr_T1k_r2w_s3__zero.json` | 45.6 KB |  |
| `results/rr_T1k_r2w_s3/final.pt` | `<work>/vega/rrline/node_results/rr_T1k_r2w_s3/final.pt` | 20.3 MB |  |
| `results/rr_T1k_r2w_s3/progress.jsonl` | `<work>/vega/rrline/node_results/rr_T1k_r2w_s3/progress.jsonl` | 358.7 KB |  |
| `results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_eval_holdout__final.json` | `<work>/vega/rrline/node_results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_eval_holdout__final.json` | 66.2 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_eval_holdout__final.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_eval_holdout__final.slip.json` | 985 B |  |
| `results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_eval_holdout__model_175079424.json` | `<work>/vega/rrline/node_results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_eval_holdout__model_175079424.json` | 66.3 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_eval_holdout__model_175079424.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_eval_holdout__model_175079424.slip.json` | 984 B |  |
| `results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_eval_holdout__zero.json` | `<work>/vega/rrline/node_results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_eval_holdout__zero.json` | 67.8 KB |  |
| `results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_eval_holdplace__final.json` | `<work>/vega/rrline/node_results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_eval_holdplace__final.json` | 60.5 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_eval_holdplace__final.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_eval_holdplace__final.slip.json` | 1.4 KB |  |
| `results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_eval_holdplace__model_175079424.json` | `<work>/vega/rrline/node_results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_eval_holdplace__model_175079424.json` | 60.6 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_eval_holdplace__model_175079424.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_eval_holdplace__model_175079424.slip.json` | 1.4 KB |  |
| `results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_eval_holdplace__zero.json` | `<work>/vega/rrline/node_results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_eval_holdplace__zero.json` | 61.5 KB |  |
| `results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_s4__final.json` | `<work>/vega/rrline/node_results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_s4__final.json` | 65.5 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_s4__final.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_s4__final.slip.json` | 1.5 KB |  |
| `results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_s4__fromstart__final.json` | `<work>/vega/rrline/node_results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_s4__fromstart__final.json` | 93.8 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_s4__fromstart__final.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_s4__fromstart__final.slip.json` | 1.5 KB |  |
| `results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_s4__fromstart__zero.json` | `<work>/vega/rrline/node_results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_s4__fromstart__zero.json` | 78.2 KB |  |
| `results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_s4__model_175079424.json` | `<work>/vega/rrline/node_results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_s4__model_175079424.json` | 65.5 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_s4__model_175079424.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_s4__model_175079424.slip.json` | 1.5 KB |  |
| `results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_s4__zero.json` | `<work>/vega/rrline/node_results/rr_T1k_r3A_s4/eval/rr_T1k_r3A_s4__zero.json` | 67.7 KB |  |
| `results/rr_T1k_r3A_s4/eval_early/fs_model_50036736.json` | `<work>/vega/rrline/node_results/rr_T1k_r3A_s4/eval_early/fs_model_50036736.json` | 53.8 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_r3A_s4/final.pt` | `<work>/vega/rrline/node_results/rr_T1k_r3A_s4/final.pt` | 20.3 MB |  |
| `results/rr_T1k_r3A_s4/progress.jsonl` | `<work>/vega/rrline/node_results/rr_T1k_r3A_s4/progress.jsonl` | 1.1 MB |  |
| `results/rr_T1k_r3A_s4/state.json` | `<work>/vega/rrline/node_results/rr_T1k_r3A_s4/state.json` | 14.0 KB | scrubbed 2 local/node path(s) |
| `results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_eval_holdout__final.json` | `<work>/vega/rrline/node_results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_eval_holdout__final.json` | 65.3 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_eval_holdout__final.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_eval_holdout__final.slip.json` | 985 B |  |
| `results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_eval_holdout__model_175079424.json` | `<work>/vega/rrline/node_results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_eval_holdout__model_175079424.json` | 65.4 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_eval_holdout__model_175079424.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_eval_holdout__model_175079424.slip.json` | 985 B |  |
| `results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_eval_holdout__zero.json` | `<work>/vega/rrline/node_results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_eval_holdout__zero.json` | 67.9 KB |  |
| `results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_eval_holdplace__final.json` | `<work>/vega/rrline/node_results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_eval_holdplace__final.json` | 60.4 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_eval_holdplace__final.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_eval_holdplace__final.slip.json` | 1.4 KB |  |
| `results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_eval_holdplace__model_175079424.json` | `<work>/vega/rrline/node_results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_eval_holdplace__model_175079424.json` | 60.5 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_eval_holdplace__model_175079424.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_eval_holdplace__model_175079424.slip.json` | 1.4 KB |  |
| `results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_eval_holdplace__zero.json` | `<work>/vega/rrline/node_results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_eval_holdplace__zero.json` | 61.8 KB |  |
| `results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_s5__final.json` | `<work>/vega/rrline/node_results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_s5__final.json` | 65.0 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_s5__final.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_s5__final.slip.json` | 1.5 KB |  |
| `results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_s5__fromstart__final.json` | `<work>/vega/rrline/node_results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_s5__fromstart__final.json` | 93.5 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_s5__fromstart__final.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_s5__fromstart__final.slip.json` | 1.5 KB |  |
| `results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_s5__fromstart__zero.json` | `<work>/vega/rrline/node_results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_s5__fromstart__zero.json` | 79.1 KB |  |
| `results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_s5__model_175079424.json` | `<work>/vega/rrline/node_results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_s5__model_175079424.json` | 65.2 KB | scrubbed 1 local/node path(s) |
| `results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_s5__model_175079424.slip.json` | `<work>/vega/rrline/node_results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_s5__model_175079424.slip.json` | 1.5 KB |  |
| `results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_s5__zero.json` | `<work>/vega/rrline/node_results/rr_T1k_r3B_s5/eval/rr_T1k_r3B_s5__zero.json` | 67.7 KB |  |
| `results/rr_T1k_r3B_s5/final.pt` | `<work>/vega/rrline/node_results/rr_T1k_r3B_s5/final.pt` | 20.3 MB |  |
| `results/rr_T1k_r3B_s5/progress.jsonl` | `<work>/vega/rrline/node_results/rr_T1k_r3B_s5/progress.jsonl` | 1.1 MB |  |
| `results/rr_T1k_r3B_s5/state.json` | `<work>/vega/rrline/node_results/rr_T1k_r3B_s5/state.json` | 14.0 KB | scrubbed 2 local/node path(s) |
| `results/rr_T1k_tc_s5/eval/contact.json` | `<node_code>/rl/runs/local_eval/tc_final/contact.json` (node_eval.sh) | 985 B | scrubbed 1 local/node path(s) |
| `results/rr_T1k_tc_s5/eval/fromstart.json` | `<node_code>/rl/runs/local_eval/tc_final/fromstart.json` (node_eval.sh) | 53.6 KB |  |
| `results/rr_T1k_tc_s5/eval/holdout.json` | `<node_code>/rl/runs/local_eval/tc_final/holdout.json` (node_eval.sh) | 65.8 KB |  |
| `results/rr_T1k_tc_s5/eval/holdplace.json` | `<node_code>/rl/runs/local_eval/tc_final/holdplace.json` (node_eval.sh) | 60.3 KB |  |
| `results/rr_T1k_tc_s5/eval/rb.json` | `<node_code>/rl/runs/local_eval/tc_final/rb.json` (node_eval.sh) | 4.2 KB |  |
| `results/rr_T1k_tc_s5/eval/train_starts.json` | `<node_code>/rl/runs/local_eval/tc_final/train_starts.json` (node_eval.sh) | 65.4 KB |  |
| `results/rr_T1k_tc_s5/final.pt` | `<node_code>/rl/runs/rr_T1k_tc_s5/final.pt` | 20.3 MB |  |
| `results/rr_T1k_tc_s5/progress.jsonl` | `<node_code>/rl/runs/rr_T1k_tc_s5/progress.jsonl` | 1.1 MB |  |
| `results/rr_T1k_tc_s5/state.json` | `<node_code>/rl/runs/rr_T1k_tc_s5/state.json` | 14.8 KB | scrubbed 2 local/node path(s) |
| `results/rr_T1k_tcq_s5/eval/contact.json` | `<node_code>/rl/runs/local_eval/rr_T1k_tcq_s5/contact.json` (node_eval.sh) | 985 B | scrubbed 1 local/node path(s) |
| `results/rr_T1k_tcq_s5/eval/fromstart.json` | `<node_code>/rl/runs/local_eval/rr_T1k_tcq_s5/fromstart.json` (node_eval.sh) | 50.9 KB |  |
| `results/rr_T1k_tcq_s5/eval/holdout.json` | `<node_code>/rl/runs/local_eval/rr_T1k_tcq_s5/holdout.json` (node_eval.sh) | 67.6 KB |  |
| `results/rr_T1k_tcq_s5/eval/holdplace.json` | `<node_code>/rl/runs/local_eval/rr_T1k_tcq_s5/holdplace.json` (node_eval.sh) | 61.8 KB |  |
| `results/rr_T1k_tcq_s5/eval/rb.json` | `<node_code>/rl/runs/local_eval/rr_T1k_tcq_s5/rb.json` (node_eval.sh) | 4.4 KB |  |
| `results/rr_T1k_tcq_s5/eval/train_starts.json` | `<node_code>/rl/runs/local_eval/rr_T1k_tcq_s5/train_starts.json` (node_eval.sh) | 67.2 KB |  |
| `results/rr_T1k_tcq_s5/final.pt` | `<node_code>/rl/runs/rr_T1k_tcq_s5/final.pt` | 20.3 MB |  |
| `results/rr_T1k_tcq_s5/progress.jsonl` | `<node_code>/rl/runs/rr_T1k_tcq_s5/progress.jsonl` | 1.1 MB |  |
| `results/rr_T1k_tcq_s5/state.json` | `<node_code>/rl/runs/rr_T1k_tcq_s5/state.json` | 15.9 KB | scrubbed 2 local/node path(s) |
| `results/rr_T1k_tcs1_s5/eval/contact.json` | `<node_code>/rl/runs/local_eval/rr_T1k_tcs1_s5/contact.json` (node_eval.sh) | 989 B | scrubbed 1 local/node path(s) |
| `results/rr_T1k_tcs1_s5/eval/fromstart.json` | `<node_code>/rl/runs/local_eval/rr_T1k_tcs1_s5/fromstart.json` (node_eval.sh) | 52.7 KB |  |
| `results/rr_T1k_tcs1_s5/eval/holdout.json` | `<node_code>/rl/runs/local_eval/rr_T1k_tcs1_s5/holdout.json` (node_eval.sh) | 65.2 KB |  |
| `results/rr_T1k_tcs1_s5/eval/holdplace.json` | `<node_code>/rl/runs/local_eval/rr_T1k_tcs1_s5/holdplace.json` (node_eval.sh) | 59.7 KB |  |
| `results/rr_T1k_tcs1_s5/eval/rb.json` | `<node_code>/rl/runs/local_eval/rr_T1k_tcs1_s5/rb.json` (node_eval.sh) | 4.3 KB |  |
| `results/rr_T1k_tcs1_s5/eval/train_starts.json` | `<node_code>/rl/runs/local_eval/rr_T1k_tcs1_s5/train_starts.json` (node_eval.sh) | 64.9 KB |  |
| `results/rr_T1k_tcs1_s5/final.pt` | `<node_code>/rl/runs/rr_T1k_tcs1_s5/final.pt` | 20.3 MB |  |
| `results/rr_T1k_tcs1_s5/progress.jsonl` | `<node_code>/rl/runs/rr_T1k_tcs1_s5/progress.jsonl` | 1.2 MB |  |
| `results/rr_T1k_tcs1_s5/state.json` | `<node_code>/rl/runs/rr_T1k_tcs1_s5/state.json` | 14.5 KB | scrubbed 2 local/node path(s) |
| `results/rr_T1k_tcs5_s5/eval/contact.json` | `<node_code>/rl/runs/local_eval/rr_T1k_tcs5_s5/contact.json` (node_eval.sh) | 989 B | scrubbed 1 local/node path(s) |
| `results/rr_T1k_tcs5_s5/eval/fromstart.json` | `<node_code>/rl/runs/local_eval/rr_T1k_tcs5_s5/fromstart.json` (node_eval.sh) | 51.6 KB |  |
| `results/rr_T1k_tcs5_s5/eval/holdout.json` | `<node_code>/rl/runs/local_eval/rr_T1k_tcs5_s5/holdout.json` (node_eval.sh) | 65.9 KB |  |
| `results/rr_T1k_tcs5_s5/eval/holdplace.json` | `<node_code>/rl/runs/local_eval/rr_T1k_tcs5_s5/holdplace.json` (node_eval.sh) | 60.4 KB |  |
| `results/rr_T1k_tcs5_s5/eval/rb.json` | `<node_code>/rl/runs/local_eval/rr_T1k_tcs5_s5/rb.json` (node_eval.sh) | 4.3 KB |  |
| `results/rr_T1k_tcs5_s5/eval/train_starts.json` | `<node_code>/rl/runs/local_eval/rr_T1k_tcs5_s5/train_starts.json` (node_eval.sh) | 65.7 KB |  |
| `results/rr_T1k_tcs5_s5/final.pt` | `<node_code>/rl/runs/rr_T1k_tcs5_s5/final.pt` | 20.3 MB |  |
| `results/rr_T1k_tcs5_s5/progress.jsonl` | `<node_code>/rl/runs/rr_T1k_tcs5_s5/progress.jsonl` | 1.2 MB |  |
| `results/rr_T1k_tcs5_s5/state.json` | `<node_code>/rl/runs/rr_T1k_tcs5_s5/state.json` | 14.5 KB | scrubbed 2 local/node path(s) |
| `results/student_plumb/eval_student_plumb.json` | `<work>/vega/rrline/eval_student_plumb.json` | 659 B | DAgger plumbing test (task E1, not included) |
| `results/zero_residual_noise/zero_s0.03.json` | `<node_code>/rl/runs/local_eval/noise/` (eval_rr.py, zero residual + action noise) | 4.4 KB |  |
| `results/zero_residual_noise/zero_s0.1.json` | `<node_code>/rl/runs/local_eval/noise/` (eval_rr.py, zero residual + action noise) | 4.3 KB |  |
| `results/zero_residual_noise/zero_s0.147.json` | `<node_code>/rl/runs/local_eval/noise/` (eval_rr.py, zero residual + action noise) | 4.3 KB |  |
| `rl/audit_starts.py` | `<work>/vega/rrline/audit_starts.py` | 1.2 KB |  |
| `rl/caps.py` | `(new)` | 3.3 KB | new |
| `rl/caps_calib.py` | `(new)` | 3.9 KB | new |
| `rl/compare_runs.py` | `(new)` | 3.0 KB | new |
| `rl/compose_ref_vs_policy.py` | `<work>/vega/rrline/compose_ref_vs_policy.py` | 3.2 KB |  |
| `rl/configs/dr_rr1.json` | `<work>/vega/rrline/configs/dr_rr1.json` | 299 B |  |
| `rl/configs/dr_rr1_kv3.json` | `<work>/vega/rrline/configs/dr_rr1_kv3.json` | 296 B |  |
| `rl/configs/dr_rr1_kv3_zeta.json` | `<work>/vega/rrline/configs/dr_rr1_kv3_zeta.json` | 328 B |  |
| `rl/configs/rr_T1k_eval_holdout_lp.json` | `<work>/vega/rrline/configs/rr_T1k_eval_holdout_lp.json` | 1.1 KB | path fix |
| `rl/configs/rr_T1k_eval_holdplace_lp.json` | `<work>/vega/rrline/configs/rr_T1k_eval_holdplace_lp.json` | 4.6 KB | path fix |
| `rl/configs/rr_T1k_full_lp_s0.json` | `<work>/vega/rrline/configs/rr_T1k_full_lp_s0.json` | 4.9 KB | path fix |
| `rl/configs/rr_T1k_r2_eval_holdout.json` | `<work>/vega/rrline/configs/rr_T1k_r2_eval_holdout.json` | 1.2 KB | path fix |
| `rl/configs/rr_T1k_r2_eval_holdplace.json` | `<work>/vega/rrline/configs/rr_T1k_r2_eval_holdplace.json` | 4.7 KB | path fix |
| `rl/configs/rr_T1k_r2_s2.json` | `<work>/vega/rrline/configs/rr_T1k_r2_s2.json` | 4.4 KB | path fix |
| `rl/configs/rr_T1k_r2_s2_evalzeta.json` | `<work>/vega/rrline/configs/rr_T1k_r2_s2_evalzeta.json` | 4.3 KB | path fix |
| `rl/configs/rr_T1k_r3A_eval_holdout.json` | `<work>/vega/rrline/configs/rr_T1k_r3A_eval_holdout.json` | 1.2 KB | path fix |
| `rl/configs/rr_T1k_r3A_eval_holdplace.json` | `<work>/vega/rrline/configs/rr_T1k_r3A_eval_holdplace.json` | 4.7 KB | path fix |
| `rl/configs/rr_T1k_r3A_s4.json` | `<work>/vega/rrline/configs/rr_T1k_r3A_s4.json` | 5.6 KB | path fix |
| `rl/configs/rr_T1k_r3B_eval_holdout.json` | `<work>/vega/rrline/configs/rr_T1k_r3B_eval_holdout.json` | 1.2 KB | path fix |
| `rl/configs/rr_T1k_r3B_eval_holdplace.json` | `<work>/vega/rrline/configs/rr_T1k_r3B_eval_holdplace.json` | 4.7 KB | path fix |
| `rl/configs/rr_T1k_r3B_s5.json` | `<work>/vega/rrline/configs/rr_T1k_r3B_s5.json` | 5.6 KB | path fix |
| `rl/configs/rr_T1k_tc_eval_holdout.json` | `(new)` | 1.5 KB | new |
| `rl/configs/rr_T1k_tc_eval_holdplace.json` | `(new)` | 5.0 KB | new |
| `rl/configs/rr_T1k_tc_s5.json` | `(new)` | 6.0 KB | new |
| `rl/configs/rr_T1k_tcq_eval_holdout.json` | `(new)` | 1.6 KB | new |
| `rl/configs/rr_T1k_tcq_eval_holdplace.json` | `(new)` | 5.1 KB | new |
| `rl/configs/rr_T1k_tcq_s5.json` | `(new)` | 6.8 KB | new |
| `rl/configs/rr_T1k_tcs1_s5.json` | `(new)` | 5.4 KB | new |
| `rl/configs/rr_T1k_tcs5_s5.json` | `(new)` | 5.4 KB | new |
| `rl/configs/rr_student_plumb.json` | `<work>/vega/rrline/configs/rr_student_plumb.json` | 1.1 KB |  |
| `rl/configs/rr_teacher_smoke.json` | `<work>/vega/rrline/configs/rr_teacher_smoke.json` | 505 B |  |
| `rl/contact_ref.py` | `<work>/vega/rrline/contact_ref.py` | 5.0 KB |  |
| `rl/data_gate.py` | `<work>/vega/rrline/data_gate.py` | 3.3 KB |  |
| `rl/distill.py` | `<work>/vega/rrline/distill.py` | 10.1 KB | path fix |
| `rl/ee_path.py` | `(new)` | 6.7 KB | new |
| `rl/ee_path_plot.py` | `(new)` | 1.3 KB | new |
| `rl/eval_contact_stability.py` | `(new)` | 6.7 KB | new |
| `rl/eval_rr.py` | `<work>/vega/rrline/eval_rr.py` | 30.8 KB | path fix; updated 11-08..11-10; updated 11-08..11-10 |
| `rl/export_task.py` | `<work>/vega/rrline/export_task.py` | 17.8 KB | path fix |
| `rl/final_table.py` | `<work>/vega/rrline/final_table.py` | 4.6 KB | path fix |
| `rl/finger_event_labels.py` | `<work>/vega/rrline/finger_event_labels.py` | 3.4 KB |  |
| `rl/jitter_env.py` | `<work>/vega/rrline/jitter_env.py` | 3.5 KB | path fix |
| `rl/jitter_whole.py` | `<work>/vega/rrline/jitter_whole.py` | 6.9 KB |  |
| `rl/make_base_variant.py` | `<work>/vega/rrline/make_base_variant.py` | 2.5 KB |  |
| `rl/make_relocatable_task.py` | `<work>/vega/rrline/make_relocatable_task.py` | 1.7 KB |  |
| `rl/record_rollouts.py` | `<work>/vega/rrline/record_rollouts.py` | 6.2 KB | path fix; updated 11-08..11-10; updated 11-08..11-10 |
| `rl/render_compare.py` | `(new)` | 9.4 KB | new |
| `rl/render_rollout.py` | `<work>/vega/rrline/render_rollout.py` | 3.9 KB | path fix |
| `rl/replay_check.py` | `<work>/vega/rrline/replay_check.py` | 3.0 KB |  |
| `rl/reward_breakdown.py` | `(new)` | 5.2 KB | new |
| `rl/seat_miss_analysis.py` | `<work>/vega/rrline/seat_miss_analysis.py` | 3.7 KB |  |
| `rl/slip_compare.py` | `<work>/vega/rrline/slip_compare.py` | 1.9 KB |  |
| `rl/slowmo_closeup.py` | `<work>/vega/rrline/slowmo_closeup.py` | 3.8 KB |  |
| `rl/smoke_r3.py` | `<work>/vega/rrline/smoke_r3.py` | 3.7 KB | path fix |
| `rl/test_arm_clamp.py` | `<work>/vega/rrline/test_arm_clamp.py` | 1.9 KB |  |
| `rl/test_caps.py` | `(new)` | 4.3 KB | new |
| `rl/test_contact_terms.py` | `(new)` | 15.5 KB | new |
| `rl/test_rate_terms.py` | `(new)` | 4.3 KB | new |
| `rl/train.py` | `<work>/vega/rrline/train.py` | 6.9 KB | path fix; updated 11-08..11-10; updated 11-08..11-10 |
| `rl/vary_summary.py` | `(new)` | 3.4 KB | new |
| `rl/vega_extras.py` | `<work>/vega/rrline/vega_extras.py` | 51.3 KB | path fix; updated 11-08..11-10; updated 11-08..11-10 |
| `rl/vega_res_env.py` | `<work>/vega/rrline/vega_res_env.py` | 40.7 KB | path fix; updated 11-08..11-10; updated 11-08..11-10 |
| `tasks/task_T1k_lp/contact_ref.json` | `<work>/vega/rrline/task_T1k_lp/contact_ref.json` | 390 B |  |
| `tasks/task_T1k_lp/contact_ref.npz` | `<work>/vega/rrline/task_T1k_lp/contact_ref.npz` | 664.6 KB |  |
| `tasks/task_T1k_lp/reference.npz` | `<work>/vega/rrline/task_T1k_lp/reference.npz` | 5.6 MB |  |
| `tasks/task_T1k_lp/scene.xml` | `<work>/vega/rrline/task_T1k_lp/scene.xml` | 1000.2 KB | 1 mesh file paths rewritten to ../../assets/... (meshes NOT included) |
| `tasks/task_T1k_lp/task.json` | `<work>/vega/rrline/task_T1k_lp/task.json` | 12.9 KB | scrubbed 2 local/node path(s) |
| `tasks/task_T1k_lp/valid_starts.npy` | `<work>/vega/rrline/task_T1k_lp/valid_starts.npy` | 9.5 KB |  |
| `tasks/task_T1k_lp/valid_starts_audit.json` | `<work>/vega/rrline/task_T1k_lp/valid_starts_audit.json` | 56 B |  |
| `tasks/task_T1k_lp_noff/contact_ref.json` | `<work>/vega/rrline/task_T1k_lp_noff/contact_ref.json` | 390 B |  |
| `tasks/task_T1k_lp_noff/contact_ref.npz` | `<work>/vega/rrline/task_T1k_lp_noff/contact_ref.npz` | 664.6 KB |  |
| `tasks/task_T1k_lp_noff/reference.npz` | `<work>/vega/rrline/task_T1k_lp_noff/reference.npz` | 7.8 MB |  |
| `tasks/task_T1k_lp_noff/scene.xml` | `<work>/vega/rrline/task_T1k_lp_noff/scene.xml` | 1000.2 KB | 1 mesh file paths rewritten to ../../assets/... (meshes NOT included) |
| `tasks/task_T1k_lp_noff/task.json` | `<work>/vega/rrline/task_T1k_lp_noff/task.json` | 14.2 KB | scrubbed 2 local/node path(s) |
| `tasks/task_T1k_lp_noff/valid_starts.npy` | `<work>/vega/rrline/task_T1k_lp_noff/valid_starts.npy` | 9.5 KB |  |
| `tasks/task_T1k_lp_noff/valid_starts_audit.json` | `<work>/vega/rrline/task_T1k_lp_noff/valid_starts_audit.json` | 56 B |  |
| `tasks/task_T1k_still3_noff/contact_ref.json` | `<work>/vega/rrline/task_T1k_still3_noff/contact_ref.json` | 392 B |  |
| `tasks/task_T1k_still3_noff/contact_ref.npz` | `<work>/vega/rrline/task_T1k_still3_noff/contact_ref.npz` | 667.1 KB |  |
| `tasks/task_T1k_still3_noff/reference.npz` | `<work>/vega/rrline/task_T1k_still3_noff/reference.npz` | 7.8 MB |  |
| `tasks/task_T1k_still3_noff/scene.xml` | `<work>/vega/rrline/task_T1k_still3_noff/scene.xml` | 1000.2 KB | 1 mesh file paths rewritten to ../../assets/... (meshes NOT included) |
| `tasks/task_T1k_still3_noff/task.json` | `<work>/vega/rrline/task_T1k_still3_noff/task.json` | 29.5 KB | scrubbed 2 local/node path(s) |
| `tasks/task_T1k_still3_noff/valid_starts.npy` | `<work>/vega/rrline/task_T1k_still3_noff/valid_starts.npy` | 9.5 KB |  |
| `tasks/task_T1k_still3_noff/valid_starts_audit.json` | `<work>/vega/rrline/task_T1k_still3_noff/valid_starts_audit.json` | 56 B |  |
| `tasks/task_kin_noprior/contact_ref.json` | `<work>/vega/tracker/task_kin_noprior/contact_ref.json` | 392 B |  |
| `tasks/task_kin_noprior/contact_ref.npz` | `<work>/vega/tracker/task_kin_noprior/contact_ref.npz` | 667.1 KB |  |
| `tasks/task_kin_noprior/reference.npz` | `<work>/vega/tracker/task_kin_noprior/reference.npz` | 1.7 MB |  |
| `tasks/task_kin_noprior/scene.xml` | `<work>/vega/tracker/task_kin_noprior/scene.xml` | 1000.2 KB | 1 mesh file paths rewritten to ../../assets/... (meshes NOT included) |
| `tasks/task_kin_noprior/task.json` | `<work>/vega/tracker/task_kin_noprior/task.json` | 33.7 KB | scrubbed 5 local/node path(s) |
| `tasks/task_kin_noprior/valid_starts.npy` | `<work>/vega/tracker/task_kin_noprior/valid_starts.npy` | 9.5 KB |  |
| `tasks/task_kin_noprior/valid_starts_audit.json` | `<work>/vega/tracker/task_kin_noprior/valid_starts_audit.json` | 56 B |  |
| `tasks/task_kin_noprior_press/contact_ref.json` | `<work>/vega/tracker/task_kin_noprior_press/contact_ref.json` | 392 B |  |
| `tasks/task_kin_noprior_press/contact_ref.npz` | `<work>/vega/tracker/task_kin_noprior_press/contact_ref.npz` | 667.1 KB |  |
| `tasks/task_kin_noprior_press/contact_ref_kin.json` | `<work>/vega/tracker/task_kin_noprior_press/contact_ref_kin.json` | 16.5 KB |  |
| `tasks/task_kin_noprior_press/contact_ref_kin.npz` | `<work>/vega/tracker/task_kin_noprior_press/contact_ref_kin.npz` | 372.0 KB |  |
| `tasks/task_kin_noprior_press/reference.npz` | `<work>/vega/tracker/task_kin_noprior_press/reference.npz` | 1.7 MB |  |
| `tasks/task_kin_noprior_press/scene.xml` | `<work>/vega/tracker/task_kin_noprior_press/scene.xml` | 1000.2 KB | 1 mesh file paths rewritten to ../../assets/... (meshes NOT included) |
| `tasks/task_kin_noprior_press/task.json` | `<work>/vega/tracker/task_kin_noprior_press/task.json` | 34.2 KB | scrubbed 5 local/node path(s) |
| `tasks/task_kin_noprior_press/valid_starts.npy` | `<work>/vega/tracker/task_kin_noprior_press/valid_starts.npy` | 9.5 KB |  |
| `tasks/task_kin_noprior_press/valid_starts_audit.json` | `<work>/vega/tracker/task_kin_noprior_press/valid_starts_audit.json` | 56 B |  |
| `tasks/task_kin_prior/contact_ref.json` | `<work>/vega/tracker/task_kin_prior/contact_ref.json` | 392 B |  |
| `tasks/task_kin_prior/contact_ref.npz` | `<work>/vega/tracker/task_kin_prior/contact_ref.npz` | 667.1 KB |  |
| `tasks/task_kin_prior/reference.npz` | `<work>/vega/tracker/task_kin_prior/reference.npz` | 1.7 MB |  |
| `tasks/task_kin_prior/scene.xml` | `<work>/vega/tracker/task_kin_prior/scene.xml` | 1000.2 KB | 1 mesh file paths rewritten to ../../assets/... (meshes NOT included) |
| `tasks/task_kin_prior/task.json` | `<work>/vega/tracker/task_kin_prior/task.json` | 33.7 KB | scrubbed 5 local/node path(s) |
| `tasks/task_kin_prior/valid_starts.npy` | `<work>/vega/tracker/task_kin_prior/valid_starts.npy` | 9.5 KB |  |
| `tasks/task_kin_prior/valid_starts_audit.json` | `<work>/vega/tracker/task_kin_prior/valid_starts_audit.json` | 56 B |  |
| `tracker/DESIGN.md` | `<work>/vega/tracker/DESIGN.md` | 10.7 KB | design note (Chinese) |
| `tracker/add_press_prior.py` | `<work>/vega/tracker/add_press_prior.py` | 5.1 KB | path fix |
| `tracker/base_voctest_kp2000.json` | `<work>/vega/tracker/base_voctest_kp2000.json` | 27.8 KB |  |
| `tracker/base_voctest_kp400.json` | `<work>/vega/tracker/base_voctest_kp400.json` | 26.6 KB |  |
| `tracker/baselines/_summary.txt` | `<work>/vega/tracker/baselines/_summary.txt` | 750 B |  |
| `tracker/baselines/stage0_noprior_evalonly_voc0.5.json` | `<work>/vega/tracker/baselines/stage0_noprior_evalonly_voc0.5.json` | 50.1 KB |  |
| `tracker/baselines/stage0_noprior_evalonly_voc0.json` | `<work>/vega/tracker/baselines/stage0_noprior_evalonly_voc0.json` | 45.6 KB |  |
| `tracker/baselines/stage0_noprior_evalonly_voc1.json` | `<work>/vega/tracker/baselines/stage0_noprior_evalonly_voc1.json` | 48.5 KB |  |
| `tracker/baselines/stage0_voc0.5.json` | `<work>/vega/tracker/baselines/stage0_voc0.5.json` | 49.8 KB |  |
| `tracker/baselines/stage0_voc0.json` | `<work>/vega/tracker/baselines/stage0_voc0.json` | 45.0 KB |  |
| `tracker/baselines/stage0_voc1.json` | `<work>/vega/tracker/baselines/stage0_voc1.json` | 49.0 KB |  |
| `tracker/build_kin_contact_labels.py` | `<work>/vega/tracker/build_kin_contact_labels.py` | 6.6 KB |  |
| `tracker/build_kin_task.py` | `<work>/vega/tracker/build_kin_task.py` | 8.6 KB | path fix |
| `tracker/configs/dr_stage0.json` | `<work>/vega/tracker/configs/dr_stage0.json` | 23 B |  |
| `tracker/configs/kin_tc_s5.json` | `(new)` | 5.7 KB | new |
| `tracker/configs/kin_tc_s5_eval_allholes.json` | `(new)` | 5.6 KB | new |
| `tracker/configs/kin_tc_s5_eval_holdout.json` | `(new)` | 2.0 KB | new |
| `tracker/configs/kin_tc_s5_eval_holdplace.json` | `(new)` | 5.4 KB | new |
| `tracker/configs/kin_tcw_s5.json` | `(new)` | 6.0 KB | new |
| `tracker/configs/kin_tcw_s5_eval_holdout.json` | `(new)` | 2.1 KB | new |
| `tracker/configs/kin_tcw_s5_eval_holdplace.json` | `(new)` | 5.5 KB | new |
| `tracker/configs/stage0.json` | `<work>/vega/tracker/configs/stage0.json` | 5.1 KB | path fix |
| `tracker/configs/stage0_noprior.json` | `<work>/vega/tracker/configs/stage0_noprior.json` | 5.1 KB | path fix |
| `tracker/configs/stage0_noprior_evalonly.json` | `<work>/vega/tracker/configs/stage0_noprior_evalonly.json` | 5.1 KB | path fix |
| `tracker/configs/stage0_noprior_r2.json` | `<work>/vega/tracker/configs/stage0_noprior_r2.json` | 5.4 KB | path fix |
| `tracker/configs/stage0_noprior_r3.json` | `<work>/vega/tracker/configs/stage0_noprior_r3.json` | 5.7 KB | path fix |
| `tracker/configs/stage0_r4.json` | `<work>/vega/tracker/configs/stage0_r4.json` | 6.1 KB | path fix |
| `tracker/eval_tracker.py` | `<work>/vega/tracker/eval_tracker.py` | 33.7 KB | path fix |
| `tracker/evals/ckpt_75M.pt` | `<work>/vega/tracker/evals/ckpt_75M.pt` | 20.3 MB | = runs/stage0_noprior/model_75005952.pt (best tracker ckpt so far) |
| `tracker/evals/go_100M_voc0.json` | `<work>/vega/tracker/evals/go_100M_voc0.json` | 54.4 KB |  |
| `tracker/evals/go_152M_voc0.json` | `<work>/vega/tracker/evals/go_152M_voc0.json` | 53.9 KB |  |
| `tracker/evals/go_152M_voc0_T1.json` | `<work>/vega/tracker/evals/go_152M_voc0_T1.json` | 54.2 KB |  |
| `tracker/evals/go_46M_voc0.json` | `<work>/vega/tracker/evals/go_46M_voc0.json` | 53.2 KB |  |
| `tracker/evals/go_50M_voc0.json` | `<work>/vega/tracker/evals/go_50M_voc0.json` | 52.9 KB |  |
| `tracker/evals/go_75M_voc0.json` | `<work>/vega/tracker/evals/go_75M_voc0.json` | 53.1 KB |  |
| `tracker/evals/go_75M_voc0_links.json` | `<work>/vega/tracker/evals/go_75M_voc0_links.json` | 54.5 KB |  |
| `tracker/evals/q21.9M_voc0.5.json` | `<work>/vega/tracker/evals/q21.9M_voc0.5.json` | 29.9 KB |  |
| `tracker/evals/q21.9M_voc0.json` | `<work>/vega/tracker/evals/q21.9M_voc0.json` | 28.2 KB |  |
| `tracker/probe_press.py` | `<work>/vega/tracker/probe_press.py` | 3.1 KB | path fix |
| `tracker/test_tracker_env.json` | `<work>/vega/tracker/test_tracker_env.json` | 1.2 KB |  |
| `tracker/test_tracker_env.py` | `<work>/vega/tracker/test_tracker_env.py` | 3.0 KB | path fix |
| `tracker/tracker_env.py` | `<work>/vega/tracker/tracker_env.py` | 11.7 KB | path fix |
| `tracker/train_tracker.py` | `<work>/vega/tracker/train_tracker.py` | 8.9 KB | path fix; updated 11-08..11-10; updated 11-08..11-10 |
| `videos/blender/KA_final_world2/KA_final_world2_head_overview.mp4` | `(new: render/mj_to_blender.py + render/blender_replay.py)` | 2.6 MB | render only |
| `videos/blender/KA_final_world2/head.mp4` | `(new: render/mj_to_blender.py + render/blender_replay.py)` | 1.9 MB | render only |
| `videos/blender/KA_final_world2/overview.mp4` | `(new: render/mj_to_blender.py + render/blender_replay.py)` | 1.4 MB | render only |
| `videos/caps_rounds/compare_world2.mp4` | `(new: rl/record_rollouts.py + rl/render_compare.py)` | 17.1 MB |  |
| `videos/caps_rounds/compare_world2_carry_slow4x.mp4` | `(new: rl/record_rollouts.py + rl/render_compare.py)` | 8.6 MB |  |
| `videos/caps_rounds/compare_world3.mp4` | `(new: rl/record_rollouts.py + rl/render_compare.py)` | 16.8 MB |  |
| `videos/caps_rounds/ee_path_world2.png` | `(new: rl/ee_path.py + rl/ee_path_plot.py)` | 380.6 KB |  |
| `videos/kin_failures/KA_world150_12_stand_yaw_p30_pipette_return.mp4` | `(new: rl/record_rollouts.py + rl/render_compare.py)` | 1.4 MB |  |
| `videos/kin_failures/KA_world160_11_stand_yaw_p45_grasp_pose_unreachable.mp4` | `(new: rl/record_rollouts.py + rl/render_compare.py)` | 1.5 MB |  |
| `videos/kin_failures/KA_world20_10_stand_yaw_m30_wrist_joint_limit.mp4` | `(new: rl/record_rollouts.py + rl/render_compare.py)` | 1.3 MB |  |
| `videos/kin_failures/KA_world22_7_tube_slips_out_when_middle_finger_opens.mp4` | `(new: rl/record_rollouts.py + rl/render_compare.py)` | 997.7 KB |  |
| `videos/kin_failures/KA_world37_1_pipette_falls_returning_to_stand.mp4` | `(new: rl/record_rollouts.py + rl/render_compare.py)` | 1.7 MB |  |
| `videos/kin_failures/KA_world38_6_tube_jammed_tilted_in_rack.mp4` | `(new: rl/record_rollouts.py + rl/render_compare.py)` | 1.6 MB |  |
| `videos/kin_failures/KA_world44_2_pipette_extraction_from_stand.mp4` | `(new: rl/record_rollouts.py + rl/render_compare.py)` | 1.3 MB |  |
| `videos/kin_failures/KA_world49_3_tube_entering_rack.mp4` | `(new: rl/record_rollouts.py + rl/render_compare.py)` | 1.4 MB |  |
| `videos/kin_failures/KA_world56_5_tube_grasp_from_rack.mp4` | `(new: rl/record_rollouts.py + rl/render_compare.py)` | 1.3 MB |  |
| `videos/kin_failures/KA_world58_9_pick_r1c0_tube_jammed_tilted.mp4` | `(new: rl/record_rollouts.py + rl/render_compare.py)` | 1.4 MB |  |
| `videos/kin_failures/KA_world59_4_tube_lost_during_press.mp4` | `(new: rl/record_rollouts.py + rl/render_compare.py)` | 1.9 MB |  |
| `videos/kin_failures/KA_world59_8_pick_r1c0_tube_lost_lifting_out.mp4` | `(new: rl/record_rollouts.py + rl/render_compare.py)` | 1.2 MB |  |
| `videos/kin_failures/all_failures_12clips.mp4` | `(new: rl/record_rollouts.py + rl/render_compare.py)` | 17.0 MB |  |
| `videos/kin_ref_final/compare_world1.mp4` | `(new: rl/record_rollouts.py + rl/render_compare.py)` | 17.4 MB |  |
| `videos/kin_ref_final/compare_world2.mp4` | `(new: rl/record_rollouts.py + rl/render_compare.py)` | 17.4 MB |  |
| `videos/kin_ref_final/compare_world2_press_slow2x.mp4` | `(new: rl/record_rollouts.py + rl/render_compare.py)` | 5.4 MB |  |
| `videos/kin_ref_final/compare_world3.mp4` | `(new: rl/record_rollouts.py + rl/render_compare.py)` | 17.2 MB |  |
| `videos/kin_ref_final/ee_path_world2.png` | `(new: rl/ee_path.py + rl/ee_path_plot.py)` | 373.1 KB |  |
| `videos/pipette_vega_wuji_R3_reference_vs_R3B_50fps.mp4` | `<projects>/pipette_vega_wuji_R3_reference_vs_R3B_50fps.mp4` | 9.5 MB |  |
| `videos/pipette_vega_wuji_R3_reference_vs_R3B_hands_closeup_with_MPC_50fps.mp4` | `<projects>/pipette_vega_wuji_R3_reference_vs_R3B_hands_closeup_with_MPC_50fps.mp4` | 14.8 MB |  |
| `videos/pipette_vega_wuji_RL_R2s2_nominal_DRon_vs_DRoff_50fps.mp4` | `<projects>/pipette_vega_wuji_RL_R2s2_nominal_DRon_vs_DRoff_50fps.mp4` | 4.3 MB |  |
| `videos/pipette_vega_wuji_RL_R3B_heldout_50fps.mp4` | `<projects>/pipette_vega_wuji_RL_R3B_heldout_50fps.mp4` | 5.3 MB |  |
| `videos/pipette_vega_wuji_RL_R3B_nominal_DRon_vs_DRoff_50fps.mp4` | `<projects>/pipette_vega_wuji_RL_R3B_nominal_DRon_vs_DRoff_50fps.mp4` | 9.5 MB |  |
| `videos/reference_layers/reference_layers_full.mp4` | `(new: rl/render_compare.py --ref, reference replay, no policy)` | 11.5 MB |  |
| `videos/reference_layers/reference_layers_press_8-12s_slow2x.mp4` | `(new: rl/render_compare.py --ref, reference replay, no policy)` | 2.9 MB |  |
| `videos/reference_layers/reference_layers_return_17-24s_slow2x.mp4` | `(new: rl/render_compare.py --ref, reference replay, no policy)` | 4.9 MB |  |

392 files.
