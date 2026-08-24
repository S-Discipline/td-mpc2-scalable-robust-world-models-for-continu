# TD-MPC2 — Reproduction on a single GPU

**Verdict: D — 仅运行成功 (the code runs; the paper's core claims are not
verified).** A claim-by-claim reproduction attempt of
[TD-MPC2: Scalable, Robust World Models for Continuous Control](https://arxiv.org/abs/2310.16828)
(Hansen, Su & Wang, ICLR 2024). It imports the official upstream
[`nicklashansen/tdmpc2`](https://github.com/nicklashansen/tdmpc2) code and adds a
fixed run command (`bash run.sh`) plus a per-branch reproduction driver.

## What was tested and the honest bottom line

**Important caveat about evidence type.** Only numbers produced by *training our
own code* count as reproduction. Re-evaluating the authors' downloadable
checkpoints (labelled **ckpt-eval** below) validates our eval harness but does
*not* reproduce the algorithm's behaviour. In this project only the two
walker-walk retraining runs were trained; the scaling and single-task numbers
came from re-evaluating the authors' checkpoints.

| Core claim | Paper result | Observed | Evidence type | Assessment |
|---|---|---|---|---|
| Capability scales with model size | mt30 18.9/28.3/54.2/59.4 | 18.7/29.9/54.4/59.4 | ckpt-eval | **Not reproduced** (no training) |
| Single-task asymptotic (4 tasks) | ~max bands | in-band ×4 | ckpt-eval | Partially corroborated |
| Online training data-efficiency | fast rise to near-max | 35 → ~980 in 100k steps | **trained** (1 seed) | Directionally consistent |
| SimNorm ablation (stability) | effect on hard/multi-task | no effect on easy walker-walk | trained (1 seed) | Inconclusive |
| Beats SAC/DreamerV3/TD-MPC | strong wins | not attempted | — | Not attempted |

The four scaling numbers above match the paper because they were measured on the
**authors' own checkpoints** — they should not be read as a reproduced scaling
result. The paper's *core* conclusions (scaling through training, beating
baselines, cross-domain robustness with one hyperparameter set) are **not**
verified by this work.

## Down-scaling and substitutions (important)

- **Compute used:** a single **NVIDIA L4 (24 GB)** GPU, configured as an SSH
  compute host (`tdmpc2-l4`, a Vast.ai machine). All jobs ran on it.
- **No model is trained except one easy task.** We did not train any multi-task
  or 5M-plus single-task agent; the scaling and single-task figures come from
  re-evaluating the authors' released checkpoints (`mt30` DMControl-only set —
  Meta-World is not installable on this box without breaking the MuJoCo/gym
  stack).
- The **317M** checkpoint, **4 domains**, and the **baseline comparisons** were
  not covered.
- The only trained results are walker-walk, 5M, 250k steps, **single seed**, with
  and without SimNorm. `torch.compile` was disabled on the L4 for speed (no
  scientific effect).

## Reproducing the commands

Each experiment runs the **same fixed command** on its own branch; a committed
`run_config.sh` selects the job. Launched with `--backend ssh --host tdmpc2-l4`.

| Branch / experiment | Purpose / change | Exact run command | Outcome | Compute |
|---|---|---|---|---|
| `orx/eval-single-task-walker-walk-baseline-infra` | Baseline infra + walker-walk ckpt-eval | `orx exp run e962e9f5-41d1-4737-b316-99a3735a4a58 --backend ssh --host tdmpc2-l4` | ckpt-eval walker-walk R 982.9 | L4, ~21 min |
| `orx/scaling-offline-eval-of-mt80-checkpoints-1m-5m-1` | ckpt-eval published mt30 @ 1/5/19/48M | `orx exp run 72690700-a3b0-4389-bc2b-00076f751a08 --backend ssh --host tdmpc2-l4` | ckpt-eval 18.66/29.89/54.36/59.38 | L4, ~4h44m |
| `orx/single-task-eval-multiple-dmcontrol-tasks-c1` | ckpt-eval published 5M ×4 tasks | `orx exp run 30b00e00-3040-4426-b34a-d62e2baec988 --backend ssh --host tdmpc2-l4` | ckpt-eval R 983/837/433/752 | L4, ~11 min |
| `orx/short-budget-retrain-walker-walk-full-vs-no-simn` | Retrain walker-walk, SimNorm ON | `orx exp run 716ba16d-0a22-42eb-be74-8f9a7cec368c --backend ssh --host tdmpc2-l4` | **trained** R ~981 (1 seed) | L4, ~2h23m |
| `orx/short-budget-train-walker-walk-no-simnorm-c3-abl` | Retrain walker-walk, SimNorm OFF | `orx exp run 6dcea292-9736-4cc9-8700-f1cfdeeef9fd --backend ssh --host tdmpc2-l4` | **trained** R ~982, inconclusive | L4, ~1h49m |

`main` is the publication surface: it hosts this README, the report, the
notebook, and the data. **No formal experiment was launched from `main` itself**
(not run as an experiment — publication surface only).

## Reports and notebook

- **Full reproduction report** with figures and claim-by-claim evidence:
  [`reports/tdmpc2-reproduction/report.md`](reports/tdmpc2-reproduction/report.md)
  (figures in [`images/`](reports/tdmpc2-reproduction/images)).
- **Interactive notebook** that redraws the scaling-vs-paper figure (from the
  embedded observed/paper values) and explains the claim and its caveats
  (no expensive reruns needed):
  [`notebooks/tdmpc2_scaling_repro.py`](notebooks/tdmpc2_scaling_repro.py).
  Run locally with `marimo edit notebooks/tdmpc2_scaling_repro.py` or
  `marimo run notebooks/tdmpc2_scaling_repro.py`.
- **Data** behind the figures:
  [`artifacts/scaling/scaling_mt30.json`](artifacts/scaling/scaling_mt30.json),
  [`artifacts/simnorm_on_curve.json`](artifacts/simnorm_on_curve.json),
  [`artifacts/simnorm_off_curve.json`](artifacts/simnorm_off_curve.json).

---

## Upstream TD-MPC2

The rest of this README is the upstream TD-MPC2 documentation (imported verbatim
from `nicklashansen/tdmpc2`). **TD-MPC2** is a scalable, robust model-based RL
algorithm. It compares favorably to existing model-free and model-based methods
across **104** continuous control tasks spanning multiple domains, with a
*single* set of hyperparameters, and supports training a single 317M-parameter
agent to perform 80 tasks across multiple domains, embodiments, and action
spaces. See the upstream repository and https://tdmpc2.com for full details,
models, and datasets.
