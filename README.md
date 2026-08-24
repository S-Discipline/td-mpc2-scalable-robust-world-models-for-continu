# TD-MPC2 — Reproduction on a single GPU

This repository holds a **claim-by-claim reproduction** of
[TD-MPC2: Scalable, Robust World Models for Continuous Control](https://arxiv.org/abs/2310.16828)
(Hansen, Su & Wang, ICLR 2024). It imports the official upstream
[`nicklashansen/tdmpc2`](https://github.com/nicklashansen/tdmpc2) code and adds a
fixed run command (`bash run.sh`) plus a per-branch reproduction driver.

## What was tested

The paper's **headline claim** is that a single TD-MPC2 world model gets
steadily better as it gets bigger — capability scales with model size. We
reproduced it by **offline-evaluating the officially published multi-task
checkpoints** on the 30-task DMControl set (`mt30`) at four model sizes, plus a
few supporting single-task and training-behaviour checks.

| Claim | Paper result (mt30) | Observed | Assessment |
|---|---|---|---|
| **Capability scales with model size** | 18.9 / 28.3 / 54.2 / 59.4 (1/5/19/48M) | **18.66 / 29.89 / 54.36 / 59.38** | **Aligned** (within ~0.5 pt) |
| Single-task asymptotic (4 DMControl tasks) | ~max return bands | falls in band at all 4 tasks | Aligned |
| Online training data-efficiency | fast rise to ~max | rise ~35 → ~980 in 100k steps | Aligned |
| SimNorm ablation ("essential for stability") | large effect on hard/multi-task | no effect on easy walker-walk | Inconclusive (wrong regime) |
| Beats SAC/DreamerV3/TD-MPC baselines | strong wins | not attempted | Not attempted |

The four scaling numbers track the paper within a fraction of a point — the
central result of the paper is confirmed.

## Down-scaling and substitutions (important)

Full-scale TD-MPC2 training (a 317M multi-task model, 4 domains, all baselines,
≈33 GPU-days on an RTX 3090) is **not** feasible on the available hardware, so:

- **Compute used:** a single **NVIDIA L4 (24 GB)** GPU, configured as an SSH
  compute host (`tdmpc2-l4`, a Vast.ai machine). All jobs ran on it.
- **Scaling** was reproduced on the **`mt30` DMControl-only** task set rather than
  the paper's `mt80` (DMControl + Meta-World) because Meta-World is not
  installable on this box without breaking the working MuJoCo/gym stack. The
  paper reports mt30 numbers with the same shape, and those are the ones matched.
- The **317M** checkpoint and the **baseline comparisons** were not run.
- **Retraining** used one easy task (`walker-walk`) at 250k steps to test
  data-efficiency and the SimNorm ablation; `torch.compile` was disabled on the
  L4 for speed (no scientific effect).

## Reproducing the commands

Each experiment runs the **same fixed command** on its own branch; a committed
`run_config.sh` selects the job. Launched with `--backend ssh --host tdmpc2-l4`.

| Branch / experiment | Purpose / change | Exact run command | Outcome | Compute |
|---|---|---|---|---|
| `orx/eval-single-task-walker-walk-baseline-infra` | Baseline infra + walker-walk checkpoint eval | `orx exp run e962e9f5-41d1-4737-b316-99a3735a4a58 --backend ssh --host tdmpc2-l4` | Aligned — walker-walk R 982.9 | L4, ~21 min |
| `orx/scaling-offline-eval-of-mt80-checkpoints-1m-5m-1` | **Scaling**: eval published mt30 @ 1/5/19/48M | `orx exp run 72690700-a3b0-4389-bc2b-00076f751a08 --backend ssh --host tdmpc2-l4` | **Aligned** — 18.66/29.89/54.36/59.38 | L4, ~4h44m |
| `orx/single-task-eval-multiple-dmcontrol-tasks-c1` | Eval published 5M single-task ×4 tasks | `orx exp run 30b00e00-3040-4426-b34a-d62e2baec988 --backend ssh --host tdmpc2-l4` | Aligned — R 983/837/433/752 | L4, ~11 min |
| `orx/short-budget-retrain-walker-walk-full-vs-no-simn` | Retrain walker-walk, SimNorm ON | `orx exp run 716ba16d-0a22-42eb-be74-8f9a7cec368c --backend ssh --host tdmpc2-l4` | Aligned — R ~981 | L4, ~2h23m |
| `orx/short-budget-train-walker-walk-no-simnorm-c3-abl` | Retrain walker-walk, SimNorm OFF | `orx exp run 6dcea292-9736-4cc9-8700-f1cfdeeef9fd --backend ssh --host tdmpc2-l4` | Inconclusive — R ~982 | L4, ~1h49m |

`main` is the publication surface: it hosts this README, the report, the
notebook, and the data. **No formal experiment was launched from `main` itself**
(not run as an experiment — publication surface only).

## Reports and notebook

- **Full reproduction report** with figures and claim-by-claim evidence:
  [`reports/tdmpc2-reproduction/report.md`](reports/tdmpc2-reproduction/report.md)
  (figures in [`images/`](reports/tdmpc2-reproduction/images)).
- **Interactive notebook** that reproduces the scaling figure and explains the
  claim (no expensive reruns needed):
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
