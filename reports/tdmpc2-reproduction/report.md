# TD-MPC2: Reproducing "Scalable, Robust World Models for Continuous Control"

**Overall verdict: D — 仅运行成功 (the code runs, but the paper's core claims
are not verified).** A claim-by-claim reproduction attempt of
[TD-MPC2 — Scalable, Robust World Models for Continuous Control](https://arxiv.org/abs/2310.16828)
(Hansen, Su & Wang, ICLR 2024), run on a single NVIDIA L4 GPU.

> **Read this first — what "reproduced" means here.**
> The numbers from *training our own code* and from *re-evaluating the authors'
> published checkpoints* are fundamentally different evidence. In this report:
> - **Only two results were produced by training our code from scratch**
>   (walker-walk, 5M, 250k steps, single seed × 2 conditions). These are the only
>   genuine reproductions.
> - **Everything else** (the scaling-normalized-scores figure below, the
>   single-task returns) came from **downloading the authors' official
>   checkpoints and re-evaluating them**. That validates our *evaluation harness*
>   and the released models' benchmark scores, but it is **not** evidence that
>   our code / the algorithm reproduces those results.

![Scaling at 4 model sizes: observed (from official checkpoints) vs paper](images/fig1_scaling.png)

Because the scaling numbers in the figure were produced by evaluating the
authors' own checkpoints (not by training), the close alignment is expected and
does **not** constitute reproduction of the scaling claim. The claim "agent
capability scales with model size" was **not** reproduced by this work.

## What is TD-MPC2 and what is it claiming?

Most reinforcement-learning (RL) algorithms work on a *single* task and need
their hyperparameters re-tuned for every new environment. This makes RL brittle
and hard to scale. TD-MPC2 is a **model-based** RL algorithm — it learns a world
model of the environment, then *plans* actions by simulating many action
sequences in that model's latent space and picking the best expected return
(Model Predictive Control). Its headline claim is that one algorithm, with one
fixed set of hyperparameters, is simultaneously:

1. **Robust** — competitive on 104 continuous-control tasks spanning four
   domains, without per-task tuning;
2. **Scalable** — a single agent's capabilities grow as the world model gets
   bigger and sees more data (like LLMs, but for robot control).

The key engineering moves that make this work: a decoder-free "implicit" world
model (it predicts returns, not future images), **discrete regression** of
rewards/values in a log space (so loss scale is reward-independent), a 5-member
Q-function ensemble, a maximum-entropy policy prior that guides the planner, and
**SimNorm** — a softmax-based normalization of the latent state claimed essential
for training stability.

## How we reproduced it

This reproduction is built on the repository's own machinery. The official code
and **published checkpoints are free**, so for the experiments that need only
evaluation we downloaded the released models and measured them; for the
training-behaviour claims we re-trained a small agent from scratch on a single
GPU. This is a deliberate down-scaling: a full-scale reproduction (a 317M
multi-task agent, 4 domains, 33 GPU-days) is out of reach on one NVIDIA L4, so we
use published-checkpoint evaluation for the scaling claim and short-budget
retraining for the algorithmic-behaviour claims, saying so at each step.

The whole pipeline runs through a single fixed command `bash run.sh`, which reads
a committed `run_config.sh` per experiment branch:

```bash
# Fixed per-node command (never differs across branches)
bash run.sh
```

Each experiment changes only its branch's `run_config.sh` (mode, task, model
size, checkpoint) or a minimal code edit for the SimNorm ablation. This keeps the
immutable-source contract clean: the logged commit is exactly the code that ran.

- **Compute:** 1 × NVIDIA L4 (24 GB), 128 vCPU, via an SSH host configured in
  `~/.ssh/config` (`tdmpc2-l4`, a Vast.ai box). Single GPU, so jobs ran
  sequentially.
- **Software:** official `nicklashansen/tdmpc2` code; `torch 2.7.1`, MuJoCo 3.1.2,
  dm-control 1.0.16, in a Python 3.12 venv. `torch.compile` disabled for the
  training runs (it is very slow on the L4's small SM count and changes no result).
- **Evidence channel:** every run prints its config, per-task metrics, and a
  final summary to its run log, cited throughout.

---

## Headline claim: agent capabilities scale with model size — NOT reproduced by this work

The paper *trains* multi-task world models at 1M, 5M, 19M, 48M and 317M parameters
on a large offline dataset and reports a single **normalized score** (mean over
tasks of reward/10 for DMControl and per-task success×100 for Meta-World). Its
central scaling claim is that capability rises monotonically with model size.

**We did not train any multi-task model.** We only *downloaded the authors'
published checkpoints* and re-evaluated them on the 30-task DMControl set
(`mt30`), 10 episodes per task, 30 tasks, at four sizes:

| Model (M params) | Paper `mt30` score | **Observed (from authors' ckpts)** | Note |
|---|---|---|---|
| 1  | 18.9  | 18.66 | evaluating authors' own model |
| 5  | 28.3  | 29.89 | evaluating authors' own model |
| 19 | 54.2  | 54.36 | evaluating authors' own model |
| 48 | 59.4  | 59.38 | evaluating authors' own model |

These numbers merely confirm that the *released checkpoints* score as the authors
report and that our `evaluate.py` harness is consistent. They are **not** produced
by training and therefore do **not** reproduce the scaling claim ("capability
scales with model size"), which requires training models of different sizes from
data. **Assessment: not reproduced (insufficient evidence).**

> Differences from the paper's evaluation setup that prevent direct comparison
> even as a checkpoint check: we used the 30-task `mt30` DMControl set (paper's
> headline uses 80 tasks, DMControl + Meta-World, which is not installable on this
> box); 10 episodes/task, single seed; not the authors' exact eval seeds/conditions.

---

## Claim: single-task RL reaches the reported performance (one hyperparameter set) — partially corroborated, not reproduced

A second claim is that with one hyperparameter set, single-task TD-MPC2 reaches
strong data-efficient performance and beats prior methods. As with scaling, we did
**not** train these — we re-evaluated the authors' published 5M single-task
checkpoints. The comparison-against-baselines part (SAC, DreamerV3, TD-MPC) was
**not** run.

![Single-task DMControl: observed (authors' checkpoints) vs paper asymptotic range](images/fig2_singletask.png)

Evaluating the authors' published 5M single-task checkpoints:

| Task          | Observed return (authors' ckpt) | Paper asymptotic (approx.) |
|---|---|---|
| walker-walk   | 982.9  | ~950–1000 |
| cheetah-run   | 837.0  | ~850–900 |
| hopper-hop    | 433.0  | ~350–500 |
| dog-run       | 752.3  | ~700–800 |

These fall inside the paper's reported bands, confirming the released models score
as reported. They are **not** produced by training, so the single-task
data-efficiency / beats-baselines claim is **not reproduced** here. Assessment:
**partially corroborated (checkpoint re-evaluation only), comparative baselines not
attempted.**

---

## Claim: data-efficient online training behaviour — the one directly reproduced result

To get genuine evidence from *training*, we retrained a 5M TD-MPC2 agent from
scratch on `walker-walk` (250k environment steps, **single seed**, 1 NVIDIA L4).
This is the only place in this report where the numbers were produced by running
`train.py` on our own code.

![Walker-walk online training: full agent vs no-SimNorm](images/fig3_training_ablation.png)

The agent goes from random-init reward (~35) to **~980** out of 1000 within about
100k steps and saturates near the maximum, matching the paper's fast,
data-efficient learning curve for this task. Assessment: **consistent in
direction with the paper's walker-walk curve, but single seed and no directly
comparable paper figure/metric (the paper's Fig. 12 curves are not reproduced in a
machine-readable table), so this is directional support only.**

---

## Claim: SimNorm is "essential to training stability" (ablation) — inconclusive

We ran the same walker-walk training with **SimNorm removed** (latent
normalization replaced by an identity — the only code change on that branch).

![Walker-walk training: the no-SimNorm ablation does not diverge here](images/fig4_dataefficiency.png)

On the easy `walker-walk` task the two arms are **indistinguishable** (single
seed each):

| Condition      | train return | eval return |
|---|---|---|
| Full (SimNorm on)  | 971.4 | 981.5 |
| No SimNorm         | 978.6 | 981.7 |

**Assessment: inconclusive under this setup.** The paper's own ablation shows
SimNorm's effect on the *hardest* tasks (Dog Run, Humanoid Walk) and in 19M
multi-task training. Our single, easy, single-seed walker-walk test is not in the
regime where the paper reports the effect, so we neither support nor refute the
claim from this evidence.

---

## Claim status summary

Evidence is labelled by how it was produced: **trained** (numbers from running
`train.py` on our code) vs **ckpt-eval** (numbers from re-evaluating the authors'
downloadable checkpoints). Only trained numbers count toward reproduction.

| Claim | Paper result | Observed (evidence type) | Assessment |
|---|---|---|---|
| **Scaling** — capability ↑ with model size | mt30 18.9/28.3/54.2/59.4 | 18.7/29.9/54.4/59.4 (ckpt-eval) | **Not reproduced** (no training evidence) |
| Single-task asymptotic | ~max bands | in-band ×4 (ckpt-eval) | Partially corroborated (ckpt only) |
| **Online training** (walker-walk) | fast rise to near-max | 35 → ~980 in 100k steps (**trained**, 1 seed) | **Directionally consistent** |
| SimNorm ablation (stability) | effect on hard/multi-task | no effect on easy walker-walk (trained, 1 seed) | Inconclusive |
| Beats SAC/DreamerV3/TD-MPC | strong wins | not attempted | Not attempted |
| 80-task + 317M scaling | 16.0 → 70.6 | 30-task set, ≤48M, ckpt-eval | Not reproduced |

**Verdict: D — 仅运行成功.** The code runs and produces results, but the paper's
*core* conclusions (scaling through training; beating baselines; cross-domain
robustness with one hyperparameter set) are not verified: they lack training
evidence, valid baselines, multiple seeds, and paper-comparable metrics. The only
directly reproduced evidence is directional (one task, one seed, walker-walk).

### What a full-scale reproduction would still need

- **Meta-World + ManiSkill2 + MyoSuite** installed to evaluate the 80-task/104-task
  claims across all four domains.
- **A larger GPU** (or more GPUs) to re-train the 5M–317M multi-task models from
  the 545M-transition dataset, and to run the baseline (SAC/DreamerV3/TD-MPC)
  comparisons the paper's data-efficiency curves rely on.
- **The hard-task / 19M-multi-task regime** to properly test the SimNorm ablation.

---

## Reproducing commands

Every experiment runs the same fixed command on a branch cloned into the worktree:

```sh
orx exp run <expId> --backend ssh --host tdmpc2-l4
```

Branches and their committed `run_config.sh`:

| Experiment | Branch | Purpose | Command | Assessment |
|---|---|---|---|---|
| Walker-walk eval (root) | `orx/eval-single-task-walker-walk-baseline-infra` | Baseline infra + walker-walk ckpt-eval | `bash run.sh` (eval walker-walk 5M) | ckpt-eval R 982.9 |
| Scaling mt30 | `orx/scaling-offline-eval-of-mt80-checkpoints-1m-5m-1` | ckpt-eval mt30 @ 1/5/19/48M | `bash run.sh` (eval mt30 ×4 sizes) | Not reproduction (ckpt-eval) |
| Single-task DMControl | `orx/single-task-eval-multiple-dmcontrol-tasks-c1` | ckpt-eval 5M ×4 tasks | `bash run.sh` (eval 4 tasks) | ckpt-eval |
| Retrain walker-walk (full) | `orx/short-budget-retrain-walker-walk-full-vs-no-simn` | 250k-step train, SimNorm on | `bash run.sh` (train walker-walk) | trained R ~981 (1 seed) |
| Retrain walker-walk (no-SimNorm) | `orx/short-budget-train-walker-walk-no-simnorm-c3-abl` | 250k-step train, SimNorm off | `bash run.sh` (train walker-walk) | trained R ~982, inconclusive |

Data behind the figures: [`artifacts/scaling/scaling_mt30.json`](../../artifacts/scaling/scaling_mt30.json),
`artifacts/simnorm_on_curve.json`, `artifacts/simnorm_off_curve.json`.
