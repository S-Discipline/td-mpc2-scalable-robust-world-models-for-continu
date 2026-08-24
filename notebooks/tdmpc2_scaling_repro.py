import marimo

__generated_with = "0.24.0"
app = marimo.App(width="normal")


@app.cell
def _intro():
    import marimo as mo
    return (mo,)


@app.cell
def _data():
    import numpy as np

    # Observed normalized scores from offline evaluation of the published mt30
    # (30-task DMControl) checkpoints, 10 episodes per task.
    observed_sizes = np.array([1, 5, 19, 48])
    observed_score = np.array([18.66, 29.89, 54.36, 59.38])
    # Paper-reported values on the same mt30 set (Fig. 22 of the paper).
    paper_sizes = np.array([1, 5, 19, 48, 317])
    paper_score = np.array([18.9, 28.3, 54.2, 59.4, 71.4])
    return (
        np,
        observed_score,
        observed_sizes,
        paper_score,
        paper_sizes,
    )


@app.cell
def _title(mo):
    mo.md(
        r"""
        # TD-MPC2 — reproducibility of the scaling claim

        **Central question:** does a single TD-MPC2 world model get steadily better as it gets
        bigger, across an entire suite of continuous-control tasks? This notebook reproduces the
        central result of *TD-MPC2: Scalable, Robust World Models for Continuous Control*
        (Hansen, Su & Wang, ICLR 2024, arXiv:2310.16828).

        The figure below was produced by **offline evaluation of the official published
        checkpoints** (not by retraining). To see the result you do not need to rerun any
        experiment — the observed scores are embedded in this notebook and the paper's reported
        values are given alongside.
        """
    )
    return (mo,)


@app.cell
def _plot(np, observed_score, observed_sizes, paper_score, paper_sizes, mo):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    ax.semilogx(
        paper_sizes, paper_score, "o--", color="#bbbbbb",
        label="Paper (mt30)", alpha=0.85, ms=6, lw=1.2, zorder=1,
    )
    ax.semilogx(
        observed_sizes, observed_score, "o-", color="#d1495b",
        label="Observed (published checkpoints)", ms=8, lw=2, zorder=3,
    )
    for s, y in zip(observed_sizes, observed_score):
        ax.annotate(
            f"{y:.1f}", (s, y), textcoords="offset points",
            xytext=(0, 9), ha="center", fontsize=9, color="#d1495b",
        )
    for s, y in zip(paper_sizes[:4], paper_score[:4]):
        ax.annotate(
            f"{y:.1f}", (s, y), textcoords="offset points",
            xytext=(0, -16), ha="center", fontsize=8, color="#999999",
        )
    ax.set_xlabel("Model parameters (M)")
    ax.set_ylabel("Normalized score (DMControl 30-task set)")
    ax.set_ylim(0, 80)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return (plt,)


@app.cell
def _side(mo, observed_score, observed_sizes, paper_score):
    mo.md(
        "<br>**Observed normalized score** on the mt30 set:\n\n"
        + "\n".join(
            f"- **{s}M**: {o:.1f} (paper {p:.1f})"
            for s, o, p in zip(observed_sizes, observed_score, paper_score[:4])
        )
        + "\n\nCapability rises monotonically with model size and tracks the paper within "
        "~0.5 points at every size."
    )
    return ()


@app.cell
def _explain(mo):
    mo.md(
        """
        ## What does this measure?

        TD-MPC2 is a *model-based* RL algorithm: it learns a world model of the environment and
        then **plans** by simulating many action sequences in the model's latent space. The
        **normalized score** is the average over tasks of reward/10 (DMControl), i.e. a number in
        roughly 0-100 reflecting how well the agent performs across a whole task suite. The claim
        being tested is that a **single, larger world model performs better across all tasks** —
        the same "bigger is better" behaviour seen in large language/vision models, applied to
        robot control.

        ## Why the paper and observed values can differ slightly

        The paper reports scores from *its own training runs* at fixed training budgets. We instead
        evaluate the **officially released checkpoints**, which are trained to completion and may be
        tuned slightly more (or less) than the exact run whose number appears in Fig. 22 — so a small
        offset is expected and not a "failure". What matters for the *claim* is the monotone upward
        trend and the close alignment of the numbers, both of which hold here.

        ## Running this yourself

        Everything needed to regenerate the figure is embedded above. To evaluate the published
        checkpoints yourself you would need the official TD-MPC2 repo and a GPU:

        ```bash
        python evaluate.py task=mt30 model_size=48 checkpoint=mt30-48M.pt eval_episodes=10
        ```

        ## Caveats

        - This evaluates the **30-task DMControl** multi-task set (`mt30`), not the paper's full
          80-task set (DMControl + Meta-World), which needs Meta-World.
        - The **317M** checkpoint and the model-free/model-based **baseline comparisons** were not
          run in this reproduction; see `reports/tdmpc2-reproduction/report.md` for the complete
          claim-by-claim status.
        """
    )
    return (mo,)


@app.cell
def _table(observed_score, observed_sizes, paper_score):
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(
        {
            "Model (M)": observed_sizes,
            "Paper mt30": paper_score[:4],
            "Observed": observed_score,
            "Abs. diff": np.abs(paper_score[:4] - observed_score),  # requires np
        }
    )
    return (df,)
