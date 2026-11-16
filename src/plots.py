"""All figures. Each function returns the matplotlib Figure and, if `save_to` is given, writes a PNG."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .data import FEATURES, TARGET
from .paper_reference import FEATURE_IMPORTANCE, METRICS as PAPER_METRICS, OPTIMIZER_LABELS

# One colour per optimizer, same everywhere. "paper" is always grey, "ours" uses the optimizer colour.
COLORS = {"GS": "#4C78A8", "RS": "#F58518", "BO": "#54A24B", "Baseline": "#7F7F7F"}
PAPER_GREY = "#B0B0B0"


def _finish(fig, save_to):
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=200, bbox_inches="tight")
    return fig


def plot_distributions(df, save_to=None):
    cols = FEATURES + [TARGET]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, c in zip(axes.ravel(), cols):
        ax.hist(df[c], bins=15, color="#4C78A8", edgecolor="white")
        ax.set_title(c)
        ax.set_ylabel("count")
    axes.ravel()[-1].axis("off")
    fig.suptitle("Distribution of each variable (107 rows)", y=1.02)
    return _finish(fig, save_to)


def plot_correlation(df, save_to=None):
    corr = df[FEATURES + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax, cbar_kws={"label": "Pearson r"})
    ax.set_title("Correlation matrix (compare with paper Fig. 4)")
    return _finish(fig, save_to)


def plot_pred_vs_actual(y_true, preds, metrics=None, save_to=None):
    names = list(preds)
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.8), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    lo = min(np.min(y_true), *(np.min(p) for p in preds.values()))
    hi = max(np.max(y_true), *(np.max(p) for p in preds.values()))
    for ax, n in zip(axes, names):
        ax.scatter(y_true, preds[n], color=COLORS.get(n, "#4C78A8"), alpha=0.85, edgecolor="white")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        title = {"Baseline": "Baseline RF", **OPTIMIZER_LABELS}.get(n, n)
        if metrics:
            title += f"\nR²={metrics[n]['R2']:.3f}, RMSE={metrics[n]['RMSE']:.3f}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("actual ALE (m)")
    axes[0].set_ylabel("predicted ALE (m)")
    return _finish(fig, save_to)


def plot_importance(ours, selected, save_to=None):
    """`ours` = DataFrame of importances (rows = features); `selected` = column name to plot."""
    order = list(FEATURE_IMPORTANCE)  # paper's ranking
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(y - 0.2, [FEATURE_IMPORTANCE[f] for f in order], height=0.4, color=PAPER_GREY, label="Paper (reported)")
    ax.barh(y + 0.2, [ours.loc[f, selected] for f in order], height=0.4, color=COLORS.get(selected, "#4C78A8"),
            label=f"Ours ({OPTIMIZER_LABELS.get(selected, selected)})")
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("RF impurity-based importance")
    ax.set_title("Feature importance: paper vs ours")
    ax.legend()
    return _finish(fig, save_to)


def plot_metric_comparison(comp, save_to=None):
    """`comp` = comparison_table() output: paper vs ours for each optimizer, for MAE/RMSE/R2/MAPE."""
    metrics = ["MAE", "RMSE", "R2", "MAPE"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    labels = list(OPTIMIZER_LABELS.values())
    inv = {v: k for k, v in OPTIMIZER_LABELS.items()}
    x = np.arange(len(labels))
    for ax, m in zip(axes, metrics):
        sub = comp[comp["metric"] == m].set_index("optimizer")
        ax.bar(x - 0.2, [sub.loc[l, "paper_reported"] for l in labels], width=0.4, color=PAPER_GREY, label="Paper")
        ax.bar(x + 0.2, [sub.loc[l, "ours"] for l in labels], width=0.4,
               color=[COLORS[inv[l]] for l in labels], label="Ours")
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace(" ", "\n") for l in labels], fontsize=9)
        ax.set_title(m)
    fig.suptitle("Paper-reported (grey, left bar) vs our (coloured, right bar) test-set results", y=1.03)
    return _finish(fig, save_to)


def plot_seed_robustness(long_df, save_to=None):
    """Box plots of test RMSE and R2 across different train/test split seeds, with the paper's value marked."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    order = ["GS", "RS", "BO"]
    for ax, m in zip(axes, ["RMSE", "R2"]):
        data = [long_df.loc[long_df["optimizer"] == o, m].to_numpy() for o in order]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5)
        for patch, o in zip(bp["boxes"], order):
            patch.set_facecolor(COLORS[o])
            patch.set_alpha(0.8)
        for i, o in enumerate(order, start=1):
            ax.scatter(np.full(len(data[i - 1]), i), data[i - 1], color="k", s=14, zorder=3)
            ax.scatter([i], [PAPER_METRICS[o][m]], marker="D", color="red", s=45, zorder=4,
                       label="Paper (reported)" if i == 1 else None)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([OPTIMIZER_LABELS[o] for o in order])
        ax.set_title(f"Test {m} over {long_df['seed'].nunique()} different splits")
    axes[0].legend()
    return _finish(fig, save_to)
