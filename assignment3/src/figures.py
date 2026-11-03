"""Report figures for A3 from committed result files only (no model is fitted)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from data import A3

RESULTS = A3 / "results"
OUT = A3 / "figures"
# Reference palette (dataviz skill): slots 1-2 validated all-pairs in light mode.
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
WITH, WITHOUT = "#2a78d6", "#eb6834"
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#e34948", "#f0efec", "#2a78d6"])
FAMILIES = [("tree", "Decision tree"), ("knn", "kNN"), ("svm_rbf", "RBF-SVM"), ("adaboost", "AdaBoost (trees)"),
            ("bagged_knn", "Bagged kNN"), ("cnn", "1D CNN"), ("transfer", "CNN + transfer")]
GROUPS = {"tree": ("A3_with_position", "B3_without_absolute_position"),
          "cnn": ("A_seq_with_position", "B_seq_without_absolute_position")}

plt.rcParams.update({"font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2, "axes.facecolor": SURFACE,
                     "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .8})


def groups_for(family):
    return GROUPS["cnn"] if family in ("cnn", "transfer") else GROUPS["tree"]


def development():
    frames = [pd.read_csv(RESULTS / p / "summary.csv") for p in ("models_v3/basic", "models_v3/ensembles", "deep_v3")]
    return pd.concat(frames, ignore_index=True).set_index("model")


def dot(ax, x, y, color, label=None):
    ax.plot(x, y, "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=3, label=label)


def legend(fig):
    handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=c, mec=SURFACE, mew=2) for c in (WITH, WITHOUT)]
    fig.legend(handles, ["with absolute position", "without absolute position"], loc="upper center", ncol=2,
               frameon=False, fontsize=8.5, bbox_to_anchor=(.5, 1.0))


def ab_figure(dev, test):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
    rows = np.arange(len(FAMILIES))[::-1]
    controls = {"Development": (dev.loc["dummy__B3_without_absolute_position", "equal_person_balanced_accuracy"],
                                dev.loc["position_stump__position_only", "equal_person_balanced_accuracy"]),
                "Locked test": (test.loc["dummy__B3_without_absolute_position", "test_equal_person_balanced_accuracy"],
                                test.loc["position_stump__position_only", "test_equal_person_balanced_accuracy"])}
    for ax, (title, source) in zip(axes, [("Development", dev), ("Locked test", test)]):
        column = "equal_person_balanced_accuracy" if source is dev else "test_equal_person_balanced_accuracy"
        dummy, stump = controls[title]
        for value, name, align in [(dummy, "majority ", "right"), (stump, " position stump", "left")]:
            ax.axvline(value, color=MUTED, lw=1, zorder=1)
            ax.text(value, len(FAMILIES) - .45, name, color=MUTED, fontsize=7.5, va="center", ha=align)
        for row, (family, label) in zip(rows, FAMILIES):
            a, b = (f"{family}__{g}" for g in groups_for(family))
            ax.plot([source.loc[a, column], source.loc[b, column]], [row, row], color=GRID, lw=2, zorder=2)
            if source is test:
                for name, color, offset in [(a, WITH, .14), (b, WITHOUT, -.14)]:
                    low, high = json.loads(source.loc[name, "test_ba_interval_95"])
                    ax.plot([low, high], [row + offset] * 2, color=color, lw=1.2, alpha=.7, zorder=2)
            dot(ax, source.loc[a, column], row, WITH)
            dot(ax, source.loc[b, column], row, WITHOUT)
        ax.set_title(title, loc="left", color=INK, fontsize=10)
        ax.set_xlim(.3, .85)
        ax.set_ylim(-.6, len(FAMILIES) - .1)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Balanced accuracy, mean over people")
    axes[0].set_yticks(rows, [label for _, label in FAMILIES])
    legend(fig)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(OUT / "ab_comparison.png", dpi=200)
    plt.close(fig)


def time_figure(dev, test):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
    specs = [("refit_seconds", "Refit on all development windows (s)"),
             ("predict_seconds_test_windows", "Predict 369 test windows (s)")]
    # Label placement per panel where points sit close together: (offset, alignment).
    placement = [{"tree": ((0, 8), "center"), "svm_rbf": ((0, -14), "center"),
                  "cnn": ((0, 8), "center"), "transfer": ((-8, 2), "right")},
                 {"cnn": ((-8, 6), "right"), "transfer": ((8, -2), "left")}]
    for ax, (column, xlabel), labels in zip(axes, specs, placement):
        for family, label in FAMILIES:
            a, b = (f"{family}__{g}" for g in groups_for(family))
            xa, xb = test.loc[a, column], test.loc[b, column]
            ya, yb = dev.loc[a, "equal_person_balanced_accuracy"], dev.loc[b, "equal_person_balanced_accuracy"]
            ax.plot([xa, xb], [ya, yb], color=GRID, lw=1.5, zorder=1)
            dot(ax, xa, ya, WITH)
            dot(ax, xb, yb, WITHOUT)
            offset, align = labels.get(family, ((6, 2), "left"))
            ax.annotate(label, (xa, ya), xytext=offset, textcoords="offset points", fontsize=7.5, color=INK2, ha=align)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("Development balanced accuracy")
    legend(fig)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(OUT / "time_vs_performance.png", dpi=200)
    plt.close(fig)


def importance_figure():
    frames = [pd.read_csv(RESULTS / p / "permutation_importance.csv") for p in ("models_v3/basic", "models_v3/ensembles")]
    table = pd.concat(frames).groupby(["feature", "model"]).importance_mean.mean().unstack()
    order = [f"{f}__{g}" for f, _ in FAMILIES[:5] for g in GROUPS["tree"]]
    names = {"tree": "Tree", "knn": "kNN", "svm_rbf": "SVM", "adaboost": "AdaBoost", "bagged_knn": "Bag-kNN"}
    table = table[order]
    table = table.loc[table.abs().max(axis=1).sort_values(ascending=False).index]
    limit = float(np.nanmax(np.abs(table.to_numpy())))
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    image = ax.imshow(table.to_numpy(), cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(order)), [f"{names[m.split('__')[0]]}\n{'A3' if 'A3' in m else 'B3'}" for m in order],
                  fontsize=7.5, rotation=0)
    ax.set_yticks(range(len(table)), table.index, fontsize=7.5)
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for boundary in np.arange(1.5, len(order) - 1, 2):
        ax.axvline(boundary, color=SURFACE, lw=3)
    for (r, c), value in np.ndenumerate(table.to_numpy()):
        if np.isnan(value):
            ax.text(c, r, "–", ha="center", va="center", color=MUTED, fontsize=7)
    bar = fig.colorbar(image, ax=ax, shrink=.6)
    bar.set_label("Drop in balanced accuracy when shuffled", color=INK2)
    bar.outline.set_visible(False)
    ax.set_title("Permutation importance on outer validation windows (mean of 3 folds)", loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "permutation_importance.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    dev = development()
    test = pd.read_csv(RESULTS / "final_test/summary.csv").set_index("model")
    test = test[test.method != "v2_basic"]
    ab_figure(dev, test)
    time_figure(dev, test)
    importance_figure()
    print(sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
