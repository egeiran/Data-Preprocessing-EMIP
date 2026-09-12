"""Figures for report.md. Palette and mark specs follow the dataviz reference."""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cleaning
import outliers as o
import transform as t

OUT = pathlib.Path("figures"); OUT.mkdir(exist_ok=True)

SURFACE = "#fcfcfb"
S1_FILL = "#9ec5f4"                            # sequential blue step 200, for histogram fill
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3
CRIT = "#d03b3b"                               # status: threshold / rejected

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 10,
    "axes.titlelocation": "left", "axes.titlepad": 10,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    "legend.fontsize": 8, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def bare(ax, grid="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis=grid, zorder=0)
    if grid == "y":
        ax.spines["bottom"].set_color(AXIS)
    return ax


def fig_holes():
    edges = [0, 20, 50, 100, 150, 400, 1e9]
    names = ["<20", "20-50", "50-100", "100-150", "150-400", ">400"]
    ms = []
    for pid, smp in cleaning.each_participant():        # all 33 files
        m = smp["quality"] != "ok"
        # a run never crosses a stimulus change, same rule as in cleaning.preprocess
        run = ((m != m.shift(fill_value=False)) | (smp["segment"] != smp["segment"].shift())).cumsum()
        sz = m.groupby(run).sum()
        ms += list(sz[sz > 0] * 1000 / smp.attrs["sample_rate"])
    s = pd.Series(ms)
    b = pd.cut(s, edges, labels=names)
    holes = s.groupby(b, observed=False).size() / len(s) * 100
    lost = s.groupby(b, observed=False).sum() / s.sum() * 100
    print(f"  {len(s):,} holes, median {s.median():.0f} ms")
    print(pd.DataFrame({"share_of_holes": holes.round(1), "share_of_lost": lost.round(1)}).to_string())

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(names)); w = 0.38
    ax.bar(x - w/2, holes, w, color=S1, label="share of holes", zorder=3)
    ax.bar(x + w/2, lost, w, color=S2, label="share of lost samples", zorder=3)
    for xi, v in zip(x - w/2, holes):
        ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=7.5, color=INK2)
    for xi, v in zip(x + w/2, lost):
        ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=7.5, color=INK2)
    bare(ax)
    ax.set_xticks(x, names)
    ax.set_xlabel("hole duration (ms)"); ax.set_ylabel("percent")
    ax.set_ylim(0, 88)
    ax.set_title("Most holes are tiny; the damage is in the few long ones")
    ax.legend(loc="upper center", ncol=2)
    fig.savefig(OUT / "fig1_holes.png"); plt.close(fig)


def fig_exclusion():
    rows = []
    for pid, smp in cleaning.each_participant():
        rows.append({"pid": pid,
                     "gap": 100 * (smp["quality"] == "gap").mean(),
                     "code_s": float(cleaning.code_seconds(smp).sum())})
    d = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.axvspan(25, 100, color=CRIT, alpha=0.06, zorder=0)
    ax.axvline(25, color=CRIT, lw=1.2, zorder=2)
    ax.scatter(d.gap, d.code_s, s=34, color=S1, edgecolor=SURFACE, linewidth=1.2, zorder=4)
    for _, r in d.iterrows():
        if r.gap > 20:
            ax.annotate(r.pid, (r.gap, r.code_s), textcoords="offset points",
                        xytext=(6, 3), fontsize=7.5, color=INK2)
    ax.text(26.5, 196, "rejected by a 25% gap rule", color=CRIT, fontsize=8)
    bare(ax, grid="both")
    ax.set_xlabel("gap % of the whole session")
    ax.set_ylabel("usable code-reading seconds")
    ax.set_title("The two exclusion metrics disagree")
    fig.savefig(OUT / "fig2_exclusion.png"); plt.close(fig)


def fig_iqr():
    df = cleaning.build_dataset()
    st = df["stimulus"].astype(str)
    lo, hi = o.iqr_bounds(df["L POR X [px]"])
    code = df.loc[st.str.endswith(("java.jpg", "java2.jpg", "scala.jpg")), "L POR X [px]"]
    cal = df.loc[st == "instruction_calibration.jpg", "L POR X [px]"]

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bins = np.linspace(0, 1920, 61)
    for data, c, lab in [(code, S1, "code reading"), (cal, S2, "calibration screen")]:
        h, _ = np.histogram(data, bins=bins)
        ax.step(bins[:-1], 100 * h / h.sum(), where="post", color=c, lw=1.8, label=lab, zorder=3)
    ax.axvspan(0, lo, color=CRIT, alpha=0.07, zorder=0)
    ax.axvspan(hi, 1920, color=CRIT, alpha=0.07, zorder=0)
    for v in (lo, hi):
        ax.axvline(v, color=CRIT, lw=1.2, zorder=2)
    ax.text(0.985, 0.93, "flagged as outliers by IQR", color=CRIT, fontsize=8,
            ha="right", transform=ax.transAxes)
    bare(ax)
    ax.set_xlim(0, 1920); ax.set_xlabel("gaze x position (px, screen is 1920 wide)")
    ax.set_ylabel("percent of samples")
    ax.set_title("The IQR fence rejects the calibration targets, not the reading data")
    ax.legend(loc="upper left")
    fig.savefig(OUT / "fig4_iqr.png"); plt.close(fig)


def fig_velocity():
    df = cleaning.build_dataset()
    v = o.velocity(df).dropna()
    n_fast = int((v > o.MAX_DEG_S).sum())
    v = v[v > 0]

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    bins = np.logspace(-1, 3.6, 70)
    ax.hist(v, bins=bins, histtype="stepfilled", facecolor=S1_FILL,
            edgecolor=S1, linewidth=1.4, zorder=3)
    ax.axvspan(o.MAX_DEG_S, bins[-1], color=CRIT, alpha=0.07, zorder=0)
    ax.axvline(o.MAX_DEG_S, color=CRIT, lw=1.2, zorder=4)
    ax.annotate(f"{o.MAX_DEG_S} deg/s physical limit\n{n_fast} samples above", (o.MAX_DEG_S, 6e4),
                textcoords="offset points", xytext=(-108, 0), fontsize=8, color=CRIT)
    ax.set_xscale("log"); ax.set_yscale("log")
    bare(ax)
    ax.set_xlabel("gaze velocity (deg/s, log scale)"); ax.set_ylabel("samples (log scale)")
    ax.set_title("Velocity outliers sit far past the physiological range")
    fig.savefig(OUT / "fig3_velocity.png"); plt.close(fig)


def fig_scaling(Xtr, continuous):
    r = Xtr[continuous].agg(["min", "max"]).T
    r["range"] = r["max"] - r["min"]
    r = r.sort_values("range")
    pick = pd.concat([r.head(5), r.tail(5)])

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    y = np.arange(len(pick))
    ax.hlines(y, 0.15, pick["range"], color=GRID, lw=1.0, zorder=2)
    ax.scatter(pick["range"], y, s=46, color=S1, edgecolor=SURFACE, linewidth=1.2, zorder=4)
    for yi, v in zip(y, pick["range"]):
        ax.text(v * 1.5, yi, f"{v:,.2f}" if v < 10 else f"{v:,.0f}",
                va="center", fontsize=7.5, color=INK2)
    ax.set_xscale("log")
    bare(ax, grid="x")
    ax.set_yticks(y, [i.replace(" [px]", "") for i in pick.index])
    ax.set_xlim(0.15, 1.1e4)
    ax.set_xlabel("range of the raw column, max - min (log scale)")
    ratio = r["range"].max() / r["range"].min()
    ax.set_title(f"Five narrowest and five widest features: a factor of ~{ratio:,.0f}")
    fig.savefig(OUT / "fig5_scaling.png"); plt.close(fig)


def fig_pca(Xtr, continuous):
    from sklearn.decomposition import PCA
    ev = PCA().fit(Xtr[continuous]).explained_variance_ratio_
    cum = 100 * ev.cumsum()
    n = len(cum)

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.axhline(90, color=AXIS, lw=1.0, zorder=2)
    ax.axhline(95, color=AXIS, lw=1.0, zorder=2)
    ax.plot(range(1, n + 1), cum, color=S1, lw=2, marker="o", ms=4.5,
            markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=4)
    k90 = int(np.argmax(cum >= 90)) + 1
    ax.scatter([k90], [cum[k90 - 1]], s=90, facecolor="none", edgecolor=CRIT,
               linewidth=1.6, zorder=5)
    ax.annotate(f"{k90} of {n} components\nreach 90%", (k90, cum[k90 - 1]),
                textcoords="offset points", xytext=(12, -18), fontsize=8, color=CRIT)
    ax.text(n, 91, "90%", ha="right", fontsize=7.5, color=MUTED)
    ax.text(n, 96, "95%", ha="right", fontsize=7.5, color=MUTED)
    bare(ax)
    ax.set_xlim(0.5, n + 0.5); ax.set_ylim(20, 103)
    ax.set_xlabel("principal components"); ax.set_ylabel("cumulative variance (%)")
    ax.set_title(f"{n} sensor columns carry about {k90} dimensions of information")
    fig.savefig(OUT / "fig6_pca.png"); plt.close(fig)


if __name__ == "__main__":
    fig_holes();     print("fig1 holes")
    fig_exclusion(); print("fig2 exclusion")
    fig_velocity();  print("fig3 velocity")
    fig_iqr();       print("fig4 iqr")

    df = t.label(o.handle())
    X, y, groups, continuous = t.encode(df)
    Xtr, Xte, ytr, yte, gtr, gte = t.split(X, y, groups)
    fig_scaling(X, continuous);   print("fig5 scaling")     # same frame as transform.report()
    Xtr_s, _, _ = t.scale(Xtr, Xte, continuous)
    fig_pca(Xtr_s, continuous);   print("fig6 pca")
