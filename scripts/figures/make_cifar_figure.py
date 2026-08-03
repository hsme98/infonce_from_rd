from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "results" / "paper" / "cifar"
OUT = ROOT / "assets"
OUT.mkdir(parents=True, exist_ok=True)

curves = pd.read_csv(RESULTS / "aggregate_curves.csv")
thresholds = pd.read_csv(RESULTS / "aggregate_thresholds.csv")
methods = ["simclr_z", "simclr_h", "pixel", "random_z", "supervised_h"]
labels = {
    "simclr_z": "InfoNCE critic $z$",
    "simclr_h": "SimCLR encoder $h$",
    "pixel": "pixel MSE",
    "random_z": "random encoder",
    "supervised_h": "supervised reference",
}
linestyles = {
    "simclr_z": "-",
    "simclr_h": "--",
    "pixel": "-.",
    "random_z": ":",
    "supervised_h": "-",
}
markers = {
    "simclr_z": "o",
    "simclr_h": "D",
    "pixel": "s",
    "random_z": "^",
    "supervised_h": None,
}

plt.rcParams.update(
    {
        "font.size": 8.0,
        "axes.titlesize": 8.8,
        "axes.labelsize": 8.0,
        "legend.fontsize": 6.7,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
fig, axes = plt.subplots(1, 3, figsize=(10.3, 2.65), constrained_layout=True)

for ax, metric, title in zip(
    axes[:2],
    ["coarse_retention", "fine_retention"],
    ["(a) CIFAR-100 superclasses", "(b) CIFAR-100 fine classes"],
):
    for method in methods:
        data = curves[(curves.method == method) & (curves.metric == metric)].sort_values("rate")
        line = ax.plot(
            data.rate,
            data["median"],
            label=labels[method],
            linestyle=linestyles[method],
            linewidth=1.45,
            marker=markers[method],
            markevery=35,
            markersize=2.6,
        )[0]
        if method in {"simclr_z", "simclr_h", "supervised_h"}:
            ax.fill_between(data.rate, data.q10, data.q90, alpha=0.07, color=line.get_color())
    ax.set_xlim(0, curves.rate.max())
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"rate $I(X;\widetilde X)$ [nats]")
    ax.set_ylabel("normalized label information retained")
    ax.set_title(title)
    ax.grid(True, alpha=0.22)
axes[1].legend(frameon=False, loc="lower right", ncol=1)

ax = axes[2]
x = np.arange(len(methods))
width = 0.36
for j, (metric, name) in enumerate(
    [("coarse_retention", "coarse"), ("fine_retention", "fine")]
):
    values, lower, upper = [], [], []
    for method in methods:
        row = thresholds[
            (thresholds.method == method)
            & (thresholds.metric == metric)
            & (np.isclose(thresholds.target, 0.8))
        ].iloc[0]
        values.append(row.median_rate)
        lower.append(row.median_rate - row.q10_rate)
        upper.append(row.q90_rate - row.median_rate)
    ax.bar(
        x + (j - 0.5) * width,
        values,
        width=width,
        label=name,
        yerr=np.vstack([lower, upper]),
        capsize=1.5,
        linewidth=0.5,
    )
ax.set_xticks(x, [labels[m].replace("$", "") for m in methods], rotation=25, ha="right")
ax.set_ylabel("rate for 80% retention [nats]")
ax.set_title("(c) semantic rate requirement")
ax.grid(True, axis="y", alpha=0.22)
ax.legend(frameon=False, loc="upper right")

fig.savefig(OUT / "cifar100_semantic_rd_paper.pdf", bbox_inches="tight")
fig.savefig(OUT / "cifar100_semantic_rd_paper.png", dpi=300, bbox_inches="tight")
plt.close(fig)
