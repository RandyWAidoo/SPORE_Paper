import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from matplotlib import RcParams 

ap = ArgumentParser()
ap.add_argument('-f', '--file', help="File to analyze", required=True)
ap.add_argument('-e', '--exclude', help="Excluded algorithms", default="")
args = ap.parse_args()
excluded = set(args.exclude.strip().split())

RcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

index = "Algorithm"
metric, metric_units = "ARI", ""
cost, cost_units = "Runtime", "(Seconds)"
df = pd.read_csv(args.file)[f"{index},{metric},{cost}".split(',')]
if excluded:
    df = df[~df["Algorithm"].isin(excluded)]
avgs = df.groupby("Algorithm")[[metric, cost]].mean()

fig, ax = plt.subplots(figsize=(8, 6))
s = 60
ax.set_title(f"Mean {metric}{r'$\uparrow$'} vs Mean {cost}{r'$\downarrow$'}")

ax.set_xscale("log")
x_mn, x_mx = min(avgs[cost]), max(avgs[cost])
xl, xu = x_mn / 1.5, x_mx * 2
ax.set_xbound(xl, xu)
ax.set_xlabel("Mean " + cost + " " + cost_units + " (Log Scale)")

y_mn, y_mx = min(avgs[metric]), max(avgs[metric])
y_r = y_mx - y_mn
y_marg = y_r * .05
yl, yu = y_mn - y_marg, y_mx + y_marg
ax.set_ybound(yl, yu)
ax.set_ylabel("Mean " + metric + " " + metric_units)

# One scatter call per algorithm so each gets a legend entry
cmap = plt.get_cmap("tab10")
for i, (name, row) in enumerate(avgs.iterrows()):
    label = f"{name}  ({row[cost]:.2f}s, {row[metric]:.2f} ARI)"
    print(label)
    ax.scatter(row[cost], row[metric], color=cmap(i), s=50, label=label)

ax.legend(title="Algorithm", loc="best", framealpha=0.9)

if avgs.shape[0] > 1:
    second_smallest_cost = sorted(avgs[cost].values)[1]
    second_largest_metric = sorted(avgs[metric].values)[-2]
    ax.axvline(second_smallest_cost, linestyle='--', color='gray', linewidth=0.8)
    ax.axhline(second_largest_metric, linestyle='--', color='gray', linewidth=0.8)

    ax.fill_between(
        [xl, second_smallest_cost],
        second_largest_metric,
        yu,
        alpha=0.1, color='green'
    )

    ax.set_xbound(xl, xu)
    ax.set_ybound(yl, yu)

from pathlib import Path
plt.savefig(Path(__file__).parent / "pareto.png", dpi=300, bbox_inches="tight", pad_inches=0.1)

plt.show()