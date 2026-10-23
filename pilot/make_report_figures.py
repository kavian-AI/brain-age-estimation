#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extra figures for report 3: literature comparison + longitudinal decomposition."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

DATA = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\data"
OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"
FEAT = ["nWBV", "eTIV", "Sex"]

# ---------------------------------------------------------- FIGURE 2
# our pilot vs published numbers
labels = [
    "chronological\nage (baseline)",
    "brain volume\n(nWBV)",
    "brain-age gap\n(OUR PILOT)",
    "brain-age gap\n(Tan 2025)",
    "64-dim features\n(Tan 2025)",
    "direct model\n(Tan 2025)",
]
vals = [0.507, 0.696, 0.709, 0.70, 0.84, 0.92]
cols = ["#c3ccd6", "#7a9cb8", "#2b5d8a", "#4a7fb0", "#d98f3f", "#c0504d"]

fig, ax = plt.subplots(figsize=(10.5, 5.2))
bars = ax.bar(range(len(vals)), vals, color=cols, width=.66)
ax.axhline(.5, color="k", lw=.9, ls=":")
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0.4, 1.0)
ax.set_ylabel("AUC", fontsize=11)
ax.set_title("Comparison: our pilot vs. published results (AD vs healthy)", fontsize=12)
for i, v in enumerate(vals):
    ax.text(i, v + .014, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")

ax.annotate("", xy=(3, 0.665), xytext=(2, 0.665),
            arrowprops=dict(arrowstyle="<->", color="#444", lw=1.3))
ax.text(2.5, 0.635, "≈ identical", ha="center", fontsize=9.5, style="italic", color="#333")
ax.annotate("", xy=(5, 0.955), xytext=(3, 0.955),
            arrowprops=dict(arrowstyle="<->", color="#c0504d", lw=1.3))
ax.text(4, 0.965, "gap = 0.22 AUC lost by compression", ha="center",
        fontsize=9.5, color="#c0504d")
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_comparison.png", dpi=170)
print("fig2 saved")

# ---------------------------------------------------------- FIGURE 3
cs = pd.read_csv(f"{DATA}/oasis_cross-sectional.csv").dropna(
    subset=["Age", "nWBV", "eTIV", "CDR"]).copy()
cs["Sex"] = (cs["M/F"] == "M").astype(int)
healthy = cs[cs.CDR == 0]
X, y = healthy[FEAT].values, healthy["Age"].values
oof = np.zeros(len(y))
for tr, te in KFold(10, shuffle=True, random_state=0).split(X):
    oof[te] = LinearRegression().fit(X[tr], y[tr]).predict(X[te])
final = LinearRegression().fit(X, y)
bias = LinearRegression().fit(y.reshape(-1, 1), oof - y)

lg = pd.read_csv(f"{DATA}/oasis_longitudinal.csv").dropna(
    subset=["Age", "nWBV", "eTIV"]).copy()
lg["Sex"] = (lg["M/F"] == "M").astype(int)
raw = final.predict(lg[FEAT].values) - lg["Age"].values
lg["gap"] = raw - bias.predict(lg["Age"].values.reshape(-1, 1))
lg["years"] = lg["MR Delay"] / 365.25

rows = []
for sid, g in lg.groupby("Subject ID"):
    if len(g) < 2 or g["years"].nunique() < 2:
        continue
    grp = g["Group"].iloc[0] if g["Group"].nunique() == 1 else "Converted"
    rows.append(dict(group=grp,
                     offset=g.loc[g["years"].idxmin(), "gap"],
                     slope=np.polyfit(g["years"], g["gap"], 1)[0]))
lon = pd.DataFrame(rows)

order = ["Nondemented", "Converted", "Demented"]
fa = ["پایدار سالم", "تبدیل‌شونده", "مبتلا"]
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.8))

for k, (metric, title, ylab) in enumerate([
        ("offset", "A. Offset (baseline gap) = 'reserve'", "brain age gap at baseline (yr)"),
        ("slope", "B. Slope (rate of change) = 'ageing rate'", "gap change per year (yr/yr)")]):
    data = [lon[lon.group == g][metric].values for g in order]
    bp = ax[k].boxplot(data, labels=["stable\nhealthy", "converters", "demented"],
                       patch_artist=True, widths=.55, showmeans=True)
    for p, c in zip(bp["boxes"], ["#cfd9e3", "#e8c07a", "#e2a6a6"]):
        p.set_facecolor(c)
    ax[k].axhline(0, color="k", lw=.8, ls=":")
    ax[k].set_title(title, fontsize=11)
    ax[k].set_ylabel(ylab, fontsize=10)
    p = stats.ttest_ind(data[1], data[0]).pvalue
    ax[k].text(.5, .96, f"converters vs stable:  p = {p:.3f}",
               transform=ax[k].transAxes, ha="center", va="top", fontsize=10,
               bbox=dict(boxstyle="round,pad=0.35", fc="#fff6e5", ec="#d9a441"))
    for i, d in enumerate(data):
        ax[k].text(i + 1, ax[k].get_ylim()[0], f"n={len(d)}", ha="center",
                   va="bottom", fontsize=8, color="#555")

plt.tight_layout()
plt.savefig(f"{OUT}/fig3_longitudinal.png", dpi=170)
print("fig3 saved")

print(lon.groupby("group").agg(n=("offset", "size"), offset=("offset", "mean"),
                               slope=("slope", "mean")).round(3).to_string())
