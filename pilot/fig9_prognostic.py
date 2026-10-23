#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"
df = pd.read_csv(f"{OUT}/exp6_prognostic.csv")

fig = plt.figure(figsize=(15, 5.0))

# ---- Panel A: KM in cognitively-normal, split by brain-age gap tertile
ax = fig.add_subplot(1, 3, 1)
cn = df[df.bdx == "CN"].copy()
q = cn.gap.quantile([1/3, 2/3]).values
cn["grp"] = np.where(cn.gap <= q[0], "low", np.where(cn.gap >= q[1], "high", "mid"))
kmf = KaplanMeierFitter()
for g, col in [("low", "#4a7fb0"), ("mid", "#8a8a8a"), ("high", "#c0504d")]:
    s = cn[cn.grp == g]
    kmf.fit(s.t, s.event, label=f"{g} gap (n={len(s)})")
    kmf.plot_survival_function(ax=ax, ci_show=False, color=col, lw=2)
ax.set_title("A. Cognitively normal at baseline:\nstaying dementia-free by brain-age gap",
             fontsize=10.5)
ax.set_xlabel("years"); ax.set_ylabel("P(still dementia-free)")
ax.set_ylim(0.8, 1.005); ax.legend(fontsize=8.5, loc="lower left")

# ---- Panel B: conversion rates
ax = fig.add_subplot(1, 3, 2)
cats = ["CN\nlow gap", "CN\nhigh gap", "CN flagged\nhealthy", "CN flagged\ndiseased\n(false alarm)"]
lo = cn[cn.gap < cn.gap.quantile(.25)]
hi = cn[cn.gap > cn.gap.quantile(.75)]
cnf = df[df.bdx == "CN"].copy()
cnf["flag"] = (cnf.direct > 0.5).astype(int)
rates = [100*lo.event.mean(), 100*hi.event.mean(),
         100*cnf[cnf.flag == 0].event.mean(), 100*cnf[cnf.flag == 1].event.mean()]
ns = [len(lo), len(hi), (cnf.flag == 0).sum(), (cnf.flag == 1).sum()]
bars = ax.bar(range(4), rates, color=["#4a7fb0", "#c0504d", "#4a7fb0", "#c0504d"], width=.62)
for i, (r, n) in enumerate(zip(rates, ns)):
    ax.text(i, r + 0.3, f"{r:.0f}%", ha="center", fontsize=11, fontweight="bold")
    ax.text(i, 0.3, f"n={n}", ha="center", fontsize=8, color="white")
ax.set_xticks(range(4)); ax.set_xticklabels(cats, fontsize=8.5)
ax.set_ylabel("% converting to dementia")
ax.set_title("B. Who actually converts?\n(brain age flags future risk in healthy)",
             fontsize=10.5)

# ---- Panel C: hazard ratios (forest)
ax = fig.add_subplot(1, 3, 3)
rows = [("gap | whole cohort", 1.35, 1.15, 1.58),
        ("gap | CN only",      1.16, 0.77, 1.75),
        ("gap | MCI only",     1.33, 1.12, 1.58),
        ("direct | whole",     2.19, 1.89, 2.54)]
ypos = range(len(rows))
for i, (lab, hr, lo_, hi_) in enumerate(rows):
    col = "#c0504d" if "gap" in lab else "#2b5d8a"
    ax.plot([lo_, hi_], [i, i], color=col, lw=2)
    ax.plot(hr, i, "o", color=col, ms=8)
    ax.text(hi_ + 0.05, i, f"HR={hr:.2f}", va="center", fontsize=8.5)
ax.axvline(1, color="k", ls=":", lw=1)
ax.set_yticks(list(ypos)); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
ax.set_xlabel("hazard ratio (per SD)")
ax.set_title("C. Independent prognostic hazard\n(gap adjusted for direct model)",
             fontsize=10.5)
ax.set_xlim(0.6, 3.0)

fig.suptitle("Does brain age add PROGNOSTIC value the direct model lacks?  "
             "ADNI, 1415 non-demented, 337 conversions", fontsize=12.5, y=1.01)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f"{OUT}/fig9_prognostic.png", dpi=165)
print("saved")
