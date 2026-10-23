#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"
z = np.load(f"{OUT}/exp4_fusion.npz")
keys = ["FAIR", "DIRECT", "MT", "TL", "ENS"]
labels = ["volumes+age\n(baseline)", "direct MLP", "multi-task\n(idea 1)",
          "pretrain→FT\n(idea 2)", "ensemble\n(idea 3)"]
cols = ["#c0504d", "#8a8a8a", "#2b5d8a", "#d98f3f", "#5a9367"]

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
for ax, (pfx, title) in zip(axes, [("diag", "ADNI: AD vs control (diagnosis)"),
                                    ("prog", "ADNI: MCI→dementia (prognosis)")]):
    vals = [z[f"{pfx}_{k}"].mean() for k in keys]
    ax.bar(range(5), vals, color=cols, width=.68)
    ax.axhline(vals[0], color="#c0504d", ls="--", lw=1.3, alpha=.8)
    lo = min(vals) - 0.03
    ax.set_ylim(lo, max(vals) + 0.02)
    ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=8.8)
    ax.set_title(title, fontsize=11.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9.5, fontweight="bold")
axes[0].set_ylabel("AUC", fontsize=12)
fig.suptitle("Do the three fusion ideas beat separate age + disease?  (tabular volumes)",
             fontsize=13, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"{OUT}/fig7_fusion.png", dpi=170)
print("saved")
