#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 5: the control experiment across three tasks."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"
r2 = np.load(f"{OUT}/exp2_regional.npz")
r3 = np.load(f"{OUT}/exp3_prognosis.npz")

tasks = [
    ("ADNI: Alzheimer's vs control\n(diagnosis, n=1193)",
     [r2["adni_R1"].mean(), r2["adni_R2"].mean(), r2["adni_R3"].mean(), r2["adni_B2"].mean()]),
    ("ADNI: MCI to dementia\n(prognosis, n=1004)",
     [r3["R1"].mean(), r3["R2"].mean(), r3["R3"].mean(), r3["B2"].mean()]),
    ("ABIDE: autism vs control\n(diagnosis, n=1099)",
     [r2["abide_R1"].mean(), r2["abide_R2"].mean(), r2["abide_R3"].mean(), r2["abide_B2"].mean()]),
]
labels = ["global gap\n(1 number)", "regional profile\n(R numbers)",
          "profile shape\n(severity removed)", "volumes + age\n(FAIR BASELINE)"]
cols = ["#2b5d8a", "#d98f3f", "#8e6bab", "#c0504d"]

fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
for ax, (title, vals) in zip(axes, tasks):
    b = ax.bar(range(4), vals, color=cols, width=.68)
    lo = min(vals) - 0.06
    ax.set_ylim(max(0.45, lo), max(vals) + 0.055)
    ax.axhline(vals[3], color="#c0504d", ls="--", lw=1.3, alpha=.85)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_title(title, fontsize=11)
    for i, v in enumerate(vals):
        ax.text(i, v + .004, f"{v:.3f}", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(3, vals[3] - .028, "nothing beats\nthis line", ha="center",
            fontsize=8.5, color="#8c3b38", style="italic")
axes[0].set_ylabel("AUC", fontsize=12)
fig.suptitle("Does the brain-age transformation add anything over its own inputs?",
             fontsize=13.5, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"{OUT}/fig5_control.png", dpi=170)
print("fig5 saved")
