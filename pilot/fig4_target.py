#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 4: target-comparison experiment results."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"

models = ["M0\nnWBV alone\n(benchmark)",
          "M1\nage target\n(brain age gap)",
          "M2\ndisease target\n(Cox)",
          "M3\ndirect\nsupervision"]
cdr = [0.628, 0.513, 0.518, 0.529]
grp = [0.642, 0.587, 0.511, 0.504]
cdr_lo = [0.628, 0.485, 0.402, 0.423]
cdr_hi = [0.628, 0.535, 0.599, 0.604]

x = np.arange(len(models))
w = 0.36
fig, ax = plt.subplots(figsize=(10.5, 5.4))

err = [np.array(cdr) - np.array(cdr_lo), np.array(cdr_hi) - np.array(cdr)]
b1 = ax.bar(x - w/2, cdr, w, yerr=err, capsize=4,
            color=["#7a9cb8", "#2b5d8a", "#d98f3f", "#c0504d"],
            label="cohort definition A (CDR-based, 15 events)")
b2 = ax.bar(x + w/2, grp, w, alpha=.45,
            color=["#7a9cb8", "#2b5d8a", "#d98f3f", "#c0504d"],
            hatch="//", edgecolor="white",
            label="cohort definition B (curated label, 13 events)")

ax.axhline(.5, color="k", lw=1, ls=":")
ax.text(3.52, .505, "chance", fontsize=8.5, color="#444", va="bottom", ha="right")
ax.axhline(cdr[0], color="#7a9cb8", lw=1.2, ls="--", alpha=.9)
ax.text(-0.42, cdr[0] + .006, "benchmark = simple brain volume",
        fontsize=9, color="#3d6b8f", style="italic")

for xi, v in zip(x - w/2, cdr):
    ax.text(xi, v + .052, f"{v:.3f}", ha="center", fontsize=9.5, fontweight="bold")
for xi, v in zip(x + w/2, grp):
    ax.text(xi, v + .012, f"{v:.3f}", ha="center", fontsize=9, color="#444")

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9.5)
ax.set_ylim(0.40, 0.75)
ax.set_ylabel("C-index (predicting conversion to dementia)", fontsize=11)
ax.set_title("Does the training target matter?  OASIS-2, n=85 at risk",
             fontsize=12.5)
ax.legend(fontsize=9, loc="upper right", framealpha=.95)
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_target.png", dpi=170)
print("fig4 saved")
