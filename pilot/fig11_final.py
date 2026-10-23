#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"
z9 = np.load(f"{OUT}/exp9_regional_rate.npz")
d9 = pd.read_csv(f"{OUT}/exp9_regional_rate.csv")
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]

fig = plt.figure(figsize=(15.5, 5.0))

# ---- A: C-index ladder
ax = fig.add_subplot(1, 3, 1)
order = ["direct_snapshot_only", "scalar_rate_only", "regional_rate_PROFILE",
         "direct_+_scalar_rate", "direct_+_regional_PROFILE"]
labs = ["direct\nsnapshot", "scalar\nrate", "regional\nrate profile",
        "direct +\nscalar rate", "direct +\nregional profile"]
vals = [z9[k].mean() for k in order]
cols = ["#2b5d8a", "#e0a458", "#d98f3f", "#7aa06f", "#5a9367"]
ax.bar(range(5), vals, color=cols, width=.68)
ax.axhline(vals[0], color="#2b5d8a", ls="--", lw=1.1, alpha=.7)
ax.set_ylim(0.74, 0.85)
ax.set_xticks(range(5)); ax.set_xticklabels(labs, fontsize=8.3)
ax.set_ylabel("C-index (predict conversion)")
for i, v in enumerate(vals):
    ax.text(i, v + .002, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
ax.set_title("A. Regional rate PROFILE beats the scalar\n(ADNI, n=1100)", fontsize=10.5)

# ---- B: per-region rate, converters vs non
ax = fig.add_subplot(1, 3, 2)
cc = [d9[d9.event == 1][f"rate_{v}"].mean() for v in V]
nc = [d9[d9.event == 0][f"rate_{v}"].mean() for v in V]
x = np.arange(len(V)); w = 0.38
ax.bar(x - w/2, cc, w, color="#c0504d", label="converters")
ax.bar(x + w/2, nc, w, color="#4a7fb0", label="non-converters")
ax.set_xticks(x)
ax.set_xticklabels([v.replace("MidTemp", "MidTemp").replace("WholeBrain", "Whole")
                    for v in V], rotation=35, ha="right", fontsize=8)
ax.set_ylabel("brain-aging rate (gap-yr / yr)")
ax.set_title("B. Where the brain ages fastest\n(hippocampus & temporal lead)", fontsize=10.5)
ax.legend(fontsize=8.5)

# ---- C: OASIS external validation
ax = fig.add_subplot(1, 3, 3)
# the +0.047 gain replicates
adni = [0.772, 0.820]
oasis = [0.633, 0.680]
x = np.arange(2); w = 0.35
b1 = ax.bar(x - w/2, [adni[0], oasis[0]], w, color="#2b5d8a", label="direct only")
b2 = ax.bar(x + w/2, [adni[1], oasis[1]], w, color="#5a9367", label="direct + rate")
ax.set_xticks(x); ax.set_xticklabels(["ADNI\n(n=1100)", "OASIS-2\n(n=85)"], fontsize=9)
ax.set_ylabel("C-index")
ax.set_ylim(0.55, 0.87)
for xi, a, b in zip(x, [adni[0], oasis[0]], [adni[1], oasis[1]]):
    ax.text(xi - w/2, a + .006, f"{a:.3f}", ha="center", fontsize=8.5)
    ax.text(xi + w/2, b + .006, f"{b:.3f}", ha="center", fontsize=8.5, fontweight="bold")
    ax.annotate(f"+{b-a:.3f}", xy=(xi, b + .03), ha="center",
                fontsize=9.5, color="#3d6b48", fontweight="bold")
ax.legend(fontsize=8.5, loc="lower right")
ax.set_title("C. The gain REPLICATES on an\nindependent cohort", fontsize=10.5)

fig.suptitle("Making brain age useful: regional RATE of aging + a combined score, "
             "validated across two cohorts", fontsize=12.5, y=1.01)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f"{OUT}/fig11_final.png", dpi=165)
print("saved")
