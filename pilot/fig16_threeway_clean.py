#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Clean two-panel version of the three-way figure.
Panel C of fig12 (the profile-rotation angle) is deliberately dropped:
experiment 17 showed that angle is an artefact of the CN profile being ~zero."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"
z = np.load(f"{OUT}/exp14_threeway.npz")
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
SHORT = ["Hipp", "Ento", "Fusi", "MidT", "Vent", "Whole"]
mCN, mMCI, mAD = z["mean_CN"], z["mean_MCI"], z["mean_AD"]

fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))

# ---- A: adjacent-pair AUCs
ax = axes[0]
tasks = ["CN vs MCI\n(early detection)", "MCI vs AD\n(staging)", "CN vs AD\n(easy)"]
scal = [0.694, 0.707, 0.869]
prof = [0.734, 0.744, 0.921]
base = [0.735, 0.760, 0.936]
x = np.arange(3); w = 0.26
ax.bar(x - w, scal, w, color="#2b5d8a", label="scalar gap")
ax.bar(x, prof, w, color="#d98f3f", label="regional profile")
ax.bar(x + w, base, w, color="#c0504d", label="volumes + age")
ax.set_xticks(x); ax.set_xticklabels(tasks, fontsize=9.5)
ax.set_ylim(0.6, 1.0); ax.set_ylabel("AUC", fontsize=11)
ax.legend(fontsize=9, loc="upper left")
ax.set_title("A. Adjacent stages are the hard part\n(ADNI, matched on age AND sex)",
             fontsize=11)
for xi, v in zip(x - w, scal):
    ax.text(xi, v + .006, f"{v:.3f}", ha="center", fontsize=8)
for xi, v in zip(x, prof):
    ax.text(xi, v + .006, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
for xi, v in zip(x + w, base):
    ax.text(xi, v + .006, f"{v:.3f}", ha="center", fontsize=8)
# highlight the tie on early detection
ax.annotate("", xy=(0 + w, base[0] + .035), xytext=(0, prof[0] + .035),
            arrowprops=dict(arrowstyle="<->", color="#3d6b48", lw=1.8))
ax.text(0 + w/2, base[0] + .045, "essentially\nEQUAL", ha="center", fontsize=8.5,
        color="#3d6b48", fontweight="bold")

# ---- B: the profile at each stage
ax = axes[1]
xx = np.arange(len(V)); w = 0.26
ax.bar(xx - w, mCN, w, color="#4a7fb0", label="CN (healthy)")
ax.bar(xx, mMCI, w, color="#e0a458", label="MCI")
ax.bar(xx + w, mAD, w, color="#c0504d", label="Alzheimer's")
ax.set_xticks(xx); ax.set_xticklabels(SHORT, fontsize=9.5)
ax.set_ylabel("regional brain-age gap (years)", fontsize=11)
ax.axhline(0, color="k", lw=.8, ls=":")
ax.legend(fontsize=9)
ax.set_title("B. Hippocampus leads at every stage,\nand its lead grows", fontsize=11)

fig.suptitle("Three-way CN → MCI → Alzheimer's within ADNI (no scanner confound)",
             fontsize=12.5, y=1.01)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f"{OUT}/fig16_threeway_clean.png", dpi=165)
print("saved fig16_threeway_clean.png")
