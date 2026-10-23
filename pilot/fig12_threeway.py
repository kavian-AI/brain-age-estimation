#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"
z = np.load(f"{OUT}/exp14_threeway.npz")
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
short = ["Hipp", "Ento", "Fusi", "MidT", "Vent", "Whole"]
mCN, mMCI, mAD = z["mean_CN"], z["mean_MCI"], z["mean_AD"]

fig = plt.figure(figsize=(15.2, 4.9))

# --- A: adjacent-pair AUCs
ax = fig.add_subplot(1, 3, 1)
tasks = ["CN vs MCI\n(early)", "MCI vs AD\n(staging)", "CN vs AD\n(easy)"]
scal = [0.694, 0.707, 0.869]
prof = [0.734, 0.744, 0.921]
base = [0.735, 0.760, 0.936]
x = np.arange(3); w = 0.26
ax.bar(x - w, scal, w, color="#2b5d8a", label="scalar gap")
ax.bar(x, prof, w, color="#d98f3f", label="regional profile")
ax.bar(x + w, base, w, color="#c0504d", label="volumes + age")
ax.set_xticks(x); ax.set_xticklabels(tasks, fontsize=9)
ax.set_ylim(0.6, 1.0); ax.set_ylabel("AUC")
ax.legend(fontsize=8.3, loc="upper left")
ax.set_title("A. Adjacent stages are the hard part\n(ADNI, age+sex matched)", fontsize=10.5)
for xi, v in zip(x - w, scal):
    ax.text(xi, v + .006, f"{v:.2f}", ha="center", fontsize=7.5)
for xi, v in zip(x, prof):
    ax.text(xi, v + .006, f"{v:.2f}", ha="center", fontsize=7.5, fontweight="bold")
for xi, v in zip(x + w, base):
    ax.text(xi, v + .006, f"{v:.2f}", ha="center", fontsize=7.5)

# --- B: the profile at each stage
ax = fig.add_subplot(1, 3, 2)
xx = np.arange(len(V)); w = 0.26
ax.bar(xx - w, mCN, w, color="#4a7fb0", label="CN")
ax.bar(xx, mMCI, w, color="#e0a458", label="MCI")
ax.bar(xx + w, mAD, w, color="#c0504d", label="AD")
ax.set_xticks(xx); ax.set_xticklabels(short, fontsize=8.5)
ax.set_ylabel("regional brain-age gap (years)")
ax.axhline(0, color="k", lw=.8, ls=":")
ax.legend(fontsize=8.5)
ax.set_title("B. Hippocampus leads, and the lead\ngrows with stage", fontsize=10.5)

# --- C: rotation vs scaling (normalised shapes)
ax = fig.add_subplot(1, 3, 3)
for m, c, lab in [(mCN, "#4a7fb0", "CN"), (mMCI, "#e0a458", "MCI"),
                  (mAD, "#c0504d", "AD")]:
    n = m / np.linalg.norm(m)
    ax.plot(range(len(V)), n, "o-", color=c, lw=2, ms=6, label=lab)
ax.set_xticks(range(len(V))); ax.set_xticklabels(short, fontsize=8.5)
ax.set_ylabel("normalised profile (shape only)")
ax.legend(fontsize=8.5)
ax.set_title("C. Shape ROTATES 31.7° from CN→MCI,\nthen only 3.9° MCI→AD", fontsize=10.5)
ax.text(0.02, 0.03, "the pattern is set EARLY,\nthen just amplifies",
        transform=ax.transAxes, fontsize=8.5, style="italic", color="#444")

fig.suptitle("Three-way CN → MCI → Alzheimer's within ADNI (no site confound)",
             fontsize=12.5, y=1.01)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f"{OUT}/fig12_threeway.png", dpi=165)
print("saved")
