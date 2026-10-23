#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figures for the three experiments that had none: E15 (add-value),
E16 (which regions), E18 (ordering + network)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"
SHORT = ["Hipp", "Ento", "Fusi", "MidT", "Vent", "Whole"]

# ================= FIG 13: E15 - does the profile ADD value?
tasks = ["AD vs CN", "CN vs MCI", "MCI→dementia"]
volonly = [0.9377, 0.7332, 0.7507]
volage = [0.9426, 0.7467, 0.7520]
volageprof = [0.9508, 0.7544, 0.7560]

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
for ax, t, a, b, c in zip(axes, tasks, volonly, volage, volageprof):
    bars = ax.bar(["volumes\nonly", "volumes\n+age", "volumes+age\n+PROFILE"],
                  [a, b, c], color=["#b9c6d2", "#c0504d", "#5a9367"], width=.62)
    lo = min(a, b, c)
    ax.set_ylim(lo - 0.012, c + 0.008)
    for i, v in enumerate([a, b, c]):
        ax.text(i, v + 0.0008, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax.annotate("", xy=(2, c), xytext=(1, b),
                arrowprops=dict(arrowstyle="->", color="#3d6b48", lw=2))
    ax.text(1.5, (b + c) / 2 + 0.002, f"+{c-b:.4f}\np<0.001", ha="center",
            fontsize=9, color="#3d6b48", fontweight="bold")
    ax.set_title(t, fontsize=11)
axes[0].set_ylabel("AUC")
fig.suptitle("FIGURE 13 (exp 15) — The profile ADDS value on top of volumes+age, "
             "in all three tasks", fontsize=12.5, y=1.01)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(f"{OUT}/fig13_addvalue.png", dpi=160)
plt.close()

# ================= FIG 14: E16 - which regions matter
solo = {"MidTemp": 0.765, "Ventricles": 0.746, "Hippocampus": 0.741,
        "WholeBrain": 0.710, "Fusiform": 0.690, "Entorhinal": 0.675}
hr = {"Hippocampus": (1.47, 8e-8), "Ventricles": (1.40, 2e-7),
      "MidTemp": (1.26, 0.003), "Entorhinal": (1.15, 0.04),
      "WholeBrain": (1.02, 0.78), "Fusiform": (0.97, 0.66)}
greedy_lbl = ["MidTemp", "+Hippo", "+Vent", "+Ento", "+Fusi", "+Whole"]
greedy_val = [0.765, 0.790, 0.801, 0.803, 0.802, 0.799]

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))

ax = axes[0]
ks = sorted(solo, key=lambda k: solo[k])
ax.barh(range(len(ks)), [solo[k] for k in ks],
        color=["#c0504d" if solo[k] > 0.73 else "#b9c6d2" for k in ks])
ax.set_yticks(range(len(ks))); ax.set_yticklabels(ks, fontsize=9)
ax.set_xlim(0.6, 0.80); ax.set_xlabel("C-index (this region's rate ALONE)")
for i, k in enumerate(ks):
    ax.text(solo[k] + .002, i, f"{solo[k]:.3f}", va="center", fontsize=8.5)
ax.set_title("A. Each region alone", fontsize=11)

ax = axes[1]
ks2 = sorted(hr, key=lambda k: hr[k][0])
cols = ["#5a9367" if hr[k][1] < 0.05 else "#c9c9c9" for k in ks2]
ax.barh(range(len(ks2)), [hr[k][0] for k in ks2], color=cols)
ax.axvline(1, color="k", ls=":", lw=1.2)
ax.set_yticks(range(len(ks2))); ax.set_yticklabels(ks2, fontsize=9)
ax.set_xlabel("hazard ratio (all regions in ONE model)")
for i, k in enumerate(ks2):
    star = "***" if hr[k][1] < .001 else ("**" if hr[k][1] < .01
                                          else ("*" if hr[k][1] < .05 else "ns"))
    ax.text(hr[k][0] + .01, i, f"{hr[k][0]:.2f} {star}", va="center", fontsize=8.5)
ax.set_xlim(0.9, 1.65)
ax.set_title("B. Independent contribution\n(grey = NOT significant)", fontsize=11)

ax = axes[2]
ax.plot(range(len(greedy_val)), greedy_val, "o-", color="#2b5d8a", lw=2.2, ms=8)
ax.set_xticks(range(len(greedy_lbl)))
ax.set_xticklabels(greedy_lbl, fontsize=8.5, rotation=25, ha="right")
ax.set_ylabel("C-index")
ax.axhline(greedy_val[1], color="#5a9367", ls="--", lw=1.3)
ax.text(3.1, greedy_val[1] - 0.004, "2 regions = 98.8% of the\nfull 6-region model",
        fontsize=9, color="#3d6b48", fontweight="bold")
for i, v in enumerate(greedy_val):
    ax.text(i, v + 0.0015, f"{v:.3f}", ha="center", fontsize=8)
ax.set_title("C. Adding regions one by one\n(last two HURT)", fontsize=11)

fig.suptitle("FIGURE 14 (exp 16) — Which regions carry the Alzheimer's signal?",
             fontsize=12.5, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.91])
plt.savefig(f"{OUT}/fig14_regions.png", dpi=160)
plt.close()

# ================= FIG 15: E18 - ordering + network
z = np.load(f"{OUT}/exp18_ordering.npz")
P, Cc, Cs = z["P"], z["Cc"], z["Cs"]

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))

ax = axes[0]
im = ax.imshow(P, cmap="RdBu_r", vmin=0.1, vmax=0.9)
ax.set_xticks(range(6)); ax.set_xticklabels(SHORT, fontsize=9, rotation=45)
ax.set_yticks(range(6)); ax.set_yticklabels(SHORT, fontsize=9)
for i in range(6):
    for j in range(6):
        if i != j and np.isfinite(P[i, j]):
            ax.text(j, i, f"{P[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(P[i, j] - .5) > .2 else "black")
ax.set_title("A. P(row becomes abnormal\nBEFORE column)", fontsize=11)
plt.colorbar(im, ax=ax, fraction=.046)
ax.add_patch(plt.Rectangle((3.5, -0.5), 1, 6, fill=False, ec="#111", lw=2.5))
ax.text(4, 6.15, "Ventricles = always LAST", ha="center", fontsize=8.5,
        fontweight="bold")

ax = axes[1]
iu = np.triu_indices(6, 1)
ax.scatter(Cs[iu], Cc[iu], s=70, color="#2b5d8a", zorder=3)
lim = [-0.05, 0.7]
ax.plot(lim, lim, "k--", lw=1.2)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("coupling in STABLE"); ax.set_ylabel("coupling in CONVERTERS")
for (i, j) in zip(*iu):
    if abs(Cc[i, j] - Cs[i, j]) > 0.15:
        ax.annotate(f"{SHORT[i]}-{SHORT[j]}", (Cs[i, j], Cc[i, j]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
ax.set_title("B. Are regions more synchronised\nin converters? (above line = yes)",
             fontsize=11)

ax = axes[2]
ang = np.linspace(0, 2 * np.pi, 7)[:-1]
pos = np.column_stack([np.cos(ang), np.sin(ang)])
prec = np.linalg.pinv(np.corrcoef(z["Rc"].T))
dd = np.sqrt(np.diag(prec))
PC = -prec / np.outer(dd, dd)
np.fill_diagonal(PC, 1)
for i in range(6):
    for j in range(i + 1, 6):
        w = PC[i, j]
        if abs(w) > 0.15:
            ax.plot(*zip(pos[i], pos[j]), color="#c0504d" if w > 0 else "#4a7fb0",
                    lw=abs(w) * 9, alpha=.7, zorder=1)
deg = np.array([np.sum(np.abs(PC[i]) > 0.15) - 1 for i in range(6)])
ax.scatter(pos[:, 0], pos[:, 1], s=300 + 260 * deg, color="#2b5d8a", zorder=3)
for i, s in enumerate(SHORT):
    ax.text(pos[i, 0], pos[i, 1], s, ha="center", va="center", color="white",
            fontsize=8.5, fontweight="bold", zorder=4)
ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.axis("off")
ax.set_title("C. Direct coupling network\n(MidTemp = the hub)", fontsize=11)

fig.suptitle("FIGURE 15 (exp 18) — Do regions age in an order, and do they talk?",
             fontsize=12.5, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.91])
plt.savefig(f"{OUT}/fig15_network.png", dpi=160)
plt.close()

print("saved fig13, fig14, fig15")
