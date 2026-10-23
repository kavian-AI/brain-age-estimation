#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure: supervising the image through an anatomical bottleneck."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"
NPZ = os.path.join(os.path.dirname(OUT), "kaggle", "brainage_bottleneck",
                   "out", "bottleneck.npz")
BLUE, RED, GREY, GREEN, ORANGE = "#2c6fbb", "#c0504d", "#9a9a9a", "#5b8c5a", "#e8a33d"
TG = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles",
      "WholeBrain", "ICV"]


def main():
    z = np.load(NPZ, allow_pickle=True)
    P, T = z["pred"], z["true"]

    fig = plt.figure(figsize=(13.6, 8.8))
    gs = fig.add_gridspec(2, 3, hspace=0.44, wspace=0.30,
                          left=0.075, right=0.975, top=0.875, bottom=0.085)

    # ---- A  stage A agreement per target
    ax = fig.add_subplot(gs[0, 0])
    r = [np.corrcoef(P[:, j], T[:, j])[0, 1] for j in range(len(TG))]
    o = np.argsort(r)
    cols = [GREEN if v > .7 else (ORANGE if v > .55 else RED) for v in np.array(r)[o]]
    ax.barh(range(len(TG)), np.array(r)[o], color=cols)
    ax.set_yticks(range(len(TG)))
    ax.set_yticklabels([TG[i] for i in o], fontsize=8.5)
    for i, v in enumerate(np.array(r)[o]):
        ax.text(v + .012, i, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("correlation with FreeSurfer (out-of-fold)")
    ax.set_title("A  Can the image reproduce the\nFreeSurfer numbers?",
                 fontsize=10, loc="left")
    ax.grid(axis="x", alpha=.25)

    # ---- B  ventricles scatter (best) vs entorhinal (worst)
    for n, (j, lab) in enumerate([(4, "Ventricles  r=0.92"),
                                  (1, "Entorhinal  r=0.44")]):
        ax = fig.add_subplot(gs[0, 1 + n])
        ax.scatter(T[:, j] / 1e3, P[:, j] / 1e3, s=5, alpha=.28,
                   color=GREEN if n == 0 else RED, edgecolors="none")
        lo = min(T[:, j].min(), P[:, j].min()) / 1e3
        hi = max(T[:, j].max(), P[:, j].max()) / 1e3
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("FreeSurfer (×10³ mm³)")
        ax.set_ylabel("predicted from image")
        ax.set_title(f"{'B' if n == 0 else 'C'}  {lab}", fontsize=10, loc="left")
        ax.grid(alpha=.2)

    # ---- D  the ladder
    ax = fig.add_subplot(gs[1, 0])
    names = ["3D CNN\n(did not train)", "voxel PCA", "BOTTLENECK\nimage→volumes",
             "true FreeSurfer\nvolumes"]
    vals = [0.507, 0.648, 0.824, 0.953]
    cols = [GREY, RED, BLUE, "#1f4e79"]
    ax.bar(range(4), vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + .012, f"{v:.3f}", ha="center", fontsize=9)
    ax.axhline(.5, color="k", ls=":", lw=.9)
    ax.set_xticks(range(4))
    ax.set_xticklabels(names, fontsize=7.8)
    ax.set_ylim(.45, 1.02)
    ax.set_ylabel("AUC")
    ax.set_title("D  AD vs CN (n=621): supervising the image\n"
                 "through anatomy more than doubles what it recovers",
                 fontsize=10, loc="left")
    ax.grid(axis="y", alpha=.25)

    # ---- E  above-chance recovery across tasks
    ax = fig.add_subplot(gs[1, 1])
    tasks = ["AD vs CN", "MCI vs CN", "AD vs MCI"]
    pca = [32.7, 38.8, np.nan]
    bot = [71.5, 63.9, 59.4]
    x = np.arange(3)
    ax.bar(x - .2, pca, .4, color=RED, label="voxel PCA")
    ax.bar(x + .2, bot, .4, color=BLUE, label="bottleneck")
    for i in range(3):
        if np.isfinite(pca[i]):
            ax.text(x[i] - .2, pca[i] + 1.4, f"{pca[i]:.0f}%", ha="center", fontsize=8)
        ax.text(x[i] + .2, bot[i] + 1.4, f"{bot[i]:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=9)
    ax.set_ylim(0, 85)
    ax.set_ylabel("% of true-volume signal recovered\n(above chance)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("E  Recovery falls as the task gets harder —\n"
                 "a trend the naive AUC ratio (86% flat) hid",
                 fontsize=10, loc="left")
    ax.grid(axis="y", alpha=.25)

    # ---- F  the rate arm
    ax = fig.add_subplot(gs[1, 2])
    arms = ["gap\nvalue", "gap\nRATE", "value\n+ RATE"]
    true = [0.639, 0.681, 0.712]
    pred = [0.576, 0.604, 0.636]
    x = np.arange(3)
    ax.bar(x - .2, true, .4, color="#1f4e79", label="true volumes")
    ax.bar(x + .2, pred, .4, color=BLUE, label="from image (bottleneck)")
    ax.axhline(.509, color=RED, ls="--", lw=1.5)
    ax.text(-0.45, .4955, "voxel-PCA rate (chance)", color=RED, fontsize=7.5,
            ha="left", va="center")
    ax.set_xticks(x)
    ax.set_xticklabels(arms, fontsize=8.5)
    ax.set_ylim(.47, .75)
    ax.set_ylabel("C-index")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("F  MCI→dementia: the rate now works from\n"
                 "images, and still beats the value",
                 fontsize=10, loc="left")
    ax.grid(axis="y", alpha=.25)

    fig.suptitle("The anatomical bottleneck: image → FreeSurfer volumes → age → disease",
                 fontsize=13.5, y=0.952)
    p = os.path.join(OUT, "fig19_bottleneck.png")
    fig.savefig(p, dpi=140)
    print("saved ->", p)


if __name__ == "__main__":
    main()
