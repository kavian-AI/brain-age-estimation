#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure for Part II: what happened when we finally used the images."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"
NL, H, W = 13, 40, 80
BG = 0.25
C1, C2, C3 = "#2c6fbb", "#c0504d", "#7f7f7f"


def oasis_geometry():
    c = np.load(os.path.join(OUT, "oasis_img_cache.npz"), allow_pickle=True)
    acq = pd.DataFrame(dict(subject=c["subject"]))
    o = acq.groupby("subject").indices
    subs = sorted(o)
    Xs = np.stack([c["X"][o[s]].mean(0) for s in subs])
    idx = pd.read_csv(os.path.join(OUT, "images_index.csv"))
    meta = idx.groupby("subject").first().reset_index()
    meta = pd.DataFrame(dict(subject=subs)).merge(meta, on="subject", how="left")
    V = Xs.reshape(-1, NL, H, W)[:, NL // 2]
    g = []
    for im in V:
        m = im > BG
        cy, cx = ndimage.center_of_mass(m)
        g.append([cx, cy, m.sum()])
    g = np.array(g)
    Z = PCA(n_components=6, random_state=0).fit_transform(
        StandardScaler().fit_transform(Xs))
    age = meta.Age.values.astype(float)
    rows = []
    for i in range(5):
        rows.append([abs(np.corrcoef(Z[:, i], g[:, 0])[0, 1]),
                     abs(np.corrcoef(Z[:, i], g[:, 2])[0, 1]),
                     abs(np.corrcoef(Z[:, i], age)[0, 1])])
    return np.array(rows)


def main():
    fig = plt.figure(figsize=(13.5, 8.4))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.26,
                          left=0.07, right=0.98, top=0.90, bottom=0.09)

    # ---- A: what the components actually encode
    ax = fig.add_subplot(gs[0, 0])
    R = oasis_geometry()
    x = np.arange(5)
    w = 0.27
    ax.bar(x - w, R[:, 0], w, label="head position", color=C2)
    ax.bar(x, R[:, 1], w, label="head size", color="#e8a33d")
    ax.bar(x + w, R[:, 2], w, label="age", color=C1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{i+1}" for i in range(5)])
    ax.set_ylabel("|correlation|")
    ax.set_title("A  OASIS: the leading components encode head geometry,\n"
                 "not brain ageing — because the slices are not registered",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=.25)

    # ---- B: the representation ladder
    ax = fig.add_subplot(gs[0, 1])
    names = ["80 PCA components\nover 41,600 pixels", "one hand-made feature\n(dark-CSF fraction)",
             "convnet\n(Kaggle T4)", "three tabular numbers\nnWBV, eTIV, sex"]
    vals = [0.526, 0.607, 0.637, 0.711]
    cols = [C2, C3, "#6a9b53", C1]
    b = ax.barh(range(4), vals, color=cols)
    ax.set_yticks(range(4))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0.45, 0.78)
    ax.axvline(0.5, color="k", lw=.8, ls=":")
    for i, v in enumerate(vals):
        ax.text(v + .004, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("disease AUC (OASIS, age-matched, n=163)")
    ax.set_title("B  The information is in the image;\n"
                 "the representation is what throws it away",
                 fontsize=10, loc="left")
    ax.grid(axis="x", alpha=.25)

    # ---- C: ADNI, registered, larger
    ax = fig.add_subplot(gs[1, 0])
    arms = ["volumes\n+ICV+sex", "voxel PCA", "both"]
    ad_gap = [0.923, 0.627, 0.873]
    ad_dir = [0.952, 0.648, 0.931]
    x = np.arange(3)
    ax.bar(x - 0.19, ad_gap, 0.38, label="as a brain-age gap", color=C1)
    ax.bar(x + 0.19, ad_dir, 0.38, label="direct model (+ age)", color="#9dc3e6")
    for i in range(3):
        ax.text(x[i] - 0.19, ad_gap[i] + .008, f"{ad_gap[i]:.3f}", ha="center", fontsize=8)
        ax.text(x[i] + 0.19, ad_dir[i] + .008, f"{ad_dir[i]:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(arms, fontsize=9)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("AUC")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.set_title("C  ADNI, AD vs CN (n=549), volumes properly registered:\n"
                 "the image does not beat — and does not add to — the volumes",
                 fontsize=10, loc="left")
    ax.grid(axis="y", alpha=.25)

    # ---- D: test-retest
    ax = fig.add_subplot(gs[1, 1])
    lab = ["linear\n(PCA)", "convnet\n(T4)"]
    win = [1.89, 1.98]
    bet = [10.95, 12.54]
    x = np.arange(2)
    ax.bar(x - 0.18, bet, 0.36, color=C3, label="between people (biology)")
    ax.bar(x + 0.18, win, 0.36, color=C2, label="same person, minutes apart (noise)")
    for i in range(2):
        ax.text(x[i] - 0.18, bet[i] + .3, f"{bet[i]:.1f} y", ha="center", fontsize=8)
        ax.text(x[i] + 0.18, win[i] + .3, f"{win[i]:.1f} y", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=9)
    ax.set_ylabel("SD of the brain-age gap (years)")
    ax.set_ylim(0, max(bet) * 1.28)
    ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=2)
    ax.set_title("D  The gap's noise floor: rescanning the same brain\n"
                 "minutes later moves it by ~1.9 years (two models agree)",
                 fontsize=10, loc="left")
    ax.grid(axis="y", alpha=.25)

    fig.suptitle("Part II — using the actual images, and what broke along the way",
                 fontsize=13, y=0.965)
    p = os.path.join(OUT, "fig17_images.png")
    fig.savefig(p, dpi=140)
    print("saved ->", p)


if __name__ == "__main__":
    main()
