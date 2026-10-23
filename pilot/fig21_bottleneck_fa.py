#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 2 for Report 4, section 4: the anatomical bottleneck.

Five panels, built from the final run (phase3 / E31) rather than the earlier
E27 numbers, so the figure and the text agree.

  A  how well each FreeSurfer number is reproduced from the image
  B  the best case (ventricles) as a scatter
  C  the worst case (entorhinal) as a scatter - and the one that matters
     clinically, which is why it is shown rather than summarised
  D  the recovery ladder on AD vs CN
  E  four routes from pixels, identical backbone and data, differing only in
     what the network is asked to predict

Labels are Persian (reshaped + bidi) to match the report.
"""

import os

import arabic_reshaper
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from bidi.algorithm import get_display

BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
NPZ = os.path.join(BASE, "kaggle", "brainage_phase3", "out", "phase3.npz")
OUT = os.path.join(BASE, "pilot")

TG = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles",
      "WholeBrain", "ICV"]
FA = {"Hippocampus": "هیپوکامپ", "Entorhinal": "قشر آنتورینال",
      "Fusiform": "شکنج دوکی", "MidTemp": "گیجگاهی میانی",
      "Ventricles": "بطن‌ها", "WholeBrain": "کل مغز", "ICV": "داخل جمجمه"}

BLUE, RED, GREEN, GREY, ORANGE = "#2c6fbb", "#c0504d", "#4a8c4a", "#9a9a9a", "#e8a33d"
CEIL = 0.952
PD = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
plt.rcParams["font.family"] = "Tahoma"


def fa(t):
    return get_display(arabic_reshaper.reshape(str(t)))


def num(x, d=3):
    """Latin digits read left-to-right inside Persian text, so numbers are
    converted explicitly rather than left for the bidi algorithm to place."""
    return f"{x:.{d}f}".translate(PD).replace(".", "٫")


def main():
    z = np.load(NPZ, allow_pickle=True)
    P, T = z["pred"], z["true"]
    r = np.array([np.corrcoef(P[:, j], T[:, j])[0, 1] for j in range(len(TG))])

    fig = plt.figure(figsize=(13.6, 8.6))
    gs = fig.add_gridspec(2, 6, hspace=0.46, wspace=1.05,
                          left=0.065, right=0.975, top=0.90, bottom=0.09)

    # ---- A: per-region agreement
    ax = fig.add_subplot(gs[0, :2])
    o = np.argsort(r)
    cols = [GREEN if v > .70 else (ORANGE if v > .55 else RED) for v in r[o]]
    ax.barh(range(len(TG)), r[o], color=cols)
    ax.set_yticks(range(len(TG)))
    ax.set_yticklabels([fa(FA[TG[i]]) for i in o], fontsize=15.3)
    for i, v in enumerate(r[o]):
        ax.text(v + .015, i, num(v, 2), va="center", fontsize=14.4)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel(fa("همبستگی با مقدار واقعی"), fontsize=16.1)
    ax.grid(axis="x", alpha=.25)
    ax.set_title(fa("الف) بازتولید اعداد FreeSurfer از روی تصویر"),
                 fontsize=14.5, loc="left", pad=8)

    # ---- B, C: best and worst, as scatters
    for n, (name, col, lab) in enumerate(
            [("Ventricles", GREEN, "ب"), ("Entorhinal", RED, "ج")]):
        j = TG.index(name)
        ax = fig.add_subplot(gs[0, 2 + 2 * n: 4 + 2 * n])
        ax.scatter(T[:, j] / 1e3, P[:, j] / 1e3, s=5, alpha=.25, color=col,
                   edgecolors="none")
        lo = min(T[:, j].min(), P[:, j].min()) / 1e3
        hi = max(T[:, j].max(), P[:, j].max()) / 1e3
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel(fa("مقدار واقعی (هزار میلی‌متر مکعب)"), fontsize=15.3)
        ax.set_ylabel(fa("پیش‌بینی از تصویر"), fontsize=15.3)
        ax.grid(alpha=.2)
        ax.set_title(fa(f"{lab}) {FA[name]} — همبستگی {num(r[j], 2)}"),
                     fontsize=14.5, loc="left", pad=8)

    # ---- D: the recovery ladder
    ax = fig.add_subplot(gs[1, :3])
    # A Latin token at the START of a Persian string is placed at the wrong end
    # by the bidi algorithm ("PCA وکسلی" reads back as "وکسلی PCA"), so this
    # label is written in Persian throughout.
    names = ["شبکه‌ی سه‌بعدی\nاز صفر", "مؤلفه‌های اصلی\nروی وکسل‌ها",
             "گلوگاه\nتشریحی", "سقف: حجم‌های\nواقعی"]
    vals = [0.507, 0.648, 0.836, CEIL]
    cols = [GREY, RED, BLUE, "#1f4e79"]
    ax.bar(range(4), vals, color=cols, width=.62)
    for i, v in enumerate(vals):
        pct = 100 * (v - .5) / (CEIL - .5)
        ax.text(i, v + .012, num(v), ha="center", fontsize=16.1)
        ax.text(i, v - .045, fa(f"{num(pct,1)}٪"), ha="center", fontsize=14.4,
                color="white", fontweight="bold")
    ax.axhline(.5, color="k", ls=":", lw=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels([fa(n) for n in names], fontsize=14.4)
    ax.set_ylim(.45, 1.02)
    ax.set_ylabel("AUC", fontsize=16.1)
    ax.grid(axis="y", alpha=.25)
    ax.set_title(fa("د) تشخیص آلزایمر: بازیابی بیش از دو برابر می‌شود"),
                 fontsize=14.5, loc="left", pad=8)

    # ---- E: four routes, same backbone and data
    ax = fig.add_subplot(gs[1, 3:])
    routes = ["گلوگاه", "گلوگاه\n(شکاف سن)", "سرتاسری", "سن\nمستقیم",
              "ویژگی\nخام"]
    rv = [0.836, 0.791, 0.724, 0.703, 0.693]
    rc = [BLUE, "#5a8fc9", ORANGE, ORANGE, GREY]
    ax.bar(range(5), rv, color=rc, width=.62)
    for i, v in enumerate(rv):
        ax.text(i, v + .006, num(v), ha="center", fontsize=15.3)
    ax.annotate("", xy=(0, .858), xytext=(2, .858),
                arrowprops=dict(arrowstyle="<->", color="#333", lw=1.2))
    ax.text(1, .866, fa(f"اختلاف {num(0.112)}"), ha="center", fontsize=14.4)
    ax.set_xticks(range(5))
    ax.set_xticklabels([fa(n) for n in routes], fontsize=14.4)
    ax.set_ylim(.66, .90)
    ax.set_ylabel("AUC", fontsize=16.1)
    ax.grid(axis="y", alpha=.25)
    ax.set_title(fa("ه) همان شبکه، همان داده — تنها هدف آموزش فرق می‌کند"),
                 fontsize=14.5, loc="left", pad=8)

    p = os.path.join(OUT, "fig21_bottleneck_fa.png")
    fig.savefig(p, dpi=150, facecolor="white")
    print("saved ->", p)
    for j, t in enumerate(TG):
        print(f"  {t:14s} r = {r[j]:.3f}")
    print(f"  mean r = {r.mean():.3f}")


if __name__ == "__main__":
    main()
