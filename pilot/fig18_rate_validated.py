#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure: the rate of brain ageing, re-validated, and what it does not beat."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"
BLUE, RED, GREY, GREEN = "#2c6fbb", "#c0504d", "#8c8c8c", "#5b8c5a"


def main():
    fig = plt.figure(figsize=(13.5, 8.6))
    gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.34,
                          left=0.145, right=0.97, top=0.89, bottom=0.09)

    # ---- A  converters age faster
    ax = fig.add_subplot(gs[0, 0])
    df = pd.read_csv(os.path.join(OUT, "exp8b_cohort.csv"))
    bins = np.linspace(-2.5, 3.5, 26)
    ax.hist(df.rate[df.event == 0], bins=bins, alpha=.72, color=GREY,
            label=f"stayed MCI (n={int((df.event==0).sum())})")
    ax.hist(df.rate[df.event == 1], bins=bins, alpha=.72, color=RED,
            label=f"converted to dementia (n={int(df.event.sum())})")
    for v, c in [(df.rate[df.event == 0].mean(), GREY),
                 (df.rate[df.event == 1].mean(), RED)]:
        ax.axvline(v, color=c, ls="--", lw=1.6)
    ax.set_xlabel("rate of brain ageing  (gap-years per year)")
    ax.set_ylabel("subjects")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("A  Converters' brains age >5x faster\n"
                 "+0.76 vs +0.14 gap-years per year  (n=562)",
                 fontsize=10, loc="left")

    # ---- B  in-sample vs out-of-fold
    ax = fig.add_subplot(gs[0, 1])
    arms = ["age only", "hipp value", "hipp RATE", "gap value", "gap RATE",
            "gap value\n+ RATE"]
    ins = [0.559, 0.712, 0.686, 0.732, 0.733, 0.784]
    oof = [0.555, 0.710, 0.683, 0.730, 0.730, 0.781]
    x = np.arange(len(arms))
    ax.bar(x - .19, ins, .38, color="#c9d9ec", label="in-sample (how E8 measured)")
    ax.bar(x + .19, oof, .38, color=BLUE, label="out-of-fold (honest)")
    ax.set_xticks(x)
    ax.set_xticklabels(arms, fontsize=7.5)
    ax.set_ylim(0.5, 0.83)
    ax.set_ylabel("C-index")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("B  The missing cross-validation was a real flaw\n"
                 "with almost no consequence: optimism is only +0.002 to +0.004",
                 fontsize=10, loc="left")
    ax.grid(axis="y", alpha=.25)

    # ---- C  bootstrap intervals
    ax = fig.add_subplot(gs[1, 0])
    tests = ["rate BEATS value", "rate ADDS to value\n(E8's actual claim)",
             "rate vs\nhippocampus RATE", "rate vs\nhippocampus VALUE"]
    est = [0.001, 0.052, 0.046, 0.021]
    lo = [-0.038, 0.029, 0.020, -0.021]
    hi = [0.044, 0.077, 0.071, 0.059]
    y = np.arange(len(tests))[::-1]
    for i, (e, l, h) in enumerate(zip(est, lo, hi)):
        c = GREEN if l > 0 else RED
        ax.plot([l, h], [y[i], y[i]], color=c, lw=3.4, solid_capstyle="round")
        ax.plot(e, y[i], "o", color=c, ms=8, zorder=3)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(tests, fontsize=8.5)
    ax.set_xlabel("difference in C-index  (95% CI, bootstrap over subjects)")
    ax.set_title("C  Bootstrap over SUBJECTS, not over CV splits:\n"
                 "E8 survives — but not against the simplest baseline",
                 fontsize=10, loc="left")
    ax.grid(axis="x", alpha=.25)

    # ---- D  landmark sweep + image arm
    ax = fig.add_subplot(gs[1, 1])
    L = [1.0, 1.5, 2.0, 2.5, 3.0]
    rate = [0.652, 0.671, 0.682, 0.659, 0.688]
    val = [0.655, 0.651, 0.627, 0.622, 0.611]
    hip = [0.598, 0.619, 0.637, 0.648, 0.644]
    img = [0.510, np.nan, 0.509, np.nan, np.nan]
    ax.plot(L, rate, "o-", color=BLUE, lw=2, label="gap RATE (volumes)")
    ax.plot(L, val, "s--", color=GREY, lw=1.6, label="gap value (volumes)")
    ax.plot(L, hip, "^:", color=GREEN, lw=1.6, label="hippocampus RATE")
    ax.plot(L, img, "x-", color=RED, lw=1.8, ms=9, label="gap RATE (IMAGE)")
    ax.axhline(0.5, color="k", lw=.8, ls=":")
    ax.set_xlabel("landmark L (years of scans used to estimate the rate)")
    ax.set_ylabel("C-index")
    ax.set_ylim(0.45, 0.73)
    ax.legend(fontsize=8, frameon=False, loc="center right")
    ax.set_title("D  The rate needs a long enough window — and the image\n"
                 "contributes nothing at this resolution (n=207 subset)",
                 fontsize=10, loc="left")
    ax.grid(alpha=.25)

    fig.suptitle("The rate of brain ageing: re-validated, and its honest limits",
                 fontsize=13, y=0.955)
    p = os.path.join(OUT, "fig18_rate_validated.png")
    fig.savefig(p, dpi=140)
    print("saved ->", p)


if __name__ == "__main__":
    main()
