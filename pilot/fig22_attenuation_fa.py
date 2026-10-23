#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 3 for Report 4, section 6: longitudinal attenuation.

  A  every image-derived rate against the true-volume ceiling, drawn WITH the
     measurement uncertainty. The bars differ; the intervals overlap. This is
     the self-correction made visible rather than asserted in prose.

  B  cross-sectional accuracy against longitudinal sensitivity, one point per
     region. If they were the same skill the points would lie on the diagonal.
     They do not, and the rank order differs - ICV is read more accurately than
     hippocampus yet reports less of its change.

  C, D  the same fact as raw data: change measured by FreeSurfer against change
     reported by the network, for the best and the worst region. The fitted
     slope IS the attenuation.
"""

import os

import arabic_reshaper
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bidi.algorithm import get_display

BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
NPZ = os.path.join(BASE, "kaggle", "brainage_phase3", "out", "phase3.npz")
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "pilot")

TG = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles",
      "WholeBrain", "ICV"]
FA = {"Hippocampus": "هیپوکامپ", "Entorhinal": "آنتورینال",
      "Fusiform": "دوکی", "MidTemp": "گیجگاهی میانی",
      "Ventricles": "بطن‌ها", "WholeBrain": "کل مغز", "ICV": "داخل جمجمه"}
BLUE, RED, GREEN, GREY, ORANGE = "#2c6fbb", "#c0504d", "#4a8c4a", "#9a9a9a", "#e8a33d"
HALF_CI = 0.055          # half-width of the C-index interval at n=203
PD = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
plt.rcParams["font.family"] = "Tahoma"


def fa(t):
    return get_display(arabic_reshaper.reshape(str(t)))


def num(x, d=3):
    return f"{x:.{d}f}".translate(PD).replace(".", "٫")


def deltas():
    """First-vs-last visit change per subject, true and predicted."""
    z = np.load(NPZ, allow_pickle=True)
    d = pd.DataFrame(dict(PTID=z["subs"].astype(str), VISCODE=z["vis"].astype(str)))
    for j, t in enumerate(TG):
        d["P_" + t] = z["pred"][:, j]
        d["T_" + t] = z["true"][:, j]
    a = pd.read_csv(os.path.join(DATA, "ADNIMERGE_05Mar2026.csv"), low_memory=False)
    k = a[["PTID", "VISCODE", "Years_bl"]].drop_duplicates(["PTID", "VISCODE"])
    d = d.merge(k, on=["PTID", "VISCODE"], how="left").dropna(subset=["Years_bl"])
    dp, dt = {t: [] for t in TG}, {t: [] for t in TG}
    for _, g in d.groupby("PTID"):
        g = g.sort_values("Years_bl")
        if len(g) < 2 or g.Years_bl.iloc[-1] - g.Years_bl.iloc[0] < 0.75:
            continue
        for t in TG:
            dp[t].append(g["P_" + t].iloc[-1] - g["P_" + t].iloc[0])
            dt[t].append(g["T_" + t].iloc[-1] - g["T_" + t].iloc[0])
    xr = {t: np.corrcoef(d["P_" + t], d["T_" + t])[0, 1] for t in TG}
    return ({t: np.array(v) for t, v in dp.items()},
            {t: np.array(v) for t, v in dt.items()}, xr)


def main():
    dp, dt, xr = deltas()
    att = {t: np.polyfit(dt[t], dp[t], 1)[0] for t in TG}
    n = len(dp[TG[0]])

    fig = plt.figure(figsize=(13.4, 8.8))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.26,
                          left=0.075, right=0.975, top=0.91, bottom=0.085)

    # ---- A: rates, drawn with their uncertainty
    ax = fig.add_subplot(gs[0, 0])
    labs = ["مؤلفه‌های\nاصلی", "گلوگاه\n(اجرای ۲)", "گلوگاه\n(اجرای ۳)",
            "شبکه‌ی\nدوقلو", "گلوگاه\n(اجرای ۱)", "سقف:\nحجم واقعی"]
    vals = [0.509, 0.521, 0.551, 0.579, 0.604, 0.672]
    cols = [GREY, BLUE, BLUE, ORANGE, BLUE, "#1f4e79"]
    ax.bar(range(6), vals, color=cols, width=.6)
    ax.errorbar(range(5), vals[:5], yerr=HALF_CI, fmt="none",
                ecolor="#333", elinewidth=1.3, capsize=5)
    for i, v in enumerate(vals):
        ax.text(i, v + (HALF_CI + .012 if i < 5 else .012), num(v),
                ha="center", fontsize=15.3)
    ax.axhline(.5, color="k", ls=":", lw=1)
    ax.set_xticks(range(6))
    ax.set_xticklabels([fa(l) for l in labs], fontsize=13.6)
    ax.set_ylim(.46, .76)
    ax.set_ylabel(fa("شاخص هم‌خوانی"), fontsize=16.1)
    ax.grid(axis="y", alpha=.25)
    ax.set_title(fa("الف) نرخ پیری از تصویر — میله‌ها فرق دارند،\n"
                    "بازه‌های اطمینان هم‌پوشانی دارند"),
                 fontsize=14.5, loc="left", pad=8)

    # ---- B: the dissociation
    ax = fig.add_subplot(gs[0, 1])
    X = np.array([xr[t] for t in TG])
    Y = np.array([att[t] for t in TG])
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=.5)
    for t in TG:
        c = GREEN if att[t] > .6 else (ORANGE if att[t] > .35 else RED)
        ax.scatter(xr[t], att[t], s=110, color=c, zorder=3,
                   edgecolors="white", linewidths=1.5)
        dy = .045 if t not in ("MidTemp", "ICV") else -.065
        ax.annotate(fa(FA[t]), (xr[t], att[t]), textcoords="offset points",
                    xytext=(0, 14 if dy > 0 else -18), ha="center", fontsize=14.4)
    ax.set_xlabel(fa("دقت در یک اسکن (همبستگی مقطعی)"), fontsize=16.1)
    ax.set_ylabel(fa("سهم تغییر واقعی که گزارش می‌شود"), fontsize=16.1)
    ax.set_xlim(.42, 1.02)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=.25)
    ax.set_title(fa("ب) دو توانایی متفاوت‌اند — ترتیب دو محور یکی نیست"),
                 fontsize=14.5, loc="left", pad=8)

    # ---- C, D: the raw data behind the slope
    for k, (t, col, lab) in enumerate([("Ventricles", GREEN, "ج"),
                                       ("Entorhinal", RED, "د")]):
        ax = fig.add_subplot(gs[1, k])
        x, y = dt[t] / 1e3, dp[t] / 1e3
        ax.scatter(x, y, s=12, alpha=.35, color=col, edgecolors="none")
        lim = np.percentile(np.abs(x), 99)
        xs = np.linspace(-lim, lim, 50)
        ax.plot(xs, xs, "k--", lw=1, label=fa("تطابق کامل"))
        b = np.polyfit(dt[t], dp[t], 1)
        ax.plot(xs, b[0] * xs + b[1] / 1e3, color="#222", lw=2,
                label=fa(f"شیب برازش: {num(b[0], 2)}"))
        ax.axhline(0, color="#bbb", lw=.8)
        ax.axvline(0, color="#bbb", lw=.8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(fa("تغییر واقعی (هزار میلی‌متر مکعب)"), fontsize=15.3)
        ax.set_ylabel(fa("تغییر گزارش‌شده"), fontsize=15.3)
        ax.legend(fontsize=14.4, frameon=False, loc="upper left")
        ax.grid(alpha=.2)
        ax.set_title(fa(f"{lab}) {FA[t]} — {num(100*att[t], 0)}٪ از تغییر"),
                     fontsize=14.5, loc="left", pad=8)

    p = os.path.join(OUT, "fig22_attenuation_fa.png")
    fig.savefig(p, dpi=150, facecolor="white")
    print("saved ->", p)
    print(f"subjects with a usable interval: {n}")
    for t in TG:
        print(f"  {t:14s} cross-sec r {xr[t]:.3f}   reports {100*att[t]:5.1f}%")


if __name__ == "__main__":
    main()
