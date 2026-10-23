#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figures for Phase 1 (English labels). Run after p1_definitions.py."""

import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p1_definitions as P  # noqa: E402

OUT = P.OUT
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})
COL = {"mtl": "#c0504d", "cortex": "#7f7f7f", "subcortical": "#4a8c4a", "ventricle": "#2c6fbb",
       "callosum": "#e8a33d", "cerebellum": "#7b5ea7", "brainstem": "#8c564b"}


def fig1(m, comp):
    age = m.age_visit.values.astype(float)
    sex, icv = m.SEX.values.astype(float), m.ICV_T.values.astype(float)
    cn = (m.DX == "CN").values
    panels = ["hippocampus", "lateral ventricles", "brainstem", "putamen"]
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.6))
    grid = np.linspace(62, 88, 200)
    for ax, name in zip(axs, panels):
        y = comp[name]
        nm = P.Norm(y[cn], age[cn], sex[cn], icv[cn])
        yadj = y[cn] - nm.b[3] * (sex[cn] - nm.sx) - nm.b[4] * (icv[cn] - nm.ic)
        ax.scatter(age[cn], yadj, s=7, alpha=0.35, color="#555555")
        mu, sd = nm.mu(grid), nm.sigma(grid)
        ax.plot(grid, mu, color="k", lw=2, label="healthy mean μ(a)")
        ax.fill_between(grid, mu - sd, mu + sd, color="#2c6fbb", alpha=0.25, label="±1 σ(a)")
        ax.fill_between(grid, mu - 2 * sd, mu + 2 * sd, color="#2c6fbb", alpha=0.10, label="±2 σ(a)")
        txt = "\n".join(f"age {a:.0f}: S={abs(nm.k(a))/nm.sigma(a):.3f} SD/yr, R={nm.sigma(a)/max(abs(nm.k(a)),1e-9):.0f} y"
                        for a in (70, 80))
        ax.text(0.02, 0.02 if name not in ("lateral ventricles", "hippocampus") else 0.86, txt, transform=ax.transAxes, fontsize=9,
                bbox=dict(fc="white", ec="0.7"))
        if name == "hippocampus":
            # D7 illustration: a 72-year-old with z = -1.33
            a0 = 70.0
            val = nm.mu(a0) - 1.0 * nm.sigma(a0)
            gg = np.linspace(a0, 88, 4000)
            A = gg[np.argmin(np.abs(nm.mu(gg) - val))]
            ax.scatter([a0], [val], color="#c0504d", s=60, zorder=5)
            ax.hlines(val, a0, A, colors="#c0504d", linestyles="--")
            y0 = ax.get_ylim()[0]
            ax.plot([A, A], [y0, val], color="#c0504d", ls=":")
            ax.set_ylim(bottom=y0)
            ax.annotate(f"70-year-old, z = −1.0\nreads back to age ≈ {A:.0f}\n(regional gap ≈ +{A-a0:.0f} y)",
                        xy=(A, val), xytext=(60.5, val - 1900), fontsize=9, color="#c0504d",
                        arrowprops=dict(arrowstyle="->", color="#c0504d"))
        ax.set_title(name)
        ax.set_xlabel("age (years)")
        ax.set_ylabel("volume (mm³), sex/ICV-adjusted")
    axs[0].legend(loc="upper right", bbox_to_anchor=(1.0, 0.85), fontsize=8)
    fig.suptitle("D1–D3: healthy normative curves (442 cognitively normal ADNI adults)", y=1.02)
    fig.savefig(os.path.join(OUT, "p1_fig1_normative_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig2():
    feat = pd.read_csv(os.path.join(OUT, "region_table_featured.csv"))
    fig, ax = plt.subplots(figsize=(10, 6))
    order = feat[feat.age == 75].sort_values("R").region.tolist()
    cmap = plt.get_cmap("tab20")
    for i, r in enumerate(order):
        d = feat[feat.region == r]
        ax.plot(d.age, d.R, "-o", color=cmap(i % 20), lw=2, ms=4, label=r)
    ax.set_yscale("log")
    ax.set_yticks([5, 10, 20, 50, 100, 300])
    ax.set_yticklabels(["5", "10", "20", "50", "100", "300"])
    for h in (10, 20):
        ax.axhline(h, color="0.6", ls="--", lw=1)
    ax.set_xlabel("age (years)")
    ax.set_ylabel("age resolution R(a) = σ(a)/|k(a)|  (years, log scale)")
    ax.set_title("D4–D5: age resolution changes with age and differs by region")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    fig.savefig(os.path.join(OUT, "p1_fig2_resolution_by_age.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig3():
    a = pd.read_csv(os.path.join(OUT, "region_table_all98.csv"))
    a = a[a.age == 75].sort_values("R").reset_index(drop=True)
    meas = pd.read_csv(os.path.join(OUT, "measurement_resolution.csv")).set_index("region")
    fig, ax = plt.subplots(figsize=(9, 17))
    y = np.arange(len(a))[::-1]
    ax.barh(y, a.R, color=[COL[c] for c in a.cls], height=0.8)
    mm = [meas.M.get(r, np.nan) for r in a.region]
    ax.scatter(mm, y, color="k", s=12, zorder=5, label="measurement resolution M = ε/|k| (OASIS repeats)")
    ax.set_yticks(y)
    ax.set_yticklabels(a.region, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("years (log scale)")
    ax.set_title("Age-resolution atlas at age 75: biological R (bars) vs measurement noise M (dots)")
    for c, col in COL.items():
        ax.barh([], [], color=col, label=c)
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(os.path.join(OUT, "p1_fig3_resolution_atlas.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig4(summary):
    art = pd.read_csv(os.path.join(OUT, "p4_artefact.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.6))
    ax = axs[0]
    ax.scatter(art.beta, art.app_gap_old, color="#c0504d", s=18,
               label=f"healthy, older than training mean (mean age {summary['P4_mean_age_old']:.0f})")
    ax.scatter(art.beta, art.app_gap_young, color="#2c6fbb", s=18,
               label=f"healthy, younger than training mean (mean age {summary['P4_mean_age_young']:.0f})")
    for col, key in (("#c0504d", "app_gap_old"), ("#2c6fbb", "app_gap_young")):
        c = np.polyfit(art.beta, art[key], 1)
        xx = np.linspace(art.beta.min(), art.beta.max(), 10)
        ax.plot(xx, np.polyval(c, xx), color=col)
    for r in ("R_hippocampus", "L_inflatvent", "L_caudate", "brainstem"):
        row = art[art.region == r].iloc[0]
        ax.annotate(r, (row.beta, row.app_gap_old), fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("region's predictability β_r (slope of predicted age on true age)")
    ax.set_ylabel("apparent regional 'age gap' after one shared correction (years)")
    ax.set_title(f"P4: same healthy people, regional 'older/younger' flips sign\n"
                 f"r = {summary['P4_corr_beta_vs_appgap_old']:.2f} (older)  vs  "
                 f"{summary['P4_corr_beta_vs_appgap_young']:.2f} (younger)")
    ax.legend(fontsize=8)
    ax = axs[1]
    V = 5.6 ** 2
    pred = V / (V + art.R75 ** 2)
    ax.scatter(pred, art.beta, s=18, color="#4a8c4a")
    lim = [0, max(pred.max(), art.beta.max()) * 1.05]
    ax.plot(lim, lim, "--", color="0.5")
    ax.set_xlabel("predicted shrinkage  V / (V + R²)   (from resolution alone)")
    ax.set_ylabel("observed shrinkage β_r of a region's age predictor")
    ax.set_title(f"P3: shrinkage is predicted by resolution (r = {summary['P4_corr_beta_obs_vs_pred']:.2f})")
    fig.savefig(os.path.join(OUT, "p1_fig4_predictability_artefact.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig5(summary):
    wb = np.load(os.path.join(OUT, "whole_brain.npz"), allow_pickle=True)
    dx = wb["DX"]
    fig, axs = plt.subplots(1, 3, figsize=(21, 5.8))
    fig.subplots_adjust(wspace=0.35)
    ax = axs[0]
    cols = {"CN": "#4a8c4a", "MCI": "#e8a33d", "Dementia": "#c0504d"}
    for d, c in cols.items():
        s = dx == d
        ax.scatter(wb["g"][s], wb["gap"][s], s=8, alpha=0.45, color=c, label=d)
    ax.set_xlabel("g: aging-like deviation (years), GLS on normative z")
    ax.set_ylabel("classic brain-age gap (ridge, bias-corrected)")
    ax.set_title(f"P6: classic gap vs g   r = {summary['corr_g_classicgap_all']:.2f} (all), "
                 f"{summary['corr_g_classicgap_CN']:.2f} (healthy)")
    ax.legend(markerscale=3)
    ax = axs[1]
    from scipy.stats import chi2
    p = 98
    data_g = [wb["g"][dx == d] for d in cols]
    data_t = [chi2.cdf(wb["T2"][dx == d], p - 1) for d in cols]
    bp = ax.boxplot(data_g, positions=[1, 2, 3], widths=0.5, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], cols.values()):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(list(cols))
    ax.set_ylabel("g (years)")
    ax2 = ax.twinx()
    ax2.plot([1, 2, 3], [np.median(t) for t in data_t], "ks--", ms=8, label="median T² centile (right axis)")
    ax2.set_ylim(0.5, 1.01)
    ax2.set_ylabel("T² healthy centile (χ²)")
    ax2.legend(loc="lower right", fontsize=9)
    ax.set_title("D11–D12: aging-like years (boxes) and atypicality (squares)")
    ax = axs[2]
    best_single = pd.read_csv(os.path.join(OUT, "region_table_all98.csv"))
    best_single = best_single[best_single.age == 75].R.min()
    labels = ["best single\nregion R", "precision floor\nR_brain (theory)", "SD of g\n(held-out healthy)",
              "ridge MAE\n(uncorrected)", "classic gap SD\n(after correction)"]
    vals = [best_single, summary["R_brain_mean_CN"], summary["SD_g_CN_empirical"],
            summary["MAE_raw_ridge_CN"], summary["SD_classic_gap_CN"]]
    ax.bar(range(5), vals, color=["#7f7f7f", "#2c6fbb", "#2c6fbb", "#e8a33d", "#e8a33d"])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.2, f"{v:.1f}", ha="center")
    ax.set_xticks(range(5))
    ax.set_xticklabels(labels, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("years")
    ax.set_title(f"D13/P3: precision in healthy people (98 regions)\nobserved shrinkage β = {summary['beta_obs_ridge']:.2f}, "
                 f"predicted {summary['beta_pred_V/(V+SDg2)_empirical']:.2f}–{summary['beta_pred_V/(V+R2)_floor']:.2f}")
    fig.savefig(os.path.join(OUT, "p1_fig5_whole_brain.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    m, T, names, comp = P.load()
    summary = json.load(open(os.path.join(OUT, "summary.json")))
    fig1(m, comp)
    fig2()
    fig3()
    fig4(summary)
    fig5(summary)
    print("done")
