# Figures for the pooled healthy normative model (p3_pooled.py outputs).
#   p3_fig1_resolution_atlas.png  resolution R(age), cross-sectional vs within-person, 8 regions
#   p3_fig2_rates.png             aging rates (%/yr) cross-sectional vs within-person
#   p3_fig3_floor_scanners.png    whole-brain floor by age | spread per scanner | calibration
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p3_out")
A = pd.read_csv(os.path.join(D, "pooled_atlas.csv"))
F = pd.read_csv(os.path.join(D, "pooled_floor.csv"))
C = pd.read_csv(os.path.join(D, "calibration_by_site.csv"))
REG = ["hippocampus", "entorhinal", "amygdala", "thalamus", "lateral ventricles", "inf. lat. ventricles",
       "putamen", "brainstem", "cortex", "total brain"]
CX, LG = "#1f77b4", "#d62728"


def fig1():
    fig, axs = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
    for ax, r in zip(axs.ravel(), REG):
        g = A[A.region == r]
        g = g[(g.age >= 20) & (g.age <= 88)]
        cross_ok = g.in_range
        ax.fill_between(g.age, g.R_cross_lo.clip(3, 300), g.R_cross_hi.clip(3, 300), color=CX, alpha=.15)
        ax.plot(g.age, g.R_cross.clip(3, 300), color=CX, lw=2, label="cross-sectional slope")
        ax.fill_between(g.age, g.R_long_lo.clip(3, 300), g.R_long_hi.clip(3, 300), color=LG, alpha=.15)
        ax.plot(g.age, g.R_long.clip(3, 300), color=LG, lw=2, label="within-person slope")
        ax.axhspan(3, 10, color="green", alpha=.06)
        ax.set_yscale("log")
        ax.set_ylim(4, 300)
        ax.set_yticks([5, 10, 20, 50, 100, 200])
        ax.set_yticklabels(["5", "10", "20", "50", "100", "200"])
        ax.set_title(f"{r}  (n={g.n.iloc[0]})", fontsize=11)
        ax.grid(alpha=.3, which="both")
    for ax in axs[1]:
        ax.set_xlabel("age (years)")
    for ax in axs[:, 0]:
        ax.set_ylabel("age resolution R (years)\n= spread / |aging rate|")
    axs[0, 0].legend(fontsize=9, loc="upper right")
    fig.suptitle("Pooled healthy atlas (1,471 people, 7 scanner/software sites): how many years one person's value "
                 "is uncertain by. Green band: R ≤ 10 y. Shaded: 95% bootstrap CI", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "p3_fig1_resolution_atlas.png"), dpi=130)


def fig2():
    fig, axs = plt.subplots(2, 5, figsize=(20, 8), sharex=True)
    for ax, r in zip(axs.ravel(), REG):
        g = A[(A.region == r) & (A.age >= 20) & (A.age <= 88)]
        ax.fill_between(g.age, g.slope_cross_lo, g.slope_cross_hi, color=CX, alpha=.15)
        ax.plot(g.age, g.slope_cross, color=CX, lw=2, label="cross-sectional")
        ax.fill_between(g.age, g.slope_long_lo, g.slope_long_hi, color=LG, alpha=.15)
        ax.plot(g.age, g.slope_long, color=LG, lw=2, label="within-person (DLBS + OASIS-2)")
        ax.axhline(0, color="k", lw=.8)
        ax.set_title(r, fontsize=11)
        ax.grid(alpha=.3)
    for ax in axs[1]:
        ax.set_xlabel("age (years)")
    for ax in axs[:, 0]:
        ax.set_ylabel("aging rate (% per year)")
    axs[0, 0].legend(fontsize=9)
    fig.suptitle("Aging rates: cross-sectional curve slope vs within-person change (head-size and scanner-drift "
                 "corrected); 95% CI", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "p3_fig2_rates.png"), dpi=130)


def fig3():
    fig, axs = plt.subplots(1, 3, figsize=(20, 5.8))
    ax = axs[0]
    ax.plot(F.age, F.R_brain_cross, color=CX, lw=2, label="floor, cross-sectional direction")
    ax.plot(F.age, F.R_brain_long, color=LG, lw=2, label="floor, within-person direction")
    ax.plot(F.age, F.SD_g_heldout_cross, color=CX, ls="--", label="achieved (held-out), cross")
    ax.plot(F.age, F.SD_g_heldout_long, color=LG, ls="--", label="achieved (held-out), within-person")
    ax.plot(F.age, F.best_single_R_long, color="gray", ls=":", lw=2, label="best single region (within-person)")
    ax.set_xlabel("age (years)")
    ax.set_ylabel("whole-brain precision (years)")
    ax.set_title("Whole-brain precision floor R_brain(age)\n(91 FastSurfer regions, 1,072 people)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    ax.set_ylim(0, 22)

    ax = axs[1]
    regs = ["hippocampus", "entorhinal", "amygdala", "thalamus", "inf. lat. ventricles", "precentral"]
    g = A[(A.age == 70) & A.region.isin(regs)].set_index("region").loc[regs]
    sites = [c[6:-1] for c in A.columns if c.startswith("sigma[")]
    x = np.arange(len(regs))
    wdt = 0.8 / len(sites)
    for i, s in enumerate(sites):
        v = g[f"sigma[{s}]"] / g["sigma"]
        ax.bar(x + i * wdt - 0.4 + wdt / 2, v, wdt, label=s)
    ax.axhline(1, color="k", lw=.8)
    ax.set_xticks(x)
    ax.set_xticklabels(regs, rotation=20, ha="right")
    ax.set_ylabel("spread on this scanner / average spread")
    ax.set_title("Between-person spread differs by scanner (age 70)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0.6, 1.5)

    ax = axs[2]
    C2 = C.set_index(C.columns[0])
    ax.barh(C2.index, C2.within_1_96 if "within_1_96" in C2 else C2["within_1.96"], color="#2ca02c")
    ax.axvline(0.95, color="k", ls="--", lw=1)
    for i, (s, row) in enumerate(C2.iterrows()):
        ax.text(0.905, i, f"SD(z) {row.sd_z:.2f}", va="center", fontsize=9)
    ax.set_xlim(0.9, 0.97)
    ax.set_xlabel("held-out z-scores within ±1.96 (target 0.95)")
    ax.set_title("Calibration on held-out healthy people (5-fold), 17 regions")
    fig.tight_layout()
    fig.savefig(os.path.join(D, "p3_fig3_floor_scanners.png"), dpi=130)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    print("figures written")
