# Figures: every-age normative charts and the model contest.
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p4_out")
T = pd.read_csv(os.path.join(D, "everyage_table.csv"))
M = pd.read_csv(os.path.join(D, "model_comparison.csv"))
meta = json.load(open(os.path.join(D, "everyage_meta.json")))
REG = ["hippocampus", "amygdala", "thalamus", "brainstem", "lateral ventricles", "cortex", "cerebral WM",
       "cerebellum cortex", "entorhinal", "total brain"]


def fig_charts():
    fig, axs = plt.subplots(2, 5, figsize=(21, 8.5))
    for ax, r in zip(axs.ravel(), REG):
        for sex, col in (("women", "#c2185b"), ("men", "#1565c0")):
            g = T[(T.region == r) & (T.sex == sex)]
            ax.fill_between(g.age, g["c2.5"], g["c97.5"], color=col, alpha=.10)
            ax.fill_between(g.age, g["c25"], g["c75"], color=col, alpha=.18)
            ax.plot(g.age, g["c50"], color=col, lw=2, label=f"{sex} (median, 50% and 95% bands)")
        ax.set_title(r)
        ax.set_xlabel("age (years)")
        ax.grid(alpha=.3)
        ax.set_xlim(5, 92)
    for ax in axs[:, 0]:
        ax.set_ylabel("volume (mm³), reference head, average scanner")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"Normative charts, every year of age ({meta['n_people']:,} healthy people, {meta['sites']} sites; "
                 f"reference head size {meta['reference_head_cm3']:.0f} cm³)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "p4_fig1_charts.png"), dpi=120)


def fig_resolution():
    fig, axs = plt.subplots(2, 5, figsize=(21, 8.5), sharey=True)
    for ax, r in zip(axs.ravel(), REG):
        g = T[(T.region == r) & (T.sex == "women")]
        for col, c, lab in (("R_cross", "#1f77b4", "cross-sectional (different people of different ages)"),
                            ("R_long", "#d62728", "within-person (same people re-scanned)")):
            if f"{col}_lo" in g:                       # 95 % bootstrap band
                ax.fill_between(g.age, g[f"{col}_lo"].clip(2, 400), g[f"{col}_hi"].clip(2, 400), color=c, alpha=.15, lw=0)
            ax.plot(g.age, g[col].clip(2, 400), color=c, lw=2, label=lab)
        ax.axhspan(2, 10, color="green", alpha=.07)
        ax.set_yscale("log")
        ax.set_ylim(3, 400)
        ax.set_yticks([5, 10, 20, 50, 100, 200])
        ax.set_yticklabels(["5", "10", "20", "50", "100", "200"])
        ax.set_xlim(5, 92)
        ax.set_title(r)
        ax.grid(alpha=.3, which="both")
        ax.set_xlabel("age (years)")
    for ax in axs[:, 0]:
        ax.set_ylabel("age resolution R (years)")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Age resolution across the whole lifespan, year by year (green: R ≤ 10 years)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "p4_fig2_resolution_lifespan.png"), dpi=120)


def fig_models():
    order = ["SHASH", "GAUSS", "QR", "MLP", "GBM", "GPR"]
    names = {"SHASH": "SHASH\n(skew+tails)", "GAUSS": "Gaussian\n(current)", "QR": "quantile\nregression",
             "MLP": "MLP\n(neural net)", "GBM": "boosted\ntrees", "GPR": "Gaussian\nprocess"}
    base = M[M.model == "GAUSS"].set_index("region").CRPS_pct
    M["dCRPS"] = 100 * (M.CRPS_pct / M.region.map(base) - 1)
    ref_slope = M[M.model == "GAUSS"].set_index("region").slope70
    M["slope_ratio"] = M.slope70 / M.region.map(ref_slope)
    agg = M.groupby("model").agg(dCRPS=("dCRPS", "mean"), cal=("cal_err", "mean"), site=("worst_site_cal", "mean"),
                                 age=("worst_ageband_cal", "mean"), sl=("slope_ratio", "median"),
                                 slsd=("slope70_sd", "mean")).loc[order]
    fig, axs = plt.subplots(1, 4, figsize=(21, 5))
    x = np.arange(len(order))
    axs[0].bar(x, agg.dCRPS, color="#4c72b0")
    axs[0].set_title("Prediction score vs Gaussian (CRPS, %; lower = better)")
    axs[1].bar(x - 0.27, agg.cal, 0.27, label="overall")
    axs[1].bar(x, agg.site, 0.27, label="worst scanner")
    axs[1].bar(x + 0.27, agg.age, 0.27, label="worst age band")
    axs[1].set_title("Calibration error of centiles (0 = perfect)")
    axs[1].legend(fontsize=8)
    axs[2].bar(x, agg.sl, color="#dd8452")
    axs[2].axhline(1, color="k", lw=.8)
    axs[2].set_title("Implied aging rate at 70, relative to Gaussian")
    axs[3].bar(x, agg.slsd, color="#55a868")
    axs[3].set_yscale("log")
    axs[3].set_title("Instability of aging rate across folds (%/yr, log)")
    for ax in axs:
        ax.set_xticks(x)
        ax.set_xticklabels([names[o] for o in order], fontsize=8)
        ax.grid(alpha=.3, axis="y")
    cm = os.path.join(D, "contest_meta.json")            # written by the run that produced model_comparison.csv
    who = (lambda c: f"{c['people']:,} adults, {c['sites']} sites")(json.load(open(cm))) if os.path.exists(cm) else "1,471 people"
    fig.suptitle(f"Model contest: 8 regions, {who}, 5-fold held-out evaluation", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "p4_fig3_model_contest.png"), dpi=120)


def fig_trajectories():
    """Repeat scans of the same people (DLBS waves, OASIS-2 visits) on the healthy chart. Each scan was put on the
    reference scale (reference head, average scanner) by keeping its own centile, so people with different heads
    and scanners share one picture."""
    J = pd.read_csv(os.path.join(D, "trajectories.csv"))
    regs = [r for r in ("hippocampus", "lateral ventricles", "cortex") if r in set(J.region)]
    fig, axs = plt.subplots(len(regs), 2, figsize=(15, 4.8 * len(regs)), squeeze=False)
    for i, r in enumerate(regs):
        for k, (sex, lab, col) in enumerate(((0.0, "women", "#c2185b"), (1.0, "men", "#1565c0"))):
            ax = axs[i, k]
            g = T[(T.region == r) & (T.sex == lab) & (T.age >= 18)]
            ax.fill_between(g.age, g["c2.5"], g["c97.5"], color=col, alpha=.08, label="healthy 2.5th-97.5th centile")
            ax.fill_between(g.age, g["c25"], g["c75"], color=col, alpha=.18, label="healthy 25th-75th centile")
            ax.plot(g.age, g["c50"], color=col, lw=2, label="healthy median")
            q = J[(J.region == r) & (J.sex == sex)]
            for _, h in q.groupby("subject"):
                h = h.sort_values("age")
                ax.plot(h.age, h.volume_reference_scale, color="0.2", lw=.7, alpha=.55, marker="o", ms=2)
            ax.set_title(f"{r}, {lab}: {q.subject.nunique()} people, each scanned 2-5 times")
            ax.set_xlim(18, 98)
            ax.grid(alpha=.3)
            ax.set_xlabel("age (years)")
            if r == "lateral ventricles":
                ax.set_yscale("log")
        axs[i, 0].set_ylabel("volume (mm³), reference head, average scanner")
    axs[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("How single people change: each line = one person's repeat scans (DLBS, OASIS-2) on the healthy chart",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "p5_fig4_trajectories.png"), dpi=110)


def fig_normalization():
    """What the model corrects for: head size, scanner, measurement noise; and how differently people age."""
    S = pd.read_csv(os.path.join(D, "site_params.csv"))
    NBd = pd.read_csv(os.path.join(D, "noise_budget.csv")).set_index("region")
    IR = pd.read_csv(os.path.join(D, "individual_rates.csv"))
    regs = list(S.region.drop_duplicates())
    y = np.arange(len(regs))
    fig, axs = plt.subplots(2, 2, figsize=(19, 12))
    ax = axs[0, 0]
    ax.barh(y, S.groupby("region").head_exponent_b.first().reindex(regs), color="#4c72b0")
    ax.axvline(1, color="k", lw=1, ls="--")
    ax.set_title("Head size: volume grows as (head size)^b   (dashed b = 1: simple proportion)")
    ax = axs[0, 1]
    ax.boxplot([S[S.region == r].mean_shift_pct.values for r in regs], vert=False, positions=y, widths=.6,
               flierprops=dict(ms=3))
    ax.axvline(0, color="k", lw=1)
    ax.set_title(f"Scanner effect: how much each of {S.site.nunique()} scanners over/under-measures a region (%)")
    ax = axs[1, 0]
    sp = T[(T.age == 70) & (T.sex == "women")].set_index("region").spread_pct.reindex(regs)
    ax.barh(y - .27, sp, .27, color="#8c8c8c", label="real differences between healthy 70-year-olds (spread)")
    ax.barh(y, NBd.eps_longitudinal_pct.reindex(regs), .27, color="#dd8452", label="noise between visits years apart")
    ax.barh(y + .27, NBd.eps_same_session_pct.reindex(regs), .27, color="#55a868", label="noise, same-session repeat")
    ax.set_xscale("log")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Measurement noise vs real differences between people (% of volume, log scale)")
    for ax in (axs[0, 0], axs[0, 1], axs[1, 0]):
        ax.set_yticks(y)
        ax.set_yticklabels(regs)
        ax.set_ylim(len(regs) - .5, -.5)
        ax.grid(alpha=.3, axis="x")
    ax = axs[1, 1]
    bands = list(IR.age_band.drop_duplicates())
    for r, c, dx in (("hippocampus", "#c44e52", -.22), ("lateral ventricles", "#4c72b0", 0), ("cortex", "#55a868", .22)):
        q = IR[IR.region == r].set_index("age_band").reindex(bands)
        x = np.arange(len(bands)) + dx
        ax.errorbar(x, q.median_rate_pct_yr, yerr=q.observed_sd, fmt="none", ecolor=c, alpha=.3, lw=7)
        ax.errorbar(x, q.median_rate_pct_yr, yerr=q.true_sd, fmt="o", color=c, capsize=5, lw=2, label=r)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([f"age {b}" for b in bands])
    ax.set_ylabel("change per year (%)")
    ax.legend(fontsize=9)
    ax.grid(alpha=.3)
    ax.set_title("Aging speed of single people: dot = median; thin bar = TRUE person-to-person spread (±1 SD)\n"
                 "after removing measurement noise; wide pale bar = what is observed with noise")
    fig.tight_layout()
    fig.savefig(os.path.join(D, "p5_fig5_normalization_noise.png"), dpi=110)


if __name__ == "__main__":
    fig_charts()
    fig_resolution()
    fig_models()
    print("ok")
