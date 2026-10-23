#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pilot study: brain age on open-access OASIS data.

Study 1 (OASIS-1, cross-sectional, n=436)
  - train a brain-age model on cognitively healthy subjects (CDR=0)
  - cross-validated accuracy + predicted-vs-chronological scatter
  - apply to impaired subjects (CDR>0), age-matched, bias-corrected
  - KEY TEST: does the brain-age gap beat normalised brain volume alone?

Study 2 (OASIS-2, longitudinal, 150 subjects / 373 sessions)
  - decompose each subject's gap into offset (reserve) and slope (rate)
  - compare converters vs stable groups

Runs in seconds on CPU. No GPU required.
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)

DATA = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\data"
OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"

FEATURES = ["nWBV", "eTIV", "Sex"]


def banner(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def boot_auc_diff(y, sa, sb, n=4000):
    """Bootstrap CI for AUC(a) - AUC(b) on paired scores."""
    y = np.asarray(y)
    idx = np.arange(len(y))
    diffs = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        diffs.append(roc_auc_score(y[b], sa[b]) - roc_auc_score(y[b], sb[b]))
    diffs = np.array(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return diffs.mean(), np.percentile(diffs, [2.5, 97.5]), min(p, 1.0)


# ---------------------------------------------------------------- STUDY 1
banner("STUDY 1 — cross-sectional brain age (OASIS-1)")

cs = pd.read_csv(f"{DATA}/oasis_cross-sectional.csv")
cs = cs.dropna(subset=["Age", "nWBV", "eTIV", "CDR"]).copy()
cs["Sex"] = (cs["M/F"] == "M").astype(int)
cs["impaired"] = (cs["CDR"] > 0).astype(int)

healthy = cs[cs.impaired == 0].copy()
impaired = cs[cs.impaired == 1].copy()
print(f"healthy (CDR=0): n={len(healthy)}, age {healthy.Age.min()}-{healthy.Age.max()}")
print(f"impaired (CDR>0): n={len(impaired)}, age {impaired.Age.min()}-{impaired.Age.max()}")

# ---- cross-validated brain-age model, trained on healthy only
X, y = healthy[FEATURES].values, healthy["Age"].values
models = {
    "Linear": LinearRegression(),
    "GradBoost": GradientBoostingRegressor(random_state=0, n_estimators=200, max_depth=2),
}

cv_pred = {}
for name, mdl in models.items():
    pred = np.zeros(len(y))
    for tr, te in KFold(10, shuffle=True, random_state=0).split(X):
        m = mdl.__class__(**mdl.get_params()).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    cv_pred[name] = pred
    mae = np.abs(pred - y).mean()
    r = np.corrcoef(pred, y)[0, 1]
    print(f"  {name:10s} MAE={mae:5.2f} yr   r={r:.3f}   R2={r**2:.3f}")

best = min(cv_pred, key=lambda k: np.abs(cv_pred[k] - y).mean())
print(f"  -> using {best}")
pred_healthy = cv_pred[best]

# ---- fit final model + bias-correction on healthy ONLY (no leakage)
final = models[best].__class__(**models[best].get_params()).fit(X, y)
# de Lange & Cole style: regress gap on chronological age in healthy, remove that trend.
# Estimated from OUT-OF-FOLD healthy predictions so it carries no in-sample optimism.
gap_healthy_raw = pred_healthy - y
bias = LinearRegression().fit(y.reshape(-1, 1), gap_healthy_raw)
print(f"  bias slope = {bias.coef_[0]:+.3f} gap-years per year of age "
      f"(regression dilution; corrected below)")


def _correct(raw, ages):
    return raw - bias.predict(np.asarray(ages).reshape(-1, 1))


def gap_of(df):
    """Out-of-sample gap for subjects the model never trained on."""
    p = final.predict(df[FEATURES].values)
    raw = p - df["Age"].values
    return p, raw, _correct(raw, df["Age"].values)


# ---- age-matched comparison (impaired are all elderly; must match)
lo = impaired.Age.min()
h_eld = healthy[healthy.Age >= lo].copy()
i_eld = impaired.copy()
print(f"\nAge-matched subset (age >= {lo}): healthy n={len(h_eld)}, impaired n={len(i_eld)}")
print(f"  mean age healthy {h_eld.Age.mean():.1f} vs impaired {i_eld.Age.mean():.1f} "
      f"(t-test p={stats.ttest_ind(h_eld.Age, i_eld.Age).pvalue:.3f})")

# CRITICAL: healthy subjects were used to train `final`, impaired were not.
# Scoring healthy in-sample would bias their gaps toward zero and inflate the
# group difference. Use their out-of-fold predictions instead.
oof = pd.Series(pred_healthy, index=healthy.index)
h_eld["pred"] = oof.loc[h_eld.index].values
h_eld["gap_raw"] = h_eld["pred"] - h_eld["Age"]
h_eld["gap"] = _correct(h_eld["gap_raw"].values, h_eld["Age"].values)
i_eld["pred"], i_eld["gap_raw"], i_eld["gap"] = gap_of(i_eld)

banner("STUDY 1 RESULT — brain age gap, impaired vs healthy")
for lab, col in (("uncorrected", "gap_raw"), ("bias-corrected", "gap")):
    a, b = i_eld[col], h_eld[col]
    t = stats.ttest_ind(a, b)
    d = (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    print(f"  {lab:15s} healthy {b.mean():+5.2f} yr | impaired {a.mean():+5.2f} yr "
          f"| diff {a.mean()-b.mean():+5.2f} | d={d:.2f} | p={t.pvalue:.2e}")

# ---------------------------------------------------------------- THE KEY TEST
banner("STUDY 1 KEY TEST — does the gap beat plain brain volume?")

comp = pd.concat([h_eld, i_eld])
ytrue = comp["impaired"].values
s_gap = comp["gap"].values          # higher gap -> more likely impaired
s_vol = -comp["nWBV"].values        # lower volume -> more likely impaired
s_age = comp["Age"].values

auc_gap = roc_auc_score(ytrue, s_gap)
auc_vol = roc_auc_score(ytrue, s_vol)
auc_age = roc_auc_score(ytrue, s_age)
print(f"  AUC  brain-age gap        = {auc_gap:.3f}")
print(f"  AUC  nWBV alone (volume)  = {auc_vol:.3f}")
print(f"  AUC  chronological age    = {auc_age:.3f}")

m, ci, p = boot_auc_diff(ytrue, s_gap, s_vol)
print(f"\n  gap - volume = {m:+.3f}  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  p={p:.3f}")
print("  ->", "gap ADDS value over volume" if p < 0.05 and m > 0
      else "gap does NOT beat simple volume")

# ---------------------------------------------------------------- STUDY 2
banner("STUDY 2 — longitudinal: reserve (offset) vs rate (slope), OASIS-2")

lg = pd.read_csv(f"{DATA}/oasis_longitudinal.csv")
lg = lg.dropna(subset=["Age", "nWBV", "eTIV"]).copy()
lg["Sex"] = (lg["M/F"] == "M").astype(int)
lg["pred"], lg["gap_raw"], lg["gap"] = gap_of(lg)
lg["years"] = lg["MR Delay"] / 365.25

rows = []
for sid, g in lg.groupby("Subject ID"):
    if len(g) < 2:
        continue
    grp = g["Group"].iloc[0] if g["Group"].nunique() == 1 else "Converted"
    sl = np.polyfit(g["years"], g["gap"], 1)[0] if g["years"].nunique() > 1 else np.nan
    rows.append(dict(sid=sid, group=grp, n=len(g),
                     offset=g.loc[g["years"].idxmin(), "gap"], slope=sl))
lon = pd.DataFrame(rows).dropna()
print(f"subjects with >=2 visits: {len(lon)}")
print(lon.groupby("group").agg(n=("sid", "size"),
                               offset=("offset", "mean"),
                               slope=("slope", "mean")).round(3).to_string())

conv = lon[lon.group == "Converted"]
stab = lon[lon.group == "Nondemented"]
if len(conv) > 2 and len(stab) > 2:
    print("\n  Converters vs stable non-demented:")
    for comp_name in ("offset", "slope"):
        t = stats.ttest_ind(conv[comp_name], stab[comp_name])
        print(f"    {comp_name:7s} conv {conv[comp_name].mean():+6.3f} | "
              f"stable {stab[comp_name].mean():+6.3f} | p={t.pvalue:.4f}")

# ---------------------------------------------------------------- FIGURES
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    ax[0].scatter(y, pred_healthy, s=14, alpha=.55, color="#2b5d8a", edgecolor="none")
    lims = [y.min() - 3, y.max() + 3]
    ax[0].plot(lims, lims, "k--", lw=1)
    mae = np.abs(pred_healthy - y).mean()
    r = np.corrcoef(pred_healthy, y)[0, 1]
    ax[0].set(xlabel="chronological age (years)", ylabel="predicted brain age (years)",
              title=f"A. Healthy subjects (OASIS-1)\nr={r:.2f}, MAE={mae:.1f} yr, n={len(y)}",
              xlim=lims, ylim=lims)

    data = [h_eld["gap"], i_eld["gap"]]
    bp = ax[1].boxplot(data, labels=["healthy\n(CDR=0)", "impaired\n(CDR>0)"],
                       patch_artist=True, widths=.55)
    for patch, c in zip(bp["boxes"], ["#cfd9e3", "#e2a6a6"]):
        patch.set_facecolor(c)
    ax[1].axhline(0, color="k", lw=.8, ls=":")
    ax[1].set(ylabel="brain age gap (years)",
              title=f"B. Age-matched gap\ndiff={i_eld['gap'].mean()-h_eld['gap'].mean():+.2f} yr")

    ax[2].bar(["brain-age\ngap", "nWBV\n(volume)", "chronological\nage"],
              [auc_gap, auc_vol, auc_age],
              color=["#2b5d8a", "#7a9cb8", "#b9c6d2"])
    ax[2].axhline(.5, color="k", lw=.8, ls=":")
    ax[2].set(ylim=(0.4, 1.0), ylabel="AUC (impaired vs healthy)",
              title="C. Does the gap beat volume?")
    for i, v in enumerate([auc_gap, auc_vol, auc_age]):
        ax[2].text(i, v + .012, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{OUT}/pilot_results.png", dpi=170)
    print(f"\nfigure saved -> {OUT}/pilot_results.png")
except Exception as e:
    print("figure skipped:", e)

print("\nDone.")
