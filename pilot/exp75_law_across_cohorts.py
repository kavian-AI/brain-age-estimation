#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DOES THE ALIGNMENT LAW TRANSFER ACROSS COHORT AND SEGMENTATION SOFTWARE?

Everything in E67-E73 is ADNI with ADNI's FreeSurfer tables. The practical promise
of the law is that geometry estimated in ONE place predicts utility in ANOTHER.
Here the geometry (healthy covariance S, disease shift mu) comes from ADNI
FreeSurfer 4.3/5.1, and the predictions are scored in OASIS-2, a different study,
scanner and population, segmented by a different program (FastSurfer).

Common feature space: 87 regions present in both atlases (DK in ADNI, DKT in
FastSurfer: 30 cortical per hemisphere, 20 subcortical, 4 ventricular, brainstem,
3rd and 4th ventricles). Normative deviations are computed within each cohort
from its own controls.

  1  random summaries (region subsets, uniform signed averages): ADNI-predicted
     AUC vs OASIS-observed AUC for CDR>0 vs CDR 0. OASIS's cases are milder than
     ADNI's dementia group, so the absolute level is expected to be lower; the
     test is whether the RANKING of summaries transfers (Spearman), and the
     calibration after a single scale factor on d'.
  2  the decomposition transferred: ADNI-defined Alzheimer direction and beta,
     applied to OASIS deviations -> Alzheimer years vs other years against CDR
     and against later MMSE decline.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp62_rich_atlas as R  # noqa: E402
import exp66_oasis_external as O  # noqa: E402
from scipy.stats import norm, pearsonr, spearmanr  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402
from sklearn.linear_model import LinearRegression, Ridge  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def adni_name_to_oasis(n):
    return {"L_brainstem": "brainstem", "brainstem": "brainstem"}.get(n, n)


def main():
    a, m, T, names = R.build("X")
    Ro, signs_o, cls_o = O.load_fs()
    common = [n for n in names if n in Ro.columns]
    print(f"common regions: {len(common)}")
    idx_a = [names.index(n) for n in common]
    sign = T.sign.values[idx_a]

    # ------------------------------------------------ ADNI geometry
    cn = m.DX.values == "CN"
    Za, _ = R.normative(m, names, cn)
    Za = Za[:, idx_a]
    ad = m.DX.values == "Dementia"
    S = LedoitWolf().fit(np.vstack([Za[cn] - Za[cn].mean(0), Za[ad] - Za[ad].mean(0)])).covariance_
    mu = Za[ad].mean(0) - Za[cn].mean(0)

    # ------------------------------------------------ OASIS deviations
    clin = pd.read_csv(os.path.join(O.BASE, "data", "oasis_longitudinal.csv")).rename(
        columns={"MRI ID": "session", "Subject ID": "subject"})
    M = Ro[Ro.mpr == "mpr-1"].merge(clin, on="session")
    M["SEX"] = (M["M/F"] == "M").astype(int)
    M = M.sort_values(["subject", "Visit"]).reset_index(drop=True)
    B = M.drop_duplicates("subject").reset_index(drop=True)
    ever = M.groupby("subject").CDR.max()
    ref = ((B.CDR == 0) & (B.subject.map(ever) == 0)).values
    y = (B.CDR > 0).astype(int).values
    Zo = O.normative(B[common].values.astype(float), B.Age.values.astype(float), B.SEX.values,
                     B.eTIV.values.astype(float), ref, B.subject.values.astype(str))
    print(f"OASIS baseline n={len(B)} (CDR>0 {y.sum()}, stable reference {ref.sum()})")

    # ------------------------------------------------ 1 random summaries
    rng = np.random.RandomState(1)
    pred, obs = [], []
    for _ in range(600):
        k = rng.choice([3, 5, 8, 12, 20, 35, 60, len(common)])
        cols = rng.choice(len(common), k, replace=False)
        w = np.zeros(len(common))
        w[cols] = sign[cols]
        dprime = abs(w @ mu) / np.sqrt(w @ S @ w)
        pred.append(dprime)
        obs.append(roc_auc_score(y, Zo @ w))
    pred, obs = np.array(pred), np.array(obs)
    auc_pred = norm.cdf(pred / np.sqrt(2))
    # one scale factor for milder cases: fit d'_OASIS = c * d'_ADNI on the observed AUCs
    d_obs = np.sqrt(2) * norm.ppf(np.clip(obs, 0.501, 0.999))
    c = (pred @ d_obs) / (pred @ pred)
    print("\n" + "=" * 90)
    print("1  600 RANDOM SUMMARIES: ADNI geometry -> OASIS FastSurfer AUC")
    print("=" * 90)
    print(f"  Spearman(predicted, observed) = {spearmanr(auc_pred, obs)[0]:.3f}   "
          f"Pearson = {pearsonr(auc_pred, obs)[0]:.3f}")
    print(f"  raw mean |error| {np.abs(auc_pred-obs).mean():.3f} (ADNI dementia is more severe than OASIS CDR>0)")
    print(f"  after ONE severity scale factor c={c:.2f} on d': mean |error| "
          f"{np.abs(norm.cdf(c*pred/np.sqrt(2))-obs).mean():.3f}")

    # ------------------------------------------------ 2 decomposition transferred
    print("\n" + "=" * 90)
    print("2  ADNI-DEFINED ALZHEIMER YEARS, APPLIED TO OASIS")
    print("=" * 90)
    v = np.linalg.solve(S, mu)
    # ADNI brain-age model restricted to the common regions, weights in z coords
    ag = m.age_visit.values.astype(float)
    Xa = np.column_stack([m[common].values, m.ICV_T.values, m.SEX.values])
    sc = StandardScaler().fit(Xa[cn])
    md = Ridge(alpha=316.0).fit(sc.transform(Xa[cn]), ag[cn])
    g_adni = md.predict(sc.transform(Xa)) - ag
    g_adni = g_adni - LinearRegression().fit(ag[cn].reshape(-1, 1), g_adni[cn]).predict(ag.reshape(-1, 1))
    w_age = LinearRegression().fit(Za, g_adni).coef_
    s_a = Za[cn] @ v
    beta = np.cov(Za[cn] @ w_age, s_a)[0, 1] / np.var(s_a, ddof=1)
    g_o = Zo @ w_age
    g_o = g_o - g_o[ref].mean()
    gd_o = beta * (Zo @ v - (Zo[ref] @ v).mean())
    go_o = g_o - gd_o
    for nm, x in [("brain-age gap (ADNI model)", g_o), ("Alzheimer years", gd_o), ("other years", go_o)]:
        print(f"    {nm:28s} CDR>0 vs 0 AUC {roc_auc_score(y, x):.3f}   "
              f"mean: CDR0 {x[y==0].mean():+.2f}  CDR>0 {x[y==1].mean():+.2f}")
    sl = {}
    for s, g in M.groupby("subject"):
        g = g.dropna(subset=["MMSE"])
        if len(g) >= 2 and g["MR Delay"].max() > 365:
            sl[s] = np.polyfit(g["MR Delay"].values / 365.25, g.MMSE.values, 1)[0]
    ms = B.subject.map(sl).values
    q = np.isfinite(ms)
    Cc = np.column_stack([np.ones(q.sum()), B.Age.values[q], B.SEX.values[q]])
    for nm, x in [("brain-age gap", g_o), ("Alzheimer years", gd_o), ("other years", go_o)]:
        rx = x[q] - Cc @ np.linalg.lstsq(Cc, x[q], rcond=None)[0]
        ry = -ms[q] - Cc @ np.linalg.lstsq(Cc, -ms[q], rcond=None)[0]
        r = np.corrcoef(rx, ry)[0, 1]
        rng = np.random.RandomState(0)
        bs = [np.corrcoef(rx[i], ry[i])[0, 1] for i in [rng.randint(0, len(rx), len(rx)) for _ in range(1000)]]
        print(f"    {nm:28s} partial r with MMSE decline {r:+.3f} [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]  (n={q.sum()})")


if __name__ == "__main__":
    main()
