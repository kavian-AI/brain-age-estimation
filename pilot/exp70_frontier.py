#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TWO USES OF THE ALIGNMENT LAW THAT BUILD SOMETHING.

PART A - PREDICT THE WHOLE SCHULZ CURVE IN CLOSED FORM.
Schulz et al. (PLOS Biol 2025) found over-regularised brain-age models detect
disease better. If the alignment law is right, the entire accuracy-versus-
sensitivity curve across ridge strength should be PREDICTABLE: for each penalty,
express the model's gap in normative-deviation coordinates, compute eta from
geometry estimated on half A, and compare with the AUC observed on half B.

PART B - A DISEASE-AWARE BRAIN AGE THAT STILL SPEAKS IN YEARS.
The field cannot give up "years": clinicians and papers read brain age as years.
So keep the target (chronological age, controls only) and change only the prior:
a relevance-weighted ridge, penalty sum_j w_j^2 / pi_j^gamma, where pi_j is how
strongly region j separates Alzheimer's from controls on HALF A. gamma = 0 is the
standard model; larger gamma lets the model reach for disease-relevant regions
when it predicts age. This traces a Pareto frontier: how much disease alignment
can be bought for how much age accuracy? Scored on half B (AD vs CN) and on MCI
conversion (never used anywhere in fitting).

PART C - A CLASSIFIER NO PATIENT EVER TRAINED.
w* = S^-1 mu from half-A summary statistics only (a mean-difference map and a
covariance), no individual fitting - against a logistic regression trained on
half A's individuals. Both scored on half B and on MCI conversion.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp53_units_and_dimension as E  # noqa: E402
import exp62_rich_atlas as R  # noqa: E402
from scipy.stats import norm, pearsonr  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def gap_model(m, names, cn, alpha, scale=None):
    """Out-of-fold ridge brain age at a FIXED penalty, optional per-feature scale."""
    f = E.folds_of(m.PTID.values)
    ag = m.age_visit.values.astype(float)
    X = np.column_stack([m[n].values for n in names] + [m.ICV_T.values, m.SEX.values])
    g = np.zeros(len(m))
    err = []
    for k in range(5):
        tr, te = np.where((f != k) & cn)[0], np.where(f == k)[0]
        sc = StandardScaler().fit(X[tr])
        Xt, Xe = sc.transform(X[tr]), sc.transform(X[te])
        if scale is not None:
            Xt, Xe = Xt * scale, Xe * scale
        md = Ridge(alpha=alpha).fit(Xt, ag[tr])
        p = md.predict(Xe)
        err.append(np.abs(p - ag[te])[cn[te]])
        bb = LinearRegression().fit(ag[tr].reshape(-1, 1), md.predict(Xt) - ag[tr])
        g[te] = (p - ag[te]) - bb.predict(ag[te].reshape(-1, 1))
    return g, np.concatenate(err).mean()


def main():
    a, m, T, names = R.build("X")
    cn = m.DX.values == "CN"
    Z, zh = R.normative(m, names, cn)
    y_all = (m.DX.values == "Dementia").astype(int)
    sel = m.DX.isin(["CN", "Dementia"]).values
    half = np.random.RandomState(0).rand(len(m)) < 0.5
    A_, B_ = sel & half, sel & ~half
    ZcA, ZdA = Z[A_ & (y_all == 0)], Z[A_ & (y_all == 1)]
    S = LedoitWolf().fit(np.vstack([ZcA - ZcA.mean(0), ZdA - ZdA.mean(0)])).covariance_
    mu = ZdA.mean(0) - ZcA.mean(0)
    d2 = mu @ np.linalg.solve(S, mu)

    def law(w):
        dp = abs(w @ mu) / np.sqrt(w @ S @ w)
        return dp ** 2 / d2, norm.cdf(dp / np.sqrt(2))

    def weights_of(g):        # the gap expressed in z coordinates, fitted on half A
        q = A_ | (half & (m.DX.values == "MCI"))
        return LinearRegression().fit(Z[q], g[q]).coef_

    yB = y_all[B_]
    pool = m[m.DX == "MCI"]
    keep, dur, ev = E.survival(a, pool.PTID.values, ["Dementia"])
    idx = pool.index[pool.PTID.isin(set(keep))].values
    o = {x: i for i, x in enumerate(keep)}
    ii = np.array([o[x] for x in m.PTID.values[idx]])
    dur, ev = dur[ii], ev[ii]
    agC = m.age_visit.values[idx]

    def conv_c(x):
        return np.nanmean([R.cidx(np.column_stack([x[idx], agC]), dur, ev, s) for s in range(3)])

    # ------------------------------------------------------------------ A
    print("=" * 96)
    print("A  THE SCHULZ CURVE, PREDICTED IN CLOSED FORM (geometry: half A; observed: half B)")
    print("=" * 96)
    print(f"  {'ridge alpha':>12s} {'age MAE':>8s} {'eta':>7s} {'predicted AUC':>14s} {'observed AUC':>13s} {'conversion C':>13s}")
    rows = []
    for al in np.logspace(-1, 6, 15):
        g, mae = gap_model(m, names, cn, al)
        w = weights_of(g)
        e, pa = law(w)
        ob = roc_auc_score(yB, g[B_])
        cc = conv_c(g)
        rows.append((al, mae, e, pa, ob, cc))
        print(f"  {al:12.1f} {mae:8.2f} {e:7.3f} {pa:14.3f} {ob:13.3f} {cc:13.3f}")
    A = pd.DataFrame(rows, columns=["alpha", "mae", "eta", "pred", "obs", "conv"])
    print(f"\n  across the curve: corr(predicted, observed AUC) = {pearsonr(A.pred, A.obs)[0]:.3f},"
          f" mean |error| {np.abs(A.pred - A.obs).mean():.3f}")
    print(f"  corr(age MAE, observed AUC) = {pearsonr(A.mae, A.obs)[0]:+.3f}"
          f"   (positive = Schulz: worse age reading, better disease detection)")
    best = A.loc[A.obs.idxmax()]
    std = A.iloc[(A.mae).idxmin()]
    print(f"  most age-accurate model: alpha {std.alpha:.1f}, MAE {std.mae:.2f}, AUC {std.obs:.3f}")
    print(f"  best disease detector:   alpha {best.alpha:.1f}, MAE {best.mae:.2f}, AUC {best.obs:.3f}")

    # ------------------------------------------------------------------ B
    print("\n" + "=" * 96)
    print("B  DISEASE-AWARE BRAIN AGE: relevance-weighted ridge, still predicting AGE in YEARS")
    print("=" * 96)
    sdA = np.sqrt(0.5 * (ZcA.var(0, ddof=1) + ZdA.var(0, ddof=1)))
    pi = np.abs(mu / sdA)
    pi = np.r_[pi / pi.mean(), 1.0, 1.0]            # ICV and sex keep average relevance
    alpha0 = 10.0
    print(f"  {'gamma':>6s} {'age MAE':>8s} {'eta':>7s} {'pred AUC':>9s} {'AD/CN AUC (B)':>14s} {'conversion C':>13s}")
    Bres = []
    for gm in [0, 0.5, 1, 1.5, 2, 3, 4, 6]:
        g, mae = gap_model(m, names, cn, alpha0, scale=pi ** (gm / 2))
        e, pa = law(weights_of(g))
        ob, cc = roc_auc_score(yB, g[B_]), conv_c(g)
        Bres.append((gm, mae, e, ob, cc, g))
        print(f"  {gm:6.1f} {mae:8.2f} {e:7.3f} {pa:9.3f} {ob:14.3f} {cc:13.3f}")
    base = Bres[0]
    for gm, mae, e, ob, cc, g in Bres[1:]:
        if mae - base[1] <= 0.5:
            best_b = (gm, mae, e, ob, cc, g)
    print(f"\n  best model within +0.5 y of the standard MAE: gamma {best_b[0]}, MAE {best_b[1]:.2f} "
          f"(+{best_b[1]-base[1]:.2f}), AD/CN {best_b[3]:.3f} ({best_b[3]-base[3]:+.3f}), "
          f"conversion {best_b[4]:.3f} ({best_b[4]-base[4]:+.3f})")
    rng = np.random.RandomState(0)
    dd = []
    for _ in range(300):
        i = rng.randint(0, len(dur), len(dur))
        if ev[i].sum() < 20:
            continue
        p1 = R.cidx(np.column_stack([best_b[5][idx][i], agC[i]]), dur[i], ev[i], 0)
        p2 = R.cidx(np.column_stack([base[5][idx][i], agC[i]]), dur[i], ev[i], 0)
        if np.isfinite(p1) and np.isfinite(p2):
            dd.append(p1 - p2)
    R.show("conversion: disease-aware minus standard", dd)
    ib = np.where(B_)[0]
    dd = []
    for _ in range(1000):
        j = rng.choice(ib, len(ib))
        if y_all[j].sum() < 10:
            continue
        dd.append(roc_auc_score(y_all[j], best_b[5][j]) - roc_auc_score(y_all[j], base[5][j]))
    R.show("AD vs CN (half B): disease-aware minus standard", dd)

    # ------------------------------------------------------------------ C
    print("\n" + "=" * 96)
    print("C  A CLASSIFIER NO PATIENT TRAINED: w* = S^-1 mu from half-A summary statistics")
    print("=" * 96)
    wstar = np.linalg.solve(S, mu)
    s_star = Z @ wstar
    lg = LogisticRegression(C=0.05, max_iter=5000).fit(StandardScaler().fit(Z[A_]).transform(Z[A_]), y_all[A_])
    s_lg = lg.decision_function(StandardScaler().fit(Z[A_]).transform(Z))
    for nm, x in [("summary-statistics w*", s_star), ("logistic on half-A individuals", s_lg),
                  ("MTL uniform (no fitting)", (Z[:, T.cls.values == 'mtl'] * T.sign.values[T.cls.values == 'mtl']).mean(1)),
                  ("hippocampus", -zh), ("standard brain age", base[5])]:
        print(f"    {nm:32s} AD/CN (B) {roc_auc_score(yB, x[B_]):.3f}   conversion C {conv_c(x):.3f}")


if __name__ == "__main__":
    main()
