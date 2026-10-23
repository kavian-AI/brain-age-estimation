#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
THE ALIGNMENT LAW: CAN ONE NUMBER PREDICT HOW WELL ANY BRAIN SUMMARY DETECTS DISEASE?

Sixty-six experiments left a pile of facts that look unrelated:
  - brain age's regional weights are rank-uncorrelated with a disease model's
    (r = -0.03 ADNI, -0.01 OASIS) yet it still reaches AUC 0.88
  - a brain-age model with MORE regions reads age better and disease WORSE
    (Schulz 2025, and E62b on the same people)
  - averaging only the MTL regions beats averaging all 98 by +0.083
  - the dual-reference model adds nothing for Alzheimer's (E63)

One piece of classical geometry would explain all of them. For deviations z with
within-group covariance S and disease shift mu, ANY linear summary s = w'z separates
the groups with
        d'(w)^2 = (w' mu)^2 / (w' S w)
and the best achievable is d*^2 = mu' S^-1 mu. Their ratio
        eta(w) = d'(w)^2 / d*^2 = cos^2 between w and S^-1 mu, measured in the S metric
is the ALIGNMENT of the summary with the disease. Shared variance (a large common
factor in S) is why orthogonal-looking weights can still align.

If this holds, eta is computable for ANY summary before it is ever tested, and it
PREDICTS the summary's disease AUC. Tested here on 98 regions:
  1  the named summaries (gap, uniform, MTL average, hippocampus, first PC)
  2  400 random region subsets, for both the brain-age gap and the uniform average
     - many summaries the law has never seen
  3  the Schulz paradox as a consequence: across subsets, does adding regions raise
     age accuracy while LOWERING the gap's alignment?
Predicted AUC under normality: AUC = Phi(d'/sqrt(2)). Everything is estimated on
one half of the subjects and checked on the other half, so the law has to predict,
not describe.
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
from scipy.stats import norm, pearsonr, spearmanr  # noqa: E402
from sklearn.linear_model import LinearRegression, RidgeCV  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def geometry(Zc, Zd, shrink=0.1):
    """Pooled within-group covariance (shrunk toward its diagonal) and mean shift."""
    S = 0.5 * (np.cov(Zc.T) + np.cov(Zd.T))
    S = (1 - shrink) * S + shrink * np.diag(np.diag(S))
    mu = Zd.mean(0) - Zc.mean(0)
    Si_mu = np.linalg.solve(S, mu)
    return S, mu, Si_mu, float(mu @ Si_mu)


def eta(w, S, mu, dstar2):
    return float((w @ mu) ** 2 / (w @ S @ w) / dstar2)


def pred_auc(w, S, mu):
    return float(norm.cdf(abs(w @ mu) / np.sqrt(w @ S @ w) / np.sqrt(2)))


def gap_weights(m, names, cols, cn, Z):
    """The brain-age gap is a linear function of raw volumes; express it in the
    z coordinates by regressing the out-of-fold gap on z across subjects."""
    ag = m.age_visit.values.astype(float)
    X = np.column_stack([m[names[j]].values for j in cols] + [m.ICV_T.values, m.SEX.values])
    f = E.folds_of(m.PTID.values)
    g = np.zeros(len(m))
    mae = []
    for k in range(5):
        tr, te = np.where((f != k) & cn)[0], np.where(f == k)[0]
        sc = StandardScaler().fit(X[tr])
        md = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(X[tr]), ag[tr])
        p = md.predict(sc.transform(X[te]))
        mae.append(np.abs(p - ag[te])[cn[te]])
        bb = LinearRegression().fit(ag[tr].reshape(-1, 1),
                                    md.predict(sc.transform(X[tr])) - ag[tr])
        g[te] = (p - ag[te]) - bb.predict(ag[te].reshape(-1, 1))
    w = LinearRegression().fit(Z[:, cols], g).coef_
    full = np.zeros(Z.shape[1])
    full[cols] = w
    return g, full, np.concatenate(mae).mean()


def main():
    a, m, T, names = R.build("X")
    cn = (m.DX.values == "CN")
    Z, zh = R.normative(m, names, cn)
    S_sign = T.sign.values
    sel = m.DX.isin(["CN", "Dementia"]).values
    y = (m.DX.values == "Dementia").astype(int)
    rng = np.random.RandomState(0)
    half = rng.rand(len(m)) < 0.5            # A: estimate geometry,  B: test
    A_, B_ = sel & half, sel & ~half
    S, mu, Si_mu, d2 = geometry(Z[A_ & (y == 0)], Z[A_ & (y == 1)])
    print(f"n={len(m)} (98 regions). geometry from half A (CN {int((A_&(y==0)).sum())}, "
          f"AD {int((A_&(y==1)).sum())}); tested on half B (CN {int((B_&(y==0)).sum())}, "
          f"AD {int((B_&(y==1)).sum())})")
    print(f"optimal separation d* = {np.sqrt(d2):.2f}  -> best possible AUC "
          f"{norm.cdf(np.sqrt(d2)/np.sqrt(2)):.3f}")
    print(f"share of within-group variance on the first common factor: "
          f"{100*np.linalg.eigvalsh(S)[-1]/np.trace(S):.0f}%")

    # ---------------------------------------------------------- 1 named summaries
    mtl = T.cls.values == "mtl"
    hippo = np.array([n in ("L_hippocampus", "R_hippocampus") for n in names])
    _, w_gap, _ = gap_weights(m, names, list(range(len(names))), cn, Z)
    sc = StandardScaler().fit(Z[cn])
    from sklearn.decomposition import PCA
    pc = PCA(1).fit(sc.transform(Z[cn]))
    w_pc = pc.components_[0] / sc.scale_
    w_pc *= np.sign(w_pc @ mu)
    named = {"brain-age gap (98)": w_gap, "uniform (98)": S_sign.astype(float),
             "uniform, MTL only": S_sign * mtl, "hippocampus L+R": S_sign * hippo,
             "first PC": w_pc, "optimal direction": Si_mu}
    print("\n" + "=" * 88)
    print("1  NAMED SUMMARIES: alignment eta, predicted AUC, observed AUC on held-out half")
    print("=" * 88)
    print(f"    {'summary':22s} {'eta':>7s} {'predicted AUC':>14s} {'observed AUC':>13s}")
    P, O = [], []
    for nm, w in named.items():
        s = Z[B_] @ w
        obs = roc_auc_score(y[B_], s if w @ mu > 0 else -s)
        e, pa = eta(w, S, mu, d2), pred_auc(w, S, mu)
        P.append(pa)
        O.append(obs)
        print(f"    {nm:22s} {e:7.3f} {pa:14.3f} {obs:13.3f}")

    # ---------------------------------------------------------- 2 random subsets
    print("\n" + "=" * 88)
    print("2  400 RANDOM REGION SUBSETS - summaries the law has never seen")
    print("=" * 88)
    rows = []
    rng = np.random.RandomState(1)
    for i in range(400):
        k = rng.choice([3, 5, 8, 12, 20, 35, 60, 98])
        cols = sorted(rng.choice(len(names), k, replace=False))
        mask = np.zeros(len(names), bool)
        mask[cols] = True
        wu = S_sign * mask
        ou = roc_auc_score(y[B_], Z[B_] @ wu)
        rows.append(("uniform", k, eta(wu, S, mu, d2), pred_auc(wu, S, mu), ou, np.nan))
        if i % 4 == 0:                        # the gap is costlier: refit the age model
            g, wg, mae = gap_weights(m, names, cols, cn, Z)
            og = roc_auc_score(y[B_], g[B_])
            rows.append(("gap", k, eta(wg, S, mu, d2), pred_auc(wg, S, mu), og, mae))
    D = pd.DataFrame(rows, columns=["summary", "k", "eta", "pred", "obs", "mae"])
    for s in ["uniform", "gap"]:
        q = D[D.summary == s]
        r, p = pearsonr(q.pred, q.obs)
        print(f"    {s:8s} n={len(q):3d} subsets   corr(predicted AUC, observed AUC) = {r:.3f}"
              f"   mean |error| = {np.abs(q.pred - q.obs).mean():.3f}")
    r, _ = pearsonr(D.pred, D.obs)
    print(f"    all summaries together: r = {r:.3f}, Spearman {spearmanr(D.pred, D.obs)[0]:.3f}")

    # ---------------------------------------------------------- 3 Schulz paradox
    print("\n" + "=" * 88)
    print("3  THE SCHULZ PARADOX AS A CONSEQUENCE: more regions -> age better, alignment worse?")
    print("=" * 88)
    G = D[D.summary == "gap"].groupby("k").agg(mae=("mae", "mean"), eta=("eta", "mean"),
                                               obs=("obs", "mean"), n=("obs", "size"))
    print(G.to_string(float_format=lambda x: f"{x:.3f}"))
    q = D[D.summary == "gap"]
    print(f"    across gap subsets: corr(age MAE, eta) = {pearsonr(q.mae, q.eta)[0]:+.3f}   "
          f"corr(age MAE, observed AUC) = {pearsonr(q.mae, q.obs)[0]:+.3f}")
    print("    (positive = a WORSE age predictor is BETTER aligned with disease)")
    D.to_csv(os.path.join(E.BASE, "pilot", "exp67_subsets.csv"), index=False)


if __name__ == "__main__":
    main()
