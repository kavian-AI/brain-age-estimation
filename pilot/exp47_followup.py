#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FOLLOW-UP ON THE THREE THINGS E46 LEFT OPEN.

A  THE HEADLINE. E46 found that ONE principal component of the regional
   deviations recovers 95% of the six-dimensional AUC on AD vs CN (0.928 of
   0.950) - while the brain-age gap, also one number, reaches only 0.906.

   If that holds up it reframes four experiments' worth of conclusions. We have
   been saying "compression hurts". It does not, much. What hurts is that the
   brain-age gap is THE WRONG ONE NUMBER. A single well-chosen linear combination
   of regional deviations does nearly as well as all six; the particular
   combination that brain age computes does worse. That is a sharper claim and a
   more useful one, so it needs a bootstrap before it is believed.

B  THE COUPLING RESULT needs a robustness check. Regional deviations became more
   correlated in AD (mean |dr| 0.119 vs null 0.046, p<0.0001). But patients have
   larger variance, and estimated correlations can be distorted by heavy tails
   even though correlation is scale-free in theory. Spearman is insensitive to
   that, so if the effect survives a rank-based estimate it is not a variance
   artefact.

C  Q4/Q5 CRASHED on a column collision - ADNIMERGE already carries PTEDUCAT, so
   the merge produced PTEDUCAT_x and the attribute lookup failed. Fixed here.
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
DATA = os.path.join(BASE, "data")
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
FA = {"Hippocampus": "هیپوکامپ", "Entorhinal": "آنتورینال", "Fusiform": "دوکی",
      "MidTemp": "گیجگاهی میانی", "Ventricles": "بطن‌ها", "WholeBrain": "کل مغز"}
FOLDS, N_BOOT = 5, 500


def load():
    a = pd.read_csv(os.path.join(DATA, "ADNIMERGE_05Mar2026.csv"), low_memory=False)
    d = a.dropna(subset=["AGE", "Years_bl", "ICV", "PTGENDER"] + V).copy()
    d["SEX"] = (d.PTGENDER == "Male").astype(int)
    d["age_visit"] = d.AGE + d.Years_bl
    return a, d.reset_index(drop=True)


def normative_z(d, seed=0):
    cn = (d.DX == "CN").values
    uq = np.unique(d.PTID.values)
    fm = dict(zip(uq, np.random.RandomState(seed).permutation(len(uq)) % FOLDS))
    f = np.array([fm[s] for s in d.PTID.values])
    age = d.age_visit.values.astype(float)
    X = np.column_stack([age, age ** 2, d.SEX.values, d.ICV.values])
    Z = np.zeros((len(d), len(V)))
    for j, v in enumerate(V):
        y = d[v].values.astype(float)
        for k in range(FOLDS):
            tr, te = np.where((f != k) & cn)[0], np.where(f == k)[0]
            sc = StandardScaler().fit(X[tr])
            m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(X[tr]), y[tr])
            sd = (y[tr] - m.predict(sc.transform(X[tr]))).std(ddof=1)
            Z[te, j] = (y[te] - m.predict(sc.transform(X[te]))) / max(sd, 1e-9)
    return Z


def scalar_gap(d, seed=0):
    cn = (d.DX == "CN").values
    uq = np.unique(d.PTID.values)
    fm = dict(zip(uq, np.random.RandomState(seed).permutation(len(uq)) % FOLDS))
    f = np.array([fm[s] for s in d.PTID.values])
    age = d.age_visit.values.astype(float)
    X = np.column_stack([d[v].values for v in V] + [d.ICV.values, d.SEX.values])
    g = np.zeros(len(d))
    for k in range(FOLDS):
        tr, te = np.where((f != k) & cn)[0], np.where(f == k)[0]
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(X[tr]), age[tr])
        b = LinearRegression().fit(age[tr].reshape(-1, 1),
                                   m.predict(sc.transform(X[tr])) - age[tr])
        g[te] = (m.predict(sc.transform(X[te])) - age[te]) \
            - b.predict(age[te].reshape(-1, 1))
    return g


def auc_cv(X, y, subs, seed=0):
    uq = np.unique(subs)
    fm = dict(zip(uq, np.random.RandomState(seed).permutation(len(uq)) % FOLDS))
    f = np.array([fm[s] for s in subs])
    o = np.zeros(len(y))
    for k in range(FOLDS):
        tr, te = f != k, f == k
        sc = StandardScaler().fit(X[tr])
        c = LogisticRegression(max_iter=4000).fit(sc.transform(X[tr]), y[tr])
        o[te] = c.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, o)


def main():
    a, d = load()
    Z = normative_z(d)
    G = scalar_gap(d)
    bl = d.index[d.VISCODE == "bl"].values
    db = d.loc[bl]

    # ================================================================== A
    print("=" * 76)
    print("A  IS THE BRAIN-AGE GAP THE WRONG ONE NUMBER?")
    print("=" * 76)
    print("  Both arms are ONE number per subject, plus age. The only difference")
    print("  is which linear combination of the same information it is.\n")
    for task, pos in [("AD vs CN", "Dementia"), ("MCI vs CN", "MCI")]:
        sel = db.DX.isin(["CN", pos]).values
        idx = bl[sel]
        y = (db.DX[sel] == pos).astype(int).values
        subs = db.PTID.values[sel]
        ag = d.age_visit.values[idx].reshape(-1, 1)
        rng = np.random.RandomState(0)
        uq = np.unique(subs)
        diffs = []
        for _ in range(N_BOOT):
            pick = set(rng.choice(uq, len(uq), replace=True))
            m = np.array([s in pick for s in subs])
            if y[m].sum() < 20 or (1 - y[m]).sum() < 20:
                continue
            zz = Z[idx][m]
            p1 = PCA(n_components=1, random_state=0).fit_transform(
                StandardScaler().fit_transform(zz))
            try:
                a_pca = auc_cv(np.column_stack([p1, ag[m]]), y[m], subs[m])
                a_gap = auc_cv(np.column_stack([G[idx][m], ag[m]]), y[m], subs[m])
                diffs.append(a_pca - a_gap)
            except Exception:
                pass
        diffs = np.array(diffs)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        pv = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
        print(f"  {task:10s} one PCA component minus brain-age gap: "
              f"{diffs.mean():+.3f}")
        print(f"             95% CI [{lo:+.3f}, {hi:+.3f}]  p={pv:.3f}  "
              f"-> {'REAL' if pv < 0.05 else 'not distinguishable'}")

    # what IS that component? report its loadings
    sel = db.DX.isin(["CN", "Dementia"]).values
    zz = StandardScaler().fit_transform(Z[bl[sel]])
    pc = PCA(n_components=1, random_state=0).fit(zz)
    print("\n  the single component, region by region (its loadings):")
    for v, w in sorted(zip(V, pc.components_[0]), key=lambda t: -abs(t[1])):
        print(f"    {FA[v]:16s} {w:+.3f}")
    print(f"    variance explained: {100*pc.explained_variance_ratio_[0]:.0f}%")

    # ================================================================== B
    print("\n" + "=" * 76)
    print("B  IS THE COUPLING RESULT A VARIANCE ARTEFACT?  (Spearman = rank based)")
    print("=" * 76)
    gz = {g: Z[bl][db.DX.values == g] for g in ["CN", "MCI", "Dementia"]}
    for name, fn in [("Pearson", lambda A: np.corrcoef(A.T)),
                     ("Spearman", lambda A: spearmanr(A).statistic)]:
        Ccn, Cad = fn(gz["CN"]), fn(gz["Dementia"])
        obs = np.abs(Cad - Ccn)[np.triu_indices(len(V), 1)].mean()
        pool = np.vstack([gz["CN"], gz["Dementia"]])
        ncn = len(gz["CN"])
        rng = np.random.RandomState(0)
        null = []
        for _ in range(400):
            p = rng.permutation(len(pool))
            null.append(np.abs(fn(pool[p[ncn:]]) - fn(pool[p[:ncn]]))[
                np.triu_indices(len(V), 1)].mean())
        null = np.array(null)
        pv = (null >= obs).mean()
        print(f"  {name:9s} observed {obs:.3f}   null {null.mean():.3f} "
              f"[{np.percentile(null,2.5):.3f}, {np.percentile(null,97.5):.3f}]"
              f"   p={pv:.4f}")
    print("\n  variance of the deviations by group (the thing that could confound):")
    for g in ["CN", "MCI", "Dementia"]:
        print(f"    {g:10s} mean SD across regions = {gz[g].std(axis=0).mean():.2f}")

    # ================================================================== C
    print("\n" + "=" * 76)
    print("C  APOE4 AND EDUCATION (the crash in E46 was a column collision)")
    print("=" * 76)
    ed = d.PTEDUCAT.values if "PTEDUCAT" in d.columns else None
    ap = d.APOE4.values if "APOE4" in d.columns else None
    if ed is None or ap is None:
        print("  columns not present in the filtered frame; skipping")
        return
    cnm = (d.DX == "CN").values & np.isfinite(ed)
    print(f"  CONTROLS only (n={cnm.sum()}) - no disease to explain anything:")
    print(f"    education vs brain-age gap   r = {np.corrcoef(ed[cnm], G[cnm])[0,1]:+.3f}")
    for j, v in enumerate(V):
        r = np.corrcoef(ed[cnm], Z[cnm, j])[0, 1]
        print(f"    education vs {FA[v]:16s} r = {r:+.3f}")
    okA = np.isfinite(ap)
    print(f"\n  APOE4 in CONTROLS:")
    for k in [0, 1, 2]:
        m = cnm & okA & (ap == k)
        if m.sum() > 25:
            print(f"    {k} allele(s): n={m.sum():4d}  mean gap {G[m].mean():+.2f} y"
                  f"   mean hippocampal z {Z[m,0].mean():+.2f}")
    m0 = cnm & okA & (ap == 0)
    m1 = cnm & okA & (ap >= 1)
    if m0.sum() > 25 and m1.sum() > 25:
        rng = np.random.RandomState(0)
        dd = []
        for _ in range(2000):
            i0 = rng.choice(np.where(m0)[0], m0.sum())
            i1 = rng.choice(np.where(m1)[0], m1.sum())
            dd.append(G[i1].mean() - G[i0].mean())
        dd = np.array(dd)
        lo, hi = np.percentile(dd, [2.5, 97.5])
        pv = 2 * min((dd <= 0).mean(), (dd >= 0).mean())
        print(f"    carriers minus non-carriers: {dd.mean():+.2f} y  "
              f"95% CI [{lo:+.2f}, {hi:+.2f}]  p={pv:.3f}")


if __name__ == "__main__":
    main()
