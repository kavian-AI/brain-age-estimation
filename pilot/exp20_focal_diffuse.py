#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IS "FOCAL vs DIFFUSE" A GENERAL RULE?

Claim under test: neurodegenerative ageing is FOCAL (a couple of regions carry
almost all the signal) while neurodevelopmental ageing is DIFFUSE (no region
carries much alone, but the ensemble does).

Conditions: Alzheimer's (ADNI) | autism (ABIDE) | ADHD (ADHD-200)
All within-cohort, ADHD/autism on ComBat-harmonised data.

CRITICAL CONTROL: "diffuse" must not be a synonym for "weak". A condition with
a weak but focal signature would look diffuse by accident. So we do not compare
raw AUCs across conditions; we compare the SHAPE of the concentration curve,
normalised to each condition's own total signal:

  concentration = (AUC of best k regions - 0.5) / (AUC of all regions - 0.5)

If 2 regions recover ~100% of a condition's achievable signal -> FOCAL.
If you need most regions to get there -> DIFFUSE.

We also report a Gini coefficient over per-region effect sizes, which is
scale-free and so immune to the weak-vs-diffuse confusion.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
DATA = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\data"
OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"
V_AD = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]


def gaps_for(df, cols, agec, sexc, icvc, ctrl_mask):
    """Per-region brain-age gap; age model fit on controls only."""
    ctrl = df[ctrl_mask]
    G = np.zeros((len(df), len(cols)))
    for r, c in enumerate(cols):
        Xh = np.column_stack([ctrl[c].values, ctrl[icvc].values, ctrl[sexc].values])
        sc = StandardScaler().fit(Xh)
        m = LinearRegression().fit(sc.transform(Xh), ctrl[agec].values)
        b = LinearRegression().fit(ctrl[agec].values.reshape(-1, 1),
                                   m.predict(sc.transform(Xh)) - ctrl[agec].values)
        X = sc.transform(np.column_stack([df[c].values, df[icvc].values,
                                          df[sexc].values]))
        G[:, r] = (m.predict(X) - df[agec].values) \
            - b.predict(df[agec].values.reshape(-1, 1))
    return G


def auc_cv(X, y, n_rep=8):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    a = []
    for rep in range(n_rep):
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y):
            s = StandardScaler().fit(X[tr])
            c = LogisticRegression(max_iter=3000).fit(s.transform(X[tr]), y[tr])
            p[te] = c.predict_proba(s.transform(X[te]))[:, 1]
        a.append(roc_auc_score(y, p))
    return float(np.mean(a))


def gini(x):
    x = np.sort(np.abs(np.asarray(x, float)))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def analyse(G, y, names, label):
    # per-region solo AUC and effect size
    solo = np.array([auc_cv(G[:, r], y) for r in range(G.shape[1])])
    eff = np.abs(solo - 0.5)
    order = np.argsort(-solo)
    full = auc_cv(G, y)
    total = max(full - 0.5, 1e-6)

    print(f"\n{'='*72}\n{label}   n={len(y)}, patients={int(y.sum())}, "
          f"regions={G.shape[1]}\n{'='*72}")
    print(f"  all regions together: AUC = {full:.3f}")
    print(f"  best single region  : {names[order[0]]}  AUC = {solo[order[0]]:.3f}")
    print(f"  Gini of per-region effect sizes = {gini(eff):.3f}  "
          f"(0 = perfectly diffuse, 1 = one region only)")

    ks = [1, 2, 3, 5, 10, len(names)]
    ks = sorted(set(k for k in ks if k <= len(names)))
    print(f"\n  {'k':>4s} {'AUC(best k)':>12s} {'% of total signal':>19s}")
    curve = []
    for k in ks:
        idx = order[:k]
        a = auc_cv(G[:, idx], y)
        frac = (a - 0.5) / total
        curve.append((k, a, frac))
        print(f"  {k:4d} {a:12.3f} {100*frac:18.0f}%")
    # how many regions to reach 90% of total signal
    need = None
    for k in range(1, len(names) + 1):
        a = auc_cv(G[:, order[:k]], y)
        if (a - 0.5) / total >= 0.90:
            need = k
            break
    print(f"\n  regions needed for 90% of achievable signal: "
          f"{need if need else '>'+str(len(names))}"
          f"  ({100*(need or len(names))/len(names):.0f}% of all regions)")
    return dict(full=full, solo=solo, gini=gini(eff), need=need,
                n_regions=len(names), curve=curve)


def main():
    res = {}

    # ---------------- Alzheimer's (ADNI) ----------------
    d = pd.read_csv(f"{DATA}/ADNIMERGE_05Mar2026.csv", low_memory=False)
    a = d.dropna(subset=V_AD + ["ICV", "AGE", "DX", "PTGENDER"]).sort_values("Years_bl")
    a = a.groupby("RID").first().reset_index()
    a = a[a.DX.isin(["CN", "Dementia"])].copy()
    a["sexn"] = (a.PTGENDER == "Male").astype(int)
    y = (a.DX == "Dementia").astype(int).values
    G = gaps_for(a, V_AD, "AGE", "sexn", "ICV", (y == 0))
    res["Alzheimer's (ADNI)"] = analyse(G, y, V_AD, "ALZHEIMER'S — neurodegenerative")

    # ---------------- autism + ADHD (harmonised) ----------------
    h = pd.read_csv(f"{OUT}/exp12_harmonised.csv")
    roi = [c for c in h.columns if c.startswith(("Left-", "Right-"))]
    for label, prefix, pos in [("AUTISM — neurodevelopmental", "ABIDE", "autism"),
                               ("ADHD — neurodevelopmental", "ADHD", "adhd")]:
        sub = h[h.site.str.startswith(prefix)]
        sub = sub[sub.group.isin([pos, "control"])].reset_index(drop=True)
        yy = (sub.group == pos).astype(int).values
        GG = gaps_for(sub, roi, "age", "sex", "icv", (yy == 0))
        res[label.split(" —")[0]] = analyse(GG, yy, roi, label)

    # ---------------- verdict ----------------
    print("\n" + "=" * 72)
    print("VERDICT: focal or diffuse?")
    print("=" * 72)
    print(f"  {'condition':22s} {'best solo':>10s} {'all':>7s} {'Gini':>7s} "
          f"{'k for 90%':>11s} {'% regions':>10s}")
    for k, v in res.items():
        pct = 100 * (v["need"] or v["n_regions"]) / v["n_regions"]
        print(f"  {k:22s} {v['solo'].max():10.3f} {v['full']:7.3f} "
              f"{v['gini']:7.3f} {str(v['need'] or '>'+str(v['n_regions'])):>11s} "
              f"{pct:9.0f}%")
    print("\n  FOCAL  = high Gini, few regions needed for 90% of signal")
    print("  DIFFUSE= low Gini, most regions needed")

    np.savez(f"{OUT}/exp20_focal_diffuse.npz",
             **{f"{k}_solo": v["solo"] for k, v in res.items()},
             **{f"{k}_gini": v["gini"] for k, v in res.items()})
    print(f"\nsaved -> {OUT}/exp20_focal_diffuse.npz")


if __name__ == "__main__":
    main()
