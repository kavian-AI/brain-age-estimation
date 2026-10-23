#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
THREE-WAY: CN -> MCI -> Alzheimer's, within ADNI.

This is the only clean differential-diagnosis test we have: one cohort, one
acquisition protocol, so there is no site confound to explain the result away.

Groups are matched on BOTH age and sex (CN is 55% female, MCI 59% male), so
neither demographic can act as a shortcut.

Questions
  1. 3-way discrimination: profile vs volumes+age vs scalar gap.
  2. The hard adjacent pairs: CN-vs-MCI (early detection) and MCI-vs-AD
     (staging). Adjacent stages are where a marker earns its keep.
  3. Does the profile SHAPE rotate along the disease continuum, or does it
     just scale up? If shape is constant and only magnitude grows, the
     "profile" adds nothing over one severity number.
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
DATA = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\data"
OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
rng = np.random.default_rng(0)


def match_age_sex(groups, tol=2.0):
    """Greedy triplet matching on age within sex. groups = list of DataFrames."""
    pools = [g.copy() for g in groups]
    used = [set() for _ in groups]
    keep = [[] for _ in groups]
    anchor = pools[int(np.argmin([len(p) for p in pools]))]
    for i, row in anchor.iterrows():
        picks = []
        ok = True
        for gi, p in enumerate(pools):
            if p is anchor:
                picks.append(i)
                continue
            cand = p[(~p.index.isin(used[gi])) & (p.PTGENDER == row.PTGENDER)
                     & ((p.AGE - row.AGE).abs() <= tol)]
            if len(cand) == 0:
                ok = False
                break
            j = (cand.AGE - row.AGE).abs().idxmin()
            picks.append(j)
        if not ok:
            continue
        for gi, j in enumerate(picks):
            used[gi].add(j)
            keep[gi].append(j)
    return [g.loc[k] for g, k in zip(groups, keep)]


def regional_gaps(ref, tgt):
    """Per-region brain age; healthy reference = ref (CN)."""
    G = np.zeros((len(tgt), len(V)))
    for r, c in enumerate(V):
        Xh = np.column_stack([ref[c].values, ref.ICV.values,
                              (ref.PTGENDER == "Male").astype(int).values])
        sc = StandardScaler().fit(Xh)
        m = LinearRegression().fit(sc.transform(Xh), ref.AGE.values)
        b = LinearRegression().fit(ref.AGE.values.reshape(-1, 1),
                                   m.predict(sc.transform(Xh)) - ref.AGE.values)
        X = sc.transform(np.column_stack([tgt[c].values, tgt.ICV.values,
                                          (tgt.PTGENDER == "Male").astype(int).values]))
        G[:, r] = (m.predict(X) - tgt.AGE.values) - b.predict(tgt.AGE.values.reshape(-1, 1))
    return G


def cv_scores(X, y, multi=False, n_rep=20):
    out = []
    for rep in range(n_rep):
        if multi:
            pred = np.zeros(len(y), dtype=int)
        else:
            pred = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y):
            s = StandardScaler().fit(X[tr])
            c = LogisticRegression(max_iter=5000, C=1.0)
            c.fit(s.transform(X[tr]), y[tr])
            if multi:
                pred[te] = c.predict(s.transform(X[te]))
            else:
                pred[te] = c.predict_proba(s.transform(X[te]))[:, 1]
        out.append(balanced_accuracy_score(y, pred) if multi
                   else roc_auc_score(y, pred))
    return np.array(out)


def main():
    d = pd.read_csv(f"{DATA}/ADNIMERGE_05Mar2026.csv", low_memory=False)
    a = d.dropna(subset=V + ["ICV", "AGE", "DX", "PTGENDER"]).sort_values("Years_bl")
    a = a.groupby("RID").first().reset_index()

    cn_all = a[a.DX == "CN"]
    mci_all = a[a.DX == "MCI"]
    ad_all = a[a.DX == "Dementia"]
    cn, mci, ad = match_age_sex([cn_all, mci_all, ad_all])
    print(f"matched triplets: CN={len(cn)}, MCI={len(mci)}, AD={len(ad)}")
    for nm, g in [("CN", cn), ("MCI", mci), ("AD", ad)]:
        print(f"  {nm}: age {g.AGE.mean():.1f}±{g.AGE.std():.1f}, "
              f"male {100*(g.PTGENDER=='Male').mean():.0f}%")
    p_age = stats.f_oneway(cn.AGE, mci.AGE, ad.AGE).pvalue
    print(f"  age ANOVA across matched groups: p={p_age:.3f} (want >0.05)")

    # reference for the age model: CN subjects NOT used in the matched set
    ref = cn_all[~cn_all.index.isin(cn.index)]
    if len(ref) < 100:
        ref = cn_all
        print("  (small CN pool: reusing all CN as reference)")
    print(f"  age-model reference (held-out CN): n={len(ref)}\n")

    allg = pd.concat([cn, mci, ad])
    G = regional_gaps(ref, allg)
    Gs = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
    Vol = np.column_stack([allg[V].values / allg.ICV.values[:, None],
                           (allg.PTGENDER == "Male").astype(int).values,
                           allg.AGE.values])
    y3 = np.array([0] * len(cn) + [1] * len(mci) + [2] * len(ad))

    reps = {"scalar gap": G.mean(1, keepdims=True),
            "regional PROFILE": G,
            "profile SHAPE": Gs,
            "volumes + age": Vol}

    print("=" * 70)
    print("1. THREE-WAY  CN vs MCI vs AD   (balanced accuracy, chance = 0.333)")
    print("=" * 70)
    for nm, X in reps.items():
        s = cv_scores(X, y3, multi=True)
        print(f"  {nm:22s} {s.mean():.3f}  [{np.percentile(s,2.5):.3f}, "
              f"{np.percentile(s,97.5):.3f}]")

    print("\n" + "=" * 70)
    print("2. ADJACENT PAIRS (the hard, clinically real comparisons)")
    print("=" * 70)
    pairs = [("CN vs MCI  (early detection)", 0, 1),
             ("MCI vs AD  (staging)", 1, 2),
             ("CN vs AD   (easy reference)", 0, 2)]
    for title, g1, g2 in pairs:
        m = np.isin(y3, [g1, g2])
        yy = (y3[m] == g2).astype(int)
        print(f"\n  {title}   n={m.sum()}")
        for nm, X in reps.items():
            s = cv_scores(X[m], yy)
            print(f"    {nm:22s} AUC = {s.mean():.3f}  "
                  f"[{np.percentile(s,2.5):.3f}, {np.percentile(s,97.5):.3f}]")

    print("\n" + "=" * 70)
    print("3. DOES THE PROFILE SHAPE ROTATE, OR JUST SCALE?")
    print("=" * 70)
    means = {}
    for nm, idx in [("CN", y3 == 0), ("MCI", y3 == 1), ("AD", y3 == 2)]:
        mg = G[idx].mean(0)
        means[nm] = mg
        print(f"  {nm:4s} magnitude={np.linalg.norm(mg):5.2f}  " +
              "  ".join(f"{v[:5]}={x:+.1f}" for v, x in zip(V, mg)))
    for x, yq in [("CN", "MCI"), ("MCI", "AD"), ("CN", "AD")]:
        u, w = means[x], means[yq]
        cos = u @ w / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-9)
        ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        print(f"  angle {x:3s}->{yq:3s}: {ang:5.1f} deg  (0 = pure scaling, "
              f"larger = pattern rotates)")

    np.savez(f"{OUT}/exp14_threeway.npz", G=G, y3=y3,
             **{f"mean_{k}": v for k, v in means.items()})
    print(f"\nsaved -> {OUT}/exp14_threeway.npz")


if __name__ == "__main__":
    main()
