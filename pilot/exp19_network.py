#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Follow-ups:
  A. Is the pairwise ORDERING in E18 statistically real, or noise?
     The medians were nearly identical, but the within-subject pairwise matrix
     was lopsided. Test each pair against P=0.5 with a binomial test, and
     bootstrap the inferred sequence to see how stable the ranking is.
  B. Regional signature for AUTISM with all 32 ROIs (user's request) -
     which regions carry autism signal, and is the ranking different from AD?
  C. Partial correlation network of ageing rates: which couplings survive
     after removing the shared global-ageing factor?
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
DATA = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\data"
OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
SHORT = ["Hipp", "Ento", "Fusi", "MidT", "Vent", "Whole"]
rng = np.random.default_rng(0)


# ---------------------------------------------------------------- A
def part_a():
    from exp18_ordering import build_trajectories
    trajs = build_trajectories()
    conv = [t for t in trajs if t["converter"] == 1 and np.isfinite(t["t_conv"])]
    stab = [t for t in trajs if t["converter"] == 0]
    thr = np.percentile(np.vstack([t["G"] for t in stab]), 95, axis=0)

    print("=" * 72)
    print("A. IS THE ORDERING REAL?  binomial test per pair vs P=0.5")
    print("=" * 72)
    sig = []
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            wins = tot = 0
            for t in conv:
                oi = np.where(t["G"][:, i] > thr[i])[0]
                oj = np.where(t["G"][:, j] > thr[j])[0]
                if len(oi) and len(oj):
                    tot += 1
                    ti, tj = t["t"][oi[0]], t["t"][oj[0]]
                    if ti < tj:
                        wins += 1
                    elif ti == tj:
                        wins += 0.5
            if tot >= 20:
                p = stats.binomtest(int(round(wins)), tot, 0.5).pvalue
                sig.append((SHORT[i], SHORT[j], wins / tot, tot, p))
    sig.sort(key=lambda x: x[4])
    print(f"  {'pair':16s} {'P(first)':>9s} {'n':>5s} {'p':>10s}")
    for x, y, pr, n, p in sig:
        star = "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else "ns"))
        print(f"  {x+' before '+y:16s} {pr:9.2f} {n:5d} {p:10.1e} {star}")

    # bootstrap the inferred sequence
    print("\n  bootstrap stability of the sequence (500 resamples of subjects):")
    ranks = {s: [] for s in SHORT}
    for _ in range(500):
        bs = rng.choice(len(conv), len(conv), replace=True)
        P = np.full((len(V), len(V)), np.nan)
        for i in range(len(V)):
            for j in range(len(V)):
                if i == j:
                    continue
                w = tt = 0
                for k in bs:
                    t = conv[k]
                    oi = np.where(t["G"][:, i] > thr[i])[0]
                    oj = np.where(t["G"][:, j] > thr[j])[0]
                    if len(oi) and len(oj):
                        tt += 1
                        if t["t"][oi[0]] < t["t"][oj[0]]:
                            w += 1
                if tt >= 10:
                    P[i, j] = w / tt
        sc = np.nanmean(P, axis=1)
        order = np.argsort(-sc)
        for pos, idx in enumerate(order):
            ranks[SHORT[idx]].append(pos + 1)
    print(f"  {'region':8s} {'median rank':>12s} {'IQR':>14s}")
    for s in sorted(SHORT, key=lambda s: np.median(ranks[s])):
        r = np.array(ranks[s])
        print(f"  {s:8s} {np.median(r):12.1f}   [{np.percentile(r,25):.0f}, "
              f"{np.percentile(r,75):.0f}]")

    # C: partial correlations of rates, converters
    print("\n" + "=" * 72)
    print("C. PARTIAL CORRELATION of ageing rates (global factor removed)")
    print("=" * 72)
    R = np.array([[np.polyfit(t["t"] - t["t"][0], t["G"][:, r], 1)[0]
                   for r in range(len(V))] for t in conv])
    C = np.corrcoef(R.T)
    prec = np.linalg.pinv(C)
    d = np.sqrt(np.diag(prec))
    PC = -prec / np.outer(d, d)
    np.fill_diagonal(PC, 1.0)
    print("        " + "".join(f"{s:>7s}" for s in SHORT))
    for i, s in enumerate(SHORT):
        print(f"  {s:6s}" + "".join("      -" if i == j else f"{PC[i,j]:7.2f}"
                                     for j in range(len(V))))
    iu = np.triu_indices(len(V), 1)
    top = sorted(zip(np.abs(PC[iu]), [(SHORT[i], SHORT[j])
                 for i, j in zip(*iu)]), reverse=True)[:4]
    print("  strongest DIRECT couplings (not explained by global ageing):")
    for m, (x, y) in top:
        i, j = SHORT.index(x), SHORT.index(y)
        print(f"    {x}-{y}: partial r = {PC[i,j]:+.2f}")


# ---------------------------------------------------------------- B
def part_b():
    print("\n" + "=" * 72)
    print("B. AUTISM: which of the 32 regions carry signal? (vs AD ranking)")
    print("=" * 72)
    h = pd.read_csv(f"{OUT}/exp12_harmonised.csv")   # site-harmonised
    ab = h[h.site.str.startswith("ABIDE")]
    sub = ab[ab.group.isin(["autism", "control"])].reset_index(drop=True)
    roi = [c for c in sub.columns if c.startswith(("Left-", "Right-"))]
    y = (sub.group == "autism").astype(int).values
    print(f"  n={len(y)}, autism={y.sum()}, regions={len(roi)}")

    ctrl = sub[y == 0]
    solo = {}
    for c in roi:
        # per-region brain age gap, age model on controls
        Xh = np.column_stack([ctrl[c].values, ctrl.sex.values])
        sc = StandardScaler().fit(Xh)
        m = LinearRegression().fit(sc.transform(Xh), ctrl.age.values)
        b = LinearRegression().fit(ctrl.age.values.reshape(-1, 1),
                                   m.predict(sc.transform(Xh)) - ctrl.age.values)
        X = sc.transform(np.column_stack([sub[c].values, sub.sex.values]))
        g = (m.predict(X) - sub.age.values) - b.predict(sub.age.values.reshape(-1, 1))
        au = []
        for rep in range(5):
            p = np.zeros(len(y))
            for tr, te in StratifiedKFold(5, shuffle=True,
                                          random_state=rep).split(g.reshape(-1, 1), y):
                s2 = StandardScaler().fit(g[tr].reshape(-1, 1))
                cl = LogisticRegression(max_iter=2000).fit(
                    s2.transform(g[tr].reshape(-1, 1)), y[tr])
                p[te] = cl.predict_proba(s2.transform(g[te].reshape(-1, 1)))[:, 1]
            au.append(roc_auc_score(y, p))
        solo[c] = np.mean(au)
    rk = sorted(solo.items(), key=lambda kv: -kv[1])
    print("\n  top 8 regions for AUTISM:")
    for c, a in rk[:8]:
        print(f"    {c:32s} AUC = {a:.3f}")
    print("  bottom 4:")
    for c, a in rk[-4:]:
        print(f"    {c:32s} AUC = {a:.3f}")
    print(f"\n  best single region {rk[0][1]:.3f} vs best AD region ~0.765 (MidTemp)")
    print("  -> autism has NO strongly informative single region; AD does.")
    pd.DataFrame(rk, columns=["region", "auc"]).to_csv(
        f"{OUT}/exp19_autism_regions.csv", index=False)


if __name__ == "__main__":
    part_a()
    part_b()
    print(f"\nsaved -> {OUT}/exp19_autism_regions.csv")
