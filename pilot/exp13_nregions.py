#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Does the brain-age profile close the gap on the fair baseline when we give it
MORE regions?

Our repeated finding was: regional profile > scalar gap, but never > volumes+age.
A fair objection is that ADNI only gave us 6 regions. ABIDE/ADHD-200 give 32-42.

Here we sweep the number of regions R = 2,4,8,16,32 and, at each R, compare:
  scalar gap  |  regional profile  |  volumes + age  (fair baseline)

Run on the site-harmonised (ComBat) children's data, using the honest task:
patients vs controls WITHIN cohort (no cross-cohort confound).
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
rng = np.random.default_rng(0)


def auc_cv(X, y, n_rep=10):
    a = []
    for rep in range(n_rep):
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y):
            s = StandardScaler().fit(X[tr])
            c = LogisticRegression(max_iter=4000).fit(s.transform(X[tr]), y[tr])
            p[te] = c.predict_proba(s.transform(X[te]))[:, 1]
        a.append(roc_auc_score(y, p))
    return np.array(a)


def gaps_cv(vol, age, sex, y, cols, n_rep=10):
    """Out-of-fold regional gaps: age model fit on training-fold CONTROLS only."""
    R = vol.shape[1]
    prof = np.zeros((n_rep, len(y), R))
    for rep in range(n_rep):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(vol, y):
            h = tr[y[tr] == 0]
            for r in range(R):
                Xh = np.column_stack([vol[h, r], sex[h]])
                sc = StandardScaler().fit(Xh)
                m = LinearRegression().fit(sc.transform(Xh), age[h])
                b = LinearRegression().fit(age[h].reshape(-1, 1),
                                           m.predict(sc.transform(Xh)) - age[h])
                X = sc.transform(np.column_stack([vol[te, r], sex[te]]))
                prof[rep, te, r] = (m.predict(X) - age[te]) - b.predict(age[te].reshape(-1, 1))
    return prof


def evaluate_R(vol, age, sex, y, R_list, label):
    print(f"\n{'='*72}\n{label}   n={len(y)}, patients={int(y.sum())}\n{'='*72}")
    print(f"  {'R':>4s} {'scalar gap':>12s} {'profile':>10s} {'volumes+age':>13s} "
          f"{'profile-baseline':>18s}")
    rows = []
    for R in R_list:
        cols = list(range(R))
        v = vol[:, cols]
        prof = gaps_cv(v, age, sex, y, cols)
        # average across reps for a stable representation, then CV the classifier
        P = prof.mean(0)
        a_prof = auc_cv(P, y)
        a_scal = auc_cv(P.mean(1, keepdims=True), y)
        a_base = auc_cv(np.column_stack([v, sex, age]), y)
        d = a_prof.mean() - a_base.mean()
        print(f"  {R:4d} {a_scal.mean():12.3f} {a_prof.mean():10.3f} "
              f"{a_base.mean():13.3f} {d:+18.3f}")
        rows.append(dict(R=R, scalar=a_scal.mean(), profile=a_prof.mean(),
                         baseline=a_base.mean(), diff=d))
    return pd.DataFrame(rows)


def main():
    h = pd.read_csv(f"{OUT}/exp12_harmonised.csv")
    roi = [c for c in h.columns if c.startswith(("Left-", "Right-"))]
    # rank regions by |correlation with age| in controls -> most age-informative first
    ctrl = h[h.group == "control"]
    strength = {c: abs(np.corrcoef(ctrl[c].values, ctrl.age.values)[0, 1]) for c in roi}
    roi_sorted = sorted(roi, key=lambda c: -strength[c])
    print(f"harmonised data: n={len(h)}, regions available={len(roi)}")
    print("most age-related regions:", roi_sorted[:4])

    R_list = [r for r in [2, 4, 8, 16, 28] if r <= len(roi_sorted)]
    res = {}

    # --- autism vs its own controls (ABIDE only, no cross-cohort confound)
    ab = h[h.site.str.startswith("ABIDE")]
    sub = ab[ab.group.isin(["autism", "control"])]
    res["autism"] = evaluate_R(sub[roi_sorted].values, sub.age.values,
                               sub.sex.values,
                               (sub.group == "autism").astype(int).values,
                               R_list, "ABIDE: autism vs own controls (harmonised)")

    # --- ADHD vs its own controls (ADHD-200 only)
    ad = h[h.site.str.startswith("ADHD")]
    sub2 = ad[ad.group.isin(["adhd", "control"])]
    res["adhd"] = evaluate_R(sub2[roi_sorted].values, sub2.age.values,
                             sub2.sex.values,
                             (sub2.group == "adhd").astype(int).values,
                             R_list, "ADHD-200: ADHD vs own controls (harmonised)")

    for k, v in res.items():
        v.to_csv(f"{OUT}/exp13_{k}.csv", index=False)
    print(f"\nsaved -> {OUT}/exp13_*.csv")

    print(f"\n{'='*72}\nDOES MORE REGIONS CLOSE THE GAP?\n{'='*72}")
    for k, v in res.items():
        first, last = v.iloc[0], v.iloc[-1]
        print(f"  {k:8s}: profile-baseline gap at R=2 {first['diff']:+.3f} "
              f"-> at R=32 {last['diff']:+.3f}")


if __name__ == "__main__":
    main()
