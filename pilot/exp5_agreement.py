#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
When the DIRECT disease model flags someone as diseased,
what brain-age gap does the AGE model assign to that same person?

Are the two models seeing the same signal, or different ones?

ADNI, AD vs CN. Everything out-of-fold (no leakage).
  - direct disease score : logistic regression on volumes + age  (the winner)
  - brain-age gap        : age model trained on CN only -> predicted - real age
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
DATA = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\data"
OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]


def main():
    d = pd.read_csv(f"{DATA}/ADNIMERGE_05Mar2026.csv", low_memory=False)
    a = d.dropna(subset=V + ["ICV", "AGE", "DX", "PTGENDER"]).sort_values("Years_bl")
    a = a.groupby("RID").first().reset_index()
    a = a[a.DX.isin(["CN", "Dementia"])].reset_index(drop=True)

    vol = a[V].values
    age = a.AGE.values
    icv = a.ICV.values
    sex = (a.PTGENDER == "Male").astype(int).values
    y = (a.DX == "Dementia").astype(int).values      # 1 = AD, 0 = CN
    n = len(y)

    direct = np.zeros(n)      # out-of-fold direct disease probability
    gap = np.zeros(n)         # out-of-fold brain-age gap (global)

    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(vol, y):
        # direct disease model: volumes + ICV + sex + age -> logistic
        feat = np.column_stack([vol, icv, sex, age])
        sc = StandardScaler().fit(feat[tr])
        clf = LogisticRegression(max_iter=3000).fit(sc.transform(feat[tr]), y[tr])
        direct[te] = clf.predict_proba(sc.transform(feat[te]))[:, 1]

        # brain-age model: trained on CN only (healthy reference)
        h = tr[y[tr] == 0]
        Xh = np.column_stack([vol[h], icv[h], sex[h]])
        sca = StandardScaler().fit(Xh)
        m = LinearRegression().fit(sca.transform(Xh), age[h])
        bias = LinearRegression().fit(age[h].reshape(-1, 1),
                                      m.predict(sca.transform(Xh)) - age[h])
        Xte = np.column_stack([vol[te], icv[te], sex[te]])
        raw = m.predict(sca.transform(Xte)) - age[te]
        gap[te] = raw - bias.predict(age[te].reshape(-1, 1))

    # operating point: flag as diseased if direct prob > 0.5
    flagged = direct > 0.5
    a_out = a.copy()
    a_out["direct"] = direct
    a_out["gap"] = gap
    a_out["true_AD"] = y
    a_out["flagged"] = flagged.astype(int)
    a_out[["RID", "DX", "true_AD", "direct", "gap", "flagged"]].to_csv(
        f"{OUT}/exp5_agreement.csv", index=False)

    print(f"n = {n}  (AD={int(y.sum())}, CN={int((1-y).sum())})")
    print(f"direct model flags {flagged.sum()} as diseased "
          f"(of which truly AD: {y[flagged].sum()}, false alarms: {(1-y[flagged]).sum()})")
    print()
    print("brain-age gap (years), by group:")
    for name, mask in [("flagged diseased  (all)", flagged),
                       ("  - correct (true AD)", flagged & (y == 1)),
                       ("  - false alarm (true CN)", flagged & (y == 0)),
                       ("flagged healthy   (all)", ~flagged),
                       ("  - missed AD", ~flagged & (y == 1)),
                       ("  - correct CN", ~flagged & (y == 0))]:
        g = gap[mask]
        if len(g):
            print(f"  {name:28s} n={len(g):4d}  mean={g.mean():+5.2f}  median={np.median(g):+5.2f}")

    r = np.corrcoef(direct, gap)[0, 1]
    print(f"\ncorrelation(direct score, brain-age gap) = {r:.3f}")

    # how many flagged-diseased have a NORMAL gap? (detected by non-age features)
    thr = np.percentile(gap[y == 0], 90)   # upper limit of normal CN range
    normal_gap_but_flagged = flagged & (gap < thr)
    print(f"\ndiseased-flagged with a NORMAL brain-age gap (< {thr:.1f} y, the CN 90th pct):")
    print(f"  {normal_gap_but_flagged.sum()} of {flagged.sum()} flagged "
          f"({100*normal_gap_but_flagged.sum()/flagged.sum():.0f}%)")
    print("  -> these are detected by the direct model but look age-normal")

    np.savez(f"{OUT}/exp5_agreement.npz", direct=direct, gap=gap, y=y,
             flagged=flagged)


if __name__ == "__main__":
    main()
