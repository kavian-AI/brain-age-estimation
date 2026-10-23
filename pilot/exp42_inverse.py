#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RUNNING THE MODEL BACKWARDS: age -> expected brain, instead of brain -> age.

The classical pipeline predicts age from the brain and calls the residual a
brain-age gap. The reverse asks what a HEALTHY brain of this age should look
like, and treats the difference from the actual brain as the signal. The two are
not the same thing, and the reverse has one structural advantage:

  Forward:  age is the dependent variable, so predictions shrink toward the mean
            age and the gap acquires a spurious correlation with age. Every
            brain-age paper then applies a regression-dilution correction, which
            is fitted on controls and carried over to patients.

  Reverse:  the brain is the dependent variable. Shrinkage happens in volume
            space, where it is harmless, and NO age-dependent correction is
            needed at all. One fewer fitted step between the data and the claim.

The reverse direction is normative modelling (Marquand and colleagues;
BrainChart, Nature 2022) and is not new. What has not been done, and is done
here, is a head-to-head against the brain-age gap on IDENTICAL folds and
identical features, scored on both disease separation AND measurement noise.

ARMS (all fitted on CN only, out of fold by subject)
  gap scalar      all volumes -> age        -> bias-corrected gap
  gap regional    each region -> age        -> six gaps
  norm z          age,sex,ICV -> each volume -> six z-scores
  norm distance   the same six z as one Mahalanobis distance
  volumes + age   the fair baseline that has beaten everything so far
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "pilot")
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
FA = {"Hippocampus": "هیپوکامپ", "Entorhinal": "آنتورینال", "Fusiform": "دوکی",
      "MidTemp": "گیجگاهی میانی", "Ventricles": "بطن‌ها", "WholeBrain": "کل مغز"}
FOLDS, SEEDS, N_BOOT, L = 5, 6, 400, 2.0


def load():
    a = pd.read_csv(os.path.join(DATA, "ADNIMERGE_05Mar2026.csv"), low_memory=False)
    d = a.dropna(subset=["AGE", "Years_bl", "ICV", "PTGENDER"] + V).copy()
    d["SEX"] = (d.PTGENDER == "Male").astype(int)
    d["age_visit"] = d.AGE + d.Years_bl
    return a, d.reset_index(drop=True)


def folds_of(subs, seed):
    uq = np.unique(subs)
    m = dict(zip(uq, np.random.RandomState(seed).permutation(len(uq)) % FOLDS))
    return np.array([m[s] for s in subs])


def build_features(d, seed=0):
    """Both directions, out of fold, fitted on CN only."""
    age = d.age_visit.values.astype(float)
    cn = (d.DX == "CN").values
    f = folds_of(d.PTID.values, seed)
    n = len(d)

    gap_all = np.zeros(n)
    gap_reg = np.zeros((n, len(V)))
    z_norm = np.zeros((n, len(V)))

    Xall = np.column_stack([d[v].values for v in V]
                           + [d.ICV.values, d.SEX.values])
    for k in range(FOLDS):
        tr = np.where((f != k) & cn)[0]
        te = np.where(f == k)[0]

        # ---------- forward: brain -> age, with the usual bias correction
        sc = StandardScaler().fit(Xall[tr])
        m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(Xall[tr]), age[tr])
        b = LinearRegression().fit(age[tr].reshape(-1, 1),
                                   m.predict(sc.transform(Xall[tr])) - age[tr])
        gap_all[te] = (m.predict(sc.transform(Xall[te])) - age[te]) \
            - b.predict(age[te].reshape(-1, 1))

        for j, v in enumerate(V):
            Xr = np.column_stack([d[v].values, d.ICV.values, d.SEX.values])
            s2 = StandardScaler().fit(Xr[tr])
            m2 = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(s2.transform(Xr[tr]), age[tr])
            b2 = LinearRegression().fit(
                age[tr].reshape(-1, 1),
                m2.predict(s2.transform(Xr[tr])) - age[tr])
            gap_reg[te, j] = (m2.predict(s2.transform(Xr[te])) - age[te]) \
                - b2.predict(age[te].reshape(-1, 1))

        # ---------- reverse: age -> expected volume, no bias correction needed
        A = np.column_stack([age, age ** 2, d.SEX.values, d.ICV.values])
        sa = StandardScaler().fit(A[tr])
        for j, v in enumerate(V):
            y = d[v].values
            mv = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sa.transform(A[tr]), y[tr])
            res_tr = y[tr] - mv.predict(sa.transform(A[tr]))
            sd = res_tr.std(ddof=1)
            z_norm[te, j] = (y[te] - mv.predict(sa.transform(A[te]))) / max(sd, 1e-9)

    return gap_all, gap_reg, z_norm


def auc_cv(X, y, subs, seed):
    f = folds_of(subs, seed + 50)
    o = np.zeros(len(y))
    for k in range(FOLDS):
        tr, te = f != k, f == k
        sc = StandardScaler().fit(X[tr])
        c = LogisticRegression(max_iter=4000).fit(sc.transform(X[tr]), y[tr])
        o[te] = c.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, o)


def within_noise(d, col_vals, name):
    """Within-subject scatter about a straight line, in non-progressing people:
    the measurement noise of whatever quantity was passed in."""
    tmp = d.copy()
    tmp["_q"] = col_vals
    st = tmp[(tmp.DX == "CN")]
    r = []
    for _, g in st.groupby("PTID"):
        g = g.sort_values("Years_bl")
        if len(g) < 4 or (g.Years_bl.max() - g.Years_bl.min()) < 1.5:
            continue
        t, y = g.Years_bl.values, g._q.values
        r.append((y - np.polyval(np.polyfit(t, y, 1), t)).std(ddof=1))
    return np.median(r)


def main():
    a, d = load()
    gap_all, gap_reg, z_norm = build_features(d)
    print(f"visits {len(d)}, subjects {d.PTID.nunique()}")

    # ---------------- sanity: does the reverse direction avoid the age bias?
    cn = (d.DX == "CN").values
    age = d.age_visit.values
    r_gap = np.corrcoef(gap_all[cn], age[cn])[0, 1]
    r_z = np.mean([abs(np.corrcoef(z_norm[cn, j], age[cn])[0, 1])
                   for j in range(len(V))])
    print("\n" + "=" * 76)
    print("SANITY: residual correlation with age in CONTROLS (should be ~0)")
    print("=" * 76)
    print(f"  forward gap, AFTER its bias correction : {r_gap:+.3f}")
    print(f"  reverse z-scores, no correction applied: {r_z:+.3f} (mean |r|)")

    # ---------------- measurement noise of each quantity
    print("\n" + "=" * 76)
    print("MEASUREMENT NOISE (within-subject, controls)")
    print("=" * 76)
    ng = within_noise(d, gap_all, "gap")
    print(f"  scalar brain-age gap : {ng:.3f}  (units: years)")
    print(f"  {'region':16s} {'forward gap (y)':>16} {'reverse z (SD)':>16}")
    for j, v in enumerate(V):
        a1 = within_noise(d, gap_reg[:, j], v)
        a2 = within_noise(d, z_norm[:, j], v)
        print(f"  {FA[v]:16s} {a1:16.3f} {a2:16.3f}")

    # ---------------- disease separation, identical folds
    bl = d[d.VISCODE == "bl"].index.values
    db = d.loc[bl]
    print("\n" + "=" * 76)
    print("DISEASE SEPARATION (baseline visits, identical folds)")
    print("=" * 76)
    for task, pos in [("AD vs CN", "Dementia"), ("MCI vs CN", "MCI")]:
        sel = db.DX.isin(["CN", pos]).values
        idx = bl[sel]
        y = (db.DX[sel] == pos).astype(int).values
        subs = db.PTID.values[sel]
        ag = d.age_visit.values[idx].reshape(-1, 1)
        arms = {
            "gap scalar": np.column_stack([gap_all[idx], ag]),
            "gap regional (6)": np.column_stack([gap_reg[idx], ag]),
            "norm z (6)": np.column_stack([z_norm[idx], ag]),
            "norm distance": np.column_stack([np.sqrt((z_norm[idx] ** 2).sum(1)), ag]),
            "volumes + age": np.column_stack(
                [d.loc[idx, V].values, d.ICV.values[idx], d.SEX.values[idx], ag]),
        }
        print(f"\n  {task}  n={len(y)} ({int((y==0).sum())} vs {int(y.sum())})")
        res = {}
        for nm, X in arms.items():
            v = np.mean([auc_cv(X, y, subs, s) for s in range(SEEDS)])
            res[nm] = v
            print(f"    {nm:20s} AUC = {v:.3f}")
        dd = res["norm z (6)"] - res["gap regional (6)"]
        print(f"    reverse minus forward (both 6-dim): {dd:+.3f}")

    # ---------------- which regions drive an individual call
    print("\n" + "=" * 76)
    print("INTERPRETABILITY: mean |z| by group, per region")
    print("=" * 76)
    print(f"  {'region':16s} {'CN':>8} {'MCI':>8} {'AD':>8}")
    for j, v in enumerate(V):
        row = [np.mean(np.abs(z_norm[bl][db.DX.values == g, j]))
               for g in ["CN", "MCI", "Dementia"]]
        print(f"  {FA[v]:16s} {row[0]:8.2f} {row[1]:8.2f} {row[2]:8.2f}")

    np.savez(os.path.join(OUT, "exp42_inverse.npz"),
             gap_all=gap_all, gap_reg=gap_reg, z_norm=z_norm,
             ptid=d.PTID.values, vis=d.VISCODE.values)
    print(f"\nsaved -> {OUT}\\exp42_inverse.npz")


if __name__ == "__main__":
    main()
