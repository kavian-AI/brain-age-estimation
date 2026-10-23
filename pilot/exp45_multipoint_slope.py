#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
THE SAME TEST WITH A MULTI-POINT SLOPE, WHICH IS THE FAIR VERSION.

E44 found that a regional rate adds nothing to a regional cross-sectional vector
(+0.007, CI crossing zero). But that rate came from only TWO scans, while Report
3 fitted a slope through three or more. A two-point difference is far noisier, so
E44 handicapped the arm it was judging.

Here every subject must have at least three scans inside the landmark window and
the slope is FITTED through all of them, exactly as Report 3 did. If the rate
still adds nothing, the conclusion holds and is no longer design-dependent. If it
recovers, E44 was an artefact of a two-point difference and must be withdrawn.

Cross-sectional comparator is the z at the LAST visit inside the window, so both
arms see exactly the same scans and the same information - only the summary
differs.
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
DATA = os.path.join(BASE, "data")
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
FA = {"Hippocampus": "هیپوکامپ", "Entorhinal": "آنتورینال", "Fusiform": "دوکی",
      "MidTemp": "گیجگاهی میانی", "Ventricles": "بطن‌ها", "WholeBrain": "کل مغز"}
L, FOLDS, N_BOOT, MIN_VIS = 2.0, 5, 300, 3


def load():
    a = pd.read_csv(os.path.join(DATA, "ADNIMERGE_05Mar2026.csv"), low_memory=False)
    d = a.dropna(subset=["AGE", "Years_bl", "ICV", "PTGENDER"] + V).copy()
    d["SEX"] = (d.PTGENDER == "Male").astype(int)
    d["age_visit"] = d.AGE + d.Years_bl
    return a, d.reset_index(drop=True)


def normative_z(d, seed=0):
    """Population z for EVERY visit, CN-trained, out of fold by subject."""
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
            tr = np.where((f != k) & cn)[0]
            te = np.where(f == k)[0]
            sc = StandardScaler().fit(X[tr])
            m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(X[tr]), y[tr])
            sd = (y[tr] - m.predict(sc.transform(X[tr]))).std(ddof=1)
            Z[te, j] = (y[te] - m.predict(sc.transform(X[te]))) / max(sd, 1e-9)
    return Z


def summarise(d, Z):
    """Per subject inside the window: z at the last visit, and a fitted slope."""
    rows = []
    for s, g in d[d.Years_bl <= L + 0.35].groupby("PTID"):
        g = g.sort_values("Years_bl")
        if len(g) < MIN_VIS or (g.Years_bl.max() - g.Years_bl.min()) < 1.2:
            continue
        t = g.Years_bl.values.astype(float)
        zz = Z[g.index.values]
        r = dict(PTID=s, age=g.age_visit.values[0], n_vis=len(g),
                 t_last=t[-1], DX0=g.DX.values[0])
        for j, v in enumerate(V):
            r["zl_" + v] = zz[-1, j]                      # cross-sectional
            r["zs_" + v] = np.polyfit(t, zz[:, j], 1)[0]  # fitted slope
        rows.append(r)
    return pd.DataFrame(rows)


def outcome(a, ids, tlast):
    hist = a[["PTID", "Years_bl", "DX"]].dropna(subset=["DX"])
    conv = hist[hist.DX == "Dementia"].groupby("PTID").Years_bl.min()
    last = hist.groupby("PTID").Years_bl.max()
    keep, dur, ev = [], [], []
    for s, t in zip(ids, tlast):
        if s not in last.index:
            continue
        t2 = (conv[s] - t) if s in conv.index else (last[s] - t)
        if t2 > 0:
            keep.append(s)
            dur.append(t2)
            ev.append(1 if s in conv.index else 0)
    return keep, np.array(dur), np.array(ev)


def cox_c(F, cols, dur, ev, seed):
    n = len(dur)
    f = np.random.RandomState(seed).permutation(n) % FOLDS
    r = np.zeros(n)
    for k in range(FOLDS):
        tr, te = f != k, f == k
        if ev[tr].sum() < 5:
            return np.nan
        D = F[cols][tr].copy()
        D["T"], D["E"] = dur[tr], ev[tr]
        try:
            m = CoxPHFitter(penalizer=0.1).fit(D, "T", "E")
            r[te] = m.predict_partial_hazard(F[cols][te]).values
        except Exception:
            return np.nan
    return concordance_index(dur, -r, ev)


def boot(F, c1, c2, dur, ev):
    rng = np.random.RandomState(0)
    out = []
    for _ in range(N_BOOT):
        i = rng.randint(0, len(dur), len(dur))
        if ev[i].sum() < 12:
            continue
        Fi = F.iloc[i].reset_index(drop=True)
        x, y = cox_c(Fi, c1, dur[i], ev[i], 0), cox_c(Fi, c2, dur[i], ev[i], 0)
        if np.isfinite(x) and np.isfinite(y):
            out.append(x - y)
    return np.array(out)


def main():
    a, d = load()
    Z = normative_z(d)
    S = summarise(d, Z)
    M = S[S.DX0 == "MCI"].reset_index(drop=True)
    keep, dur, ev = outcome(a, M.PTID.values, M.t_last.values)
    ks = {s: i for i, s in enumerate(keep)}
    sel = M.PTID.isin(ks).values
    M = M[sel].reset_index(drop=True)
    ii = np.array([ks[s] for s in M.PTID.values])
    dur, ev = dur[ii], ev[ii]

    print(f"MCI with >={MIN_VIS} scans inside {L} y: n={len(M)}, "
          f"conversions after the last one={int(ev.sum())}")
    print(f"median scans per subject in window: {M.n_vis.median():.0f}")

    F = M[["age"]].copy()
    ZL = [f"zl_{v}" for v in V]
    ZS = [f"zs_{v}" for v in V]
    for c in ZL + ZS:
        F[c] = M[c].values

    print("\n" + "=" * 74)
    print("MULTI-POINT SLOPE vs CROSS-SECTION, six against six")
    print("=" * 74)
    arms = [("age only", ["age"]),
            ("cross-section z (6)", ZL + ["age"]),
            ("fitted SLOPE (6)", ZS + ["age"]),
            ("both (12)", ZL + ZS + ["age"])]
    res = {}
    for nm, cols in arms:
        c = np.nanmean([cox_c(F, cols, dur, ev, s) for s in range(8)])
        res[nm] = cols
        print(f"  {nm:24s} C = {c:.3f}")

    print("\n" + "=" * 74)
    print("DOES A PROPERLY FITTED SLOPE ADD?  (subject-level bootstrap)")
    print("=" * 74)
    for lab, A, B in [("slope adds to cross-section?", "both (12)", "cross-section z (6)"),
                      ("cross-section beats slope?", "cross-section z (6)", "fitted SLOPE (6)")]:
        dd = boot(F, res[A], res[B], dur, ev)
        lo, hi = np.percentile(dd, [2.5, 97.5])
        p = 2 * min((dd <= 0).mean(), (dd >= 0).mean())
        print(f"\n  {lab}")
        print(f"    {dd.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  p={p:.3f}"
              f"   -> {'REAL' if p < 0.05 else 'not distinguishable'}")

    print("\n  for comparison, E44 with a TWO-point rate: +0.007 [-0.010, +0.029]")


if __name__ == "__main__":
    main()
