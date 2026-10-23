#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DOES A CROSS-SECTIONAL REGIONAL VECTOR REALLY BEAT THE RATE?

E43 reported population z (6 regions, one time point) at C = 0.791 against the
scalar gap rate at 0.712, and that looked like a cross-sectional measure beating
a longitudinal one. It is not a fair comparison: six numbers against one. Any
six-dimensional predictor has more room than a scalar, whatever it measures.

This runs the comparison at matched dimensionality:

    z at follow-up        6   cross-sectional, one visit
    z at baseline         6   cross-sectional, two years earlier
    regional rate         6   longitudinal, the change in those same six
    z follow + rate      12   does the rate ADD to the cross-section?

Report 3's actual claim was never "the rate beats the value" - a subject-level
bootstrap showed that crossed zero - it was that the rate ADDS. So the 12-column
arm is the one that tests our own prior claim, now with a regional comparator
instead of a scalar one.

Landmark: features from baseline and the ~2-year visit, conversions counted only
afterwards. CN-trained, out of fold by subject. Every comparison carries a
subject-level bootstrap.
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler

BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "pilot")
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
FA = {"Hippocampus": "هیپوکامپ", "Entorhinal": "آنتورینال", "Fusiform": "دوکی",
      "MidTemp": "گیجگاهی میانی", "Ventricles": "بطن‌ها", "WholeBrain": "کل مغز"}
L, FOLDS, N_BOOT = 2.0, 5, 500


def load():
    a = pd.read_csv(os.path.join(DATA, "ADNIMERGE_05Mar2026.csv"), low_memory=False)
    d = a.dropna(subset=["AGE", "Years_bl", "ICV", "PTGENDER"] + V).copy()
    d["SEX"] = (d.PTGENDER == "Male").astype(int)
    d["age_visit"] = d.AGE + d.Years_bl
    return a, d.reset_index(drop=True)


def make_pairs(d):
    rows = []
    for s, g in d.groupby("PTID"):
        g = g.sort_values("Years_bl")
        b = g.iloc[0]
        if b.Years_bl > 0.4:
            continue
        w = g[(g.Years_bl >= 0.8) & (g.Years_bl <= L + 0.35)]
        if w.empty:
            continue
        f = w.iloc[(w.Years_bl - L).abs().argsort().iloc[0]]
        r = dict(PTID=s, dt=f.Years_bl - b.Years_bl, age=b.age_visit,
                 SEX=b.SEX, ICV=b.ICV, DX0=b.DX, t_follow=f.Years_bl)
        for v in V:
            r["b_" + v], r["f_" + v] = b[v], f[v]
        rows.append(r)
    P = pd.DataFrame(rows)
    return P[P.dt >= 0.8].reset_index(drop=True)


def norm_z(P, which, seed=0):
    """Population normative z at 'b' (baseline) or 'f' (follow-up)."""
    cn = (P.DX0 == "CN").values
    uq = np.unique(P.PTID.values)
    fm = dict(zip(uq, np.random.RandomState(seed).permutation(len(uq)) % FOLDS))
    f = np.array([fm[s] for s in P.PTID.values])
    age = P.age.values + (P.dt.values if which == "f" else 0)
    Z = np.zeros((len(P), len(V)))
    X = np.column_stack([age, age ** 2, P.SEX.values, P.ICV.values])
    for j, v in enumerate(V):
        y = P[which + "_" + v].values
        for k in range(FOLDS):
            tr = np.where((f != k) & cn)[0]
            te = np.where(f == k)[0]
            sc = StandardScaler().fit(X[tr])
            m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(X[tr]), y[tr])
            sd = (y[tr] - m.predict(sc.transform(X[tr]))).std(ddof=1)
            Z[te, j] = (y[te] - m.predict(sc.transform(X[te]))) / max(sd, 1e-9)
    return Z


def scalar_gap(P, seed=0):
    cn = (P.DX0 == "CN").values
    uq = np.unique(P.PTID.values)
    fm = dict(zip(uq, np.random.RandomState(seed).permutation(len(uq)) % FOLDS))
    f = np.array([fm[s] for s in P.PTID.values])
    Xb = np.column_stack([P["b_" + v].values for v in V] + [P.ICV.values, P.SEX.values])
    Xf = np.column_stack([P["f_" + v].values for v in V] + [P.ICV.values, P.SEX.values])
    ab, af = P.age.values, P.age.values + P.dt.values
    gb, gf = np.zeros(len(P)), np.zeros(len(P))
    for k in range(FOLDS):
        tr, te = np.where((f != k) & cn)[0], np.where(f == k)[0]
        sc = StandardScaler().fit(Xb[tr])
        m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(Xb[tr]), ab[tr])
        b = LinearRegression().fit(ab[tr].reshape(-1, 1),
                                   m.predict(sc.transform(Xb[tr])) - ab[tr])
        gb[te] = (m.predict(sc.transform(Xb[te])) - ab[te]) - b.predict(ab[te].reshape(-1, 1))
        gf[te] = (m.predict(sc.transform(Xf[te])) - af[te]) - b.predict(af[te].reshape(-1, 1))
    return gb, (gf - gb) / P.dt.values


def outcome(a, ids, tf):
    hist = a[["PTID", "Years_bl", "DX"]].dropna(subset=["DX"])
    conv = hist[hist.DX == "Dementia"].groupby("PTID").Years_bl.min()
    last = hist.groupby("PTID").Years_bl.max()
    keep, dur, ev = [], [], []
    for s, t in zip(ids, tf):
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


def boot(F, c1, c2, dur, ev, seed=0):
    rng = np.random.RandomState(seed)
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
    P = make_pairs(d)
    Zb, Zf = norm_z(P, "b"), norm_z(P, "f")
    gb, grate = scalar_gap(P)

    pos_mci = np.where(P.DX0.values == "MCI")[0]
    keep, dur, ev = outcome(a, P.PTID.values[pos_mci], P.t_follow.values[pos_mci])
    ks = set(keep)
    pos = np.array([p for p in pos_mci if P.PTID.values[p] in ks])
    order = {s: i for i, s in enumerate(keep)}
    ii = np.array([order[P.PTID.values[p]] for p in pos])
    dur, ev = dur[ii], ev[ii]

    F = pd.DataFrame(dict(age=P.age.values[pos]))
    for j, v in enumerate(V):
        F["zf_" + v] = Zf[pos, j]
        F["zb_" + v] = Zb[pos, j]
        F["zr_" + v] = (Zf[pos, j] - Zb[pos, j]) / P.dt.values[pos]
    F["gap"] = gb[pos]
    F["grate"] = grate[pos]

    ZF = [f"zf_{v}" for v in V]
    ZB = [f"zb_{v}" for v in V]
    ZR = [f"zr_{v}" for v in V]

    print(f"MCI n={len(dur)}, conversions after the follow-up visit={int(ev.sum())}")
    print("\n" + "=" * 74)
    print("MATCHED DIMENSIONALITY: six against six")
    print("=" * 74)
    arms = [("age only", ["age"], 1),
            ("scalar gap", ["gap", "age"], 1),
            ("scalar gap RATE", ["grate", "age"], 1),
            ("z at baseline (6)", ZB + ["age"], 6),
            ("z at follow-up (6)", ZF + ["age"], 6),
            ("regional RATE (6)", ZR + ["age"], 6),
            ("z follow + rate (12)", ZF + ZR + ["age"], 12)]
    res = {}
    for nm, cols, dim in arms:
        c = np.nanmean([cox_c(F, cols, dur, ev, s) for s in range(8)])
        res[nm] = (c, cols)
        print(f"  {nm:24s} dim={dim:2d}   C = {c:.3f}")

    print("\n" + "=" * 74)
    print("THE COMPARISONS THAT MATTER (subject-level bootstrap)")
    print("=" * 74)
    tests = [
        ("cross-section beats rate?", "z at follow-up (6)", "regional RATE (6)"),
        ("does the rate ADD to it?", "z follow + rate (12)", "z at follow-up (6)"),
        ("is follow-up better than baseline?", "z at follow-up (6)", "z at baseline (6)"),
        ("regional beats scalar gap?", "z at follow-up (6)", "scalar gap"),
    ]
    for label, A, B in tests:
        dd = boot(F, res[A][1], res[B][1], dur, ev)
        lo, hi = np.percentile(dd, [2.5, 97.5])
        p = 2 * min((dd <= 0).mean(), (dd >= 0).mean())
        print(f"\n  {label}")
        print(f"    {A} minus {B}")
        print(f"    {dd.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  p={p:.3f}"
              f"   -> {'REAL' if p < 0.05 else 'not distinguishable'}")


if __name__ == "__main__":
    main()
