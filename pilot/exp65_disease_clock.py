#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
REPLACE THE TARGET: A DISEASE CLOCK INSTEAD OF A BRAIN AGE.

Every thread of this project ends at the same place. The brain-age gap fails on
Alzheimer's not because of its model, its population or its weighting scheme, but
because of its TARGET: it is trained to predict chronological age, and the
regions that predict age best are not the regions Alzheimer's damages (E61/E62,
rank r = -0.03). And its unit is dishonest: one year of gap carries the hazard of
5.6 years of age (E53a), and the conversion factor changes between populations.

So keep everything a brain-age model is - regional structure in, a number of
YEARS out - and change only what the years mean. Train on the longitudinal visits
of people who converted from MCI to Alzheimer's, with the target being each
visit's time RELATIVE TO CONVERSION (negative before, zero at conversion). The
output is a "disease time": how far along the road to dementia this brain looks.

Two questions:
  1  Prognosis. On held-out MCI at baseline, does disease time predict
     conversion better than brain age, the MTL-selected average (the best
     unsupervised number found, E62b) and the hippocampus?
  2  THE UNIT. Among held-out converters, regress their ACTUAL remaining time to
     conversion on their predicted disease time. An honest unit has slope 1: one
     predicted year is one real year. Brain age fails that test by a factor of
     ~5.6. Does disease time pass it?

Honest caveat stated up front: disease time is trained on converters' timing,
so it uses disease information the gap never sees. Beating the gap is expected;
the unit test is the part that could genuinely fail.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp53_units_and_dimension as E  # noqa: E402
from lifelines import CoxPHFitter  # noqa: E402
from lifelines.utils import concordance_index  # noqa: E402
from sklearn.linear_model import LinearRegression, RidgeCV  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

V = E.V
SIGN = np.array([-1., -1., -1., -1., +1., -1.])
FOLDS = 5


def main():
    a = pd.read_csv(os.path.join(E.DATA, "ADNIMERGE_05Mar2026.csv"), low_memory=False)
    d = a.dropna(subset=["AGE", "Years_bl", "ICV", "PTGENDER", "DX"] + V).copy()
    d["SEX"] = (d.PTGENDER == "Male").astype(int)
    d["age_visit"] = d.AGE + d.Years_bl
    d = d.sort_values(["PTID", "Years_bl"]).reset_index(drop=True)
    dx0 = d.groupby("PTID").DX.first()
    conv_t = d[d.DX == "Dementia"].groupby("PTID").Years_bl.min()

    # normative deviations for EVERY visit, CN-referenced, out of fold by subject
    f = E.folds_of(d.PTID.values)
    cn = (d.DX.values == "CN")
    ag = d.age_visit.values.astype(float)
    Xc = np.column_stack([ag, ag ** 2, d.SEX.values, d.ICV.values])
    Y = np.column_stack([d[v].values for v in V]).astype(float)
    Z = np.zeros_like(Y)
    for k in range(FOLDS):
        tr, te = np.where((f != k) & cn)[0], np.where(f == k)[0]
        sc = StandardScaler().fit(Xc[tr])
        lr = LinearRegression().fit(sc.transform(Xc[tr]), Y[tr])
        sd = (Y[tr] - lr.predict(sc.transform(Xc[tr]))).std(0, ddof=1)
        Z[te] = (Y[te] - lr.predict(sc.transform(Xc[te]))) / sd
    for j, v in enumerate(V):
        d["z_" + v] = Z[:, j]
    ZC = [f"z_{v}" for v in V]

    # training rows: visits of MCI->AD converters, from 6 y before to 2 y after
    d["conv_at"] = d.PTID.map(conv_t)
    d["t_rel"] = d.Years_bl - d.conv_at
    is_conv = (d.PTID.map(dx0) == "MCI") & d.conv_at.notna()
    trn = is_conv & d.t_rel.between(-6, 2)
    print(f"disease-clock training pool: {d[trn].PTID.nunique()} converters, "
          f"{int(trn.sum())} visits, time relative to conversion "
          f"{d.t_rel[trn].min():.1f} to {d.t_rel[trn].max():.1f} y")

    # baseline MCI: the people to score
    base = d[(d.VISCODE == "bl") & (d.PTID.map(dx0) == "MCI")].copy()
    keep, dur, ev = E.survival(a, base.PTID.values, ["Dementia"])
    base = base[base.PTID.isin(set(keep))].copy()
    o = {s: i for i, s in enumerate(keep)}
    base["tt"] = [dur[o[s]] for s in base.PTID]
    base["ee"] = [ev[o[s]] for s in base.PTID]

    # disease time, out of fold: never trained on the subject being scored
    fb = E.folds_of(base.PTID.values, seed=3)
    fmap = dict(zip(base.PTID, fb))
    dtime = np.zeros(len(base))
    for k in range(FOLDS):
        test_ids = set(base.PTID[fb == k])
        tr_rows = trn & ~d.PTID.isin(test_ids)
        Xtr = np.column_stack([d.loc[tr_rows, ZC].values, d.loc[tr_rows, "age_visit"].values])
        sc = StandardScaler().fit(Xtr)
        mdl = RidgeCV(alphas=np.logspace(-2, 4, 30)).fit(sc.transform(Xtr),
                                                        d.loc[tr_rows, "t_rel"].values)
        q = (fb == k)
        dtime[q] = mdl.predict(sc.transform(np.column_stack([base.loc[q, ZC].values,
                                                             base.loc[q, "age_visit"].values])))
    base["dtime"] = dtime

    # the comparators on the same people
    a2, b2 = E.load()
    gap = pd.Series(E.brain_age_gap(b2), index=b2.PTID)
    base["gap"] = base.PTID.map(gap).values
    base["uni6"] = (base[ZC].values * SIGN).mean(1)
    base["hipp"] = -base["z_Hippocampus"].values
    base = base.dropna(subset=["gap"])

    print(f"\nscored: {len(base)} MCI at baseline, {int(base.ee.sum())} converted")
    print("\n" + "=" * 86)
    print("1  PROGNOSIS (C-index, univariate + age, 5 fold seeds)")
    print("=" * 86)

    def cidx(col, seed=0, frame=None):
        F = base if frame is None else frame
        ff = np.random.RandomState(seed).permutation(len(F)) % 3
        r = np.zeros(len(F))
        for k in range(3):
            tr, te = ff != k, ff == k
            D = pd.DataFrame({"x": F[col].values[tr], "age": F.age_visit.values[tr],
                              "T": F.tt.values[tr], "E": F.ee.values[tr]})
            m = CoxPHFitter(penalizer=0.1).fit(D, "T", "E")
            r[te] = m.predict_partial_hazard(pd.DataFrame({"x": F[col].values[te],
                                                           "age": F.age_visit.values[te]})).values
        return concordance_index(F.tt.values, -r, F.ee.values)

    for nm, col in [("brain-age gap (target: age)", "gap"), ("hippocampus", "hipp"),
                    ("uniform, 6 ADNI regions", "uni6"), ("DISEASE TIME (target: conversion)", "dtime")]:
        print(f"    {nm:36s} C = {np.mean([cidx(col, s) for s in range(5)]):.3f}")
    rng = np.random.RandomState(0)
    for other in ["gap", "uni6", "hipp"]:
        dd = []
        for _ in range(200):
            i = rng.randint(0, len(base), len(base))
            F = base.iloc[i].reset_index(drop=True)
            if F.ee.sum() < 25:
                continue
            dd.append(cidx("dtime", 0, F) - cidx(other, 0, F))
        dd = np.array(dd)
        lo, hi = np.percentile(dd, [2.5, 97.5])
        p = 2 * min((dd <= 0).mean(), (dd >= 0).mean())
        print(f"      disease time minus {other:5s} {dd.mean():+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.3f}"
              f"  {'REAL' if p < .05 else '-'}")

    print("\n" + "=" * 86)
    print("2  IS THE UNIT HONEST?  held-out converters: actual years remaining vs predicted")
    print("=" * 86)
    C = base[base.ee == 1].copy()
    C["remaining"] = C.tt                      # years from baseline to conversion
    C["pred_remaining"] = -C.dtime             # disease time is negative before conversion
    lr = LinearRegression().fit(C[["pred_remaining"]], C.remaining)
    rng = np.random.RandomState(1)
    sl = []
    for _ in range(1000):
        i = rng.randint(0, len(C), len(C))
        sl.append(LinearRegression().fit(C[["pred_remaining"]].iloc[i], C.remaining.iloc[i]).coef_[0])
    lo, hi = np.percentile(sl, [2.5, 97.5])
    print(f"  converters n={len(C)}: actual = {lr.intercept_:+.2f} + {lr.coef_[0]:.2f} x predicted"
          f"   slope 95% CI [{lo:.2f}, {hi:.2f}]   r = {np.corrcoef(C.pred_remaining, C.remaining)[0,1]:.3f}")
    print(f"  mean predicted remaining {C.pred_remaining.mean():.2f} y vs actual {C.remaining.mean():.2f} y")
    print(f"  median absolute error {np.median(np.abs(C.pred_remaining - C.remaining)):.2f} y")
    print(f"  honest unit needs slope 1; brain age fails the analogous test by x5.6 (E53a)")

    NC = base[base.ee == 0]
    print(f"\n  non-converters (n={len(NC)}): mean disease time {NC.dtime.mean():+.2f} y  "
          f"vs converters {C.dtime.mean():+.2f} y")
    # hazard per predicted year, compared with hazard per year of brain-age gap
    for nm, col in [("disease time", "dtime"), ("brain-age gap", "gap")]:
        cf = CoxPHFitter().fit(pd.DataFrame({"x": base[col], "age": base.age_visit,
                                             "T": base.tt, "E": base.ee}), "T", "E")
        print(f"  HR per year of {nm:14s} {np.exp(cf.params_['x']):.3f}  "
              f"[{np.exp(cf.confidence_intervals_.loc['x'].iloc[0]):.3f}, "
              f"{np.exp(cf.confidence_intervals_.loc['x'].iloc[1]):.3f}]")


if __name__ == "__main__":
    main()
