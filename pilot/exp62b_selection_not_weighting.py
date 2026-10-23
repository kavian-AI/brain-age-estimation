#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WAS IT THE WEIGHTING, OR WAS IT THE SELECTION?

E62 reversed the headline. On a neutral 98-region atlas the uniform average LOSES
to the brain-age gap on diagnosis (-0.025, p=0.007) and ties it on conversion,
and prognosis is no longer one-dimensional (k=20 beats k=1 by +0.049).

Two explanations, and they predict different things:

  SAMPLE   TADPOLE covers ADNI1/GO/2 only (n=1369), while every six-region result
           was on ADNIMERGE including ADNI3 (n=1930). Maybe the effect simply
           does not exist in this sample.
           -> predicts: on the SAME 1369 people, six-region uniform does not beat
              the six-region gap either.

  SELECTION  ADNIMERGE's six regions are not a neutral sample of the brain. ADNI
           chose them BECAUSE they are Alzheimer-relevant: hippocampus,
           entorhinal, fusiform, middle temporal, ventricles, whole brain. Averaging
           them uniformly puts about two thirds of the weight on the medial
           temporal lobe. On 98 neutral regions, uniform puts 16% there - the same
           share the age model gives it. The disease knowledge was in the choice of
           regions, not in the equal weights.
           -> predicts: on the same people, six-region uniform still beats the
              gap, AND a uniform average over only the MTL regions of the rich
              atlas recovers the win.

Also measured here, because the design makes it free: on the same people, is a
98-region brain-age model a BETTER age predictor and a WORSE disease detector
than a six-region one? That is the trade-off Schulz et al. (PLOS Biology 2025)
report; E60 could not test it properly because all seven models there used the
same six inputs.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp53_units_and_dimension as E  # noqa: E402
import exp62_rich_atlas as R  # noqa: E402
from lifelines import CoxPHFitter  # noqa: E402
from lifelines.utils import concordance_index  # noqa: E402
from sklearn.linear_model import LinearRegression, RidgeCV  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

V6 = E.V
SIGN6 = np.array([-1., -1., -1., -1., +1., -1.])
N_AUC, N_COX = 300, 200


def six_region(m, cn):
    """The six ADNIMERGE regions, deviations and gap, on the SAME subjects."""
    f = E.folds_of(m.PTID.values)
    ag = m.age_visit.values.astype(float)
    X = np.column_stack([ag, ag ** 2, m.SEX.values, m.ICV.values])
    Y = np.column_stack([m[v].values for v in V6]).astype(float)
    Z = np.zeros_like(Y)
    for k in range(5):
        tr, te = np.where((f != k) & cn)[0], np.where(f == k)[0]
        sc = StandardScaler().fit(X[tr])
        lr = LinearRegression().fit(sc.transform(X[tr]), Y[tr])
        sd = (Y[tr] - lr.predict(sc.transform(X[tr]))).std(0, ddof=1)
        Z[te] = (Y[te] - lr.predict(sc.transform(X[te]))) / sd
    Xa = np.column_stack([Y, m.ICV.values, m.SEX.values])
    g = np.zeros(len(m))
    mae = []
    for k in range(5):
        tr, te = np.where((f != k) & cn)[0], np.where(f == k)[0]
        sc = StandardScaler().fit(Xa[tr])
        md = RidgeCV(alphas=np.logspace(-2, 5, 40)).fit(sc.transform(Xa[tr]), ag[tr])
        p = md.predict(sc.transform(Xa[te]))
        mae.append(np.abs(p - ag[te])[cn[te]])
        bb = LinearRegression().fit(ag[tr].reshape(-1, 1),
                                    md.predict(sc.transform(Xa[tr])) - ag[tr])
        g[te] = (p - ag[te]) - bb.predict(ag[te].reshape(-1, 1))
    return Z, g, np.concatenate(mae).mean()


def rich_gap_mae(m, names, cn):
    f = E.folds_of(m.PTID.values)
    ag = m.age_visit.values.astype(float)
    X = np.column_stack([m[n].values for n in names] + [m.ICV_T.values, m.SEX.values])
    mae = []
    for k in range(5):
        tr, te = np.where((f != k) & cn)[0], np.where(f == k)[0]
        sc = StandardScaler().fit(X[tr])
        md = RidgeCV(alphas=np.logspace(-2, 5, 40)).fit(sc.transform(X[tr]), ag[tr])
        mae.append(np.abs(md.predict(sc.transform(X[te])) - ag[te])[cn[te]])
    return np.concatenate(mae).mean()


def main():
    a, m, T, names = R.build("X")
    cn = (m.DX.values == "CN")
    print(f"SAME subjects throughout: n={len(m)} (CN {cn.sum()}, MCI {(m.DX=='MCI').sum()},"
          f" AD {(m.DX=='Dementia').sum()})\n")

    Z98, zh = R.normative(m, names, cn)
    S98 = T.sign.values
    g98 = R.rich_gap(m, names, cn)
    Z6, g6, mae6 = six_region(m, cn)
    mae98 = rich_gap_mae(m, names, cn)
    mtl = T.cls.values == "mtl"
    arms = {
        "gap (6 regions)": g6,
        "uniform (6 regions)": (Z6 * SIGN6).mean(1),
        "gap (98 regions)": g98,
        "uniform (98 regions)": (Z98 * S98).mean(1),
        "uniform (16 MTL regions only)": (Z98[:, mtl] * S98[mtl]).mean(1),
        "uniform (82 NON-MTL regions)": (Z98[:, ~mtl] * S98[~mtl]).mean(1),
        "hippocampus": -zh,
    }
    print("share of weight a uniform average puts on the medial temporal lobe:")
    print(f"  six ADNIMERGE regions: 4 of 6 = {100*4/6:.0f}% "
          f"(hippocampus, entorhinal, fusiform, middle temporal)")
    print(f"  98 TADPOLE regions:    {mtl.sum()} of {len(mtl)} = {100*mtl.mean():.0f}%\n")

    print("=" * 90)
    print("AGE ACCURACY vs DISEASE UTILITY, same people (Schulz et al. predict a trade-off)")
    print("=" * 90)
    sel = m.DX.isin(["CN", "Dementia"]).values
    y = (m.DX.values[sel] == "Dementia").astype(int)
    subs = m.PTID.values[sel]
    ag = m.age_visit.values[sel].reshape(-1, 1)
    for nm, g, mae in [("6-region brain age", g6, mae6), ("98-region brain age", g98, mae98)]:
        print(f"  {nm:22s} age MAE in controls {mae:.2f} y   "
              f"AD vs CN AUC {E.auc_cv(np.column_stack([g[sel], ag]), y, subs):.3f}")

    print("\n" + "=" * 90)
    print("EVERY ARM, THREE ENDPOINTS, SAME PEOPLE")
    print("=" * 90)
    pool = m[m.DX == "MCI"]
    keep, dur, ev = E.survival(a, pool.PTID.values, ["Dementia"])
    idx = pool.index[pool.PTID.isin(set(keep))].values
    o = {s: i for i, s in enumerate(keep)}
    ii = np.array([o[s] for s in m.PTID.values[idx]])
    dur, ev = dur[ii], ev[ii]
    agC = m.age_visit.values[idx]
    res = {}
    print(f"  {'arm':32s} {'AD vs CN':>9s} {'MCI vs CN':>10s} {'MCI->AD C':>10s}")
    for nm, x in arms.items():
        r1 = E.auc_cv(np.column_stack([x[sel], ag]), y, subs)
        s2 = m.DX.isin(["CN", "MCI"]).values
        y2 = (m.DX.values[s2] == "MCI").astype(int)
        r2 = E.auc_cv(np.column_stack([x[s2], m.age_visit.values[s2].reshape(-1, 1)]),
                      y2, m.PTID.values[s2])
        r3 = np.nanmean([R.cidx(np.column_stack([x[idx], agC]), dur, ev, s) for s in range(3)])
        res[nm] = (r1, r2, r3)
        print(f"  {nm:32s} {r1:9.3f} {r2:10.3f} {r3:10.3f}")

    print("\n" + "=" * 90)
    print("THE DECISIVE CONTRASTS (subject-level bootstrap)")
    print("=" * 90)
    tests = [("uniform (6 regions)", "gap (6 regions)", "SAMPLE test: does 6-region uniform still win here?"),
             ("uniform (16 MTL regions only)", "gap (98 regions)", "SELECTION test: MTL-only average vs rich gap"),
             ("uniform (16 MTL regions only)", "uniform (98 regions)", "same weighting, different selection"),
             ("uniform (16 MTL regions only)", "hippocampus", "MTL average vs hippocampus alone"),
             ("gap (98 regions)", "gap (6 regions)", "does a richer age model detect disease better?")]
    uq = np.unique(subs)
    for A, B, lab in tests:
        print(f"\n  {lab}")
        rng = np.random.RandomState(0)
        d = []
        for _ in range(N_AUC):
            pick = set(rng.choice(uq, len(uq), replace=True))
            mm = np.array([s in pick for s in subs])
            if y[mm].sum() < 20 or (1 - y[mm]).sum() < 20:
                continue
            d.append(E.auc_cv(np.column_stack([arms[A][sel][mm], ag[mm]]), y[mm], subs[mm])
                     - E.auc_cv(np.column_stack([arms[B][sel][mm], ag[mm]]), y[mm], subs[mm]))
        R.show(f"AD vs CN   {A} minus {B}"[:70], d)
        rng = np.random.RandomState(1)
        d = []
        for _ in range(N_COX):
            i = rng.randint(0, len(dur), len(dur))
            if ev[i].sum() < 20:
                continue
            p = R.cidx(np.column_stack([arms[A][idx][i], agC[i]]), dur[i], ev[i], 0)
            q = R.cidx(np.column_stack([arms[B][idx][i], agC[i]]), dur[i], ev[i], 0)
            if np.isfinite(p) and np.isfinite(q):
                d.append(p - q)
        R.show(f"MCI->AD    {A} minus {B}"[:70], d)


if __name__ == "__main__":
    main()
