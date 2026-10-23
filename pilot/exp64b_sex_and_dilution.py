#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TWO RELATIONSHIPS FROM E64 THAT NEED A MECHANISM BEFORE THEY ARE BELIEVED.

1  THE SEX BIAS. Healthy men carry a brain-age gap 1.31 years higher than healthy
   women (95% CI [+0.86, +1.72]). Sex is an INPUT to the age model, so an
   unbiased linear model should leave zero mean gap in each sex. Two suspects in
   the recipe itself: ridge shrinks the sex coefficient, and the standard bias
   correction regresses the gap on AGE only. If refitting without shrinkage, or
   correcting on age AND sex, removes the 1.31 years, the "older male brain" is an
   artefact of how brain age is computed - and E64-F's finding that the gap
   misses small-headed women converters should shrink with it.

2  THE TAU PARADOX. In cognitively normal people a higher gap goes with LOWER CSF
   amyloid (looks pathological) AND LOWER CSF tau (looks protective). Alzheimer's
   biology cannot produce that pair: amyloid falls and tau rises. A physical
   explanation can. A more atrophied brain has larger CSF spaces, and every
   protein measured in CSF is more DILUTED. Two predictions follow:
     - the associations vanish for RATIOS such as p-tau/Abeta42, where a common
       dilution factor cancels
     - they vanish after controlling for ventricular size
   Amyloid PET, which is not a CSF concentration, already showed nothing (E64-A).
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp53_units_and_dimension as E  # noqa: E402
import exp64_what_it_measures as W  # noqa: E402
from sklearn.linear_model import LinearRegression, RidgeCV  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

V = E.V
N_BOOT = 1000


def gap_variant(b, kind):
    f = E.folds_of(b.PTID.values)
    cn = (b.DX.values == "CN")
    ag = b.age_visit.values.astype(float)
    X = np.column_stack([b[v].values for v in V] + [b.ICV.values, b.SEX.values])
    g = np.zeros(len(b))
    for k in range(5):
        tr, te = np.where((f != k) & cn)[0], np.where(f == k)[0]
        sc = StandardScaler().fit(X[tr])
        mdl = (LinearRegression() if kind == "ols"
               else RidgeCV(alphas=np.logspace(-2, 5, 30))).fit(sc.transform(X[tr]), ag[tr])
        raw_tr = mdl.predict(sc.transform(X[tr])) - ag[tr]
        raw_te = mdl.predict(sc.transform(X[te])) - ag[te]
        if kind == "ridge+age&sex correction":
            C_tr = np.column_stack([ag[tr], b.SEX.values[tr]])
            C_te = np.column_stack([ag[te], b.SEX.values[te]])
        else:
            C_tr, C_te = ag[tr].reshape(-1, 1), ag[te].reshape(-1, 1)
        bb = LinearRegression().fit(C_tr, raw_tr)
        g[te] = raw_te - bb.predict(C_te)
    return g


def main():
    a, b = E.load()
    cn = b.DX.values == "CN"
    print("=" * 90)
    print("1  WHERE DOES THE 1.3-YEAR SEX GAP COME FROM?")
    print("=" * 90)
    fitted = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(
        StandardScaler().fit_transform(np.column_stack([b[v].values[cn] for v in V]
                                                       + [b.ICV.values[cn], b.SEX.values[cn]])),
        b.age_visit.values[cn])
    print(f"  ridge alpha chosen by cross-validation: {fitted.alpha_:.1f}")
    for kind in ["ridge (standard recipe)", "ols", "ridge+age&sex correction"]:
        g = gap_variant(b, kind)
        gm, gw = g[cn & (b.SEX.values == 1)], g[cn & (b.SEX.values == 0)]
        rng = np.random.RandomState(0)
        dd = [rng.choice(gm, len(gm)).mean() - rng.choice(gw, len(gw)).mean()
              for _ in range(N_BOOT)]
        lo, hi = np.percentile(dd, [2.5, 97.5])
        sel = b.DX.isin(["CN", "Dementia"]).values
        aucs = []
        for sx in [1, 0]:
            q = sel & (b.SEX.values == sx)
            aucs.append(E.auc_cv(np.column_stack([g[q], b.age_visit.values[q].reshape(-1, 1)]),
                                 (b.DX.values[q] == "Dementia").astype(int), b.PTID.values[q]))
        print(f"  {kind:28s} healthy men {gm.mean():+.2f}  women {gw.mean():+.2f}"
              f"   difference {np.mean(dd):+.2f} [{lo:+.2f},{hi:+.2f}]"
              f"   AD/CN AUC men {aucs[0]:.3f} women {aucs[1]:.3f}")

    # does the correction rescue the converters the gap was missing?
    pool = b[b.DX == "MCI"]
    keep, dur, ev = E.survival(a, pool.PTID.values, ["Dementia"])
    M = pool[pool.PTID.isin(set(keep))].copy()
    o = {s: i for i, s in enumerate(keep)}
    M["ee"] = [ev[o[s]] for s in M.PTID]
    Z = E.normative_z(b)
    hipp = pd.Series(-Z[:, 0], index=b.index)
    print("\n  converters missed by the gap but flagged by the hippocampus:")
    for kind in ["ridge (standard recipe)", "ridge+age&sex correction"]:
        g = pd.Series(gap_variant(b, kind), index=b.index)
        gr = g.reindex(M.index).rank(pct=True)
        hr_ = hipp.reindex(M.index).rank(pct=True)
        conv = M.ee == 1
        missed = conv & (gr < .5) & (hr_ >= .5)
        print(f"    {kind:28s} {int(missed.sum()):3d} missed, of whom women "
              f"{100*(1-M.SEX[missed].mean()):.0f}%")

    print("\n" + "=" * 90)
    print("2  IS THE TAU PARADOX CSF DILUTION?  (cognitively normal only)")
    print("=" * 90)
    gap = E.brain_age_gap(b)
    bl = a[a.VISCODE == "bl"].drop_duplicates("PTID").set_index("PTID")
    D = b[cn].copy()
    for c in ["ABETA", "PTAU", "TAU", "AV45"]:
        D[c] = W.num(bl.reindex(D.PTID)[c]).values
    D["gap"] = gap[cn]
    D["vent_z"] = Z[cn, 4]
    D["hipp"] = -Z[cn, 0]
    tests = [("CSF Abeta42 (low=bad)", -D.ABETA), ("CSF p-tau", D.PTAU), ("CSF total tau", D.TAU),
             ("RATIO p-tau / Abeta42", D.PTAU / D.ABETA), ("RATIO t-tau / Abeta42", D.TAU / D.ABETA),
             ("amyloid PET (not CSF)", D.AV45)]
    C0 = np.column_stack([D.age_visit.values, D.SEX.values])
    C1 = np.column_stack([D.age_visit.values, D.SEX.values, D.vent_z.values])
    print(f"  {'biomarker':26s} {'gap, age+sex':>22s} {'gap, + ventricle size':>24s}"
          f" {'ventricle size alone':>22s}")
    for nm, y in tests:
        y = y.values.astype(float)
        r0, n0 = W.partial_r(D.gap.values, y, C0)
        l0, h0 = W.boot_pr(D.gap.values, y, C0, n=600)
        r1, _ = W.partial_r(D.gap.values, y, C1)
        l1, h1 = W.boot_pr(D.gap.values, y, C1, n=600)
        rv, _ = W.partial_r(D.vent_z.values, y, C0)
        lv, hv = W.boot_pr(D.vent_z.values, y, C0, n=600)
        s = lambda lo, hi: "*" if lo > 0 or hi < 0 else " "
        print(f"  {nm:26s} {r0:+.3f}{s(l0,h0)} [{l0:+.2f},{h0:+.2f}]  {r1:+.3f}{s(l1,h1)} [{l1:+.2f},{h1:+.2f}]"
              f"   {rv:+.3f}{s(lv,hv)} (n={n0})")
    print("  * = 95% bootstrap CI excludes zero.  Dilution predicts: raw CSF concentrations")
    print("  move together, ratios do not, PET does not, and controlling ventricle size removes it.")


if __name__ == "__main__":
    main()
