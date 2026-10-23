#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IS WHOLE-BRAIN VOLUME THE ENTIRE MECHANISM?

E59's leave-one-out gave one row that explains everything else. On AD vs CN,
dropping WholeBrain moved the brain-age gap from 0.914 to 0.939 and the uniform
average from 0.928 to 0.939 - they TIE, and the advantage disappears (p=0.912).
Every other region left the advantage intact.

That is a mechanism, not a coincidence. Whole-brain volume is the single best
predictor of chronological age, so an age-trained model loads on it hardest - the
CN-fitted coefficients in E49 gave WholeBrain -1.49, the largest magnitude of the
six. But whole-brain volume is comparatively uninformative for Alzheimer's
relative to medial temporal structures. The gap therefore spends its weight in the
wrong place, and that misallocation IS the +0.016.

If this is right it has to hold on the endpoint that actually matters. Prognosis
is where the uniform average earns its keep (E59: +0.032 over hippocampus alone,
+0.019 over the gap), so the same removal is run on MCI -> AD conversion. If
dropping WholeBrain also closes the gap's deficit there, the mechanism is general.
If it does not, the AD vs CN row is a quirk of that task and must be reported as
one.
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp53_units_and_dimension as E
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler

V = E.V
SIGN = np.array([-1., -1., -1., -1., +1., -1.])
FOLDS, N_COX = 5, 300

a, b = E.load()
Z = E.normative_z(b)
cn_all = (b.DX.values == "CN")

def gap_subset(cols):
    f = E.folds_of(b.PTID.values)
    ag = b.age_visit.values.astype(float)
    X = np.column_stack([b[V[j]].values for j in cols] + [b.ICV.values, b.SEX.values])
    g = np.zeros(len(b))
    for k in range(FOLDS):
        tr, te = np.where((f != k) & cn_all)[0], np.where(f == k)[0]
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(X[tr]), ag[tr])
        bb = LinearRegression().fit(ag[tr].reshape(-1, 1),
                                    m.predict(sc.transform(X[tr])) - ag[tr])
        g[te] = (m.predict(sc.transform(X[te])) - ag[te]) - bb.predict(ag[te].reshape(-1, 1))
    return g

def coefs_of(cols):
    ag = b.age_visit.values.astype(float)
    X = np.column_stack([b[V[j]].values for j in cols] + [b.ICV.values, b.SEX.values])
    sc = StandardScaler().fit(X[cn_all])
    m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(X[cn_all]), ag[cn_all])
    return m.coef_

def cidx(X, d_, e_, seed=0):
    f = np.random.RandomState(seed).permutation(len(d_)) % 3
    r = np.zeros(len(d_)); nm = [f"x{j}" for j in range(X.shape[1])]
    for k in range(3):
        tr, te = f != k, f == k
        if e_[tr].sum() < 5: return np.nan
        D = pd.DataFrame(X[tr], columns=nm); D["T"], D["E"] = d_[tr], e_[tr]
        try:
            mm = CoxPHFitter(penalizer=1.0).fit(D, "T", "E")
            r[te] = mm.predict_partial_hazard(pd.DataFrame(X[te], columns=nm)).values
        except Exception: return np.nan
    return concordance_index(d_, -r, e_)

print("=" * 80)
print("WHAT DOES THE AGE MODEL ACTUALLY WEIGHT?  (CN-fitted, standardised)")
print("=" * 80)
c = coefs_of(list(range(6)))
for v, w in sorted(zip(V + ["ICV", "SEX"], c), key=lambda t: -abs(t[1])):
    print(f"  {v:14s} {w:+.3f}")

pool = b[b.DX == "MCI"]
keep, dur, ev = E.survival(a, pool.PTID.values, ["Dementia"])
idx = pool.index[pool.PTID.isin(set(keep))].values
o = {s: i for i, s in enumerate(keep)}
ii = np.array([o[s] for s in b.PTID.values[idx]])
dur, ev = dur[ii], ev[ii]
age_m = b.age_visit.values[idx]

print("\n" + "=" * 80)
print(f"MCI -> AD CONVERSION  n={len(idx)}, {int(ev.sum())} events")
print("=" * 80)
for label, cols in [("all six regions", list(range(6))),
                    ("WITHOUT WholeBrain", [0, 1, 2, 3, 4])]:
    g = gap_subset(cols)
    u = (Z[:, cols] * SIGN[cols]).mean(1)
    Mg = np.column_stack([g[idx], age_m]); Mu = np.column_stack([u[idx], age_m])
    cg = np.nanmean([cidx(Mg, dur, ev, s) for s in range(5)])
    cu = np.nanmean([cidx(Mu, dur, ev, s) for s in range(5)])
    rng = np.random.RandomState(0); dd = []
    for _ in range(N_COX):
        i = rng.randint(0, len(dur), len(dur))
        if ev[i].sum() < 25: continue
        p, q = cidx(Mu[i], dur[i], ev[i], 0), cidx(Mg[i], dur[i], ev[i], 0)
        if np.isfinite(p) and np.isfinite(q): dd.append(p - q)
    dd = np.array(dd); lo, hi = np.percentile(dd, [2.5, 97.5])
    pv = 2 * min((dd <= 0).mean(), (dd >= 0).mean())
    print(f"\n  {label}")
    print(f"    gap C = {cg:.3f}   uniform C = {cu:.3f}")
    print(f"    uniform minus gap {dd.mean():+.4f} [{lo:+.4f},{hi:+.4f}] p={pv:.3f}"
          f"  {'HOLDS' if pv < 0.05 and dd.mean() > 0 else 'does not hold'}")
