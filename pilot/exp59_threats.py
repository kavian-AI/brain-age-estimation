#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
THE FOUR THINGS A REVIEWER WOULD USE TO KILL THIS.

The claim is: average the regional normative deviations with equal weight instead
of weighting them by how well they predict chronological age. Before it is written
up, the obvious attacks have to be run and survived - or not.

A  IS IT JUST THE HIPPOCAMPUS?  The existential one. E53a already showed that in
   MCI a single hippocampal deviation carries a bigger hazard ratio (1.306 per SD)
   than the brain-age gap (1.119 per year). If the average does not beat the one
   region everybody already measures, the whole thing is an elaborate way of
   saying look at the hippocampus. Run on all three tasks.

B  DOES IT DEPEND ON EXACTLY THESE SIX REGIONS?  Leave one out, six times, with
   the brain-age model refitted on the same five so the comparison stays fair. If
   the advantage needs one particular region, it is a coincidence of this feature
   set rather than a property of the weighting.

C  DOES THE ADVANTAGE GROW WITH THE NUMBER OF REGIONS?  Every subset of size 2..6,
   gap refitted on the same subset each time. ADNIMERGE ships six regions and no
   full segmentation, so the hundred-region question cannot be answered directly -
   but the slope across 2..6 says which way it is heading.

E  HOW MUCH IS LEFT ON THE TABLE?  A supervised fit on the same six deviations is
   the ceiling. The distance from uniform to that ceiling is what a cleverer
   weighting could still buy, and it bounds how interesting the remaining problem
   is.
"""

import itertools
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
FA = {"Hippocampus": "هیپوکامپ", "Entorhinal": "آنتورینال", "Fusiform": "دوکی",
      "MidTemp": "گیجگاهی میانی", "Ventricles": "بطن‌ها", "WholeBrain": "کل مغز"}
SIGN = np.array([-1., -1., -1., -1., +1., -1.])
N_BOOT, N_COX, FOLDS = 400, 250, 5

a, b = E.load()
Z = E.normative_z(b)
gap = E.brain_age_gap(b)
cn_all = (b.DX.values == "CN")
uni = (Z * SIGN).mean(1)
hipp = -Z[:, 0]                       # sign-corrected, high = worse


def gap_subset(cols):
    """Brain-age gap refitted using ONLY these regions. Same recipe otherwise."""
    f = E.folds_of(b.PTID.values)
    ag = b.age_visit.values.astype(float)
    X = np.column_stack([b[V[j]].values for j in cols]
                        + [b.ICV.values, b.SEX.values])
    g = np.zeros(len(b))
    for k in range(FOLDS):
        tr, te = np.where((f != k) & cn_all)[0], np.where(f == k)[0]
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(X[tr]), ag[tr])
        bb = LinearRegression().fit(ag[tr].reshape(-1, 1),
                                    m.predict(sc.transform(X[tr])) - ag[tr])
        g[te] = (m.predict(sc.transform(X[te])) - ag[te]) \
            - bb.predict(ag[te].reshape(-1, 1))
    return g


def cidx(X, d_, e_, seed=0):
    f = np.random.RandomState(seed).permutation(len(d_)) % 3
    r = np.zeros(len(d_))
    nm = [f"x{j}" for j in range(X.shape[1])]
    for k in range(3):
        tr, te = f != k, f == k
        if e_[tr].sum() < 5:
            return np.nan
        D = pd.DataFrame(X[tr], columns=nm)
        D["T"], D["E"] = d_[tr], e_[tr]
        try:
            mm = CoxPHFitter(penalizer=1.0).fit(D, "T", "E")
            r[te] = mm.predict_partial_hazard(pd.DataFrame(X[te], columns=nm)).values
        except Exception:
            return np.nan
    return concordance_index(d_, -r, e_)


def task_auc(pos):
    sel = b.DX.isin(["CN", pos]).values
    return (sel, (b.DX.values[sel] == pos).astype(int), b.PTID.values[sel],
            b.age_visit.values[sel].reshape(-1, 1))


pool = b[b.DX == "MCI"]
keep, dur, ev = E.survival(a, pool.PTID.values, ["Dementia"])
idxC = pool.index[pool.PTID.isin(set(keep))].values
_o = {s: i for i, s in enumerate(keep)}
_ii = np.array([_o[s] for s in b.PTID.values[idxC]])
durC, evC = dur[_ii], ev[_ii]
ageC = b.age_visit.values[idxC]


def boot_auc(xA, xB, sel, y, subs, ag, n=N_BOOT):
    rng = np.random.RandomState(0)
    uq = np.unique(subs)
    out = []
    for _ in range(n):
        pick = set(rng.choice(uq, len(uq), replace=True))
        m = np.array([s in pick for s in subs])
        if y[m].sum() < 25 or (1 - y[m]).sum() < 25:
            continue
        try:
            out.append(E.auc_cv(np.column_stack([xA[m], ag[m]]), y[m], subs[m])
                       - E.auc_cv(np.column_stack([xB[m], ag[m]]), y[m], subs[m]))
        except Exception:
            pass
    return np.array(out)


def boot_cox(xA, xB, n=N_COX):
    rng = np.random.RandomState(1)
    out = []
    MA, MB = np.column_stack([xA, ageC]), np.column_stack([xB, ageC])
    for _ in range(n):
        i = rng.randint(0, len(durC), len(durC))
        if evC[i].sum() < 25:
            continue
        p, q = cidx(MA[i], durC[i], evC[i], 0), cidx(MB[i], durC[i], evC[i], 0)
        if np.isfinite(p) and np.isfinite(q):
            out.append(p - q)
    return np.array(out)


def show(lab, d):
    if len(d) < 20:
        print(f"    {lab:44s} too few draws")
        return
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    print(f"    {lab:44s} {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.3f}"
          f" {'REAL' if p < 0.05 else 'not distinguishable'}")


# =================================================================== A
print("=" * 84)
print("A  IS IT JUST THE HIPPOCAMPUS?")
print("=" * 84)
for pos in ["Dementia", "MCI"]:
    sel, y, subs, ag = task_auc(pos)
    print(f"\n  {'AD' if pos == 'Dementia' else 'MCI'} vs CN  n={len(y)}")
    for nm, x in [("brain-age gap", gap), ("hippocampus alone", hipp),
                  ("uniform average", uni)]:
        print(f"    {nm:22s} AUC = "
              f"{E.auc_cv(np.column_stack([x[sel], ag]), y, subs):.3f}")
    show("uniform minus hippocampus alone",
         boot_auc(uni[sel], hipp[sel], sel, y, subs, ag))
    show("hippocampus alone minus gap",
         boot_auc(hipp[sel], gap[sel], sel, y, subs, ag))

print(f"\n  MCI to AD conversion  n={len(durC)}, {int(evC.sum())} events")
for nm, x in [("brain-age gap", gap), ("hippocampus alone", hipp),
              ("uniform average", uni)]:
    M = np.column_stack([x[idxC], ageC])
    print(f"    {nm:22s} C = "
          f"{np.nanmean([cidx(M, durC, evC, s) for s in range(5)]):.3f}")
show("uniform minus hippocampus alone", boot_cox(uni[idxC], hipp[idxC]))
show("hippocampus alone minus gap", boot_cox(hipp[idxC], gap[idxC]))

# =================================================================== B
print("\n" + "=" * 84)
print("B  LEAVE ONE REGION OUT - does it need a particular region?")
print("=" * 84)
sel, y, subs, ag = task_auc("Dementia")
print("  AD vs CN, uniform over 5 regions vs a gap REFITTED on the same 5")
for drop in range(6):
    cols = [j for j in range(6) if j != drop]
    u5 = (Z[:, cols] * SIGN[cols]).mean(1)
    g5 = gap_subset(cols)
    d = boot_auc(u5[sel], g5[sel], sel, y, subs, ag, n=250)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    print(f"    without {FA[V[drop]]:14s} "
          f"uniform {E.auc_cv(np.column_stack([u5[sel], ag]), y, subs):.3f}"
          f"  gap {E.auc_cv(np.column_stack([g5[sel], ag]), y, subs):.3f}"
          f"  diff {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.3f}"
          f" {'HOLDS' if p < 0.05 and d.mean() > 0 else '-'}")

# =================================================================== C
print("\n" + "=" * 84)
print("C  DOES THE ADVANTAGE GROW WITH THE NUMBER OF REGIONS?")
print("=" * 84)
print("  every subset of each size; the gap is refitted on the SAME subset")
for k in range(2, 7):
    advs = []
    for cols in itertools.combinations(range(6), k):
        cols = list(cols)
        u = (Z[:, cols] * SIGN[cols]).mean(1)
        g = gap_subset(cols)
        advs.append(E.auc_cv(np.column_stack([u[sel], ag]), y, subs)
                    - E.auc_cv(np.column_stack([g[sel], ag]), y, subs))
    advs = np.array(advs)
    print(f"    {k} regions ({len(advs):2d} subsets)  "
          f"mean advantage {advs.mean():+.4f}"
          f"   range [{advs.min():+.4f}, {advs.max():+.4f}]"
          f"   positive in {100*(advs > 0).mean():.0f}% of subsets")

# =================================================================== E
print("\n" + "=" * 84)
print("E  HOW MUCH IS LEFT ON THE TABLE?")
print("=" * 84)
for pos in ["Dementia", "MCI"]:
    sel2, y2, subs2, ag2 = task_auc(pos)
    a_u = E.auc_cv(np.column_stack([uni[sel2], ag2]), y2, subs2)
    a_s = E.auc_cv(np.column_stack([Z[sel2], ag2]), y2, subs2)
    print(f"  {'AD' if pos == 'Dementia' else 'MCI'} vs CN   "
          f"uniform {a_u:.3f}   supervised on all six {a_s:.3f}"
          f"   headroom {a_s - a_u:+.3f}")
Mu = np.column_stack([uni[idxC], ageC])
Ms = np.column_stack([Z[idxC], ageC])
cu = np.nanmean([cidx(Mu, durC, evC, s) for s in range(5)])
cs = np.nanmean([cidx(Ms, durC, evC, s) for s in range(5)])
print(f"  conversion   uniform {cu:.3f}   supervised on all six {cs:.3f}"
      f"   headroom {cs - cu:+.3f}")
