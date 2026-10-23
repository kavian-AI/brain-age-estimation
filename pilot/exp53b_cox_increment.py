#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
E53 PART B, ENDPOINT 2, THE ONE CONTRAST THAT WOULD NOT CONVERGE.

The Cox bootstrap inside exp53 returned nothing: with six correlated components
plus age, duplicated rows in a bootstrap draw make the partial likelihood
ill-conditioned and lifelines gives up. Fixed here by raising the penalty and by
scoring with a single fold split instead of averaging several, which is all a
bootstrap contrast needs since both arms see the identical split.

Only two questions, so it finishes:
    does k=6 beat k=1?      i.e. is PROGNOSIS more than one-dimensional?
    does k=2 beat k=1?
The classification endpoint already answered its own version inside exp53.
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

N_BOOT = 250


def c_index(X, dur, ev, seed=0):
    n = len(dur)
    f = np.random.RandomState(seed).permutation(n) % 3
    r = np.zeros(n)
    nm = [f"x{j}" for j in range(X.shape[1])]
    for k in range(3):
        tr, te = f != k, f == k
        if ev[tr].sum() < 5:
            return np.nan
        D = pd.DataFrame(X[tr], columns=nm)
        D["T"], D["E"] = dur[tr], ev[tr]
        try:
            m = CoxPHFitter(penalizer=1.0).fit(D, "T", "E")
            r[te] = m.predict_partial_hazard(pd.DataFrame(X[te], columns=nm)).values
        except Exception:
            return np.nan
    return concordance_index(dur, -r, ev)


def main():
    a, b = E.load()
    Z = E.normative_z(b)
    gap = E.brain_age_gap(b)
    pool = b[b.DX == "MCI"]
    keep, dur, ev = E.survival(a, pool.PTID.values, ["Dementia"])
    idx = pool.index[pool.PTID.isin(set(keep))].values
    o = {s: i for i, s in enumerate(keep)}
    ii = np.array([o[s] for s in b.PTID.values[idx]])
    dur, ev = dur[ii], ev[ii]
    P6, _ = E.rotate(Z[b.DX.values == "CN"], Z[idx], 6)
    age = b.age_visit.values[idx]

    arms = {"gap": np.column_stack([gap[idx], age]),
            "k=1": np.column_stack([P6[:, :1], age]),
            "k=2": np.column_stack([P6[:, :2], age]),
            "k=6": np.column_stack([P6, age])}
    print(f"MCI n={len(dur)}, conversions={int(ev.sum())}\n")
    print("point estimates, mean over 5 fold seeds:")
    for k, X in arms.items():
        print(f"  {k:5s} C = {np.nanmean([c_index(X, dur, ev, s) for s in range(5)]):.3f}")

    rng = np.random.RandomState(0)
    pairs = [("k=1", "gap"), ("k=2", "k=1"), ("k=6", "k=1")]
    res = {p: [] for p in pairs}
    ok = 0
    for _ in range(N_BOOT):
        i = rng.randint(0, len(dur), len(dur))
        if ev[i].sum() < 25:
            continue
        v = {k: c_index(X[i], dur[i], ev[i], 0) for k, X in arms.items()}
        if any(not np.isfinite(x) for x in v.values()):
            continue
        ok += 1
        for p in pairs:
            res[p].append(v[p[0]] - v[p[1]])
    print(f"\nbootstrap draws that converged: {ok}/{N_BOOT}")
    for (A, B) in pairs:
        d = np.array(res[(A, B)])
        if len(d) < 20:
            print(f"  {A} minus {B}: too few")
            continue
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        print(f"  {A:4s} minus {B:4s}  {d.mean():+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]  p={p:.3f}  "
              f"{'REAL' if p < 0.05 else 'not distinguishable'}")


if __name__ == "__main__":
    main()
