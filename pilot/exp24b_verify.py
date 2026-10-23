#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stress-test the one result in Part II that is positive.

exp24 says the RATE of the tabular brain-age gap predicts MCI -> dementia
conversion better than the gap's value (+0.054 C-index) and better than the rate
of hippocampal atrophy (+0.045). That replicates E8, so it is exactly the claim
most worth trying to break before it goes in a report.

The confidence intervals in exp24 come from repeated cross-validation, which
measures split noise and not sampling noise - with 207 subjects the real
uncertainty is dominated by WHICH subjects we have. Three checks here:

  1  bootstrap over SUBJECTS, the honest interval for the difference
  2  a sweep over the landmark L, because a result that exists at exactly one
     choice of an arbitrary parameter is a result about that choice
  3  a label permutation, to confirm the pipeline does not manufacture signal
"""

import os
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

import exp24_image_rate as E

warnings.filterwarnings("ignore")
OUT = E.OUT
N_BOOT = 400


def build(L, d, hist, scorers_cols):
    w = d[d.Years_bl <= L + 0.25]
    f = {}
    for nm in scorers_cols:
        f[f"rate_{nm}"] = E.slopes(w, f"gap_{nm}")
        f[f"value_{nm}"] = w.sort_values("Years_bl").groupby("PTID")[f"gap_{nm}"].first()
    w2 = w.copy()
    w2["hipp"] = w2.Hippocampus / w2.ICV
    f["rate_hipp"] = E.slopes(w2, "hipp") * 1e3
    f["value_hipp"] = w2.sort_values("Years_bl").groupby("PTID").hipp.first() * 1e3
    F = pd.DataFrame(f).dropna()

    fut = hist[hist.PTID.isin(F.index) & (hist.Years_bl > L)]
    conv = fut[fut.DX == "Dementia"].groupby("PTID").Years_bl.min()
    last = fut.groupby("PTID").Years_bl.max()
    sub = sorted(set(last.index) & set(F.index))
    F = F.loc[sub]
    dur = np.array([conv[s] - L if s in conv.index else last[s] - L for s in sub])
    ev = np.array([1 if s in conv.index else 0 for s in sub])
    ok = dur > 0
    F, dur, ev = F[ok], dur[ok], ev[ok]
    F = F.copy()
    F["__age"] = d.drop_duplicates("PTID").set_index("PTID").AGE.loc[F.index].values
    return F, dur, ev


def cind(F, cols, dur, ev, seed=0):
    """One 5-fold CV C-index."""
    n = len(dur)
    r = np.random.RandomState(seed)
    fold = r.permutation(n) % 5
    risk = np.zeros(n)
    for k in range(5):
        tr, te = fold != k, fold == k
        if ev[tr].sum() < 5:
            return np.nan
        D = F[cols][tr].copy()
        D["T"], D["E"] = dur[tr], ev[tr]
        try:
            cph = CoxPHFitter(penalizer=0.1).fit(D, "T", "E")
            risk[te] = cph.predict_partial_hazard(F[cols][te]).values
        except Exception:
            return np.nan
    return concordance_index(dur, -risk, ev)


def main():
    X, m, hist = E.load()
    complete = (m.DX.notna() & m.ICV.notna() & m[E.V].notna().all(1)
                & m.age_visit.notna())
    m = m[complete].reset_index(drop=True)
    X = X[np.where(complete.values)[0]]
    tab = np.column_stack([m[E.V].values, m.ICV.values, m.SEX.values])

    cn = m[m.DX == "CN"].drop_duplicates("PTID")
    icn = cn.index.values
    sc = {nm: E.fit_age_model(X[icn], tab[icn], cn.age_visit.values, ui, ut)
          for nm, ui, ut in [("TAB", False, True), ("IMG", True, False)]}

    bl = m[m.VISCODE == "bl"][["PTID", "DX"]]
    mci = set(bl[bl.DX == "MCI"].PTID)
    d = m[m.PTID.isin(mci)].copy()
    di = np.where(m.PTID.isin(mci).values)[0]
    for nm, fn in sc.items():
        d[f"gap_{nm}"] = fn(X[di], tab[di], d.age_visit.values)

    A, B, C = ["rate_TAB", "__age"], ["value_TAB", "__age"], ["rate_hipp", "__age"]

    # ---------------------------------------------------------- 2  landmark sweep
    print("=" * 74)
    print("LANDMARK SWEEP - is the result specific to L = 2.0?")
    print("=" * 74)
    print(f"  {'L':>5} {'n':>5} {'ev':>4}  {'rate':>6} {'value':>6} {'hipp':>6}   "
          f"{'rate-value':>11} {'rate-hipp':>10}")
    keep = {}
    for L in (1.0, 1.5, 2.0, 2.5, 3.0):
        F, dur, ev = build(L, d, hist, sc)
        if ev.sum() < 25:
            print(f"  {L:5.1f} {len(F):5d} {int(ev.sum()):4d}   too few events")
            continue
        r = np.nanmean([cind(F, A, dur, ev, s) for s in range(12)])
        v = np.nanmean([cind(F, B, dur, ev, s) for s in range(12)])
        h = np.nanmean([cind(F, C, dur, ev, s) for s in range(12)])
        print(f"  {L:5.1f} {len(F):5d} {int(ev.sum()):4d}  {r:6.3f} {v:6.3f} {h:6.3f}   "
              f"{r-v:+11.3f} {r-h:+10.3f}")
        keep[L] = (F, dur, ev)

    # ---------------------------------------------------- 1  bootstrap over subjects
    L = 2.0
    F, dur, ev = keep[L]
    print("\n" + "=" * 74)
    print(f"BOOTSTRAP OVER SUBJECTS at L={L}  ({N_BOOT} resamples, n={len(F)})")
    print("=" * 74)
    dv, dh = [], []
    rng = np.random.RandomState(0)
    for b in range(N_BOOT):
        ix = rng.randint(0, len(F), len(F))
        Fb, db, eb = F.iloc[ix].reset_index(drop=True), dur[ix], ev[ix]
        if eb.sum() < 20:
            continue
        r = cind(Fb, A, db, eb, b)
        v = cind(Fb, B, db, eb, b)
        h = cind(Fb, C, db, eb, b)
        if np.isfinite(r) and np.isfinite(v):
            dv.append(r - v)
        if np.isfinite(r) and np.isfinite(h):
            dh.append(r - h)
    for lbl, a in [("rate - value (tabular)", np.array(dv)),
                   ("rate - hippocampus rate", np.array(dh))]:
        lo, hi = np.percentile(a, [2.5, 97.5])
        print(f"  {lbl:26s} {a.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"P(>0) = {(a > 0).mean():.3f}"
              f"{'   *** holds' if lo > 0 else '   <- CROSSES ZERO'}")

    # ---------------------------------------------------------- 3  permutation
    print("\n" + "=" * 74)
    print("PERMUTATION - does the pipeline invent signal from nothing?")
    print("=" * 74)
    null = []
    rng = np.random.RandomState(1)
    for b in range(150):
        p = rng.permutation(len(F))
        null.append(cind(F, A, dur[p], ev[p], b))
    null = np.array([x for x in null if np.isfinite(x)])
    obs = np.nanmean([cind(F, A, dur, ev, s) for s in range(12)])
    print(f"  observed C = {obs:.3f}")
    print(f"  permuted   = {null.mean():.3f} "
          f"[{np.percentile(null,2.5):.3f}, {np.percentile(null,97.5):.3f}]")
    print(f"  p = {(null >= obs).mean():.4f}")


if __name__ == "__main__":
    main()
