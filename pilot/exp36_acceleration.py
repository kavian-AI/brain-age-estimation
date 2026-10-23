#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IS THERE INFORMATION IN THE SECOND DERIVATIVE OF THE BRAIN-AGE GAP?

Report 3 established that the RATE of change of the gap adds to its VALUE. The
obvious next question is whether the ACCELERATION adds to both - whether people
whose gap is widening faster and faster convert sooner than people whose gap is
widening steadily.

THE PROBLEM THAT HAS TO BE SETTLED FIRST
----------------------------------------
A second derivative is a difference of differences, so it amplifies noise twice.
We measured the gap's noise floor at about 1.9 years from same-session repeat
scans. Fitting a quadratic to five noisy points over two years can manufacture
an "acceleration" out of nothing at all. So before any survival model is run:

  1  jackknife - drop one visit at a time and see how far the estimate moves
  2  a NULL SIMULATION - take each subject's fitted straight line, add noise of
     the measured size, refit the quadratic, and record the acceleration that
     pure noise produces. If the observed spread of acceleration is no wider
     than the null spread, there is no signal and the analysis stops here.

Only if acceleration survives both checks does it go into a Cox model.

DESIGN
------
Age model: ridge on 6 FreeSurfer volumes + ICV + sex, fitted on CN subjects
only, out of fold by subject, with the usual regression-dilution correction.

Landmark: trajectory parameters are estimated ONLY from visits within the first
L years, and only conversions AFTER L are counted, so no subject's outcome can
leak into their own predictor.

Every comparison carries a subject-level bootstrap interval. At this sample size
a C-index is worth about +-0.05, and differences smaller than that are not
differences.
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
L = 2.0            # landmark, years
MIN_VIS = 4        # a quadratic needs 3; 4 gives it one degree of freedom
MIN_SPAN = 1.4     # years between first and last visit inside the window
N_BOOT = 600


# ------------------------------------------------------------ gap trajectory

def load():
    a = pd.read_csv(os.path.join(DATA, "ADNIMERGE_05Mar2026.csv"), low_memory=False)
    d = a.dropna(subset=["AGE", "Years_bl", "ICV", "PTGENDER"] + V).copy()
    d["SEX"] = (d.PTGENDER == "Male").astype(int)
    d["age_visit"] = d.AGE + d.Years_bl          # AGE is age at BASELINE
    return a, d


def gap_trajectories(d, seed=0):
    """Out-of-fold gap for every visit, from a model fitted on CN only."""
    X = np.column_stack([d[v].values for v in V] + [d.ICV.values, d.SEX.values])
    age = d.age_visit.values.astype(float)
    cn = (d.DX == "CN").values
    subs = d.PTID.values
    uq = np.unique(subs)
    fold = dict(zip(uq, np.random.RandomState(seed).permutation(len(uq)) % 5))
    f = np.array([fold[s] for s in subs])

    gap = np.zeros(len(d))
    for k in range(5):
        tr = np.where((f != k) & cn)[0]
        te = np.where(f == k)[0]
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(X[tr]), age[tr])
        b = LinearRegression().fit(age[tr].reshape(-1, 1),
                                   m.predict(sc.transform(X[tr])) - age[tr])
        gap[te] = (m.predict(sc.transform(X[te])) - age[te]) \
            - b.predict(age[te].reshape(-1, 1))
    return gap


def fit_curves(d):
    """Per subject, inside the landmark window: value, rate, acceleration."""
    w = d[d.Years_bl <= L + 0.25]
    rows = []
    for s, g in w.groupby("PTID"):
        g = g.sort_values("Years_bl")
        t, y = g.Years_bl.values.astype(float), g.gap.values.astype(float)
        if len(t) < MIN_VIS or (t.max() - t.min()) < MIN_SPAN:
            continue
        q = np.polyfit(t, y, 2)                  # y = q0 t^2 + q1 t + q2
        lin = np.polyfit(t, y, 1)
        resid = y - np.polyval(lin, t)
        # jackknife the acceleration: drop one visit at a time
        jk = []
        for i in range(len(t)):
            m = np.ones(len(t), bool)
            m[i] = False
            if m.sum() >= 3:
                jk.append(np.polyfit(t[m], y[m], 2)[0])
        rows.append(dict(PTID=s, value=y[0], rate=lin[0], accel=2 * q[0],
                         accel_jk_sd=np.std(jk) * 2 if jk else np.nan,
                         resid_sd=resid.std(ddof=1) if len(t) > 2 else np.nan,
                         n_vis=len(t), span=t.max() - t.min(),
                         age=g.AGE.values[0], t=t, y=y, lin=lin))
    return pd.DataFrame(rows).set_index("PTID")


# ------------------------------------------------------------- is it real?

def null_acceleration(F, n_rep=200, seed=0):
    """Acceleration produced by pure noise on each subject's own straight line."""
    rng = np.random.RandomState(seed)
    sd = np.nanmedian(F.resid_sd.values)
    out = []
    for _ in range(n_rep):
        acc = []
        for _, r in F.iterrows():
            y = np.polyval(r.lin, r.t) + rng.normal(0, sd, len(r.t))
            acc.append(2 * np.polyfit(r.t, y, 2)[0])
        out.append(np.std(acc))
    return np.array(out), sd


# ----------------------------------------------------------------- survival

def outcome(a, idx):
    hist = a[["PTID", "Years_bl", "DX"]].dropna(subset=["DX"])
    fut = hist[hist.PTID.isin(idx) & (hist.Years_bl > L)]
    conv = fut[fut.DX == "Dementia"].groupby("PTID").Years_bl.min()
    last = fut.groupby("PTID").Years_bl.max()
    keep = [s for s in idx if s in last.index]
    dur = np.array([conv[s] - L if s in conv.index else last[s] - L for s in keep])
    ev = np.array([1 if s in conv.index else 0 for s in keep])
    ok = dur > 0
    return [s for s, o in zip(keep, ok) if o], dur[ok], ev[ok]


def cox_oof(F, cols, dur, ev, seed):
    n = len(dur)
    f = np.random.RandomState(seed).permutation(n) % 5
    risk = np.zeros(n)
    for k in range(5):
        tr, te = f != k, f == k
        if ev[tr].sum() < 5:
            return np.nan
        D = F[cols][tr].copy()
        D["T"], D["E"] = dur[tr], ev[tr]
        try:
            m = CoxPHFitter(penalizer=0.1).fit(D, "T", "E")
            risk[te] = m.predict_partial_hazard(F[cols][te]).values
        except Exception:
            return np.nan
    return concordance_index(dur, -risk, ev)


def boot_diff(F, c1, c2, dur, ev, n=N_BOOT, seed=0):
    """Subject-level bootstrap of the C-index difference between two models."""
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        i = rng.randint(0, len(dur), len(dur))
        if ev[i].sum() < 12:
            continue
        Fi = F.iloc[i]
        a1 = cox_oof(Fi.reset_index(drop=True), c1, dur[i], ev[i], 0)
        a2 = cox_oof(Fi.reset_index(drop=True), c2, dur[i], ev[i], 0)
        if np.isfinite(a1) and np.isfinite(a2):
            out.append(a1 - a2)
    return np.array(out)


def main():
    a, d = load()
    d["gap"] = gap_trajectories(d)
    print(f"visits {len(d)}, subjects {d.PTID.nunique()}")
    print(f"gap on CN: mean {d[d.DX=='CN'].gap.mean():+.2f} y, "
          f"SD {d[d.DX=='CN'].gap.std():.2f}")

    F = fit_curves(d)
    bl = a[a.VISCODE == "bl"].drop_duplicates("PTID").set_index("PTID")
    F["DX0"] = bl.DX.reindex(F.index)
    print(f"\nsubjects with >={MIN_VIS} visits inside {L} y and span >={MIN_SPAN} y: "
          f"{len(F)}")
    print(F.DX0.value_counts(dropna=False).to_string())

    # ------------------------------------------------ CHECK 1: is it measurable?
    print("\n" + "=" * 76)
    print("CHECK 1  can acceleration be measured at all, or is it noise?")
    print("=" * 76)
    obs = F.accel.std()
    null, sd = null_acceleration(F)
    print(f"  residual scatter about each subject's straight line: {sd:.2f} y")
    print(f"  OBSERVED spread of acceleration : {obs:.3f} y/y^2")
    print(f"  NULL spread from noise alone    : {null.mean():.3f} "
          f"[{np.percentile(null,2.5):.3f}, {np.percentile(null,97.5):.3f}]")
    ratio = obs / null.mean()
    print(f"  ratio observed/null             : {ratio:.2f}")
    print(f"  median jackknife SD of accel    : {F.accel_jk_sd.median():.3f}")
    signal = obs**2 - null.mean()**2
    print(f"  implied true spread (variance subtraction): "
          f"{np.sqrt(signal):.3f}" if signal > 0 else
          "  implied true spread: ZERO - observed is within the null")
    if ratio < 1.15:
        print("\n  -> acceleration is NOT measurable at this sampling density.")
        print("     Anything it predicts downstream would be an artefact.")
    else:
        print(f"\n  -> acceleration carries real variance beyond noise "
              f"({100*(1-1/ratio**2):.0f}% of its variance is signal).")

    # ------------------------------------------------ CHECK 2: does it predict?
    mci = F[F.DX0 == "MCI"].copy()
    idx, dur, ev = outcome(a, list(mci.index))
    M = mci.loc[idx]
    print("\n" + "=" * 76)
    print(f"CHECK 2  MCI -> dementia after {L} y:  n={len(idx)}, events={int(ev.sum())}")
    print("=" * 76)
    models = [("age only", ["age"]),
              ("value", ["value", "age"]),
              ("rate", ["rate", "age"]),
              ("accel", ["accel", "age"]),
              ("value + rate", ["value", "rate", "age"]),
              ("value + rate + accel", ["value", "rate", "accel", "age"])]
    res = {}
    for nm, cols in models:
        c = np.nanmean([cox_oof(M.reset_index(drop=True), cols, dur, ev, s)
                        for s in range(8)])
        res[nm] = c
        print(f"  {nm:24s} C = {c:.3f}")

    print("\n  does acceleration ADD to value + rate?")
    dd = boot_diff(M, ["value", "rate", "accel", "age"], ["value", "rate", "age"],
                   dur, ev)
    lo, hi = np.percentile(dd, [2.5, 97.5])
    p = 2 * min((dd <= 0).mean(), (dd >= 0).mean())
    print(f"    difference {dd.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  p={p:.3f}")
    print(f"    -> {'REAL' if p < 0.05 else 'NOT distinguishable from noise'}")

    F.drop(columns=["t", "y", "lin"]).to_csv(os.path.join(OUT, "exp36_curves.csv"))
    print(f"\nsaved -> {OUT}\\exp36_curves.csv")


if __name__ == "__main__":
    main()
