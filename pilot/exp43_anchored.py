#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ANCHORING THE NORMATIVE MODEL TO THE PATIENT'S OWN BASELINE SCAN.

E42 ran the model backwards - age -> expected brain - and found it essentially
tied with the forward brain-age gap. Both ask the same population question: what
should a healthy person of this age look like? The answer is dominated by
between-person variation in head size, shape and reserve, none of which is
disease.

This asks a within-person question instead: given THIS patient's own baseline
scan and the time elapsed, what should their brain look like now? Everything
stable about that individual cancels, and what remains is their own change.

    population :  volume(t)  ~  age, sex, ICV
    anchored   :  volume(t)  ~  volume(baseline), elapsed time, age, sex, ICV

Both fitted on CN only, out of fold by subject, so neither has seen the person
it is judging.

ARMS, all predicting conversion AFTER the follow-up visit (landmark design, so a
subject's own outcome cannot leak into their predictor):

    population z      six deviations from the age-matched norm
    anchored z        six deviations from the person's own predicted trajectory
    raw change        the plain difference in volume, no model at all
    brain-age gap     the classical scalar, for reference
    gap rate          its slope, which has been the strongest arm so far

The raw-change arm matters: if anchoring adds nothing over simply subtracting the
baseline, then the model is decoration.
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
L = 2.0          # landmark: features from <= L years, outcome after L
FOLDS, N_BOOT = 5, 400


def load():
    a = pd.read_csv(os.path.join(DATA, "ADNIMERGE_05Mar2026.csv"), low_memory=False)
    d = a.dropna(subset=["AGE", "Years_bl", "ICV", "PTGENDER"] + V).copy()
    d["SEX"] = (d.PTGENDER == "Male").astype(int)
    d["age_visit"] = d.AGE + d.Years_bl
    return a, d.reset_index(drop=True)


def make_pairs(d):
    """Baseline plus the visit closest to L years, per subject."""
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
            r["b_" + v] = b[v]
            r["f_" + v] = f[v]
        rows.append(r)
    P = pd.DataFrame(rows)
    return P[(P.dt >= 0.8)].reset_index(drop=True)


def fit_z(P, anchored, seed=0):
    """Out-of-fold z-scores from a CN-trained model."""
    cn = (P.DX0 == "CN").values
    uq = np.unique(P.PTID.values)
    fm = dict(zip(uq, np.random.RandomState(seed).permutation(len(uq)) % FOLDS))
    f = np.array([fm[s] for s in P.PTID.values])
    Z = np.zeros((len(P), len(V)))
    for j, v in enumerate(V):
        y = P["f_" + v].values
        base = [P.age.values, P.SEX.values, P.ICV.values, P.dt.values]
        if anchored:
            base = [P["b_" + v].values] + base
        X = np.column_stack(base)
        for k in range(FOLDS):
            tr = np.where((f != k) & cn)[0]
            te = np.where(f == k)[0]
            sc = StandardScaler().fit(X[tr])
            m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(X[tr]), y[tr])
            res = y[tr] - m.predict(sc.transform(X[tr]))
            sd = res.std(ddof=1)
            Z[te, j] = (y[te] - m.predict(sc.transform(X[te]))) / max(sd, 1e-9)
    return Z


def gap_arms(P, seed=0):
    """Classical scalar gap at baseline and follow-up, and its rate."""
    cn = (P.DX0 == "CN").values
    uq = np.unique(P.PTID.values)
    fm = dict(zip(uq, np.random.RandomState(seed).permutation(len(uq)) % FOLDS))
    f = np.array([fm[s] for s in P.PTID.values])
    gb = np.zeros(len(P))
    gf = np.zeros(len(P))
    Xb = np.column_stack([P["b_" + v].values for v in V]
                         + [P.ICV.values, P.SEX.values])
    Xf = np.column_stack([P["f_" + v].values for v in V]
                         + [P.ICV.values, P.SEX.values])
    ab, af = P.age.values, P.age.values + P.dt.values
    for k in range(FOLDS):
        tr = np.where((f != k) & cn)[0]
        te = np.where(f == k)[0]
        sc = StandardScaler().fit(Xb[tr])
        m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(Xb[tr]), ab[tr])
        b = LinearRegression().fit(ab[tr].reshape(-1, 1),
                                   m.predict(sc.transform(Xb[tr])) - ab[tr])
        gb[te] = (m.predict(sc.transform(Xb[te])) - ab[te]) \
            - b.predict(ab[te].reshape(-1, 1))
        gf[te] = (m.predict(sc.transform(Xf[te])) - af[te]) \
            - b.predict(af[te].reshape(-1, 1))
    return gb, gf, (gf - gb) / P.dt.values


def outcome(a, ids, t_follow):
    hist = a[["PTID", "Years_bl", "DX"]].dropna(subset=["DX"])
    conv = hist[hist.DX == "Dementia"].groupby("PTID").Years_bl.min()
    last = hist.groupby("PTID").Years_bl.max()
    dur, ev, keep = [], [], []
    for s, tf in zip(ids, t_follow):
        if s not in last.index:
            continue
        if s in conv.index:
            t = conv[s] - tf
            e = 1
        else:
            t = last[s] - tf
            e = 0
        if t > 0:
            dur.append(t)
            ev.append(e)
            keep.append(s)
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


def main():
    a, d = load()
    P = make_pairs(d)
    print(f"subjects with a baseline and a ~{L:.0f}-year follow-up: {len(P)}")
    print(P.DX0.value_counts(dropna=False).to_string())
    print(f"median interval {P.dt.median():.2f} y")

    Zpop = fit_z(P, anchored=False)
    Zanc = fit_z(P, anchored=True)
    gb, gf, rate = gap_arms(P)

    # how much of the follow-up volume does each model explain in CONTROLS?
    cn = (P.DX0 == "CN").values
    print("\n" + "=" * 76)
    print("HOW WELL EACH MODEL PREDICTS A CONTROL'S FOLLOW-UP VOLUME")
    print("=" * 76)
    print(f"  {'region':16s} {'population':>12} {'anchored':>10} {'gain':>8}")
    for j, v in enumerate(V):
        y = P["f_" + v].values
        sd = np.std(y)
        c1 = np.corrcoef(y[cn] - Zpop[cn, j] * sd, y[cn])[0, 1]
        c2 = np.corrcoef(y[cn] - Zanc[cn, j] * sd, y[cn])[0, 1]
        print(f"  {FA[v]:16s} {c1:12.3f} {c2:10.3f} {c2 - c1:+8.3f}")

    # Work with POSITIONS IN P throughout, so the z matrices, the gap arms and
    # the outcome vector can never fall out of alignment. The earlier version
    # indexed three different ways in three lines and would have silently
    # mismatched rows.
    pos_mci = np.where(P.DX0.values == "MCI")[0]
    keep, dur, ev = outcome(a, P.PTID.values[pos_mci], P.t_follow.values[pos_mci])
    keep_set = set(keep)
    pos = np.array([p for p in pos_mci if P.PTID.values[p] in keep_set])
    order = {s: i for i, s in enumerate(keep)}
    ii = np.array([order[P.PTID.values[p]] for p in pos])
    dur, ev = dur[ii], ev[ii]

    F = pd.DataFrame(dict(age=P.age.values[pos]))
    for j, v in enumerate(V):
        F["pop_" + v] = Zpop[pos, j]
        F["anc_" + v] = Zanc[pos, j]
        F["raw_" + v] = ((P["f_" + v].values[pos] - P["b_" + v].values[pos])
                         / P.dt.values[pos])
    F["gap"] = gb[pos]
    F["rate"] = rate[pos]

    print("\n" + "=" * 76)
    print(f"MCI -> dementia AFTER the follow-up visit: n={len(dur)}, "
          f"events={int(ev.sum())}")
    print("=" * 76)
    arms = [("age only", ["age"]),
            ("brain-age gap", ["gap", "age"]),
            ("gap rate", ["rate", "age"]),
            ("raw change (6)", [f"raw_{v}" for v in V] + ["age"]),
            ("population z (6)", [f"pop_{v}" for v in V] + ["age"]),
            ("anchored z (6)", [f"anc_{v}" for v in V] + ["age"])]
    res = {}
    for nm, cols in arms:
        c = np.nanmean([cox_c(F, cols, dur, ev, s) for s in range(8)])
        res[nm] = c
        print(f"  {nm:20s} C = {c:.3f}")

    # the comparison the experiment exists for
    rng = np.random.RandomState(0)
    for a_name, b_name in [("anchored z (6)", "population z (6)"),
                           ("anchored z (6)", "raw change (6)")]:
        ca = dict(arms)[a_name]
        cb = dict(arms)[b_name]
        diffs = []
        for _ in range(N_BOOT):
            i = rng.randint(0, len(dur), len(dur))
            if ev[i].sum() < 12:
                continue
            Fi = F.iloc[i].reset_index(drop=True)
            x = cox_c(Fi, ca, dur[i], ev[i], 0)
            y = cox_c(Fi, cb, dur[i], ev[i], 0)
            if np.isfinite(x) and np.isfinite(y):
                diffs.append(x - y)
        diffs = np.array(diffs)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
        print(f"\n  {a_name} minus {b_name}:")
        print(f"    {diffs.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  p={p:.3f}"
              f"  -> {'REAL' if p < 0.05 else 'not distinguishable'}")


if __name__ == "__main__":
    main()
