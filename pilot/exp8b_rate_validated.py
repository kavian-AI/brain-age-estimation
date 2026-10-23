#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Re-validate E8, the report's headline result, with honest uncertainty.

E8 reported that adding the RATE of brain ageing to a direct snapshot model
raises the C-index from 0.772 to 0.820 (+0.048), on n=1100 with 160 conversions.
Two problems with how that number was produced, both found while testing the
image version of the same experiment:

  1  NO CROSS-VALIDATION. exp8_rate.py fits the Cox model on the whole cohort
     and then computes the C-index on that same cohort. Adding any variable to a
     Cox model can only improve in-sample fit, so an in-sample "gain" is not
     evidence that the variable helps on new patients.

  2  NO CONFIDENCE INTERVAL. On the image-matched subsample (n=207) the same
     comparison gave +0.047 - almost exactly E8's point estimate - but a
     subject-level bootstrap put the 95% interval at [-0.020, +0.115], crossing
     zero. Point estimates without intervals cannot distinguish those cases.

This script repeats E8 on the full tabular cohort with:
  - out-of-fold C-index (5-fold, one row per subject)
  - bootstrap over subjects for every reported difference
  - a permutation check
  - the fair baseline E8 never ran: the rate of hippocampal atrophy. If the
    brain-age rate only tracks the hippocampus shrinking, then measuring the
    hippocampus twice is the cheaper instrument and the brain-age machinery
    earns nothing.
"""

import os
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "pilot")
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
LANDMARK = 2.0
MIN_SCANS = 2
N_BOOT = 500


def cindex_oof(F, cols, dur, ev, seed):
    n = len(dur)
    fold = np.random.RandomState(seed).permutation(n) % 5
    risk = np.zeros(n)
    for k in range(5):
        tr, te = fold != k, fold == k
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


def cindex_insample(F, cols, dur, ev):
    D = F[cols].copy()
    D["T"], D["E"] = dur, ev
    m = CoxPHFitter(penalizer=0.1).fit(D, "T", "E")
    return concordance_index(dur, -m.predict_partial_hazard(F[cols]).values, ev)


def slope(t, v):
    ok = np.isfinite(t) & np.isfinite(v)
    if ok.sum() < MIN_SCANS or np.ptp(t[ok]) < 0.5:
        return np.nan
    return np.polyfit(t[ok], v[ok], 1)[0]


def main():
    a = pd.read_csv(os.path.join(DATA, "ADNIMERGE_05Mar2026.csv"), low_memory=False)
    need = ["PTID", "VISCODE", "AGE", "PTGENDER", "DX", "ICV", "Years_bl"] + V
    d = a[need].dropna(subset=["AGE", "PTGENDER", "ICV", "Years_bl"] + V).copy()
    d["age_visit"] = d.AGE + d.Years_bl
    d["SEX"] = (d.PTGENDER == "Male").astype(int)
    print(f"visits with complete volumetrics: {len(d):,} "
          f"from {d.PTID.nunique():,} subjects")

    # age model on CN subjects only, one scan each, disjoint from the MCI cohort
    cn = d[d.DX == "CN"].drop_duplicates("PTID")
    Xc = np.column_stack([cn[V].values, cn.ICV.values, cn.SEX.values])
    sc = StandardScaler().fit(Xc)
    mdl = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(Xc),
                                                     cn.age_visit.values)
    bias = LinearRegression().fit(
        cn.age_visit.values.reshape(-1, 1),
        mdl.predict(sc.transform(Xc)) - cn.age_visit.values)
    print(f"age model trained on {len(cn)} CN subjects")

    Xa = np.column_stack([d[V].values, d.ICV.values, d.SEX.values])
    d["gap"] = ((mdl.predict(sc.transform(Xa)) - d.age_visit.values)
                - bias.predict(d.age_visit.values.reshape(-1, 1)))
    d["hipp"] = 1e3 * d.Hippocampus / d.ICV

    bl = d[d.VISCODE == "bl"][["PTID", "DX"]].rename(columns={"DX": "DX_bl"})
    mci = set(bl[bl.DX_bl == "MCI"].PTID)
    w = d[d.PTID.isin(mci) & (d.Years_bl <= LANDMARK + 0.25)]
    hist = a[["PTID", "Years_bl", "DX"]].dropna(subset=["DX"])

    rows = []
    for s, g in w.groupby("PTID"):
        g = g.sort_values("Years_bl")
        r = slope(g.Years_bl.values, g.gap.values)
        rh = slope(g.Years_bl.values, g.hipp.values)
        if not np.isfinite(r) or not np.isfinite(rh):
            continue
        rows.append(dict(PTID=s, rate=r, value=g.gap.values[0],
                         rate_hipp=rh, value_hipp=g.hipp.values[0],
                         age=g.AGE.values[0]))
    F = pd.DataFrame(rows).set_index("PTID")

    fut = hist[hist.PTID.isin(F.index) & (hist.Years_bl > LANDMARK)]
    conv = fut[fut.DX == "Dementia"].groupby("PTID").Years_bl.min()
    last = fut.groupby("PTID").Years_bl.max()
    sub = sorted(set(last.index) & set(F.index))
    F = F.loc[sub]
    dur = np.array([conv[s] - LANDMARK if s in conv.index else last[s] - LANDMARK
                    for s in sub])
    ev = np.array([1 if s in conv.index else 0 for s in sub])
    ok = dur > 0
    F, dur, ev = F[ok], dur[ok], ev[ok]
    print(f"\nlandmark cohort: n={len(F)}, conversions after "
          f"{LANDMARK} y = {int(ev.sum())}")
    print(f"converters' rate {F.rate[ev == 1].mean():+.2f} vs "
          f"non-converters {F.rate[ev == 0].mean():+.2f} gap-years per year")

    arms = {
        "age only": ["age"],
        "hippocampus value": ["value_hipp", "age"],
        "hippocampus RATE": ["rate_hipp", "age"],
        "gap value": ["value", "age"],
        "gap RATE": ["rate", "age"],
        "gap value + RATE": ["value", "rate", "age"],
        "hipp value + gap RATE": ["value_hipp", "rate", "age"],
    }

    print("\n" + "=" * 74)
    print("IN-SAMPLE (how E8 computed it)   vs   OUT-OF-FOLD (honest)")
    print("=" * 74)
    oof = {}
    for nm, cols in arms.items():
        ins = cindex_insample(F, cols, dur, ev)
        o = np.nanmean([cindex_oof(F, cols, dur, ev, s) for s in range(15)])
        oof[nm] = o
        print(f"  {nm:24s} in-sample {ins:.3f}   out-of-fold {o:.3f}   "
              f"optimism {ins - o:+.3f}")

    print("\n" + "=" * 74)
    print(f"BOOTSTRAP OVER SUBJECTS ({N_BOOT} resamples)")
    print("=" * 74)
    tests = [("does the RATE beat the VALUE?", "gap RATE", "gap value"),
             ("does the RATE add to the VALUE?", "gap value + RATE", "gap value"),
             ("rate vs hippocampus rate (fair baseline)", "gap RATE", "hippocampus RATE"),
             ("gap rate vs hippocampus value", "gap RATE", "hippocampus value")]
    rng = np.random.RandomState(0)
    boots = {k: [] for _, k, _ in tests}
    boots.update({v: [] for _, _, v in tests})
    store = {k: [] for k in set(list(boots))}
    for b in range(N_BOOT):
        ix = rng.randint(0, len(F), len(F))
        Fb, db, eb = F.iloc[ix].reset_index(drop=True), dur[ix], ev[ix]
        if eb.sum() < 20:
            continue
        for k in store:
            store[k].append(cindex_oof(Fb, arms[k], db, eb, b))
    for lbl, k1, k2 in tests:
        x = np.array(store[k1], float) - np.array(store[k2], float)
        x = x[np.isfinite(x)]
        lo, hi = np.percentile(x, [2.5, 97.5])
        v = "*** holds" if lo > 0 else "<- crosses zero"
        print(f"  {lbl:42s} {x.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]  "
              f"P(>0)={(x > 0).mean():.3f}  {v}")

    print("\n" + "=" * 74)
    print("PERMUTATION")
    print("=" * 74)
    rng = np.random.RandomState(1)
    null = [cindex_oof(F, arms["gap RATE"], dur[p], ev[p], i)
            for i, p in enumerate(rng.permutation(len(F)) for _ in range(150))]
    null = np.array([x for x in null if np.isfinite(x)])
    print(f"  observed gap RATE C = {oof['gap RATE']:.3f}")
    print(f"  permuted            = {null.mean():.3f} "
          f"[{np.percentile(null, 2.5):.3f}, {np.percentile(null, 97.5):.3f}]")
    print(f"  p = {(null >= oof['gap RATE']).mean():.4f}")

    F.assign(dur=dur, event=ev).to_csv(os.path.join(OUT, "exp8b_cohort.csv"))
    print(f"\nsaved -> {OUT}\\exp8b_cohort.csv")


if __name__ == "__main__":
    main()
