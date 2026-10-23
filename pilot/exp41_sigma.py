#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MEASURING sigma: how much do patients differ in WHEN their brain starts deviating?

E40 showed that the entire feasibility of the regional-cascade question turns on
this one number, which nobody has measured. At sigma = 1.5 y, ADNI is enough to
detect a 1-year separation (98% power). At sigma = 3 y it is not (30%), and we
would need roughly 4,000 converters.

THREE INDEPENDENT ESTIMATES
---------------------------
A  Shape of the group curve. If every patient deviated on the same schedule the
   group mean would be a sharp ramp. Spread the onsets out and the same
   individual trajectories average into a gentle, smeared slope. So sigma can be
   read off the CURVATURE of the group mean by fitting (amplitude, mean onset,
   sigma) and finding which sigma reproduces the observed shape.

B  Between-subject spread at each time point. At a fixed number of years before
   conversion, patients differ because they are at different points on their own
   trajectory (onset heterogeneity) and because of measurement noise. Subtracting
   the measured noise variance leaves the part caused by differing onsets.

C  Per-patient onsets, estimated one subject at a time, then corrected for the
   noise in each individual estimate. The crudest method but the most direct.

Agreement between three methods that fail in different ways is the only thing
that would make this number trustworthy. Disagreement is reported as such.
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler

BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "pilot")
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
FA = {"Hippocampus": "هیپوکامپ", "Entorhinal": "آنتورینال", "Fusiform": "دوکی",
      "MidTemp": "گیجگاهی میانی", "Ventricles": "بطن‌ها", "WholeBrain": "کل مغز"}
BINS = [(-0.5, 0.75), (0.75, 1.75), (1.75, 2.75), (2.75, 3.75),
        (3.75, 5.0), (5.0, 7.0)]
MID = np.array([(lo + hi) / 2 for lo, hi in BINS])
NOISE = {"Hippocampus": 0.42, "Entorhinal": 0.40, "Fusiform": 0.42,
         "MidTemp": 0.39, "Ventricles": 0.10, "WholeBrain": 0.36}
N_BOOT = 300


def load_gaps():
    a = pd.read_csv(os.path.join(DATA, "ADNIMERGE_05Mar2026.csv"), low_memory=False)
    d = a.dropna(subset=["AGE", "Years_bl", "ICV", "PTGENDER"] + V).copy()
    d["SEX"] = (d.PTGENDER == "Male").astype(int)
    d["age_visit"] = d.AGE + d.Years_bl
    age = d.age_visit.values.astype(float)
    cn = (d.DX == "CN").values
    subs = d.PTID.values
    uq = np.unique(subs)
    fmap = dict(zip(uq, np.random.RandomState(0).permutation(len(uq)) % 5))
    f = np.array([fmap[s] for s in subs])
    for v in V:
        X = np.column_stack([d[v].values, d.ICV.values, d.SEX.values])
        g = np.zeros(len(d))
        for k in range(5):
            tr = np.where((f != k) & cn)[0]
            te = np.where(f == k)[0]
            sc = StandardScaler().fit(X[tr])
            m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(X[tr]), age[tr])
            b = LinearRegression().fit(age[tr].reshape(-1, 1),
                                       m.predict(sc.transform(X[tr])) - age[tr])
            g[te] = (m.predict(sc.transform(X[te])) - age[te]) \
                - b.predict(age[te].reshape(-1, 1))
        d["gap_" + v] = g

    bl = a[a.VISCODE == "bl"].drop_duplicates("PTID").set_index("PTID")
    mci = set(bl[bl.DX == "MCI"].index)
    dx = a[["PTID", "Years_bl", "DX"]].dropna(subset=["DX"])
    conv = dx[dx.DX == "Dementia"].groupby("PTID").Years_bl.min()
    conv = conv[conv.index.isin(mci)]
    d["is_mci"] = d.PTID.isin(mci)
    d["converter"] = d.PTID.isin(conv.index)
    d["t_before"] = np.nan
    m = d.converter.values
    d.loc[m, "t_before"] = conv.reindex(d.PTID[m]).values - d.Years_bl[m].values
    return d


def group_curve(d, v, subjects=None):
    col = "gap_" + v
    c = d[d.converter]
    if subjects is not None:
        c = c[c.PTID.isin(subjects)]
    ref = d[d.is_mci & ~d.converter][col].mean()
    return np.array([
        c[(c.t_before >= lo) & (c.t_before < hi)][col].mean() - ref
        for lo, hi in BINS])


def between_sd(d, v):
    col = "gap_" + v
    c = d[d.converter]
    return np.array([
        c[(c.t_before >= lo) & (c.t_before < hi)][col].std(ddof=1)
        for lo, hi in BINS])


def model_curve(A, T0, sigma, rng, n=4000):
    """Group mean when individual onsets are spread with SD sigma."""
    T = np.clip(T0 + rng.normal(0, sigma, n), 0.3, 15.0)
    x = np.clip((T[None, :] - MID[:, None]) / T[None, :], 0, 1)
    return A * x.mean(axis=1)


def fit_sigma_shape(obs, rng):
    """Grid search over (A, T0, sigma) for the best fit to the group curve."""
    best, grid = None, []
    for sig in np.arange(0.0, 6.01, 0.25):
        for T0 in np.arange(2.0, 14.01, 0.5):
            m = model_curve(1.0, T0, sig, rng)
            if m[0] <= 1e-6:
                continue
            A = obs[0] / m[0]                       # scale to match at conversion
            err = np.sum((A * m - obs) ** 2)
            grid.append((err, sig, T0, A))
            if best is None or err < best[0]:
                best = (err, sig, T0, A)
    return best, grid


def main():
    d = load_gaps()
    nconv = d[d.converter].PTID.nunique()
    print(f"converters {nconv}, stable MCI {d[d.is_mci & ~d.converter].PTID.nunique()}")
    rng = np.random.RandomState(0)

    print("\n" + "=" * 78)
    print("A) sigma from the SHAPE of the group curve")
    print("=" * 78)
    print(f"  {'region':16s} {'sigma (y)':>10} {'mean onset':>11} "
          f"{'fit error':>10}")
    shape_sig = {}
    for v in V:
        obs = group_curve(d, v)
        if not np.all(np.isfinite(obs)):
            continue
        best, _ = fit_sigma_shape(obs, rng)
        shape_sig[v] = best[1]
        print(f"  {FA[v]:16s} {best[1]:10.2f} {best[2]:11.1f} {best[0]:10.3f}")

    # bootstrap the hippocampus estimate, the one with the strongest signal
    subs = d[d.converter].PTID.unique()
    bs = []
    for _ in range(120):
        samp = rng.choice(subs, len(subs), replace=True)
        o = group_curve(d, "Hippocampus", set(samp))
        if np.all(np.isfinite(o)):
            bs.append(fit_sigma_shape(o, rng)[0][1])
    bs = np.array(bs)
    print(f"\n  hippocampus sigma, 95% CI over subjects: "
          f"[{np.percentile(bs,2.5):.2f}, {np.percentile(bs,97.5):.2f}]")

    print("\n" + "=" * 78)
    print("B) sigma from BETWEEN-SUBJECT spread, after removing measurement noise")
    print("=" * 78)
    print(f"  {'region':16s}" + "".join(f"{-m:>8.1f}" for m in MID))
    for v in V:
        sd = between_sd(d, v)
        excess = np.sqrt(np.clip(sd ** 2 - NOISE[v] ** 2, 0, None))
        print(f"  {FA[v]:16s}" + "".join(f"{x:8.2f}" for x in excess))
    print("\n  (this is the spread of the gap itself, in gap-years, not of the")
    print("   onset in calendar years - it bounds sigma but does not equal it)")

    print("\n" + "=" * 78)
    print("C) sigma from PER-PATIENT onsets")
    print("=" * 78)
    col = "gap_Hippocampus"
    thr = 2.0 * NOISE["Hippocampus"]
    ref = d[d.is_mci & ~d.converter][col].mean()
    ons, ses = [], []
    for s, g in d[d.converter].groupby("PTID"):
        g = g.sort_values("t_before")
        t, y = g.t_before.values, g[col].values - ref
        if len(t) < 4 or (t.max() - t.min()) < 1.5:
            continue
        b = np.polyfit(t, y, 1)
        if b[0] >= -1e-6:                 # must decline going back in time
            continue
        on = (thr - b[1]) / b[0]
        if 0 < on < 15:
            ons.append(on)
            resid = y - np.polyval(b, t)
            ses.append(resid.std(ddof=1) / abs(b[0]))
    ons, ses = np.array(ons), np.array(ses)
    raw = ons.std(ddof=1)
    est_noise = np.sqrt(np.mean(ses ** 2))
    true = np.sqrt(max(raw ** 2 - est_noise ** 2, 0))
    print(f"  patients with a usable individual fit: {len(ons)}")
    print(f"  raw spread of individual onsets        {raw:.2f} y")
    print(f"  typical noise in one onset estimate    {est_noise:.2f} y")
    print(f"  corrected sigma                        {true:.2f} y")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    a_hip = shape_sig.get("Hippocampus", np.nan)
    print(f"  A) curve shape, hippocampus : {a_hip:.2f} y  "
          f"[{np.percentile(bs,2.5):.2f}, {np.percentile(bs,97.5):.2f}]")
    print(f"  C) per-patient onsets       : {true:.2f} y")
    lo = min(a_hip, true)
    hi = max(a_hip, true)
    print(f"\n  the two direct estimates bracket sigma at roughly {lo:.1f}-{hi:.1f} y")
    if hi <= 1.7:
        print("  -> ADNI IS ENOUGH. 98% power for a 1-year separation at n=379.")
    elif lo >= 2.5:
        print("  -> ADNI IS NOT ENOUGH for small separations: ~30% power at 1 year.")
        print("     Only separations of about 2 years or more are answerable here.")
    else:
        print("  -> BORDERLINE. Powered for a 2-year separation, not for 1 year.")
        print("     The proposal must state which size of effect it claims to test.")


if __name__ == "__main__":
    main()
