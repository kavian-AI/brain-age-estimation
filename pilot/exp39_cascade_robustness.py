#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HOW ROBUST IS THE "PRECISION EXPLAINS THE CASCADE" RESULT?

E38 used one detection threshold (2x each region's own noise) and one noise
estimator. A finding that depends on either choice is not a finding. This sweeps
the threshold from 1 to 4 and reports, at each value, what fraction of the
observed onset spread is reproduced by measurement precision alone.

It also repeats the whole thing with a second, independent noise estimator -
the scatter between CONSECUTIVE visit pairs rather than about a fitted line -
because a residual about a fitted line is slightly optimistic (the line was fitted
to those same points).

OASIS cannot serve as the replication cohort here: OASIS-2 carries only whole
brain volume (nWBV), not the regional parcellation this analysis needs. That is a
real limitation of the finding and is stated rather than worked around.
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
N_SIM = 300


def load():
    a = pd.read_csv(os.path.join(DATA, "ADNIMERGE_05Mar2026.csv"), low_memory=False)
    d = a.dropna(subset=["AGE", "Years_bl", "ICV", "PTGENDER"] + V).copy()
    d["SEX"] = (d.PTGENDER == "Male").astype(int)
    d["age_visit"] = d.AGE + d.Years_bl
    return a, d


def regional_gaps(d, seed=0):
    age = d.age_visit.values.astype(float)
    cn = (d.DX == "CN").values
    subs = d.PTID.values
    uq = np.unique(subs)
    fmap = dict(zip(uq, np.random.RandomState(seed).permutation(len(uq)) % 5))
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
    return d


def align(a, d):
    bl = a[a.VISCODE == "bl"].drop_duplicates("PTID").set_index("PTID")
    mci = set(bl[bl.DX == "MCI"].index)
    dx = a[["PTID", "Years_bl", "DX"]].dropna(subset=["DX"])
    conv = dx[dx.DX == "Dementia"].groupby("PTID").Years_bl.min()
    conv = conv[conv.index.isin(mci)]
    d = d.copy()
    d["is_mci"] = d.PTID.isin(mci)
    d["converter"] = d.PTID.isin(conv.index)
    d["t_before"] = np.nan
    m = d.converter.values
    d.loc[m, "t_before"] = conv.reindex(d.PTID[m]).values - d.Years_bl[m].values
    return d


def noise_residual(d, v):
    """Scatter about each person's own fitted line (slightly optimistic)."""
    st = d[(d.DX == "CN") | (d.is_mci & ~d.converter)]
    r = []
    for _, g in st.groupby("PTID"):
        g = g.sort_values("Years_bl")
        if len(g) < 4 or (g.Years_bl.max() - g.Years_bl.min()) < 1.5:
            continue
        t, y = g.Years_bl.values, g["gap_" + v].values
        r.append((y - np.polyval(np.polyfit(t, y, 1), t)).std(ddof=1))
    return np.median(r)


def noise_successive(d, v):
    """Scatter of consecutive-visit differences, detrended by the person's own
    mean slope. Does not fit anything to the points it then measures, so it does
    not inherit the optimism of a residual."""
    st = d[(d.DX == "CN") | (d.is_mci & ~d.converter)]
    r = []
    for _, g in st.groupby("PTID"):
        g = g.sort_values("Years_bl")
        if len(g) < 3:
            continue
        t, y = g.Years_bl.values, g["gap_" + v].values
        dt = np.diff(t)
        ok = dt > 0.2
        if ok.sum() < 2:
            continue
        rate = (y[-1] - y[0]) / (t[-1] - t[0])
        step = (np.diff(y) - rate * dt)[ok]
        r.append(step.std(ddof=1) / np.sqrt(2))     # difference of two measures
    return np.median(r)


def dev_curve(d, v):
    col = "gap_" + v
    conv = d[d.converter]
    ref = d[d.is_mci & ~d.converter][col].mean()
    return np.array([
        conv[(conv.t_before >= lo) & (conv.t_before < hi)][col].mean() - ref
        for lo, hi in BINS])


def onset(curve, thresh):
    if not np.isfinite(curve[0]) or curve[0] < thresh:
        return np.nan
    for i in range(1, len(curve)):
        if not np.isfinite(curve[i]):
            return MID[i - 1]
        if curve[i] < thresh:
            f = (curve[i - 1] - thresh) / (curve[i - 1] - curve[i])
            return MID[i - 1] + f * (MID[i] - MID[i - 1])
    return MID[-1]


def spread_and_null(curves, noise, n_per_bin, k, rng):
    obs = {v: onset(curves[v], k * noise[v]) for v in V}
    fin = [obs[v] for v in V if np.isfinite(obs[v])]
    if len(fin) < 2:
        return np.nan, np.nan, np.nan
    s_obs = max(fin) - min(fin)
    shape = curves["Hippocampus"] / curves["Hippocampus"][0]
    sims = []
    for _ in range(N_SIM):
        on = []
        for v in V:
            se = np.array([noise[v] / np.sqrt(max(1, n)) for n in n_per_bin])
            sim = curves[v][0] * shape + rng.normal(0, se)
            o = onset(sim, k * noise[v])
            if np.isfinite(o):
                on.append(o)
        if len(on) >= 2:
            sims.append(max(on) - min(on))
    sims = np.array(sims)
    return s_obs, sims.mean(), (sims >= s_obs).mean()


def main():
    a, d = load()
    d = regional_gaps(d)
    d = align(a, d)
    curves = {v: dev_curve(d, v) for v in V}
    n_per_bin = np.array([((d.converter) & (d.t_before >= lo)
                           & (d.t_before < hi)).sum() for lo, hi in BINS])

    n_res = {v: noise_residual(d, v) for v in V}
    n_suc = {v: noise_successive(d, v) for v in V}
    print("=" * 74)
    print("TWO INDEPENDENT NOISE ESTIMATORS")
    print("=" * 74)
    print(f"  {'region':16s} {'about a line':>13} {'consecutive':>13} {'ratio':>7}")
    for v in V:
        print(f"  {FA[v]:16s} {n_res[v]:13.3f} {n_suc[v]:13.3f} "
              f"{n_suc[v]/n_res[v]:7.2f}")
    print(f"\n  ventricles are measured "
          f"{np.mean([n_res[x] for x in V if x!='Ventricles'])/n_res['Ventricles']:.1f}x "
          f"more precisely than the average region (line estimator)")

    rng = np.random.RandomState(0)
    for label, noise in [("scatter about a fitted line", n_res),
                         ("consecutive-visit differences", n_suc)]:
        print("\n" + "=" * 74)
        print(f"THRESHOLD SWEEP  -  noise estimator: {label}")
        print("=" * 74)
        print(f"  {'k':>4} {'observed spread':>16} {'null spread':>13} "
              f"{'% explained':>12} {'p':>7}")
        for k in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
            so, sn, p = spread_and_null(curves, noise, n_per_bin, k, rng)
            if not np.isfinite(so):
                print(f"  {k:4.1f} {'no region detected':>16}")
                continue
            print(f"  {k:4.1f} {so:16.2f} {sn:13.2f} {100*sn/so:11.0f}% {p:7.3f}")

    print("\n" + "=" * 74)
    print("READING")
    print("=" * 74)
    print("  If the '% explained' column stays high across the sweep and across")
    print("  both estimators, then measurement precision accounts for most of the")
    print("  apparent cascade regardless of how detection is defined - which is")
    print("  the claim. If it swings with k, the claim is threshold-dependent and")
    print("  must be stated as such.")


if __name__ == "__main__":
    main()
