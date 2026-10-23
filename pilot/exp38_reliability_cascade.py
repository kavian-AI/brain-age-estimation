#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DOES THE REGIONAL CASCADE SURVIVE CORRECTION FOR MEASUREMENT RELIABILITY?

The regional brain-age cascade is published (Alzheimer's & Dementia; and at
network level, medRxiv 2025). Our own E37 reproduced it independently. What no
published version does is correct for the fact that REGIONS ARE NOT MEASURED
EQUALLY WELL.

This matters mechanically. A deviation is detected when it rises above the noise
of its own measurement. A region measured precisely reveals its deviation early;
a noisy region hides the same deviation for years. So part of any observed
ordering may be an ordering of measurement precision rather than of biology.

WHAT IS DONE HERE
-----------------
1  Per-region measurement noise, estimated from within-subject scatter about
   each person's own straight line, using CN and STABLE MCI only - people whose
   trajectory should be close to straight, so residual scatter is mostly
   measurement error rather than disease curvature.

2  The deviation curves from E37 re-expressed in units of each region's own
   noise, so "detectable" means the same thing for every region.

3  A NULL SIMULATION that is the real test: give every region the SAME true
   onset and the same trajectory shape, scale each to the deviation it actually
   reaches, add its own measured noise, and read off the apparent onsets. If
   noise alone reproduces an ordering as wide as the observed one, the cascade
   is an artefact of precision.

FALSIFIES THE CASCADE IF: the observed spread of onsets falls inside the spread
that noise alone produces.
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
SNR_THRESH = 2.0      # a deviation must be twice its own measurement noise
N_SIM = 400


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
    return d, conv


def region_noise(d):
    """Within-subject scatter about a straight line, in people who should be
    close to straight: CN and stable MCI. This is measurement noise, not
    biology - disease curvature is excluded by construction."""
    stable = d[((d.DX == "CN") | (d.is_mci & ~d.converter))]
    out = {}
    for v in V:
        res = []
        for _, g in stable.groupby("PTID"):
            g = g.sort_values("Years_bl")
            if len(g) < 4 or (g.Years_bl.max() - g.Years_bl.min()) < 1.5:
                continue
            t, y = g.Years_bl.values, g["gap_" + v].values
            r = y - np.polyval(np.polyfit(t, y, 1), t)
            res.append(r.std(ddof=1))
        out[v] = np.median(res)
    return out


def dev_curve(d, v):
    col = "gap_" + v
    conv = d[d.converter]
    ref = d[d.is_mci & ~d.converter][col].mean()
    return np.array([
        conv[(conv.t_before >= lo) & (conv.t_before < hi)][col].mean() - ref
        for lo, hi in BINS])


def onset_from(curve, thresh):
    """Years before conversion at which the curve last stayed above thresh."""
    if not np.isfinite(curve[0]) or curve[0] < thresh:
        return np.nan
    for i in range(1, len(curve)):
        if not np.isfinite(curve[i]):
            return MID[i - 1]
        if curve[i] < thresh:
            f = (curve[i - 1] - thresh) / (curve[i - 1] - curve[i])
            return MID[i - 1] + f * (MID[i] - MID[i - 1])
    return MID[-1]


def main():
    a, d = load()
    d = regional_gaps(d)
    d, conv = align(a, d)
    print(f"converters {len(conv)}, stable MCI "
          f"{d[d.is_mci & ~d.converter].PTID.nunique()}, "
          f"CN {d[d.DX=='CN'].PTID.nunique()}")

    noise = region_noise(d)
    curves = {v: dev_curve(d, v) for v in V}
    n_per_bin = np.array([((d.converter) & (d.t_before >= lo)
                           & (d.t_before < hi)).sum() for lo, hi in BINS])

    print("\n" + "=" * 78)
    print("PER-REGION MEASUREMENT NOISE (within-subject, non-progressing people)")
    print("=" * 78)
    print(f"  {'region':16s} {'noise (y)':>10} {'dev at conv':>12} "
          f"{'SNR at conv':>12}")
    for v in V:
        print(f"  {FA[v]:16s} {noise[v]:10.2f} {curves[v][0]:12.2f} "
              f"{curves[v][0]/noise[v]:12.1f}")

    print("\n" + "=" * 78)
    print(f"ONSET with a COMMON criterion: deviation >= {SNR_THRESH} x its own noise")
    print("=" * 78)
    # Per-PERSON noise, not the standard error of the bin mean. The SE shrinks
    # with sample size, so using it makes every region detectable in every bin
    # and the comparison collapses (first attempt: all six onsets came out at
    # exactly the boundary). The question is whether a deviation would be
    # visible in one patient, which is set by that region's own measurement
    # noise and nothing else.
    obs = {}
    for v in V:
        obs[v] = onset_from(curves[v], SNR_THRESH * noise[v])
    fixed = {v: onset_from(curves[v], 1.0) for v in V}
    relat = {v: onset_from(curves[v], 0.5 * curves[v][0]) for v in V}
    CENS = MID[-1]

    def show(dic, label):
        print("\n  " + label)
        rows = sorted(V, key=lambda v: -(dic[v] if np.isfinite(dic[v]) else -9))
        for v in rows:
            x = dic[v]
            tag = f"{x:5.2f}" if np.isfinite(x) else "never"
            cens = "  (censored: still above at the edge)" if np.isfinite(x)                 and x >= CENS - 1e-6 else ""
            print(f"    {FA[v]:16s} {tag} y{cens}")
        return rows

    r_fix = show(fixed, "A) fixed threshold, 1 gap-year  (what the literature uses)")
    r_rel = show(relat, "B) relative threshold, 50% of each region's own final deviation")
    r_snr = show(obs, f"C) reliability-adjusted: {SNR_THRESH} x that region's OWN noise")

    print("\n  ordering under each criterion:")
    for lab, r in [("fixed   ", r_fix), ("relative", r_rel), ("reliab. ", r_snr)]:
        print(f"    {lab}: " + "  >  ".join(FA[v] for v in r))

    vpos = {lab: {v: i for i, v in enumerate(r)}
            for lab, r in [("fix", r_fix), ("rel", r_rel), ("snr", r_snr)]}
    print("\n  rank of each region under the three criteria:")
    print(f"    {'region':16s} {'fixed':>7} {'relative':>9} {'reliability':>12}")
    for v in V:
        print(f"    {FA[v]:16s} {vpos['fix'][v]+1:7d} {vpos['rel'][v]+1:9d} "
              f"{vpos['snr'][v]+1:12d}")

    order = [v for v in r_snr if np.isfinite(obs[v])]
    fin = [obs[v] for v in order]
    spread_obs = (max(fin) - min(fin)) if len(fin) >= 2 else 0.0
    print(f"\n  spread of onsets under the reliability criterion: "
          f"{spread_obs:.2f} y")

    # ---------------------------------------------------------------- null
    print("\n" + "=" * 78)
    print("NULL: every region given the SAME true onset, differing only in noise")
    print("=" * 78)
    # a common shape, scaled per region to the deviation it actually reaches
    shape = curves["Hippocampus"] / curves["Hippocampus"][0]
    rng = np.random.RandomState(0)
    spreads, orders = [], []
    for _ in range(N_SIM):
        on = {}
        for v in V:
            # sampling noise on the bin mean, but the DETECTION threshold uses
            # per-person noise - identical to how the observed onsets were read,
            # otherwise the null is not comparable to the observation
            se = np.array([noise[v] / np.sqrt(max(1, n)) for n in n_per_bin])
            sim = curves[v][0] * shape + rng.normal(0, se)
            on[v] = onset_from(sim, SNR_THRESH * noise[v])
        ok = [v for v in V if np.isfinite(on[v])]
        if len(ok) >= 2:
            spreads.append(max(on[v] for v in ok) - min(on[v] for v in ok))
            orders.append(sorted(ok, key=lambda v: -on[v]))
    spreads = np.array(spreads)
    lo, hi = np.percentile(spreads, [2.5, 97.5])
    print(f"  spread produced by noise alone: {spreads.mean():.2f} y "
          f"[{lo:.2f}, {hi:.2f}]")
    p = (spreads >= spread_obs).mean()
    print(f"  observed spread {spread_obs:.2f} y   p = {p:.3f}")
    print()
    if p < 0.05:
        print("  -> THE CASCADE SURVIVES. The ordering is wider than differences in")
        print("     measurement precision can produce, so it reflects timing.")
    else:
        print("  -> THE CASCADE DOES NOT SURVIVE. An ordering this wide arises from")
        print("     unequal measurement precision alone, with no difference in")
        print("     true onset. The published orderings may be reading precision.")

    # how often does noise alone reproduce the exact observed first/last?
    first, last = order[0], order[-1]
    same = np.mean([o[0] == first and o[-1] == last for o in orders])
    print(f"\n  noise alone puts {FA[first]} first AND {FA[last]} last "
          f"in {100*same:.0f}% of simulations")

    pd.DataFrame({FA[v]: curves[v] for v in V},
                 index=[f"{m:.1f}" for m in MID]).to_csv(
        os.path.join(OUT, "exp38_curves.csv"))
    pd.Series({FA[v]: noise[v] for v in V}).to_csv(
        os.path.join(OUT, "exp38_noise.csv"))
    print(f"\nsaved -> {OUT}\\exp38_*.csv")


if __name__ == "__main__":
    main()
