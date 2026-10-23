#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DO REGIONAL BRAIN-AGE GAPS DEVIATE IN A FIXED ORDER BEFORE CONVERSION?

Report 3 showed that regional gaps beat a single scalar gap, and that midtemp,
hippocampus and ventricles carry nearly all of the prognostic signal. What it
never asked is WHEN each region departs from normal.

The question here is a staging question rather than a prediction question. If
every region drifted away from normal together, the regional profile would be a
severity axis and nothing more. If instead some regions leave normal years
earlier than others, and the ORDER is the same across people, that ordering is
itself a finding - a brain-age version of the biomarker cascade.

METHOD
------
Time is re-expressed for every converter as years BEFORE their conversion visit,
so that subjects at different disease stages are aligned to a common clock. For
each region a separate age model is fitted (that region + ICV + sex -> age, on CN
only, out of fold), giving one gap per region per visit. Stable MCI subjects who
never convert provide the reference: at each time bin the converter mean is
expressed as a difference from the never-converter mean, so ordinary ageing and
any model bias cancel.

The onset for a region is the earliest time bin at which its deviation is
significantly above zero AND stays above. Ordering is bootstrapped over subjects,
because with seven regions an ordering can look convincing by chance.

WHAT WOULD FALSIFY IT
---------------------
If the bootstrap onset intervals all overlap, there is no ordering - only a
common severity axis - and that is the honest result.
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
N_BOOT = 500


THRESH_ABS = 1.0      # gap-years
REL_FRAC = 0.5        # or half of the deviation the region reaches at conversion


def crossing(c, thresh):
    """How many years before conversion the curve last stayed above `thresh`.

    Scans outward FROM conversion, not inward: the far bins are below threshold
    by construction (disease has not started), so a scan that begins there stops
    immediately and reports nothing - which is what the first version of this
    function did. Crossing time is interpolated between bin centres rather than
    snapped to them, because the bins are a year wide and the whole point is to
    compare timings.
    """
    mid = [(lo + hi) / 2 for lo, hi in BINS]
    if not np.isfinite(c[0]) or c[0] < thresh:
        return np.nan
    for i in range(1, len(c)):
        if not np.isfinite(c[i]):
            return mid[i - 1]
        if c[i] < thresh:
            f = (c[i - 1] - thresh) / (c[i - 1] - c[i])
            return mid[i - 1] + f * (mid[i] - mid[i - 1])
    return mid[-1]


def load():
    a = pd.read_csv(os.path.join(DATA, "ADNIMERGE_05Mar2026.csv"), low_memory=False)
    d = a.dropna(subset=["AGE", "Years_bl", "ICV", "PTGENDER"] + V).copy()
    d["SEX"] = (d.PTGENDER == "Male").astype(int)
    d["age_visit"] = d.AGE + d.Years_bl
    return a, d


def regional_gaps(d, seed=0):
    """One gap per region: age predicted from that region + ICV + sex alone."""
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


def align_to_conversion(a, d):
    """Years BEFORE conversion for converters; stable MCI as the reference."""
    bl = a[a.VISCODE == "bl"].drop_duplicates("PTID").set_index("PTID")
    mci = set(bl[bl.DX == "MCI"].index)
    dx = a[["PTID", "Years_bl", "DX"]].dropna(subset=["DX"])
    conv = dx[dx.DX == "Dementia"].groupby("PTID").Years_bl.min()
    conv = conv[conv.index.isin(mci)]
    stable = mci - set(conv.index)

    d = d[d.PTID.isin(mci)].copy()
    d["converter"] = d.PTID.isin(conv.index)
    d["t_before"] = np.nan
    m = d.converter.values
    d.loc[m, "t_before"] = (conv.reindex(d.PTID[m]).values
                            - d.Years_bl[m].values)
    return d, len(conv), len(stable)


def deviation_curve(d, region, boot_subjects=None):
    """Converter mean minus stable mean, per time bin, for one region."""
    col = "gap_" + region
    conv = d[d.converter]
    stab = d[~d.converter]
    if boot_subjects is not None:
        conv = conv[conv.PTID.isin(boot_subjects)]
    ref = stab[col].mean()
    out = []
    for lo, hi in BINS:
        s = conv[(conv.t_before >= lo) & (conv.t_before < hi)][col]
        out.append(s.mean() - ref if len(s) >= 8 else np.nan)
    return np.array(out)


def main():
    a, d = load()
    d = regional_gaps(d)
    d, n_conv, n_stab = align_to_conversion(a, d)
    print(f"MCI subjects: {n_conv} converters, {n_stab} stable")
    mid = [-(lo + hi) / 2 for lo, hi in BINS]
    print(f"time bins (years before conversion): "
          f"{[f'{lo:.1f}-{hi:.1f}' for lo, hi in BINS]}")
    print("\nvisits per bin:")
    for lo, hi in BINS:
        n = ((d.converter) & (d.t_before >= lo) & (d.t_before < hi)).sum()
        print(f"  {lo:.2f}-{hi:.2f} y before: {n:4d} visits")

    print("\n" + "=" * 78)
    print("REGIONAL DEVIATION (converter mean - stable mean), in gap-years")
    print("=" * 78)
    hdr = "  ".join(f"{-(lo+hi)/2:>6.1f}" for lo, hi in BINS)
    print(f"  {'region':16s} {hdr}")
    curves = {}
    for v in V:
        c = deviation_curve(d, v)
        curves[v] = c
        print(f"  {FA[v]:16s} " + "  ".join(
            f"{x:6.2f}" if np.isfinite(x) else "   nan" for x in c))

    # ---- bootstrap: when does each region first and persistently exceed 1 year?
    print("\n" + "=" * 78)
    print("ONSET: earliest bin where deviation exceeds 1 gap-year and stays above")
    print("=" * 78)
    conv_subs = d[d.converter].PTID.unique()
    rng = np.random.RandomState(0)
    onsets = {v: [] for v in V}
    onsets_rel = {v: [] for v in V}
    for _ in range(N_BOOT):
        bs = rng.choice(conv_subs, len(conv_subs), replace=True)
        for v in V:
            c = deviation_curve(d, v, boot_subjects=set(bs))
            onsets[v].append(crossing(c, THRESH_ABS))
            rel = crossing(c, REL_FRAC * c[0]) if np.isfinite(c[0]) and c[0] > 0                 else np.nan
            onsets_rel[v].append(rel)
    def summarise(store, label):
        print("\n  " + label)
        print(f"    {'region':16s} {'onset (y)':>10} {'95% CI':>18} {'detected':>9}")
        rows = []
        for v in V:
            arr = np.array(store[v], dtype=float)
            det = np.isfinite(arr).mean()
            if det < 0.5:
                print(f"    {FA[v]:16s} {'never':>10} {'':>18} {100*det:8.0f}%")
                continue
            arr = arr[np.isfinite(arr)]
            lo, hi = np.percentile(arr, [2.5, 97.5])
            print(f"    {FA[v]:16s} {np.median(arr):10.2f} "
                  f"{f'[{lo:.2f}, {hi:.2f}]':>18} {100*det:8.0f}%")
            rows.append((v, np.median(arr)))
        rows.sort(key=lambda r: -r[1])
        print("    ordering, earliest first: "
              + "  >  ".join(FA[v] for v, _ in rows))
        return rows, store

    rows_abs, _ = summarise(onsets, "A) absolute threshold: deviation >= 1 gap-year")
    rows_rel, _ = summarise(
        onsets_rel,
        "B) relative threshold: 50% of the deviation reached at conversion\n"
        "     (removes magnitude - a region with a smaller deviation crosses a\n"
        "      FIXED threshold late even with identical timing, so this is the\n"
        "      real test of whether the ORDER differs)")

    for rows, store, lab in [(rows_abs, onsets, "absolute"),
                             (rows_rel, onsets_rel, "relative")]:
        if len(rows) < 2:
            continue
        first, last = rows[0][0], rows[-1][0]
        diff = np.array([store[first][i] - store[last][i] for i in range(N_BOOT)
                         if np.isfinite(store[first][i])
                         and np.isfinite(store[last][i])])
        if len(diff) < 50:
            continue
        lo, hi = np.percentile(diff, [2.5, 97.5])
        pv = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
        print(f"\n  separation on the {lab} threshold, "
              f"{FA[first]} vs {FA[last]}:")
        print(f"    {diff.mean():+.2f} y   95% CI [{lo:+.2f}, {hi:+.2f}]   p={pv:.3f}")
        print(f"    -> {'REAL ORDERING' if pv < 0.05 else 'onsets overlap - no ordering'}")

    pd.DataFrame(curves, index=[f"{-(lo+hi)/2:.1f}" for lo, hi in BINS]).to_csv(
        os.path.join(OUT, "exp37_cascade.csv"))
    print(f"\nsaved -> {OUT}\\exp37_cascade.csv")


if __name__ == "__main__":
    main()
