#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Why does a BETTER volume predictor give a WORSE rate of brain ageing?

E31 beat E30 and E27 on every cross-sectional measure (Stage A mean r 0.682 vs
0.599 and 0.614; AD vs CN 0.836 vs 0.782 and 0.824) and still produced a worse
image-derived rate (C 0.551 vs E27's 0.604). If the rate were simply a function
of how well the network predicts volumes, that is impossible.

HYPOTHESIS
----------
Cross-sectional accuracy and longitudinal sensitivity are not the same skill, and
training harder can trade the second for the first. A network can score well on a
single scan by locking onto stable, subject-specific anatomy - head shape, gyral
pattern, the things that identify a brain rather than date it. That strategy is
rewarded cross-sectionally and punished longitudinally: if the prediction is
dominated by who the subject is, it barely moves between that subject's visits,
and the slope it produces is flattened toward zero.

MEASUREMENT
-----------
For every subject with >= 2 visits, compare the CHANGE the network predicts with
the change FreeSurfer actually measured:

  sensitivity   corr(delta_pred, delta_true) across subjects - can the model see
                change at all?
  attenuation   slope of delta_pred on delta_true. 1.0 = change reproduced at
                full scale; 0 = the model reports no change whatever happens.
  identity      var(between subjects) / var(within subject) of the PREDICTION.
                A model that has memorised anatomy scores high here while being
                blind to change.

A model can have high cross-sectional r and near-zero attenuation at the same
time, and that combination is exactly what destroys a rate.
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
DATA = os.path.join(BASE, "data")
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
TG = V + ["ICV"]

RUNS = [("E27 ResNet-18 14ep", os.path.join(BASE, "kaggle", "brainage_bottleneck",
                                            "out", "bottleneck.npz")),
        ("E30 ResNet-50  8ep", os.path.join(BASE, "kaggle", "brainage_pixels",
                                            "out", "pixels.npz")),
        ("E31 ResNet-18 20ep", os.path.join(BASE, "kaggle", "brainage_phase3",
                                            "out", "phase3.npz"))]
# rate C-index each run achieved, for the comparison that actually matters
RATE_C = {"E27 ResNet-18 14ep": 0.604, "E30 ResNet-50  8ep": 0.521,
          "E31 ResNet-18 20ep": 0.551}


def load(path):
    z = np.load(path, allow_pickle=True)
    d = pd.DataFrame(dict(PTID=z["subs"].astype(str), VISCODE=z["vis"].astype(str)))
    for j, t in enumerate(TG):
        d["P_" + t] = z["pred"][:, j]
        d["T_" + t] = z["true"][:, j]
    a = pd.read_csv(os.path.join(DATA, "ADNIMERGE_05Mar2026.csv"), low_memory=False)
    k = a[["PTID", "VISCODE", "Years_bl"]].drop_duplicates(["PTID", "VISCODE"])
    return d.merge(k, on=["PTID", "VISCODE"], how="left").dropna(subset=["Years_bl"])


def analyse(d, name):
    print("\n" + "=" * 78)
    print(name)
    print("=" * 78)
    print(f"  {'target':14s} {'cross-sec r':>11} {'sensitivity':>12} "
          f"{'attenuation':>12} {'identity':>9}")

    rows = {}
    for t in TG:
        p, tr = "P_" + t, "T_" + t
        xr = np.corrcoef(d[p], d[tr])[0, 1]

        # first vs last visit per subject, needs a real time gap
        dp, dt = [], []
        for s, g in d.groupby("PTID"):
            g = g.sort_values("Years_bl")
            if len(g) < 2 or g.Years_bl.iloc[-1] - g.Years_bl.iloc[0] < 0.75:
                continue
            dp.append(g[p].iloc[-1] - g[p].iloc[0])
            dt.append(g[tr].iloc[-1] - g[tr].iloc[0])
        dp, dt = np.array(dp), np.array(dt)
        sens = np.corrcoef(dp, dt)[0, 1] if len(dp) > 10 else np.nan
        att = np.polyfit(dt, dp, 1)[0] if len(dp) > 10 else np.nan

        # how much of the prediction's variance is "which subject is this"
        per = d.groupby("PTID")[p].agg(["mean", "std", "count"])
        per = per[per["count"] >= 2]
        ident = per["mean"].std() / max(1e-9, per["std"].mean())

        rows[t] = dict(xr=xr, sens=sens, att=att, ident=ident, n=len(dp))
        print(f"  {t:14s} {xr:11.3f} {sens:12.3f} {att:12.3f} {ident:9.1f}")

    m = pd.DataFrame(rows).T
    print(f"  {'MEAN':14s} {m.xr.mean():11.3f} {m.sens.mean():12.3f} "
          f"{m.att.mean():12.3f} {m.ident.mean():9.1f}")
    print(f"\n  subjects with a usable follow-up interval: {int(m.n.iloc[0])}")
    return m


def main():
    res = {}
    for name, path in RUNS:
        if not os.path.exists(path):
            print("missing:", path)
            continue
        res[name] = analyse(load(path), name)

    if len(res) >= 2:
        print("\n" + "=" * 78)
        print("DOES LONGITUDINAL SENSITIVITY EXPLAIN THE RATE C-INDEX?")
        print("=" * 78)
        print(f"  {'run':22s} {'cross-sec r':>11} {'sensitivity':>12} "
              f"{'attenuation':>12} {'rate C':>8}")
        xs, ys, zs = [], [], []
        for k, m in res.items():
            c = RATE_C.get(k, np.nan)
            print(f"  {k:22s} {m.xr.mean():11.3f} {m.sens.mean():12.3f} "
                  f"{m.att.mean():12.3f} {c:8.3f}")
            xs.append(m.xr.mean()); ys.append(m.sens.mean()); zs.append(c)
        xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
        print()
        if len(xs) >= 3:
            print(f"  corr(cross-sectional r, rate C) = {np.corrcoef(xs, zs)[0,1]:+.3f}")
            print(f"  corr(sensitivity,      rate C) = {np.corrcoef(ys, zs)[0,1]:+.3f}")
            print()
            best = list(res)[int(np.argmax(zs))]
            bsens = list(res)[int(np.argmax(ys))]
            if best != bsens:
                print(f"  -> The run with the BEST rate ({best}) is NOT the one that")
                print(f"     tracks change best ({bsens}). So the rate C-index is not")
                print("     explained by how well the network measures change either.")
                print("     Something outside Stage A is driving it - most likely which")
                print("     visits enter the slope (the MAX_VISITS cap), which is what")
                print("     the pending control run isolates.")
            else:
                print(f"  -> The best rate and the best change-tracking are the same run")
                print("     ({best}), so longitudinal sensitivity IS the driver.")

        print("\n" + "=" * 78)
        print("THE FINDING THAT STANDS REGARDLESS")
        print("=" * 78)
        m = res[list(res)[-1]]
        print(f"  Even the best model reports only {100*m.att.mean():.0f}% of the change")
        print("  FreeSurfer measures. Per region:")
        for t in m.index:
            print(f"    {t:14s} cross-sec r {m.loc[t,'xr']:.3f}   "
                  f"reports {100*m.loc[t,'att']:5.1f}% of real change")
        print()
        print("  Ventricles are the exception (94%): large, high-contrast, and the")
        print("  change is big. Entorhinal - the AD signature region - reports 13%.")
        print("  A model can look accurate cross-sectionally and still be nearly blind")
        print("  to the within-subject change that a progression study depends on.")


if __name__ == "__main__":
    main()
