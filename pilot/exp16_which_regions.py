#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
THE USER'S IDEA: instead of "the whole brain aged by X years", say
"region A aged 5 years and that matters; region B aged 2 years and that is
irrelevant to Alzheimer's".

Two questions:
  1. RANKING: which regional ageing RATES actually carry conversion signal?
  2. SUFFICIENCY: how few regions do we need before performance saturates?
     If 2 regions do the job of 6, the other 4 are noise for this purpose.

Method: landmark design (rates from >=3 scans, conversion strictly after).
Each region's rate is evaluated (a) alone, (b) added to the direct model,
and (c) via greedy forward selection to find the minimal informative set.
"""

import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
DATA = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\data"
OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]


def build():
    d = pd.read_csv(f"{DATA}/ADNIMERGE_05Mar2026.csv", low_memory=False)
    d = d.sort_values(["RID", "Years_bl"])
    cnb = d.dropna(subset=V + ["ICV", "AGE", "PTGENDER", "DX"]).groupby("RID").first()
    cnb = cnb[cnb.DX == "CN"]
    models = []
    for v in V:
        X = np.column_stack([cnb[v].values, cnb.ICV.values,
                             (cnb.PTGENDER == "Male").astype(int).values])
        sc = StandardScaler().fit(X)
        m = LinearRegression().fit(sc.transform(X), cnb.AGE.values)
        b = LinearRegression().fit(cnb.AGE.values.reshape(-1, 1),
                                   m.predict(sc.transform(X)) - cnb.AGE.values)
        models.append((sc, m, b))

    def gap(r, vol, age, icv, sex):
        sc, m, b = models[r]
        X = sc.transform(np.column_stack([vol, icv, sex]))
        return (m.predict(X) - age) - b.predict(np.asarray(age).reshape(-1, 1))

    rows = []
    for rid, g in d.groupby("RID"):
        g = g.sort_values("Years_bl")
        if g.iloc[0]["DX"] == "Dementia":
            continue
        gv = g.dropna(subset=V + ["ICV", "AGE", "PTGENDER"])
        if gv.Years_bl.nunique() < 3 or (gv.Years_bl.max() - gv.Years_bl.min()) < 1.0:
            continue
        t = gv.Years_bl.values - gv.Years_bl.values[0]
        sex = (gv.PTGENDER == "Male").astype(int).values
        rec = dict(rid=rid, bdx=g.iloc[0]["DX"])
        for r, v in enumerate(V):
            gg = gap(r, gv[v].values, gv.AGE.values, gv.ICV.values, sex)
            rec[f"rate_{v}"] = np.polyfit(t, gg, 1)[0]
        b0 = gv.iloc[0]
        landmark = gv.Years_bl.max()
        after = g[g.Years_bl > landmark]
        conv = after[after["DX"] == "Dementia"]
        ev = len(conv) > 0
        tt = (conv.iloc[0]["Years_bl"] - landmark) if ev else (g.Years_bl.max() - landmark)
        rec.update(event=int(ev), t=max(tt, 0.01), age=b0["AGE"],
                   sex=1 if b0["PTGENDER"] == "Male" else 0, icv=b0["ICV"])
        for v in V:
            rec[v] = b0[v]
        rows.append(rec)
    return pd.DataFrame(rows)


def cindex_cv(df, cols, n_rep=15):
    X = df[cols].values
    out = []
    for rep in range(n_rep):
        risk = np.zeros(len(df))
        for tr, te in StratifiedKFold(5, shuffle=True,
                                      random_state=rep).split(X, df.event.values):
            sc = StandardScaler().fit(X[tr])
            tr_df = pd.DataFrame(sc.transform(X[tr]), columns=cols)
            tr_df["t"] = df.t.values[tr]; tr_df["event"] = df.event.values[tr]
            cph = CoxPHFitter(penalizer=0.1).fit(tr_df, "t", "event")
            risk[te] = cph.predict_partial_hazard(
                pd.DataFrame(sc.transform(X[te]), columns=cols)).values
        out.append(concordance_index(df.t, -risk, df.event))
    return np.array(out)


def main():
    df = build()
    print(f"cohort n={len(df)}, conversions={int(df.event.sum())}\n")
    rates = [f"rate_{v}" for v in V]

    # ---------- 1. each region's rate ALONE
    print("=" * 70)
    print("1. WHICH REGIONS CARRY SIGNAL?  (rate of that region alone)")
    print("=" * 70)
    solo = {}
    for v in V:
        c = cindex_cv(df, [f"rate_{v}"])
        conv = df[df.event == 1][f"rate_{v}"].mean()
        non = df[df.event == 0][f"rate_{v}"].mean()
        solo[v] = c.mean()
        print(f"  {v:12s} C = {c.mean():.3f}   rate: converters {conv:+.2f} vs "
              f"non {non:+.2f}  (diff {conv-non:+.2f})")

    ranked = sorted(V, key=lambda v: -solo[v])
    print(f"\n  ranking: {' > '.join(ranked)}")

    # ---------- 2. hazard ratios with ALL regions in one model
    print("\n" + "=" * 70)
    print("2. INDEPENDENT CONTRIBUTION (all regional rates in one Cox model)")
    print("=" * 70)
    z = df.copy()
    for c in rates:
        z[c + "_z"] = (z[c] - z[c].mean()) / z[c].std()
    cph = CoxPHFitter(penalizer=0.05).fit(
        z[[c + "_z" for c in rates] + ["t", "event"]], "t", "event")
    s = cph.summary
    for v in V:
        nm = f"rate_{v}_z"
        hr = np.exp(s.loc[nm, "coef"]); p = s.loc[nm, "p"]
        star = "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else "  ns"))
        print(f"  {v:12s} HR = {hr:.2f}   p = {p:.2e}  {star}")

    # ---------- 3. greedy forward selection: minimal sufficient set
    print("\n" + "=" * 70)
    print("3. HOW FEW REGIONS ARE ENOUGH?  (greedy forward selection)")
    print("=" * 70)
    chosen, remaining, prev = [], list(rates), 0.0
    while remaining:
        best, best_c = None, -1
        for r in remaining:
            c = cindex_cv(df, chosen + [r]).mean()
            if c > best_c:
                best, best_c = r, c
        gain = best_c - prev
        chosen.append(best); remaining.remove(best)
        print(f"  + {best.replace('rate_',''):12s} -> C = {best_c:.3f}   "
              f"(gain {gain:+.3f})")
        prev = best_c
    full = cindex_cv(df, rates).mean()
    top2 = cindex_cv(df, chosen[:2]).mean()
    print(f"\n  all 6 regions: C = {full:.3f}")
    print(f"  best 2 alone : C = {top2:.3f}   ({100*top2/full:.1f}% of the full model)")

    df.to_csv(f"{OUT}/exp16_rates.csv", index=False)
    print(f"\nsaved -> {OUT}/exp16_rates.csv")


if __name__ == "__main__":
    main()
