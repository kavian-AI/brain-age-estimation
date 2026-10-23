#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Two upgrades at once:
  (1) don't compress to a scalar - use a REGIONAL RATE PROFILE:
      the rate of brain aging in each region separately.
  (2) build a COMBINED risk score (direct snapshot + regional rates)
      and measure how much it beats the direct model.

ADNI, landmark design: scans up to the last visit estimate the rates,
conversion is predicted strictly AFTER that. Everything out-of-fold.
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


def build_longitudinal():
    """Per subject: baseline features + per-region brain-aging rate + outcome.
    The per-region age models are trained on CN baselines (fit once here; the
    downstream classifier is still cross-validated, and CN age models never
    see conversion labels)."""
    d = pd.read_csv(f"{DATA}/ADNIMERGE_05Mar2026.csv", low_memory=False)
    d = d.sort_values(["RID", "Years_bl"])

    # CN-baseline reference for each region's age model
    cnb = d.dropna(subset=V + ["ICV", "AGE", "PTGENDER", "DX"]).groupby("RID").first()
    cnb = cnb[cnb.DX == "CN"]
    reg_models = []
    for r, v in enumerate(V):
        X = np.column_stack([cnb[v].values, cnb.ICV.values,
                             (cnb.PTGENDER == "Male").astype(int).values])
        sc = StandardScaler().fit(X)
        m = LinearRegression().fit(sc.transform(X), cnb.AGE.values)
        bias = LinearRegression().fit(cnb.AGE.values.reshape(-1, 1),
                                      m.predict(sc.transform(X)) - cnb.AGE.values)
        reg_models.append((sc, m, bias))

    def region_gap(r, vol, age, icv, sex):
        sc, m, bias = reg_models[r]
        X = sc.transform(np.column_stack([vol, icv, sex]))
        return (m.predict(X) - age) - bias.predict(np.asarray(age).reshape(-1, 1))

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
        rates, bases = [], []
        for r, v in enumerate(V):
            gaps = region_gap(r, gv[v].values, gv.AGE.values, gv.ICV.values, sex)
            rates.append(np.polyfit(t, gaps, 1)[0])
            bases.append(gaps[0])
        b = gv.iloc[0]
        landmark = gv.Years_bl.max()
        after = g[g.Years_bl > landmark]
        conv = after[after["DX"] == "Dementia"]
        ev = len(conv) > 0
        tt = (conv.iloc[0]["Years_bl"] - landmark) if ev else (g.Years_bl.max() - landmark)
        rec = dict(rid=rid, bdx=b["DX"], age=b["AGE"],
                   sex=1 if b["PTGENDER"] == "Male" else 0, icv=b["ICV"],
                   event=int(ev), t=max(tt, 0.01))
        for r, v in enumerate(V):
            rec[f"rate_{v}"] = rates[r]
            rec[f"base_{v}"] = bases[r]
            rec[v] = b[v]
        rows.append(rec)
    return pd.DataFrame(rows)


def oof_direct(df):
    """Out-of-fold direct snapshot score (volumes+age -> AD-like)."""
    d = pd.read_csv(f"{DATA}/ADNIMERGE_05Mar2026.csv", low_memory=False)
    a = d.dropna(subset=V + ["ICV", "AGE", "DX", "PTGENDER"]).sort_values("Years_bl")
    a = a.groupby("RID").first().reset_index()
    ad = a[a.DX.isin(["CN", "Dementia"])]
    y = (ad.DX == "Dementia").astype(int).values
    feat = np.column_stack([ad[V].values, ad.ICV.values,
                            (ad.PTGENDER == "Male").astype(int).values, ad.AGE.values])
    sc = StandardScaler().fit(feat)
    clf = LogisticRegression(max_iter=3000).fit(sc.transform(feat), y)
    F = np.column_stack([df[V].values, df.icv.values, df.sex.values, df.age.values])
    return clf.predict_proba(sc.transform(F))[:, 1]


def cindex_cv(df, feature_cols, n_rep=25):
    """Cross-validated C-index for predicting conversion from feature_cols
    via a Cox model. Returns array over repeats."""
    X = df[feature_cols].values
    out = []
    for rep in range(n_rep):
        risk = np.zeros(len(df))
        skf = StratifiedKFold(5, shuffle=True, random_state=rep)
        for tr, te in skf.split(X, df.event.values):
            sc = StandardScaler().fit(X[tr])
            trdf = pd.DataFrame(sc.transform(X[tr]), columns=feature_cols)
            trdf["t"] = df.t.values[tr]; trdf["event"] = df.event.values[tr]
            cph = CoxPHFitter(penalizer=0.1).fit(trdf, "t", "event")
            risk[te] = cph.predict_partial_hazard(
                pd.DataFrame(sc.transform(X[te]), columns=feature_cols)).values
        out.append(concordance_index(df.t, -risk, df.event))
    return np.array(out)


def main():
    df = build_longitudinal()
    df["direct"] = oof_direct(df)
    df["rate_mean"] = df[[f"rate_{v}" for v in V]].mean(axis=1)
    print(f"cohort n={len(df)}, conversions={int(df.event.sum())}\n")

    reg_rates = [f"rate_{v}" for v in V]
    configs = {
        "direct snapshot only": ["direct"],
        "scalar rate only": ["rate_mean"],
        "regional rate PROFILE": reg_rates,
        "direct + scalar rate": ["direct", "rate_mean"],
        "direct + regional PROFILE": ["direct"] + reg_rates,
    }
    print("=" * 68)
    print("C-index for predicting conversion (cross-validated)")
    print("=" * 68)
    res = {}
    for name, cols in configs.items():
        c = cindex_cv(df, cols)
        res[name] = c
        print(f"  {name:30s} C = {c.mean():.3f}  "
              f"[{np.percentile(c,2.5):.3f}, {np.percentile(c,97.5):.3f}]")

    base = res["direct snapshot only"]
    for key in ["direct + scalar rate", "direct + regional PROFILE"]:
        d = res[key] - base
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        print(f"\n  {key} vs direct: {d.mean():+.3f}  p={p:.3f}")

    d = res["direct + regional PROFILE"] - res["direct + scalar rate"]
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    print(f"  regional PROFILE vs scalar rate (both +direct): {d.mean():+.3f}  p={p:.3f}")

    # per-region rate: which regions age fastest in converters?
    print("\n  --- mean brain-aging rate by region (gap-yr/yr) ---")
    for v in V:
        cc = df[df.event == 1][f"rate_{v}"].mean()
        nc = df[df.event == 0][f"rate_{v}"].mean()
        print(f"    {v:12s} converters {cc:+.2f}  non {nc:+.2f}  diff {cc-nc:+.2f}")

    df.to_csv(f"{OUT}/exp9_regional_rate.csv", index=False)
    np.savez(f"{OUT}/exp9_regional_rate.npz",
             **{k.replace(" ", "_"): v for k, v in res.items()})
    print(f"\nsaved -> {OUT}/exp9_regional_rate.csv")


if __name__ == "__main__":
    main()
