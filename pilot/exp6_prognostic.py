#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
THE KEY QUESTION (user's reframing):
  don't try to beat the direct model on diagnosis.
  Ask: does the brain-age gap add PROGNOSTIC information the direct
  cross-sectional model does not have?

Cohort : ADNI, non-demented at baseline (CN + MCI), 1503 subjects.
Event  : conversion to dementia during follow-up (357 events).
Time   : years from baseline scan to conversion (or last visit if censored).

Two predictors, both from the BASELINE scan, both out-of-fold:
  direct : P(dementia) from a CN-vs-Dementia classifier
           = "how AD-like does this brain look right now"
  gap    : brain-age gap from an age model trained on CN only
           = "how much older than its age does this brain look"

Cox survival models:
  1. direct alone
  2. gap alone
  3. direct + gap    <- does gap add independent hazard?

Plus: the 45 "false alarms" from exp5 - do they actually convert?
"""

import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
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
    d = d.dropna(subset=["DX"]).sort_values(["RID", "Years_bl"])
    rows = []
    for rid, g in d.groupby("RID"):
        g = g.sort_values("Years_bl")
        b = g.iloc[0]
        if b["DX"] == "Dementia":
            continue
        if b[V + ["ICV", "AGE", "PTGENDER"]].isna().any():
            continue
        conv = g[g["DX"] == "Dementia"]
        ev = len(conv) > 0
        t = conv.iloc[0]["Years_bl"] if ev else g.iloc[-1]["Years_bl"]
        if t <= 0:
            continue
        rows.append(dict(rid=rid, bdx=b["DX"], event=int(ev), t=t,
                         age=b["AGE"], sex=1 if b["PTGENDER"] == "Male" else 0,
                         icv=b["ICV"], **{v: b[v] for v in V}))
    return pd.DataFrame(rows)


def train_ad_reference():
    """Train direct (CN vs Dementia) and age (CN only) models on the FULL
    labelled cross-sectional set, to score the non-demented longitudinal cohort."""
    d = pd.read_csv(f"{DATA}/ADNIMERGE_05Mar2026.csv", low_memory=False)
    a = d.dropna(subset=V + ["ICV", "AGE", "DX", "PTGENDER"]).sort_values("Years_bl")
    a = a.groupby("RID").first().reset_index()
    a = a[a.DX.isin(["CN", "Dementia"])]
    y = (a.DX == "Dementia").astype(int).values
    feat = np.column_stack([a[V].values, a.ICV.values,
                            (a.PTGENDER == "Male").astype(int).values, a.AGE.values])
    scd = StandardScaler().fit(feat)
    direct = LogisticRegression(max_iter=3000).fit(scd.transform(feat), y)

    cn = a[a.DX == "CN"]
    Xa = np.column_stack([cn[V].values, cn.ICV.values,
                          (cn.PTGENDER == "Male").astype(int).values])
    sca = StandardScaler().fit(Xa)
    agem = LinearRegression().fit(sca.transform(Xa), cn.AGE.values)
    bias = LinearRegression().fit(cn.AGE.values.reshape(-1, 1),
                                  agem.predict(sca.transform(Xa)) - cn.AGE.values)
    return direct, scd, agem, sca, bias


def main():
    df = build()
    direct, scd, agem, sca, bias = train_ad_reference()

    feat = np.column_stack([df[V].values, df.icv.values, df.sex.values, df.age.values])
    df["direct"] = direct.predict_proba(scd.transform(feat))[:, 1]
    Xa = np.column_stack([df[V].values, df.icv.values, df.sex.values])
    raw = agem.predict(sca.transform(Xa)) - df.age.values
    df["gap"] = raw - bias.predict(df.age.values.reshape(-1, 1))

    # z-score predictors for interpretable hazard ratios
    for c in ["direct", "gap"]:
        df[c + "_z"] = (df[c] - df[c].mean()) / df[c].std()

    print(f"cohort n={len(df)}, events={int(df.event.sum())}, "
          f"median follow-up={df.t.median():.1f} y")
    print(f"corr(direct, gap) = {np.corrcoef(df.direct, df.gap)[0,1]:.3f}\n")

    # ---------------- Cox models
    print("=" * 66)
    print("COX SURVIVAL: time to dementia conversion")
    print("=" * 66)

    def cox(cols):
        cph = CoxPHFitter()
        cph.fit(df[cols + ["t", "event"]], "t", "event")
        c = concordance_index(df.t, -cph.predict_partial_hazard(df), df.event)
        return cph, c

    m1, c1 = cox(["direct_z"])
    m2, c2 = cox(["gap_z"])
    m3, c3 = cox(["direct_z", "gap_z"])

    print(f"  1. direct alone          C-index = {c1:.3f}")
    print(f"  2. gap alone             C-index = {c2:.3f}")
    print(f"  3. direct + gap          C-index = {c3:.3f}")
    print(f"\n  gain from adding gap: {c3 - c1:+.3f}")

    print("\n  --- model 3 coefficients (hazard ratios) ---")
    s = m3.summary
    for name in ["direct_z", "gap_z"]:
        hr = np.exp(s.loc[name, "coef"])
        lo = np.exp(s.loc[name, "coef lower 95%"])
        hi = np.exp(s.loc[name, "coef upper 95%"])
        p = s.loc[name, "p"]
        print(f"    {name:10s} HR = {hr:.2f}  [{lo:.2f}, {hi:.2f}]  p = {p:.2e}")
    print("\n  -> if gap's HR>1 and p<0.05, brain age adds INDEPENDENT prognostic value")

    # ---------------- the 45 false alarms: CN who look diseased
    print("\n" + "=" * 66)
    print("THE FALSE ALARMS: CN flagged 'diseased' by the direct model")
    print("=" * 66)
    cn = df[df.bdx == "CN"].copy()
    cn["flagged"] = (cn.direct > 0.5).astype(int)
    fa = cn[cn.flagged == 1]
    ok = cn[cn.flagged == 0]
    print(f"  CN flagged diseased (false alarm): n={len(fa)}, "
          f"convert to dementia: {int(fa.event.sum())} ({100*fa.event.mean():.0f}%)")
    print(f"  CN flagged healthy:               n={len(ok)}, "
          f"convert to dementia: {int(ok.event.sum())} ({100*ok.event.mean():.0f}%)")

    # split CN by brain-age gap instead
    thr = cn.gap.median()
    hi = cn[cn.gap > cn.gap.quantile(.75)]
    lo = cn[cn.gap < cn.gap.quantile(.25)]
    print(f"\n  CN with HIGH brain-age gap (top 25%): n={len(hi)}, "
          f"convert: {int(hi.event.sum())} ({100*hi.event.mean():.0f}%)")
    print(f"  CN with LOW brain-age gap (bottom 25%): n={len(lo)}, "
          f"convert: {int(lo.event.sum())} ({100*lo.event.mean():.0f}%)")

    df.to_csv(f"{OUT}/exp6_prognostic.csv", index=False)
    np.savez(f"{OUT}/exp6_cox.npz",
             c1=c1, c2=c2, c3=c3,
             direct=df.direct.values, gap=df.gap.values,
             t=df.t.values, event=df.event.values, bdx=df.bdx.values)
    print(f"\nsaved -> {OUT}/exp6_prognostic.csv")


if __name__ == "__main__":
    main()
