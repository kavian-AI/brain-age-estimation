#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The one thing the direct snapshot model CANNOT do: measure the RATE of brain
aging over time. This is intrinsically longitudinal and intrinsically a
brain-age quantity.

For each subject with >=3 volume scans:
  - brain-age gap at every visit  (age model trained on CN baseline)
  - baseline gap           = "how old the brain looks now" (reserve)
  - gap SLOPE              = "how fast the brain is aging"  (rate)   <- new signal

Question: does the RATE of brain aging predict future dementia conversion,
ADDING to the baseline direct diagnostic snapshot?

Design avoids leakage: we predict conversion AFTER the last scan used to
estimate the slope (landmark analysis), and the age model never sees patients.
"""

import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
DATA = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\data"
OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]


def train_reference():
    d = pd.read_csv(f"{DATA}/ADNIMERGE_05Mar2026.csv", low_memory=False)
    a = d.dropna(subset=V + ["ICV", "AGE", "DX", "PTGENDER"]).sort_values("Years_bl")
    a = a.groupby("RID").first().reset_index()
    ad = a[a.DX.isin(["CN", "Dementia"])]
    y = (ad.DX == "Dementia").astype(int).values
    feat = np.column_stack([ad[V].values, ad.ICV.values,
                            (ad.PTGENDER == "Male").astype(int).values, ad.AGE.values])
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


def gap_of(row_vol, row_age, row_icv, row_sex, agem, sca, bias):
    X = sca.transform(np.column_stack([row_vol, row_icv, row_sex]))
    raw = agem.predict(X) - row_age
    return raw - bias.predict(np.array(row_age).reshape(-1, 1))


def main():
    direct, scd, agem, sca, bias = train_reference()
    d = pd.read_csv(f"{DATA}/ADNIMERGE_05Mar2026.csv", low_memory=False)
    d = d.sort_values(["RID", "Years_bl"])

    rows = []
    LANDMARK_MIN = 1.0            # need >=1y of scan history to estimate a slope
    for rid, g in d.groupby("RID"):
        g = g.sort_values("Years_bl")
        if g.iloc[0]["DX"] == "Dementia":
            continue
        gv = g.dropna(subset=V + ["ICV", "AGE", "PTGENDER"])
        if gv.Years_bl.nunique() < 3:
            continue
        span = gv.Years_bl.max() - gv.Years_bl.min()
        if span < LANDMARK_MIN:
            continue
        # gap at each scan visit
        gaps = gap_of(gv[V].values, gv.AGE.values, gv.ICV.values,
                      (gv.PTGENDER == "Male").astype(int).values, agem, sca, bias)
        t = gv.Years_bl.values
        slope = np.polyfit(t - t[0], gaps, 1)[0]      # gap-years per real year
        base_gap = gaps[0]
        b = gv.iloc[0]
        feat = np.array([[*b[V].values.astype(float), b["ICV"],
                          1.0 if b["PTGENDER"] == "Male" else 0.0, b["AGE"]]])
        d_score = direct.predict_proba(scd.transform(feat))[0, 1]

        landmark = gv.Years_bl.max()                  # predict conversion AFTER here
        after = g[g.Years_bl > landmark]
        conv = after[after["DX"] == "Dementia"]
        ev = len(conv) > 0
        if ev:
            tt = conv.iloc[0]["Years_bl"] - landmark
        else:
            tt = g.Years_bl.max() - landmark
        if tt <= 0:
            tt = 0.01
        rows.append(dict(rid=rid, bdx=b["DX"], base_gap=base_gap, rate=slope,
                         direct=d_score, t=tt, event=int(ev)))
    df = pd.DataFrame(rows)
    for c in ["base_gap", "rate", "direct"]:
        df[c + "_z"] = (df[c] - df[c].mean()) / df[c].std()

    print(f"landmark cohort n={len(df)}, post-landmark conversions={int(df.event.sum())}")
    print(f"  brain-aging rate: mean={df.rate.mean():+.2f} gap-yr/yr, "
          f"SD={df.rate.std():.2f}")
    print(f"  converters rate={df[df.event==1].rate.mean():+.2f} vs "
          f"non-converters rate={df[df.event==0].rate.mean():+.2f}")
    print(f"  corr(direct, rate)={np.corrcoef(df.direct, df.rate)[0,1]:.3f}   "
          f"corr(base_gap, rate)={np.corrcoef(df.base_gap, df.rate)[0,1]:.3f}\n")

    print("=" * 66)
    print("COX: does brain-aging RATE add to the direct snapshot?")
    print("=" * 66)

    def cox(cols):
        m = CoxPHFitter().fit(df[cols + ["t", "event"]], "t", "event")
        c = concordance_index(df.t, -m.predict_partial_hazard(df), df.event)
        return m, c

    _, c_d = cox(["direct_z"])
    _, c_b = cox(["base_gap_z"])
    _, c_r = cox(["rate_z"])
    m_dr, c_dr = cox(["direct_z", "rate_z"])
    m_all, c_all = cox(["direct_z", "base_gap_z", "rate_z"])

    print(f"  direct snapshot alone        C = {c_d:.3f}")
    print(f"  baseline gap alone           C = {c_b:.3f}")
    print(f"  brain-aging RATE alone       C = {c_r:.3f}")
    print(f"  direct + rate                C = {c_dr:.3f}   (gain {c_dr-c_d:+.3f})")
    print(f"  direct + base_gap + rate     C = {c_all:.3f}   (gain {c_all-c_d:+.3f})")

    print("\n  --- hazard ratios in the full model ---")
    s = m_all.summary
    for nm in ["direct_z", "base_gap_z", "rate_z"]:
        hr = np.exp(s.loc[nm, "coef"])
        p = s.loc[nm, "p"]
        lo = np.exp(s.loc[nm, "coef lower 95%"])
        hi = np.exp(s.loc[nm, "coef upper 95%"])
        print(f"    {nm:12s} HR={hr:.2f} [{lo:.2f},{hi:.2f}]  p={p:.2e}")

    df.to_csv(f"{OUT}/exp8_rate.csv", index=False)
    np.savez(f"{OUT}/exp8_rate.npz", c_d=c_d, c_b=c_b, c_r=c_r, c_dr=c_dr, c_all=c_all,
             rate=df.rate.values, event=df.event.values, direct=df.direct.values,
             base_gap=df.base_gap.values)
    print(f"\nsaved -> {OUT}/exp8_rate.csv")


if __name__ == "__main__":
    main()
