#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
External validation on OASIS-2 (independent cohort, different scanner,
different features - only whole-brain nWBV, not FreeSurfer regions).

We cannot transfer the exact ADNI model (features differ), so we validate the
PHENOMENON: does the rate of brain aging predict conversion, adding to a
cross-sectional snapshot?  Reference models are built from OASIS-1.

OASIS-1: cross-sectional, CN vs AD  -> trains age model (CN) + direct model.
OASIS-2: longitudinal              -> compute per-visit gap -> rate; test.
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
F = ["nWBV", "eTIV", "sex"]


def train_oasis1():
    d = pd.read_csv(f"{DATA}/oasis_cross-sectional.csv")
    d = d.dropna(subset=["Age", "nWBV", "eTIV", "CDR"]).copy()
    d["sex"] = (d["M/F"] == "M").astype(int)
    # age model on healthy (CDR=0)
    h = d[d.CDR == 0]
    sca = StandardScaler().fit(h[F].values)
    agem = LinearRegression().fit(sca.transform(h[F].values), h.Age.values)
    bias = LinearRegression().fit(h.Age.values.reshape(-1, 1),
                                  agem.predict(sca.transform(h[F].values)) - h.Age.values)
    # direct model CN vs impaired, add age as feature
    y = (d.CDR > 0).astype(int).values
    Xd = np.column_stack([d[F].values, d.Age.values])
    scd = StandardScaler().fit(Xd)
    direct = LogisticRegression(max_iter=3000).fit(scd.transform(Xd), y)
    return sca, agem, bias, scd, direct


def main():
    sca, agem, bias, scd, direct = train_oasis1()

    lg = pd.read_csv(f"{DATA}/oasis_longitudinal.csv").sort_values(
        ["Subject ID", "MR Delay"])
    lg = lg.dropna(subset=["Age", "nWBV", "eTIV", "CDR"]).copy()
    lg["sex"] = (lg["M/F"] == "M").astype(int)

    rows = []
    for sid, g in lg.groupby("Subject ID"):
        g = g.sort_values("MR Delay")
        if g.iloc[0]["CDR"] > 0:            # already impaired at baseline
            continue
        if g["MR Delay"].nunique() < 2:
            continue
        yrs = g["MR Delay"].values / 365.25
        X = g[F].values
        gap = (agem.predict(sca.transform(X)) - g.Age.values) \
            - bias.predict(g.Age.values.reshape(-1, 1))
        rate = np.polyfit(yrs - yrs[0], gap, 1)[0]
        base = g.iloc[0]
        d_score = direct.predict_proba(
            scd.transform(np.array([[*base[F].values, base["Age"]]])))[0, 1]
        conv = g[g["CDR"] > 0]
        ev = len(conv) > 0
        t = (conv["MR Delay"].iloc[0] if ev else g["MR Delay"].iloc[-1]) / 365.25
        rows.append(dict(sid=sid, base_gap=gap[0], rate=rate, direct=d_score,
                         event=int(ev), t=max(t, 0.01)))
    df = pd.DataFrame(rows)
    print(f"OASIS-2 validation cohort: n={len(df)}, conversions={int(df.event.sum())}")
    print(f"  brain-aging rate: converters {df[df.event==1].rate.mean():+.2f} vs "
          f"non {df[df.event==0].rate.mean():+.2f} gap-yr/yr")
    print(f"  baseline gap:     converters {df[df.event==1].base_gap.mean():+.2f} vs "
          f"non {df[df.event==0].base_gap.mean():+.2f}")

    for c in ["direct", "base_gap", "rate"]:
        df[c + "_z"] = (df[c] - df[c].mean()) / (df[c].std() + 1e-9)

    def cox(cols):
        m = CoxPHFitter(penalizer=0.1).fit(df[cols + ["t", "event"]], "t", "event")
        c = concordance_index(df.t, -m.predict_partial_hazard(df), df.event)
        return m, c

    print("\n" + "=" * 60)
    print("OASIS-2: does the rate replicate as a predictor?")
    print("=" * 60)
    _, c_d = cox(["direct_z"])
    _, c_r = cox(["rate_z"])
    _, c_dr = cox(["direct_z", "rate_z"])
    print(f"  direct snapshot alone   C = {c_d:.3f}")
    print(f"  brain-aging rate alone  C = {c_r:.3f}")
    print(f"  direct + rate           C = {c_dr:.3f}   (gain {c_dr-c_d:+.3f})")

    m, _ = cox(["direct_z", "rate_z"])
    s = m.summary
    for nm in ["direct_z", "rate_z"]:
        hr = np.exp(s.loc[nm, "coef"]); p = s.loc[nm, "p"]
        print(f"    {nm:10s} HR={hr:.2f}  p={p:.3f}")

    # simple risk tertiles by rate
    print("\n  conversion rate by brain-aging-rate tertile:")
    df["tert"] = pd.qcut(df.rate, 3, labels=["slow", "mid", "fast"])
    for tv in ["slow", "mid", "fast"]:
        s = df[df.tert == tv]
        print(f"    {tv:5s} n={len(s):3d}  converted {int(s.event.sum())} "
              f"({100*s.event.mean():.0f}%)")

    df.to_csv(f"{OUT}/exp10_oasis.csv", index=False)
    print(f"\nsaved -> {OUT}/exp10_oasis.csv")


if __name__ == "__main__":
    main()
