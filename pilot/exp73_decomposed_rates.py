#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DECOMPOSED BRAIN AGE OVER TIME: CAN ALZHEIMER-SHAPED YEARS MEASURE A RATE?

Grodem, Smith, Vidal-Pineiro, Fjell (medRxiv 2026) argue brain age cannot measure
individual aging rates: the age objective suppresses exactly the features whose
change differs between people. E64-B found every cross-sectional summary blind to
preclinical Alzheimer's (amyloid-positive healthy people). Two questions follow
from the decomposition (E69):

  1  RATE. In cognitively normal people, do amyloid-POSITIVE individuals accumulate
     Alzheimer-shaped years faster than amyloid-negative ones - even though their
     total brain-age slope may not differ? If yes, the disease component measures
     a disease-specific rate that brain age dilutes.
  2  UNITS. E53a found one year of gap worth 5.6 years of age in MCI and nothing
     in controls: a population-dependent unit. If a gap-year is a MIXTURE of heavy
     disease-years and light other-years whose composition differs by population,
     the hazard per DISEASE-year should be far more stable across populations than
     the hazard per gap-year.

Every visit of 1735 ADNI subjects with the 98-region cross-sectional FreeSurfer
table. The normative model, the age model and the decomposition are fitted on
BASELINE data out of fold by subject, then applied unchanged to every later visit.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp53_units_and_dimension as E  # noqa: E402
import exp64_what_it_measures as W  # noqa: E402
import tadpole_atlas as TA  # noqa: E402
from lifelines import CoxPHFitter  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402
from sklearn.linear_model import LinearRegression, Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

FOLDS = 5


def load_all_visits():
    T = TA.region_table("CV")
    XP = TA.PIPE["X"]
    cols = [f"ST{c}{k}_{XP}" for c, k in zip(T.code, T.kind)]
    t = pd.read_csv(TA.TADPOLE, low_memory=False, usecols=["PTID", "VISCODE"] + cols + [f"ST10CV_{XP}"])
    R = pd.DataFrame({"PTID": t.PTID, "VISCODE": t.VISCODE,
                      "ICV_T": pd.to_numeric(t[f"ST10CV_{XP}"], errors="coerce")})
    for nm, c in zip(T.name, cols):
        R[nm] = pd.to_numeric(t[c], errors="coerce").values
    a = pd.read_csv(os.path.join(E.DATA, "ADNIMERGE_05Mar2026.csv"), low_memory=False)
    A = a[["PTID", "VISCODE", "AGE", "Years_bl", "PTGENDER", "DX"]]
    d = R.merge(A, on=["PTID", "VISCODE"]).dropna(subset=list(T.name) + ["ICV_T", "AGE", "Years_bl"])
    d["SEX"] = (d.PTGENDER == "Male").astype(int)
    d["age_visit"] = d.AGE + d.Years_bl
    d = d.sort_values(["PTID", "Years_bl"]).reset_index(drop=True)
    d["DX0"] = d.PTID.map(d.groupby("PTID").DX.first())
    d["is_bl"] = ~d.PTID.duplicated()
    return a, d, T


def main():
    a, d, T = load_all_visits()
    names = list(T.name)
    subs = d.PTID.values
    uq = np.unique(subs)
    fm = dict(zip(uq, np.random.RandomState(0).permutation(len(uq)) % FOLDS))
    f = np.array([fm[s] for s in subs])
    ag = d.age_visit.values.astype(float)
    Xn = np.column_stack([ag, ag ** 2, d.SEX.values, d.ICV_T.values])
    Y = d[names].values.astype(float)
    Xa = np.column_stack([Y, d.ICV_T.values, d.SEX.values])
    bl = d.is_bl.values
    cnb, adb = bl & (d.DX.values == "CN"), bl & (d.DX.values == "Dementia")
    Z = np.zeros_like(Y)
    gap = np.zeros(len(d))
    gd = np.zeros(len(d))
    for k in range(FOLDS):
        tr, te = f != k, f == k
        c_tr, a_tr = tr & cnb, tr & adb
        s1 = StandardScaler().fit(Xn[c_tr])
        lr = LinearRegression().fit(s1.transform(Xn[c_tr]), Y[c_tr])
        sd = (Y[c_tr] - lr.predict(s1.transform(Xn[c_tr]))).std(0, ddof=1)
        Zk = (Y - lr.predict(s1.transform(Xn))) / sd
        Z[te] = Zk[te]
        s2 = StandardScaler().fit(Xa[c_tr])
        md = Ridge(alpha=316.0).fit(s2.transform(Xa[c_tr]), ag[c_tr])
        bb = LinearRegression().fit(ag[c_tr].reshape(-1, 1), md.predict(s2.transform(Xa[c_tr])) - ag[c_tr])
        gk = (md.predict(s2.transform(Xa)) - ag) - bb.predict(ag.reshape(-1, 1))
        S = LedoitWolf().fit(np.vstack([Zk[c_tr] - Zk[c_tr].mean(0), Zk[a_tr] - Zk[a_tr].mean(0)])).covariance_
        v = np.linalg.solve(S, Zk[a_tr].mean(0) - Zk[c_tr].mean(0))
        s_all = Zk @ v
        beta = np.cov(gk[c_tr], s_all[c_tr])[0, 1] / np.var(s_all[c_tr], ddof=1)
        gap[te] = gk[te]
        gd[te] = beta * (s_all[te] - s_all[c_tr].mean())
    d["gap"], d["gd"], d["go"] = gap, gd, gap - gd
    print(f"visits {len(d)}, subjects {d.PTID.nunique()}")

    # --------------------------------------------------------- per-subject slopes
    rows = []
    for s, g in d.groupby("PTID"):
        t = g.Years_bl.values
        if len(g) < 3 or t.max() - t.min() < 2:
            continue
        r = dict(PTID=s, DX0=g.DX0.iloc[0], age=g.age_visit.iloc[0], SEX=g.SEX.iloc[0], n=len(g))
        for c in ["gap", "gd", "go"]:
            r["b_" + c] = g[c].iloc[0]
            r["sl_" + c] = np.polyfit(t, g[c].values, 1)[0]
        if len(g) >= 6:
            h1, h2 = g.iloc[0::2], g.iloc[1::2]
            for c in ["gd", "go", "gap"]:
                r["h1_" + c] = np.polyfit(h1.Years_bl, h1[c], 1)[0]
                r["h2_" + c] = np.polyfit(h2.Years_bl, h2[c], 1)[0]
        rows.append(r)
    P = pd.DataFrame(rows)
    b0 = a[a.VISCODE == "bl"].drop_duplicates("PTID").set_index("PTID")
    for c in ["AV45", "ABETA"]:
        P[c] = W.num(b0.reindex(P.PTID)[c]).values
    P["A"] = np.where(np.isfinite(P.AV45), P.AV45 > 1.11,
                      np.where(np.isfinite(P.ABETA), P.ABETA < 980, np.nan)).astype(float)
    print(f"subjects with >=3 visits over >=2 y: {len(P)}")
    print("\n  reliability of individual slopes (split-half, Spearman-Brown), n with >=6 visits:")
    H = P.dropna(subset=["h1_gd"])
    for c, nm in [("gap", "total"), ("gd", "Alzheimer years"), ("go", "other years")]:
        r = np.corrcoef(H["h1_" + c], H["h2_" + c])[0, 1]
        print(f"    {nm:16s} r = {r:.3f} -> reliability {2*r/(1+r):.3f}   (n={len(H)})")

    print("\n" + "=" * 92)
    print("1  PRECLINICAL ALZHEIMER'S: amyloid+ vs amyloid- cognitively normal, SLOPES (years per year)")
    print("=" * 92)
    Q = P[(P.DX0 == "CN") & P.A.notna()]
    print(f"  A+ n={int((Q.A==1).sum())}, A- n={int((Q.A==0).sum())}")
    for c, nm in [("gap", "total brain age"), ("gd", "Alzheimer years"), ("go", "other years")]:
        C = np.column_stack([np.ones(len(Q)), Q.age, Q.SEX])
        res = Q["sl_" + c].values - C @ np.linalg.lstsq(C, Q["sl_" + c].values, rcond=None)[0]
        pos, neg = res[Q.A.values == 1], res[Q.A.values == 0]
        rng = np.random.RandomState(0)
        bs = [rng.choice(pos, len(pos)).mean() - rng.choice(neg, len(neg)).mean() for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        sdv = res.std()
        print(f"    {nm:16s} A+ minus A- slope {pos.mean()-neg.mean():+.3f} y/y [{lo:+.3f},{hi:+.3f}]"
              f"  = {(pos.mean()-neg.mean())/sdv:+.2f} SD   p(MW)={mannwhitneyu(pos, neg).pvalue:.4f}"
              f"  {'DETECTS' if lo > 0 else '-'}")
    print("  (cross-sectional baseline levels, for contrast:)")
    for c, nm in [("gap", "total brain age"), ("gd", "Alzheimer years")]:
        print(f"    {nm:16s} baseline A+ minus A- {Q['b_'+c][Q.A==1].mean()-Q['b_'+c][Q.A==0].mean():+.2f} y"
              f"  p={mannwhitneyu(Q['b_'+c][Q.A==1], Q['b_'+c][Q.A==0]).pvalue:.3f}")

    print("\n" + "=" * 92)
    print("2  UNITS: hazard per year of each component, in two populations")
    print("=" * 92)
    B = d[d.is_bl].copy()
    for pop, targets in [("MCI", ["Dementia"]), ("CN", ["MCI", "Dementia"])]:
        M = B[B.DX == pop]
        keep, dur, ev = E.survival(a, M.PTID.values, targets)
        o = {x: i for i, x in enumerate(keep)}
        M = M[M.PTID.isin(o)].copy()
        M["T"] = [dur[o[x]] for x in M.PTID]
        M["E"] = [ev[o[x]] for x in M.PTID]
        out = []
        for c in ["gap", "gd", "go"]:
            cf = CoxPHFitter().fit(M[[c, "age_visit", "T", "E"]], "T", "E")
            ci = np.exp(cf.confidence_intervals_.loc[c])
            out.append(f"{c}: HR/year {np.exp(cf.params_[c]):.3f} [{ci.iloc[0]:.3f},{ci.iloc[1]:.3f}]")
        cfa = CoxPHFitter().fit(M[["age_visit", "T", "E"]], "T", "E")
        print(f"  {pop:4s} (n={len(M)}, events {int(M.E.sum())}):  " + "   ".join(out)
              + f"   chronological age HR/year {np.exp(cfa.params_['age_visit']):.3f}")
        print(f"        mean composition of the gap in this group: Alzheimer {M.gd.mean():+.2f}, other {M.go.mean():+.2f}")


if __name__ == "__main__":
    main()
