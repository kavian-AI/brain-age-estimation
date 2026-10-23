#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DECOMPOSED BRAIN AGE: HOW MANY OF YOUR EXTRA YEARS ARE ALZHEIMER'S?

A brain-age gap is one number: "your brain looks 5 years older". The alignment
law (E67) says that number is a projection w'z of the regional deviations, and
that only part of w points along the disease direction. So the gap splits,
additively and in the same unit (years), into

    gap = gap_disease + gap_other

  gap_disease = beta * s, where s = (S^-1 mu)'z is the disease discriminant score
                and beta = cov(gap, s) / var(s) in controls
  gap_other   = gap - gap_disease: by construction uncorrelated with s in
                controls, so its expected Alzheimer shift is zero (w_other' mu = 0)

The split is fitted out of fold (S, mu and beta from training subjects only).
AD-vs-CN separation lives in gap_disease BY CONSTRUCTION - that part is not a
finding. Everything else is an independent test the split never saw:
  - MCI -> AD conversion (the fitting used only CN and AD)
  - amyloid PET, CSF p-tau, FDG in CN and MCI
  - cognitive decline in healthy people (E64-C: the gap predicts it - which part?)
  - the 1.3-year sex bias (E64b) and head size - which part carries them?
  - education and APOE4
And the descriptive headline: of an Alzheimer patient's extra years, how many are
Alzheimer-shaped, and of a healthy person's gap, how much is disease at all?
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp53_units_and_dimension as E  # noqa: E402
import exp62_rich_atlas as R  # noqa: E402
import exp64_what_it_measures as W  # noqa: E402
from lifelines import CoxPHFitter  # noqa: E402
from lifelines.utils import concordance_index  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

N_BOOT = 1000


def decompose(m, Z, gap):
    f = E.folds_of(m.PTID.values, seed=11)
    dx = m.DX.values
    s = np.zeros(len(m))
    gd = np.zeros(len(m))
    for k in range(5):
        tr, te = f != k, f == k
        cn_tr, ad_tr = tr & (dx == "CN"), tr & (dx == "Dementia")
        S = LedoitWolf().fit(np.vstack([Z[cn_tr] - Z[cn_tr].mean(0),
                                        Z[ad_tr] - Z[ad_tr].mean(0)])).covariance_
        mu = Z[ad_tr].mean(0) - Z[cn_tr].mean(0)
        v = np.linalg.solve(S, mu)
        s_tr, s_te = Z[cn_tr] @ v, Z[te] @ v
        beta = np.cov(gap[cn_tr], s_tr)[0, 1] / np.var(s_tr, ddof=1)
        s[te] = s_te
        gd[te] = beta * (s_te - s_tr.mean())      # centred on controls
    return gd, gap - gd, s


def boot_diff_r(x1, x2, y, C, n=N_BOOT):
    ok = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y) & np.all(np.isfinite(C), 1)
    x1, x2, y, C = x1[ok], x2[ok], y[ok], C[ok]
    rng = np.random.RandomState(0)
    out = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        out.append(W.partial_r(x1[i], y[i], C[i])[0] - W.partial_r(x2[i], y[i], C[i])[0])
    return np.percentile(out, [2.5, 97.5])


def main():
    a, m, T, names = R.build("X")
    cn = m.DX.values == "CN"
    Z, _ = R.normative(m, names, cn)
    gap = R.rich_gap(m, names, cn)
    gd, go, s = decompose(m, Z, gap)
    m = m.assign(gap=gap, g_dis=gd, g_oth=go)

    print("=" * 92)
    print("HOW MANY OF THE EXTRA YEARS ARE ALZHEIMER-SHAPED?  (98-region brain age, out of fold)")
    print("=" * 92)
    print(f"  {'group':10s} {'n':>5s} {'total gap':>10s} {'= disease years':>16s} {'+ other years':>14s}")
    for g in ["CN", "MCI", "Dementia"]:
        q = m.DX.values == g
        print(f"  {g:10s} {q.sum():5d} {gap[q].mean():+10.2f} {gd[q].mean():+16.2f} {go[q].mean():+14.2f}")
    print(f"\n  in healthy people, share of gap VARIANCE that is disease-shaped: "
          f"{100*np.var(gd[cn])/np.var(gap[cn]):.0f}%   corr(disease, other) in CN "
          f"{np.corrcoef(gd[cn], go[cn])[0,1]:+.3f}")
    sel = m.DX.isin(["CN", "Dementia"]).values
    y = (m.DX.values[sel] == "Dementia").astype(int)
    for nm, x in [("total gap", gap), ("disease years", gd), ("other years", go)]:
        print(f"  AD vs CN AUC  {nm:14s} {roc_auc_score(y, x[sel]):.3f}"
              + ("   (by construction)" if nm != "total gap" else ""))

    # ------------------------------------------------------------ conversion
    print("\n" + "=" * 92)
    print("INDEPENDENT TEST 1 - MCI -> AD CONVERSION (never used in the split)")
    print("=" * 92)
    pool = m[m.DX == "MCI"]
    keep, dur, ev = E.survival(a, pool.PTID.values, ["Dementia"])
    idx = pool.index[pool.PTID.isin(set(keep))].values
    o = {x: i for i, x in enumerate(keep)}
    ii = np.array([o[x] for x in m.PTID.values[idx]])
    dur, ev = dur[ii], ev[ii]
    agC = m.age_visit.values[idx]
    for nm, x in [("total gap", gap), ("disease years", gd), ("other years", go)]:
        c = np.nanmean([R.cidx(np.column_stack([x[idx], agC]), dur, ev, sd) for sd in range(5)])
        D = pd.DataFrame({"x": x[idx] / np.std(x[idx]), "age": agC, "T": dur, "E": ev})
        cf = CoxPHFitter().fit(D, "T", "E")
        print(f"    {nm:14s} C = {c:.3f}   HR per SD {np.exp(cf.params_['x']):.2f} "
              f"[{np.exp(cf.confidence_intervals_.loc['x'].iloc[0]):.2f}, "
              f"{np.exp(cf.confidence_intervals_.loc['x'].iloc[1]):.2f}]  p={cf.summary.loc['x','p']:.2g}")
    D = pd.DataFrame({"dis": gd[idx] / np.std(gd[idx]), "oth": go[idx] / np.std(go[idx]),
                      "age": agC, "T": dur, "E": ev})
    cf = CoxPHFitter().fit(D, "T", "E")
    print(f"    both in one model:  disease HR {np.exp(cf.params_['dis']):.2f} p={cf.summary.loc['dis','p']:.2g}"
          f"   other HR {np.exp(cf.params_['oth']):.2f} p={cf.summary.loc['oth','p']:.2g}")
    rng = np.random.RandomState(0)
    dd = []
    for _ in range(300):
        i = rng.randint(0, len(dur), len(dur))
        if ev[i].sum() < 20:
            continue
        p1 = R.cidx(np.column_stack([gd[idx][i], agC[i]]), dur[i], ev[i], 0)
        p2 = R.cidx(np.column_stack([gap[idx][i], agC[i]]), dur[i], ev[i], 0)
        if np.isfinite(p1) and np.isfinite(p2):
            dd.append(p1 - p2)
    R.show("disease years minus total gap (C-index)", dd)

    # ------------------------------------------------------------ biology
    bl = a[a.VISCODE == "bl"].drop_duplicates("PTID").set_index("PTID")
    for c in ["AV45", "PTAU", "ABETA", "FDG", "PTEDUCAT", "APOE4"]:
        m[c] = W.num(bl.reindex(m.PTID)[c]).values
    print("\n" + "=" * 92)
    print("INDEPENDENT TEST 2 - BIOLOGY (partial r, age + sex)")
    print("=" * 92)
    for grp in ["CN", "MCI"]:
        q = m.DX.values == grp
        C = np.column_stack([m.age_visit.values[q], m.SEX.values[q]])
        print(f"  {grp}")
        for bn, bv in [("amyloid PET", m.AV45.values), ("CSF p-tau", m.PTAU.values),
                       ("FDG-PET (low=bad)", -m.FDG.values), ("education", m.PTEDUCAT.values),
                       ("APOE4 alleles", m.APOE4.values)]:
            cells = []
            for x in [gap, gd, go]:
                r, n = W.partial_r(x[q], bv[q], C)
                lo, hi = W.boot_pr(x[q], bv[q], C, n=400)
                cells.append(f"{r:+.3f}{'*' if lo > 0 or hi < 0 else ' '}")
            print(f"    {bn:20s} total {cells[0]}  disease {cells[1]}  other {cells[2]}  (n={n})")

    print("\n" + "=" * 92)
    print("INDEPENDENT TEST 3 - COGNITIVE DECLINE IN HEALTHY PEOPLE (partial r, age+sex+education)")
    print("=" * 92)
    rows = []
    for sid, g in a[a.PTID.isin(m.PTID[cn])].groupby("PTID"):
        for cog in ["mPACCdigit", "ADAS13"]:
            gg = g[["Years_bl", cog]].copy()
            gg["v"] = W.num(gg[cog])
            gg = gg.dropna(subset=["Years_bl", "v"])
            if len(gg) >= 3 and gg.Years_bl.max() - gg.Years_bl.min() >= 2:
                rows.append((sid, cog, np.polyfit(gg.Years_bl.values, gg.v.values, 1)[0]))
    Sdf = pd.DataFrame(rows, columns=["PTID", "cog", "slope"])
    for cog, worse in [("mPACCdigit", -1), ("ADAS13", 1)]:
        Tt = Sdf[Sdf.cog == cog].merge(m[["PTID", "gap", "g_dis", "g_oth", "age_visit", "SEX",
                                          "PTEDUCAT"]], on="PTID")
        yv = worse * Tt.slope.values
        C = np.column_stack([Tt.age_visit, Tt.SEX, Tt.PTEDUCAT]).astype(float)
        cells = []
        for col in ["gap", "g_dis", "g_oth"]:
            r, n = W.partial_r(Tt[col].values, yv, C)
            lo, hi = W.boot_pr(Tt[col].values, yv, C, n=600)
            cells.append(f"{r:+.3f} [{lo:+.2f},{hi:+.2f}]")
        lo, hi = boot_diff_r(Tt.g_dis.values, Tt.g_oth.values, yv, C)
        print(f"  {cog:10s} n={n}  total {cells[0]}  disease {cells[1]}  other {cells[2]}"
              f"   disease-minus-other [{lo:+.2f},{hi:+.2f}]")

    print("\n" + "=" * 92)
    print("INDEPENDENT TEST 4 - WHERE DO THE ARTEFACTS LIVE?  (healthy people)")
    print("=" * 92)
    for nm, x in [("total gap", gap), ("disease years", gd), ("other years", go)]:
        mm, ww = x[cn & (m.SEX.values == 1)], x[cn & (m.SEX.values == 0)]
        r_icv = np.corrcoef(x[cn], m.ICV_T.values[cn])[0, 1]
        print(f"    {nm:14s} men minus women {mm.mean()-ww.mean():+.2f} y    corr with ICV {r_icv:+.3f}")


if __name__ == "__main__":
    main()
