#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DOES ALZHEIMER'S POINT IN A DIFFERENT DIRECTION AT EACH STAGE?

E73 found amyloid-positive healthy people change faster, but NOT along the
dementia-derived Alzheimer direction. If the disease's structural signature
rotates between stages, then any single disease axis - and brain age, which is a
fixed axis - is tuned to one stage and partly blind to the others.

Three directions, each estimated from data:
  PRECLINICAL  rate:  mean regional slope, amyloid+ minus amyloid- healthy people
  PRODROMAL    rate:  mean regional slope, MCI who convert minus MCI who do not
  DEMENTIA     level: mean regional deviation, AD minus CN at baseline
Questions:
  1  the angles between them (Euclidean cosine of standardised effect maps, and
     split-half reliability of each map so an angle can be judged against noise)
  2  which regions define the preclinical direction
  3  out of fold: does projecting a healthy person's regional slopes onto the
     preclinical direction identify amyloid status better than the brain-age slope,
     the Alzheimer-year slope or the hippocampal slope?
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
import exp73_decomposed_rates as X  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def main():
    a, d, T = X.load_all_visits()
    names = list(T.name)
    S = T.sign.values
    ag = d.age_visit.values.astype(float)
    Xn = np.column_stack([ag, ag ** 2, d.SEX.values, d.ICV_T.values])
    Y = d[names].values.astype(float)
    cnb = d.is_bl.values & (d.DX.values == "CN")
    s1 = StandardScaler().fit(Xn[cnb])
    lr = LinearRegression().fit(s1.transform(Xn[cnb]), Y[cnb])
    sd = (Y[cnb] - lr.predict(s1.transform(Xn[cnb]))).std(0, ddof=1)
    Z = (Y - lr.predict(s1.transform(Xn))) / sd * S          # signed: higher = worse
    for j, n in enumerate(names):
        d["z_" + n] = Z[:, j]
    zc = ["z_" + n for n in names]

    # per-subject regional slopes of the (age-adjusted) deviations
    rows, slopes = [], []
    for s, g in d.groupby("PTID"):
        t = g.Years_bl.values
        if len(g) < 3 or t.max() - t.min() < 2:
            continue
        tc = t - t.mean()
        slopes.append((tc @ (g[zc].values - g[zc].values.mean(0))) / (tc @ tc))
        rows.append(dict(PTID=s, DX0=g.DX0.iloc[0], age=g.age_visit.iloc[0], SEX=g.SEX.iloc[0]))
    P = pd.DataFrame(rows)
    SL = np.array(slopes)
    b0 = a[a.VISCODE == "bl"].drop_duplicates("PTID").set_index("PTID")
    for c in ["AV45", "ABETA"]:
        P[c] = W.num(b0.reindex(P.PTID)[c]).values
    P["A"] = np.where(np.isfinite(P.AV45), P.AV45 > 1.11,
                      np.where(np.isfinite(P.ABETA), P.ABETA < 980, np.nan)).astype(float)
    conv = set(a[a.DX == "Dementia"].PTID)
    P["conv"] = P.PTID.isin(conv).astype(int)

    # residualise slopes on age and sex so directions are not age artefacts
    Cov = np.column_stack([np.ones(len(P)), P.age, P.SEX])
    SLr = SL - Cov @ np.linalg.lstsq(Cov, SL, rcond=None)[0]

    def dmap(A_, B_, M):
        return (M[A_].mean(0) - M[B_].mean(0)) / np.sqrt(0.5 * (M[A_].var(0, ddof=1) + M[B_].var(0, ddof=1)))

    cn = (P.DX0 == "CN").values & P.A.notna().values
    pre_pos, pre_neg = cn & (P.A.values == 1), cn & (P.A.values == 0)
    mci = (P.DX0 == "MCI").values
    pro_pos, pro_neg = mci & (P.conv.values == 1), mci & (P.conv.values == 0)
    bl = d[d.is_bl]
    Zb = bl[zc].values
    dem = dmap((bl.DX == "Dementia").values, (bl.DX == "CN").values, Zb)
    pre = dmap(pre_pos, pre_neg, SLr)
    pro = dmap(pro_pos, pro_neg, SLr)
    print(f"preclinical: A+ {pre_pos.sum()} vs A- {pre_neg.sum()} healthy | prodromal: converters "
          f"{pro_pos.sum()} vs stable {pro_neg.sum()} MCI | dementia: baseline AD vs CN")

    # split-half reliability of each map (random halves of subjects)
    def rel(A_, B_, M, reps=200):
        rng = np.random.RandomState(0)
        ia, ib = np.where(A_)[0], np.where(B_)[0]
        r = []
        for _ in range(reps):
            pa, pb = rng.permutation(ia), rng.permutation(ib)
            h1 = dmap(np.isin(np.arange(len(M)), pa[:len(pa)//2]), np.isin(np.arange(len(M)), pb[:len(pb)//2]), M)
            h2 = dmap(np.isin(np.arange(len(M)), pa[len(pa)//2:]), np.isin(np.arange(len(M)), pb[len(pb)//2:]), M)
            r.append(np.corrcoef(h1, h2)[0, 1])
        rr = np.mean(r)
        return 2 * rr / (1 + rr)
    R_pre, R_pro = rel(pre_pos, pre_neg, SLr), rel(pro_pos, pro_neg, SLr)
    R_dem = rel((bl.DX == "Dementia").values, (bl.DX == "CN").values, Zb, reps=50)
    print(f"map reliability (split-half, Spearman-Brown): preclinical {R_pre:.2f}, prodromal {R_pro:.2f}, dementia {R_dem:.2f}")

    print("\n" + "=" * 90)
    print("1  ANGLES BETWEEN STAGES  (correlation of regional effect maps; disattenuated in brackets)")
    print("=" * 90)
    for (na, ma, ra), (nb, mb, rb) in [(("preclinical", pre, R_pre), ("prodromal", pro, R_pro)),
                                       (("preclinical", pre, R_pre), ("dementia", dem, R_dem)),
                                       (("prodromal", pro, R_pro), ("dementia", dem, R_dem))]:
        r = np.corrcoef(ma, mb)[0, 1]
        print(f"    {na:12s} vs {nb:10s} r = {r:+.3f}   [ceiling-corrected {r/np.sqrt(max(ra,1e-3)*max(rb,1e-3)):+.2f}]")

    print("\n" + "=" * 90)
    print("2  WHAT DEFINES EACH STAGE: top regions (standardised effect)")
    print("=" * 90)
    for nm, mp in [("preclinical rate", pre), ("prodromal rate", pro), ("dementia level", dem)]:
        top = np.argsort(-mp)[:8]
        print(f"  {nm:17s} " + ", ".join(f"{names[j]} {mp[j]:+.2f}" for j in top))
    cls = T.cls.values
    for nm, mp in [("preclinical", pre), ("prodromal", pro), ("dementia", dem)]:
        pos = np.clip(mp, 0, None)
        print(f"  share of positive effect in MTL - {nm:11s} {100*pos[cls=='mtl'].sum()/pos.sum():.0f}%"
              f"   cortex {100*pos[cls=='cortex'].sum()/pos.sum():.0f}%   ventricles {100*pos[cls=='ventricle'].sum()/pos.sum():.0f}%")

    print("\n" + "=" * 90)
    print("3  OUT OF FOLD: which slope best identifies amyloid+ in healthy people?")
    print("=" * 90)
    Q = np.where(cn)[0]
    yA = P.A.values[Q]
    f = np.random.RandomState(1).permutation(len(Q)) % 5
    score_pre = np.zeros(len(Q))
    for k in range(5):
        tr, te = f != k, f == k
        mp = dmap(np.isin(np.arange(len(P)), Q[tr][yA[tr] == 1]), np.isin(np.arange(len(P)), Q[tr][yA[tr] == 0]), SLr)
        score_pre[te] = SLr[Q[te]] @ np.clip(mp, 0, None)
    hip = np.array([n in ("L_hippocampus", "R_hippocampus") for n in names])
    mtl = cls == "mtl"
    cands = {"preclinical direction (out of fold)": score_pre,
             "dementia direction (level map as weights)": SLr[Q] @ np.clip(dem, 0, None),
             "uniform over all 98 regions": SLr[Q].mean(1),
             "MTL average": SLr[Q][:, mtl].mean(1),
             "hippocampus": SLr[Q][:, hip].mean(1)}
    rng = np.random.RandomState(2)
    for nm, sc in cands.items():
        au = roc_auc_score(yA, sc)
        bs = []
        for _ in range(1000):
            i = rng.randint(0, len(yA), len(yA))
            if yA[i].min() != yA[i].max():
                bs.append(roc_auc_score(yA[i], sc[i]))
        print(f"    {nm:42s} AUC {au:.3f} [{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}]")


if __name__ == "__main__":
    main()
