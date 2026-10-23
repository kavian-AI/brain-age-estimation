#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DO NON-ALZHEIMER BRAIN YEARS IDENTIFY NON-ALZHEIMER NEURODEGENERATION?

E63 pre-registered that people who "fit neither" the healthy nor the Alzheimer
reference would be enriched for amyloid-NEGATIVE neurodegeneration (SNAP) - and it
failed (OR 1.36, p=0.36). But that test used coarse categories. E69 now gives a
continuous split of every gap into Alzheimer-shaped years and other years, and
showed amyloid and APOE4 load only on the former.

The sharper version of the same hypothesis: among people with CLEAR
neurodegeneration (high total gap), the COMPOSITION of the gap should tell
amyloid-positive from amyloid-negative. Pre-registered prediction: the share of
the gap made of other years is higher in amyloid-negative atrophy (SNAP).

Also: do SNAP-like people convert to dementia less, as the SNAP literature reports?
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
import exp69_decomposed_brain_age as D  # noqa: E402
from lifelines import CoxPHFitter  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


def main():
    a, m, T, names = R.build("X")
    cn = m.DX.values == "CN"
    Z, _ = R.normative(m, names, cn)
    gap = R.rich_gap(m, names, cn)
    gd, go, s = D.decompose(m, Z, gap)
    bl = a[a.VISCODE == "bl"].drop_duplicates("PTID").set_index("PTID")
    for c in ["AV45", "ABETA", "PTAU"]:
        m[c] = W.num(bl.reindex(m.PTID)[c]).values
    Apos = np.where(np.isfinite(m.AV45), m.AV45 > 1.11,
                    np.where(np.isfinite(m.ABETA), m.ABETA < 980, np.nan)).astype(float)
    m = m.assign(gap=gap, gd=gd, go=go, A=Apos)

    for grp in ["MCI", "CN"]:
        q = (m.DX.values == grp) & np.isfinite(m.A.values)
        M = m[q].copy()
        thr = np.percentile(M.gap, 66.7)
        H = M[M.gap >= thr]                     # clear neurodegeneration: top third
        print("=" * 90)
        print(f"{grp}: top third of total gap (>= {thr:+.1f} y), n={len(H)}, "
              f"amyloid-negative {int((H.A==0).sum())}, amyloid-positive {int((H.A==1).sum())}")
        print("=" * 90)
        for nm, col in [("total gap", "gap"), ("Alzheimer-shaped years", "gd"), ("other years", "go")]:
            neg, pos = H[col][H.A == 0], H[col][H.A == 1]
            p = mannwhitneyu(neg, pos).pvalue
            print(f"    {nm:24s} A- {neg.mean():+6.2f}   A+ {pos.mean():+6.2f}   p={p:.4f}")
        frac = H.go / H.gap
        auc = roc_auc_score(1 - H.A, H.go - H.gd)
        print(f"    share of the gap that is 'other': A- {frac[H.A==0].mean():.2f} vs A+ {frac[H.A==1].mean():.2f}"
              f"   p={mannwhitneyu(frac[H.A==0], frac[H.A==1]).pvalue:.4f}")
        print(f"    AUC of (other - Alzheimer years) for amyloid-NEGATIVE among the atrophied: {auc:.3f}")
        rng = np.random.RandomState(0)
        bs = []
        for _ in range(2000):
            i = rng.randint(0, len(H), len(H))
            if H.A.values[i].min() == H.A.values[i].max():
                continue
            bs.append(roc_auc_score(1 - H.A.values[i], (H.go - H.gd).values[i]))
        print(f"      95% CI [{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}]")
        # whole group, not only the atrophied: logistic on both parts
        auc_all = roc_auc_score(M.A, M.gd), roc_auc_score(M.A, M.go), roc_auc_score(M.A, M.gap)
        print(f"    amyloid-POSITIVE across all {grp}: AUC Alzheimer years {auc_all[0]:.3f}, "
              f"other years {auc_all[1]:.3f}, total gap {auc_all[2]:.3f}")

    # conversion of SNAP-like vs AD-like atrophy in MCI
    print("\n" + "=" * 90)
    print("DO SNAP-LIKE ATROPHIED MCI CONVERT LESS?")
    print("=" * 90)
    M = m[m.DX.values == "MCI"].copy()
    keep, dur, ev = E.survival(a, M.PTID.values, ["Dementia"])
    o = {x: i for i, x in enumerate(keep)}
    M["tt"] = [dur[o[x]] if x in o else np.nan for x in M.PTID]
    M["ee"] = [ev[o[x]] if x in o else np.nan for x in M.PTID]
    M = M.dropna(subset=["tt", "ee"])
    H = M[M.gap >= np.percentile(M.gap, 66.7)].copy()
    H["snap_like"] = (H.go > H.gd).astype(int)
    for lab, qq in [("mostly Alzheimer-shaped", H.snap_like == 0), ("mostly other (SNAP-like)", H.snap_like == 1)]:
        print(f"    {lab:26s} n={qq.sum():3d}  converted {100*H.ee[qq].mean():5.1f}%   "
              f"amyloid+ {100*np.nanmean(H.A[qq]):5.1f}%")
    cf = CoxPHFitter().fit(pd.DataFrame({"x": H.snap_like, "gap": H.gap, "age": H.age_visit,
                                         "T": H.tt, "E": H.ee}), "T", "E")
    print(f"    HR SNAP-like vs AD-like at the same total gap: {np.exp(cf.params_['x']):.2f} "
          f"[{np.exp(cf.confidence_intervals_.loc['x'].iloc[0]):.2f}, "
          f"{np.exp(cf.confidence_intervals_.loc['x'].iloc[1]):.2f}] p={cf.summary.loc['x','p']:.4f}")


if __name__ == "__main__":
    main()
