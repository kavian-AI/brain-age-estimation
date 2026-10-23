#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IS THE DEVIATION AXIS DISEASE-SPECIFIC, OR IS IT ONE GENERIC ILLNESS AXIS?

E48 found that a particular rotation of the regional deviations beats the
brain-age gap on Alzheimer's. The obvious next question is whether that rotation
is ABOUT Alzheimer's at all. If the same axis separates cases from controls in
autism and in ADHD - two conditions with nothing pathological in common with AD -
then what we found is a generic "something is wrong with this brain" direction,
and calling it a disease axis is a mistake. If the axes point in different
directions, then each condition needs its own rotation and E48's result is
specific.

    TIER 1   ADNI vs ABIDE vs ADHD-200, in the three regions all three share
    TIER 2   ABIDE vs ADHD-200, in the full shared aseg space (~25 regions)

Tier 2 exists because tier 1 has only three dimensions, and three dimensions can
hide a real difference. Tier 2 cannot include ADNI - ADNIMERGE ships six regions
and no full segmentation - but it can say whether two different conditions in the
SAME feature space share an axis.

THE HONEST CAVEAT, stated up front: ABIDE is 6-58 and ADHD-200 is 7-21, while
ADNI is 55-91. The normative model in the young cohorts describes DEVELOPMENT and
in ADNI it describes AGING. Any tier-1 difference is therefore confounded with
life stage and cannot be attributed to the disease alone. Tier 2 does not have
this problem: both cohorts are young.

Null, as always: two disjoint halves of ONE cohort. Same disease, same process,
different sample. If a cross-disease cosine sits inside that, the axes are
indistinguishable.
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
DATA = os.path.join(BASE, "data")
N_NULL = 400
SHARED3 = ["Hippocampus", "Ventricles", "WholeBrain"]
FA3 = {"Hippocampus": "هیپوکامپ", "Ventricles": "بطن‌ها", "WholeBrain": "کل مغز"}


# --------------------------------------------------------------- cohorts
def adni():
    a = pd.read_csv(os.path.join(DATA, "ADNIMERGE_05Mar2026.csv"), low_memory=False)
    d = a[a.VISCODE == "bl"].drop_duplicates("PTID")
    d = d.dropna(subset=["AGE", "ICV", "PTGENDER", "DX"] + SHARED3)
    d = d[d.DX.isin(["CN", "Dementia"])]
    return pd.DataFrame(dict(
        age=d.AGE.values, sex=(d.PTGENDER == "Male").astype(int).values,
        icv=d.ICV.values, case=(d.DX == "Dementia").astype(int).values,
        site=d.SITE.astype(str).values,
        Hippocampus=d.Hippocampus.values, Ventricles=d.Ventricles.values,
        WholeBrain=d.WholeBrain.values)).reset_index(drop=True)


def _pair(d, a, b):
    return d[a].values + d[b].values


# The two young cohorts were merged from different FreeSurfer table dumps, so the
# SAME quantity is present under different column names and each file has only
# one of them populated. Picked per file after checking coverage:
#     ABIDE     ICV = IntraCranialVol (1099/1103), BrainSegNotVent is empty
#     ADHD-200  ICV = EstimatedTotalIntraCranialVol (658/659), IntraCranialVol empty
# SupraTentorial is complete in BOTH, so it is the whole-brain measure here. It is
# not identical to ADNI's WholeBrain - it excludes cerebellum and brainstem - so
# the tier-1 whole-brain loading is approximate across cohorts. Stated, not hidden.
def _young(d, age_c, sex_c, icv_c, case, site_c):
    need = [age_c, icv_c, site_c, "Left-Hippocampus", "Right-Hippocampus",
            "Left-Lateral-Ventricle", "Right-Lateral-Ventricle",
            "Left-Inf-Lat-Vent", "Right-Inf-Lat-Vent", "SupraTentorial"]
    d = d.dropna(subset=need + [sex_c]).reset_index(drop=True)
    out = pd.DataFrame(dict(
        age=d[age_c].values.astype(float), sex=d[sex_c].values.astype(int),
        icv=d[icv_c].values, case=case(d).astype(int),
        site=d[site_c].astype(str).values,
        Hippocampus=_pair(d, "Left-Hippocampus", "Right-Hippocampus"),
        Ventricles=_pair(d, "Left-Lateral-Ventricle", "Right-Lateral-Ventricle")
        + _pair(d, "Left-Inf-Lat-Vent", "Right-Inf-Lat-Vent"),
        WholeBrain=d["SupraTentorial"].values))
    return out, d


def abide():
    d = pd.read_csv(os.path.join(DATA, "abide_merged.csv"), low_memory=False)
    return _young(d, "AGE_AT_SCAN", "SEX", "IntraCranialVol",
                  lambda x: (x.DX_GROUP == 1).values, "SITE_ID")


def adhd():
    d = pd.read_csv(os.path.join(DATA, "adhd200_merged.csv"), low_memory=False)
    d = d[d.DX.astype(str).isin(["0", "1", "2", "3"])].reset_index(drop=True)
    return _young(d, "Age", "Gender", "EstimatedTotalIntraCranialVol",
                  lambda x: (x.DX.astype(str) != "0").values, "Site")


# --------------------------------------------------------------- the axis
def design(df):
    """age, age^2, sex, ICV, plus a dummy per site - scanners differ enormously
    in ABIDE and ADHD-200 and an unadjusted site effect would dominate."""
    X = [df.age.values, df.age.values ** 2, df.sex.values, df.icv.values]
    s = pd.get_dummies(df.site.values, drop_first=True).values.astype(float)
    return np.column_stack(X + ([s] if s.shape[1] else []))


def axis_of(df, regions):
    """Case-minus-control mean deviation vector, in control-SD units.

    Normative model fitted on the CONTROLS of this cohort only, then the mean
    z of the cases. The direction of that vector is the cohort's disease axis.
    """
    ctl = df.case.values == 0
    X = design(df)
    sc = StandardScaler().fit(X[ctl])
    out = np.zeros(len(regions))
    for j, v in enumerate(regions):
        y = df[v].values.astype(float)
        m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(X[ctl]), y[ctl])
        r = y - m.predict(sc.transform(X))
        s = r[ctl].std(ddof=1)
        out[j] = r[~ctl].mean() / max(s, 1e-9)
    return out


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def null_axis_cos(df, regions, n1, n2, rng, n_null=N_NULL):
    """Cosine between the axes of two DISJOINT halves of the SAME cohort."""
    n = len(df)
    out = []
    for _ in range(n_null):
        p = rng.permutation(n)
        A, B = df.iloc[p[:n1]], df.iloc[p[n1:n1 + n2]]
        if A.case.sum() < 15 or B.case.sum() < 15 \
                or (1 - A.case).sum() < 15 or (1 - B.case).sum() < 15:
            continue
        try:
            c = cosine(axis_of(A, regions), axis_of(B, regions))
            if np.isfinite(c):
                out.append(c)
        except Exception:
            pass
    return np.array(out)


def report(label, obs, null):
    if len(null) < 30:
        print(f"    {label:34s} cos = {obs:+.4f}   null failed (n={len(null)})")
        return
    lo = np.percentile(null, 2.5)
    p = (null <= obs).mean()
    tag = "SAME axis" if obs >= lo else "DIFFERENT axis"
    print(f"    {label:34s} cos = {obs:+.4f}   null[2.5%] = {lo:+.4f}"
          f"   p = {p:.3f}   -> {tag}")


def main():
    A = adni()
    B, Braw = abide()
    C, Craw = adhd()
    print("=" * 78)
    print("COHORTS")
    print("=" * 78)
    for nm, d in [("ADNI (AD vs CN)", A), ("ABIDE (autism)", B),
                  ("ADHD-200", C)]:
        print(f"  {nm:20s} n={len(d):5d}  cases={int(d.case.sum()):4d}  "
              f"controls={int((1-d.case).sum()):4d}  "
              f"age {d.age.min():.0f}-{d.age.max():.0f} "
              f"(median {d.age.median():.0f})  sites={d.site.nunique()}")

    # ============================================================== TIER 1
    print("\n" + "=" * 78)
    print("TIER 1  THE THREE REGIONS ALL THREE COHORTS SHARE")
    print("=" * 78)
    print("  Each axis is: mean deviation of cases, in that cohort's own")
    print("  control-SD units, after age, age^2, sex, ICV and site.\n")
    ax = {}
    for nm, d in [("ADNI", A), ("ABIDE", B), ("ADHD", C)]:
        ax[nm] = axis_of(d, SHARED3)
        print(f"  {nm:6s} " + "  ".join(f"{FA3[v]} {w:+.3f}"
                                        for v, w in zip(SHARED3, ax[nm]))
              + f"   |axis| = {np.linalg.norm(ax[nm]):.3f}")

    print("\n  --- are these the same direction? ---")
    rng = np.random.RandomState(0)
    frames = {"ADNI": A, "ABIDE": B, "ADHD": C}
    for i, j in [("ADNI", "ABIDE"), ("ADNI", "ADHD"), ("ABIDE", "ADHD")]:
        n = min(len(frames[i]), len(frames[j]))
        big = frames[i] if len(frames[i]) >= len(frames[j]) else frames[j]
        h = min(n, len(big) // 2)
        null = null_axis_cos(big, SHARED3, h, h, rng)
        report(f"{i} vs {j}", cosine(ax[i], ax[j]), null)

    # ============================================================== TIER 2
    print("\n" + "=" * 78)
    print("TIER 2  ABIDE vs ADHD-200 IN THE FULL SHARED SEGMENTATION")
    print("=" * 78)
    drop = ("vessel", "hypointensities", "Optic-Chiasm", "5th-Ventricle",
            "choroid-plexus", "CSF")
    common = [c for c in Braw.columns
              if c in Craw.columns
              and Braw[c].dtype.kind == "f" and Craw[c].dtype.kind == "f"
              and c.startswith(("Left-", "Right-", "3rd", "4th", "Brain-Stem",
                                "CC_"))
              and not any(k in c for k in drop)]
    common = [c for c in common
              if Braw[c].notna().mean() > .95 and Craw[c].notna().mean() > .95]
    print(f"  {len(common)} structures in common:")
    print("    " + ", ".join(common))

    Bf = pd.concat([B[["age", "sex", "icv", "case", "site"]],
                    Braw[common].reset_index(drop=True)], axis=1).dropna()
    Cf = pd.concat([C[["age", "sex", "icv", "case", "site"]],
                    Craw[common].reset_index(drop=True)], axis=1).dropna()
    aB, aC = axis_of(Bf, common), axis_of(Cf, common)
    print(f"\n  |axis| autism {np.linalg.norm(aB):.3f}   "
          f"ADHD {np.linalg.norm(aC):.3f}")
    print("\n  the five structures each condition deviates on most:")
    for nm, v in [("autism", aB), ("ADHD", aC)]:
        top = sorted(zip(common, v), key=lambda t: -abs(t[1]))[:5]
        print(f"    {nm:8s} " + "  ".join(f"{c} {w:+.2f}" for c, w in top))

    print("\n  --- same direction? ---")
    big = Bf if len(Bf) >= len(Cf) else Cf
    h = min(len(Bf), len(Cf), len(big) // 2)
    null = null_axis_cos(big, common, h, h, rng, n_null=200)
    report("ABIDE vs ADHD-200", cosine(aB, aC), null)

    # a within-cohort floor: how well does one cohort agree with ITSELF?
    print("\n  for scale, the null itself (same cohort, two disjoint halves):")
    if len(null) >= 30:
        print(f"    median {np.median(null):+.3f}   "
              f"2.5% {np.percentile(null,2.5):+.3f}   "
              f"97.5% {np.percentile(null,97.5):+.3f}")


if __name__ == "__main__":
    main()
