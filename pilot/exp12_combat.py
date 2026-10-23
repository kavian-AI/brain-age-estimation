#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Does the autism-vs-ADHD result survive site harmonisation?

The raw cross-cohort result (AUC 0.925) was contaminated: healthy ABIDE vs
healthy ADHD-200 controls separated at AUC 0.942, i.e. the classifier was
reading scanner, not disease.

Fix: ComBat harmonisation across the 27 ACQUISITION SITES (20 ABIDE + 7
ADHD-200), preserving age and sex as biological covariates.

Critical design choice: harmonise by SITE, never by COHORT. Cohort is perfectly
confounded with diagnosis here (all autism in one, all ADHD in the other), so
harmonising on cohort would delete the disease signal along with the batch
effect. Sites are nested within cohorts, which lets ComBat remove scanner
effects while leaving between-group biology intact.

Diagnostic run afterwards: healthy-vs-healthy AUC should collapse toward 0.5.
If it does not, the comparison is not salvageable.
"""

import warnings

import numpy as np
import pandas as pd
from neuroCombat import neuroCombat
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
DATA = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\data"
OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"


def icv_of(df):
    return df["IntraCranialVol"].combine_first(df["EstimatedTotalIntraCranialVol"])


def load():
    ab = pd.read_csv(f"{DATA}/abide_merged.csv")
    ad = pd.read_csv(f"{DATA}/adhd200_merged.csv")
    roi = [c for c in ab.columns if c.startswith(("Left-", "Right-")) and c in ad.columns]
    roi = [c for c in roi if ab[c].notna().all() and ad[c].notna().all()
           and ab[c].std() > 0 and ad[c].std() > 0]
    ab["ICVc"] = icv_of(ab); ad["ICVc"] = icv_of(ad)
    ad["DXn"] = pd.to_numeric(ad["DX"], errors="coerce")
    ab = ab.dropna(subset=roi + ["ICVc", "AGE_AT_SCAN", "DX_GROUP", "SEX", "SITE_ID"])
    ad = ad.dropna(subset=roi + ["ICVc", "Age", "DXn", "Gender", "Site"])

    A = pd.DataFrame({"age": ab.AGE_AT_SCAN.values,
                      "sex": (ab.SEX == 1).astype(int).values,
                      "icv": ab.ICVc.values,
                      "site": "ABIDE_" + ab.SITE_ID.astype(str).values,
                      # 1 = autism, 2 = control  ->  patient flag / group label
                      "patient": (ab.DX_GROUP == 1).astype(int).values,
                      "group": np.where(ab.DX_GROUP == 1, "autism", "control")})
    for c in roi:
        A[c] = ab[c].values
    B = pd.DataFrame({"age": ad.Age.values,
                      "sex": (ad.Gender == 1).astype(int).values,
                      "icv": ad.ICVc.values,
                      "site": "ADHD_" + ad.Site.astype(int).astype(str).values,
                      "patient": (ad.DXn > 0).astype(int).values,
                      "group": np.where(ad.DXn > 0, "adhd", "control")})
    for c in roi:
        B[c] = ad[c].values
    return pd.concat([A, B], ignore_index=True), roi


def harmonise(df, roi):
    """ComBat on ICV-normalised volumes, batch = site, covariates = age + sex."""
    X = (df[roi].values / df["icv"].values[:, None]).T      # features x samples
    covars = df[["site", "age", "sex"]].copy()
    out = neuroCombat(dat=X, covars=covars, batch_col="site",
                      continuous_cols=["age"], categorical_cols=["sex"])["data"]
    h = df.copy()
    h[roi] = out.T
    return h


def auc_cv(X, y, n_rep=10):
    a = []
    for rep in range(n_rep):
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y):
            s = StandardScaler().fit(X[tr])
            c = LogisticRegression(max_iter=4000).fit(s.transform(X[tr]), y[tr])
            p[te] = c.predict_proba(s.transform(X[te]))[:, 1]
        a.append(roc_auc_score(y, p))
    return np.mean(a)


def regional_gaps(hdf, roi, ref_mask, tgt_mask):
    """Per-region brain age; healthy reference = ref_mask rows."""
    R = len(roi)
    ref = hdf[ref_mask]; tgt = hdf[tgt_mask]
    g = np.zeros((len(tgt), R))
    for r, c in enumerate(roi):
        Xh = np.column_stack([ref[c].values, ref["sex"].values])
        sc = StandardScaler().fit(Xh)
        m = LinearRegression().fit(sc.transform(Xh), ref["age"].values)
        bias = LinearRegression().fit(ref["age"].values.reshape(-1, 1),
                                      m.predict(sc.transform(Xh)) - ref["age"].values)
        X = sc.transform(np.column_stack([tgt[c].values, tgt["sex"].values]))
        g[:, r] = (m.predict(X) - tgt["age"].values) \
            - bias.predict(tgt["age"].values.reshape(-1, 1))
    return g


def age_match(a, b, tol=1.0):
    ia, ib, used = [], [], set()
    for i in np.argsort(a):
        cand = [j for j in range(len(b)) if j not in used and abs(b[j] - a[i]) <= tol]
        if not cand:
            continue
        j = min(cand, key=lambda k: abs(b[k] - a[i]))
        used.add(j); ia.append(i); ib.append(j)
    return np.array(ia), np.array(ib)


def report(df, roi, tag):
    """The decisive diagnostic + the disease comparisons."""
    ctrl = df[df.group == "control"]
    ctrl_ab = ctrl[ctrl.site.str.startswith("ABIDE")]
    ctrl_ad = ctrl[ctrl.site.str.startswith("ADHD")]
    Xc = np.vstack([ctrl_ab[roi].values / ctrl_ab.icv.values[:, None],
                    ctrl_ad[roi].values / ctrl_ad.icv.values[:, None]]) \
        if tag == "RAW" else np.vstack([ctrl_ab[roi].values, ctrl_ad[roi].values])
    Xc = np.column_stack([Xc, np.concatenate([ctrl_ab.age, ctrl_ad.age])])
    yc = np.concatenate([np.zeros(len(ctrl_ab)), np.ones(len(ctrl_ad))]).astype(int)
    site_auc = auc_cv(Xc, yc)

    aut = df[df.group == "autism"]; adhd = df[df.group == "adhd"]
    ia, ib = age_match(aut.age.values, adhd.age.values)
    A = aut.iloc[ia]; B = adhd.iloc[ib]
    M = pd.concat([A, B])
    y = np.concatenate([np.zeros(len(A)), np.ones(len(B))]).astype(int)
    Xv = (M[roi].values / M.icv.values[:, None]) if tag == "RAW" else M[roi].values
    vol_auc = auc_cv(np.column_stack([Xv, M.sex.values, M.age.values]), y)

    ref = df.group == "control"
    idx = M.index
    G = regional_gaps(df, roi, ref.values, df.index.isin(idx))
    order = df[df.index.isin(idx)].index
    ymap = dict(zip(M.index, y))
    yG = np.array([ymap[i] for i in order])
    prof_auc = auc_cv(G, yG)
    Gs = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
    shape_auc = auc_cv(Gs, yG)

    print(f"\n{'='*68}\n{tag}  (n matched = {len(y)}, regions = {len(roi)})\n{'='*68}")
    print(f"  ** healthy ABIDE vs healthy ADHD-200 (SITE CONFOUND) AUC = {site_auc:.3f}")
    print(f"     autism vs ADHD  - volumes + age                   AUC = {vol_auc:.3f}")
    print(f"     autism vs ADHD  - regional brain-age PROFILE      AUC = {prof_auc:.3f}")
    print(f"     autism vs ADHD  - profile SHAPE                   AUC = {shape_auc:.3f}")
    return dict(site=site_auc, vol=vol_auc, profile=prof_auc, shape=shape_auc)


def main():
    df, roi = load()
    print(f"pooled n={len(df)}, sites={df.site.nunique()}, regions={len(roi)}")
    print(df.group.value_counts().to_string())

    raw = report(df, roi, "RAW")
    hdf = harmonise(df, roi)
    har = report(hdf, roi, "ComBat HARMONISED")

    print(f"\n{'='*68}\nVERDICT\n{'='*68}")
    print(f"  site confound: {raw['site']:.3f} -> {har['site']:.3f}")
    if har["site"] > 0.75:
        print("  ComBat did NOT remove the cohort effect; comparison unsalvageable.")
    else:
        print("  site effect substantially reduced.")
    print(f"  autism vs ADHD (profile): {raw['profile']:.3f} -> {har['profile']:.3f}")
    print("  -> any drop is the part that was scanner, not biology.")

    np.savez(f"{OUT}/exp12_combat.npz",
             **{f"raw_{k}": v for k, v in raw.items()},
             **{f"har_{k}": v for k, v in har.items()})
    hdf.to_csv(f"{OUT}/exp12_harmonised.csv", index=False)
    print(f"\nsaved -> {OUT}/exp12_combat.npz")


if __name__ == "__main__":
    main()
