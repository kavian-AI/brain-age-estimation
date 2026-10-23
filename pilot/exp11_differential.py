#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DIFFERENTIAL DIAGNOSIS with a regional brain-age profile.

Almost all brain-age work asks "patient vs control". This asks the harder,
clinically real question: given that someone is NOT healthy, WHICH condition
do they have?  If the regional brain-age profile carries a disease-specific
signature, it should separate disease A from disease B.

Cohorts (all age-matched WITHIN each comparison, so the answer cannot be
"the model just read off chronological age"):
  autism (ABIDE)  vs  ADHD (ADHD-200)      -> both children, cleanly matched
  Alzheimer's     vs  MCI (ADNI)           -> both elderly

Representations compared, identical folds/classifier:
  profile   : regional brain-age gap per region
  shape     : profile with global severity removed (unit norm)
  volumes   : ICV-normalised volumes + sex + age   [the fair baseline]

Key control: for the patient-vs-patient task we also report what a model using
ONLY chronological age achieves, to expose any residual age confound.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
DATA = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\data"
OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"
rng = np.random.default_rng(0)


def regional_gaps(vol_h, age_h, icv_h, sex_h, vol, age, icv, sex):
    """Per-region brain age trained on the healthy reference only."""
    R = vol.shape[1]
    g = np.zeros((vol.shape[0], R))
    for r in range(R):
        Xh = np.column_stack([vol_h[:, r], icv_h, sex_h])
        sc = StandardScaler().fit(Xh)
        m = LinearRegression().fit(sc.transform(Xh), age_h)
        bias = LinearRegression().fit(age_h.reshape(-1, 1),
                                      m.predict(sc.transform(Xh)) - age_h)
        X = sc.transform(np.column_stack([vol[:, r], icv, sex]))
        g[:, r] = (m.predict(X) - age) - bias.predict(age.reshape(-1, 1))
    return g


def age_match(a_age, b_age, tol=1.5):
    """Greedy 1:1 nearest-age matching; returns index arrays."""
    ia, ib, used = [], [], set()
    order = np.argsort(a_age)
    for i in order:
        cand = [j for j in range(len(b_age))
                if j not in used and abs(b_age[j] - a_age[i]) <= tol]
        if not cand:
            continue
        j = min(cand, key=lambda k: abs(b_age[k] - a_age[i]))
        used.add(j)
        ia.append(i); ib.append(j)
    return np.array(ia), np.array(ib)


def evaluate(vol_h, age_h, icv_h, sex_h, vol, age, icv, sex, y, title, n_rep=20):
    res = {k: [] for k in ("profile", "shape", "volumes", "age_only")}
    for rep in range(n_rep):
        oof = {k: np.zeros(len(y)) for k in res}
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(vol, y):
            G_tr = regional_gaps(vol_h, age_h, icv_h, sex_h,
                                 vol[tr], age[tr], icv[tr], sex[tr])
            G_te = regional_gaps(vol_h, age_h, icv_h, sex_h,
                                 vol[te], age[te], icv[te], sex[te])

            def fp(Ftr, Fte):
                s = StandardScaler().fit(Ftr)
                c = LogisticRegression(max_iter=4000, C=1.0)
                c.fit(s.transform(Ftr), y[tr])
                return c.predict_proba(s.transform(Fte))[:, 1]

            oof["profile"][te] = fp(G_tr, G_te)
            n_tr = G_tr / (np.linalg.norm(G_tr, axis=1, keepdims=True) + 1e-9)
            n_te = G_te / (np.linalg.norm(G_te, axis=1, keepdims=True) + 1e-9)
            oof["shape"][te] = fp(n_tr, n_te)
            Vtr = vol[tr] / icv[tr][:, None]
            Vte = vol[te] / icv[te][:, None]
            oof["volumes"][te] = fp(np.column_stack([Vtr, sex[tr], age[tr]]),
                                    np.column_stack([Vte, sex[te], age[te]]))
            oof["age_only"][te] = fp(age[tr].reshape(-1, 1), age[te].reshape(-1, 1))
        for k in res:
            res[k].append(roc_auc_score(y, oof[k]))

    print(f"\n{'='*70}\n{title}\n  n={len(y)}, group A={int((y==0).sum())}, "
          f"group B={int(y.sum())}, regions={vol.shape[1]}\n{'='*70}")
    lbl = {"profile": "regional brain-age PROFILE",
           "shape": "profile SHAPE (severity removed)",
           "volumes": "volumes + age  [fair baseline]",
           "age_only": "chronological age ONLY  [confound check]"}
    for k in ("profile", "shape", "volumes", "age_only"):
        a = np.array(res[k])
        print(f"  {lbl[k]:42s} AUC = {a.mean():.3f}  "
              f"[{np.percentile(a,2.5):.3f}, {np.percentile(a,97.5):.3f}]")
    return res


def main():
    out = {}

    # ---------------- children: autism vs ADHD (age-matched) ----------------
    try:
        ab = pd.read_csv(f"{DATA}/abide_merged.csv")
        ad = pd.read_csv(f"{DATA}/adhd200_merged.csv")
    except FileNotFoundError as e:
        print("missing file:", e)
        return

    roi = [c for c in ab.columns if c.startswith(("Left-", "Right-"))]
    roi = [c for c in roi if c in ad.columns]
    roi = [c for c in roi if ab[c].notna().all() and ad[c].notna().all()
           and ab[c].std() > 0 and ad[c].std() > 0]
    print(f"shared ROIs between ABIDE and ADHD-200: {len(roi)}")

    # FreeSurfer 5.1 (ABIDE) writes IntraCranialVol; 5.3 (ADHD-200) writes
    # EstimatedTotalIntraCranialVol. Each cohort populates only one, so
    # coalesce rather than picking a single name.
    def icv_of(df):
        return df["IntraCranialVol"].combine_first(
            df["EstimatedTotalIntraCranialVol"])
    ab["ICVc"] = icv_of(ab)
    ad["ICVc"] = icv_of(ad)
    ad["DXn"] = pd.to_numeric(ad["DX"], errors="coerce")   # 'pending' -> NaN

    ab = ab.dropna(subset=roi + ["ICVc", "AGE_AT_SCAN", "DX_GROUP", "SEX"])
    ad = ad.dropna(subset=roi + ["ICVc", "Age", "DXn", "Gender"])

    # ABIDE: DX_GROUP 1=autism, 2=control ; ADHD200: DX 0=control, >0 = ADHD
    aut = ab[ab.DX_GROUP == 1]
    ctrl_ab = ab[ab.DX_GROUP == 2]
    adhd = ad[ad.DXn > 0]
    ctrl_ad = ad[ad.DXn == 0]
    print(f"autism={len(aut)}, ABIDE controls={len(ctrl_ab)}, "
          f"ADHD={len(adhd)}, ADHD200 controls={len(ctrl_ad)}")

    ia, ib = age_match(aut.AGE_AT_SCAN.values, adhd.Age.values, tol=1.0)
    A = aut.iloc[ia]; B = adhd.iloc[ib]
    print(f"age-matched pairs: {len(ia)}  "
          f"(autism mean age {A.AGE_AT_SCAN.mean():.1f}, ADHD {B.Age.mean():.1f})")

    if len(ia) >= 60:
        # healthy reference pooled from BOTH cohorts' controls
        vol_h = np.vstack([ctrl_ab[roi].values, ctrl_ad[roi].values])
        age_h = np.concatenate([ctrl_ab.AGE_AT_SCAN.values, ctrl_ad.Age.values])
        icv_h = np.concatenate([ctrl_ab.ICVc.values,
                                ctrl_ad.ICVc.values])
        sex_h = np.concatenate([(ctrl_ab.SEX == 1).astype(int).values,
                                (ctrl_ad.Gender == 1).astype(int).values])
        vol = np.vstack([A[roi].values, B[roi].values])
        age = np.concatenate([A.AGE_AT_SCAN.values, B.Age.values])
        icv = np.concatenate([A.ICVc.values, B.ICVc.values])
        sex = np.concatenate([(A.SEX == 1).astype(int).values,
                              (B.Gender == 1).astype(int).values])
        y = np.concatenate([np.zeros(len(A)), np.ones(len(B))]).astype(int)
        out["autism_vs_adhd"] = evaluate(vol_h, age_h, icv_h, sex_h,
                                         vol, age, icv, sex, y,
                                         "CHILDREN: autism vs ADHD (age-matched)")

    # ---------------- elderly: Alzheimer's vs MCI ----------------
    adni = pd.read_csv(f"{DATA}/ADNIMERGE_05Mar2026.csv", low_memory=False)
    V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
    a = adni.dropna(subset=V + ["ICV", "AGE", "DX", "PTGENDER"]).sort_values("Years_bl")
    a = a.groupby("RID").first().reset_index()
    cn = a[a.DX == "CN"]
    dem = a[a.DX == "Dementia"]
    mci = a[a.DX == "MCI"]
    im, ic = age_match(dem.AGE.values, mci.AGE.values, tol=1.0)
    D = dem.iloc[im]; M = mci.iloc[ic]
    print(f"\nAD/MCI age-matched pairs: {len(im)} "
          f"(AD {D.AGE.mean():.1f}, MCI {M.AGE.mean():.1f})")
    vol_h = cn[V].values
    out["ad_vs_mci"] = evaluate(
        vol_h, cn.AGE.values, cn.ICV.values,
        (cn.PTGENDER == "Male").astype(int).values,
        np.vstack([D[V].values, M[V].values]),
        np.concatenate([D.AGE.values, M.AGE.values]),
        np.concatenate([D.ICV.values, M.ICV.values]),
        np.concatenate([(D.PTGENDER == "Male").astype(int).values,
                        (M.PTGENDER == "Male").astype(int).values]),
        np.concatenate([np.zeros(len(D)), np.ones(len(M))]).astype(int),
        "ELDERLY: Alzheimer's vs MCI (age-matched)")

    np.savez(f"{OUT}/exp11_differential.npz",
             **{f"{k}_{m}": np.array(v) for k, r in out.items() for m, v in r.items()})
    print(f"\nsaved -> {OUT}/exp11_differential.npz")


if __name__ == "__main__":
    main()
