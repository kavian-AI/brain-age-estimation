#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DOES THE BETTER SINGLE NUMBER GENERALISE, OR IS IT FITTED TO THIS COHORT?

E46/E47 showed that one principal component of the regional deviations beats the
brain-age gap on AD vs CN (+0.024, 95% CI [+0.016, +0.032]). The rotation was
fitted and tested inside the same cohort. "A better single number exists" is
established by that; "this number transfers" is not, and without transfer the
result is a curiosity.

ADNI gives three genuine domain shifts to test against, and they are much harder
than a random split:

    by STUDY PHASE       ADNI1 -> ADNI2/GO/3   different recruitment waves
    by FIELD STRENGTH    1.5 T -> 3 T          different physics
    by FREESURFER        4.3/5.1 -> 6.0        different segmentation software

Everything is fitted on the training cohort only: the normative model (on its CN
subjects), the rotation, and the brain-age model. Nothing from the test cohort
touches any fitted object. The rotation is also fitted TWICE - once on all
training subjects and once on training CONTROLS ONLY, because a rotation learned
without ever seeing a patient is the stronger claim.

Scored on the test cohort with a subject-level bootstrap.
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
DATA = os.path.join(BASE, "data")
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
FA = {"Hippocampus": "هیپوکامپ", "Entorhinal": "آنتورینال", "Fusiform": "دوکی",
      "MidTemp": "گیجگاهی میانی", "Ventricles": "بطن‌ها", "WholeBrain": "کل مغز"}
N_BOOT = 400


def load():
    a = pd.read_csv(os.path.join(DATA, "ADNIMERGE_05Mar2026.csv"), low_memory=False)
    keep = ["PTID", "VISCODE", "AGE", "PTGENDER", "DX", "ICV", "Years_bl",
            "ORIGPROT", "FLDSTRENG", "FSVERSION"] + V
    d = a[keep].dropna(subset=["AGE", "Years_bl", "ICV", "PTGENDER"] + V).copy()
    d = d[d.VISCODE == "bl"].drop_duplicates("PTID")
    d["SEX"] = (d.PTGENDER == "Male").astype(int)
    d["age_visit"] = d.AGE + d.Years_bl
    return d.reset_index(drop=True)


def fit_apply(tr, te, rotation_on="all"):
    """Fit everything on `tr`, produce features for `te`. Nothing leaks."""
    cn = (tr.DX == "CN").values
    agT = tr.age_visit.values.astype(float)
    agE = te.age_visit.values.astype(float)

    # --- normative model, fitted on TRAIN controls only
    Xn_T = np.column_stack([agT, agT ** 2, tr.SEX.values, tr.ICV.values])
    Xn_E = np.column_stack([agE, agE ** 2, te.SEX.values, te.ICV.values])
    Zt = np.zeros((len(tr), len(V)))
    Ze = np.zeros((len(te), len(V)))
    sc = StandardScaler().fit(Xn_T[cn])
    for j, v in enumerate(V):
        y = tr[v].values.astype(float)
        m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(Xn_T[cn]), y[cn])
        sd = (y[cn] - m.predict(sc.transform(Xn_T[cn]))).std(ddof=1)
        Zt[:, j] = (y - m.predict(sc.transform(Xn_T))) / max(sd, 1e-9)
        Ze[:, j] = (te[v].values - m.predict(sc.transform(Xn_E))) / max(sd, 1e-9)

    # --- brain-age model, fitted on TRAIN controls only
    Xa_T = np.column_stack([tr[v].values for v in V] + [tr.ICV.values, tr.SEX.values])
    Xa_E = np.column_stack([te[v].values for v in V] + [te.ICV.values, te.SEX.values])
    sa = StandardScaler().fit(Xa_T[cn])
    ma = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sa.transform(Xa_T[cn]), agT[cn])
    bb = LinearRegression().fit(agT[cn].reshape(-1, 1),
                                ma.predict(sa.transform(Xa_T[cn])) - agT[cn])
    gap_e = (ma.predict(sa.transform(Xa_E)) - agE) - bb.predict(agE.reshape(-1, 1))

    # --- rotation, fitted on TRAIN (all, or controls only)
    src = Zt[cn] if rotation_on == "cn" else Zt
    sz = StandardScaler().fit(src)
    pca = PCA(n_components=1, random_state=0).fit(sz.transform(src))
    pc_e = pca.transform(sz.transform(Ze))
    return Ze, gap_e, pc_e, pca.components_[0]


def auc_of(X, y):
    sc = StandardScaler().fit(X)
    c = LogisticRegression(max_iter=4000).fit(sc.transform(X), y)
    return roc_auc_score(y, c.predict_proba(sc.transform(X))[:, 1])


def evaluate(tr, te, label):
    print("\n" + "=" * 76)
    print(f"{label}")
    print(f"  train n={len(tr)} ({int((tr.DX=='CN').sum())} CN), "
          f"test n={len(te)} ({int((te.DX=='CN').sum())} CN, "
          f"{int((te.DX=='Dementia').sum())} AD)")
    print("=" * 76)
    sel = te.DX.isin(["CN", "Dementia"]).values
    T = te[sel].reset_index(drop=True)
    y = (T.DX == "Dementia").astype(int).values
    if y.sum() < 20 or (1 - y).sum() < 20:
        print("  too few in one class; skipped")
        return
    for rot in ["all", "cn"]:
        Ze, gap, pc, load_ = fit_apply(tr, T, rotation_on=rot)
        ag = T.age_visit.values.reshape(-1, 1)
        a_six = auc_of(np.column_stack([Ze, ag]), y)
        a_pc = auc_of(np.column_stack([pc, ag]), y)
        a_gap = auc_of(np.column_stack([gap, ag]), y)
        rng = np.random.RandomState(0)
        dif = []
        for _ in range(N_BOOT):
            i = rng.randint(0, len(y), len(y))
            if y[i].sum() < 15 or (1 - y[i]).sum() < 15:
                continue
            try:
                dif.append(auc_of(np.column_stack([pc[i], ag[i]]), y[i])
                           - auc_of(np.column_stack([gap[i], ag[i]]), y[i]))
            except Exception:
                pass
        dif = np.array(dif)
        lo, hi = np.percentile(dif, [2.5, 97.5])
        pv = 2 * min((dif <= 0).mean(), (dif >= 0).mean())
        tag = "all train subjects" if rot == "all" else "train CONTROLS only"
        print(f"\n  rotation fitted on {tag}")
        print(f"    six-dim {a_six:.3f} | one component {a_pc:.3f} | "
              f"brain-age gap {a_gap:.3f}")
        print(f"    component minus gap {dif.mean():+.3f}  95% CI "
              f"[{lo:+.3f}, {hi:+.3f}]  p={pv:.3f}  "
              f"-> {'HOLDS' if pv < 0.05 and dif.mean() > 0 else 'does not hold'}")
        if rot == "cn":
            print("    loadings: " + "  ".join(
                f"{FA[v]} {w:+.2f}" for v, w in zip(V, load_)))


def main():
    d = load()
    print(f"baseline scans with complete volumetrics: {len(d)}")
    print(d.ORIGPROT.value_counts().to_string())

    # 1. random split by subject, as a sanity reference
    rng = np.random.RandomState(0)
    m = rng.rand(len(d)) < 0.5
    evaluate(d[m], d[~m], "1. RANDOM half/half split (easiest case)")

    # 2. study phase: ADNI1 -> everything later
    tr = d[d.ORIGPROT == "ADNI1"]
    te = d[d.ORIGPROT != "ADNI1"]
    if len(tr) > 100 and len(te) > 100:
        evaluate(tr, te, "2. STUDY PHASE: train ADNI1 -> test ADNI2/GO/3")

    # 3. field strength
    f15 = d.FLDSTRENG.astype(str).str.contains("1.5", na=False)
    f30 = d.FLDSTRENG.astype(str).str.contains("^3 ", na=False, regex=True)
    if f15.sum() > 100 and f30.sum() > 100:
        evaluate(d[f15], d[f30], "3. FIELD STRENGTH: train 1.5 T -> test 3 T")

    # 4. FreeSurfer version
    old = d.FSVERSION.astype(str).str.contains("4.3|5.1", na=False)
    new = d.FSVERSION.astype(str).str.contains("6.0", na=False)
    if old.sum() > 100 and new.sum() > 100:
        evaluate(d[old], d[new], "4. SOFTWARE: train FreeSurfer 4.3/5.1 -> test 6.0")


if __name__ == "__main__":
    main()
