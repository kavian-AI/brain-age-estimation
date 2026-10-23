#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DOES E48 SURVIVE WHEN THE VOLUMES COME FROM THE IMAGE?

Everything we have claimed rests on six FreeSurfer volumes read out of a CSV. But
those volumes were COMPUTED from the images, and anything deployable has to
recompute them from images too. If the better-single-number result only exists at
FreeSurfer's precision, it is a fact about a table, not about brains.

The bottleneck kernel already produced what this needs: a CNN fine-tuned to output
the six volumes + ICV directly from ADNI scans, out of fold by subject. So the
whole E48 pipeline can be rerun with the network's volumes in place of
FreeSurfer's, on the same subjects, with the same normative model, the same
controls-only rotation and the same bootstrap.

    if the component still beats the gap  -> E48 is about brains, and it survives
                                             the pipeline you would actually ship
    if it does not                        -> E48 lives inside FreeSurfer's
                                             precision, and the claim must narrow

True volumes are recomputed here on the identical rows rather than quoted, so the
two arms differ ONLY in where the six numbers came from.
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
PRED = os.path.join(BASE, "kaggle", "brainage_bottleneck", "out",
                    "predicted_volumes.csv")
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
FA = {"Hippocampus": "هیپوکامپ", "Entorhinal": "آنتورینال", "Fusiform": "دوکی",
      "MidTemp": "گیجگاهی میانی", "Ventricles": "بطن‌ها", "WholeBrain": "کل مغز"}
FOLDS, N_BOOT = 5, 500


def folds_of(ids, seed=0):
    uq = np.unique(ids)
    fm = dict(zip(uq, np.random.RandomState(seed).permutation(len(uq)) % FOLDS))
    return np.array([fm[s] for s in ids])


def load_pair():
    """The same rows, twice: once with image-derived volumes, once with true."""
    P = pd.read_csv(PRED)
    A = pd.read_csv(os.path.join(DATA, "ADNIMERGE_05Mar2026.csv"), low_memory=False)
    A = A[["PTID", "VISCODE", "AGE", "Years_bl", "PTGENDER", "DX", "ICV"] + V]
    A = A.dropna(subset=["AGE", "Years_bl", "ICV", "PTGENDER", "DX"] + V)
    P = P.dropna(subset=["DX", "ICV", "age_visit"] + V)
    M = P.merge(A, on=["PTID", "VISCODE"], suffixes=("_img", "_true"))
    M = M[M.DX_true.isin(["CN", "MCI", "Dementia"])]
    # one visit per subject - the earliest available - to keep rows independent
    M = M.sort_values(["PTID", "Years_bl_true"]).drop_duplicates("PTID")
    M = M.reset_index(drop=True)
    # AGE, Years_bl, DX and ICV all exist in both frames, so the merge suffixed
    # them. Take the ADNIMERGE side as ground truth for everything except the
    # six volumes and ICV, which is the only thing the two arms may differ in.
    M["SEX"] = (M.PTGENDER == "Male").astype(int)
    M["age_visit"] = M.AGE_true + M.Years_bl_true
    M["DX"] = M.DX_true
    return M


def cols_for(src):
    return [f"{v}_{src}" for v in V], f"ICV_{src}"


def normative_z(M, src):
    vc, ic = cols_for(src)
    f = folds_of(M.PTID.values)
    cn = (M.DX == "CN").values
    ag = M.age_visit.values.astype(float)
    X = np.column_stack([ag, ag ** 2, M.SEX.values, M[ic].values])
    Z = np.zeros((len(M), len(V)))
    for j, c in enumerate(vc):
        y = M[c].values.astype(float)
        for k in range(FOLDS):
            tr, te = np.where((f != k) & cn)[0], np.where(f == k)[0]
            sc = StandardScaler().fit(X[tr])
            m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(X[tr]), y[tr])
            s = (y[tr] - m.predict(sc.transform(X[tr]))).std(ddof=1)
            Z[te, j] = (y[te] - m.predict(sc.transform(X[te]))) / max(s, 1e-9)
    return Z


def brain_age_gap(M, src):
    vc, ic = cols_for(src)
    f = folds_of(M.PTID.values)
    cn = (M.DX == "CN").values
    ag = M.age_visit.values.astype(float)
    X = np.column_stack([M[c].values for c in vc] + [M[ic].values, M.SEX.values])
    g = np.zeros(len(M))
    for k in range(FOLDS):
        tr, te = np.where((f != k) & cn)[0], np.where(f == k)[0]
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(X[tr]), ag[tr])
        b = LinearRegression().fit(ag[tr].reshape(-1, 1),
                                   m.predict(sc.transform(X[tr])) - ag[tr])
        g[te] = (m.predict(sc.transform(X[te])) - ag[te]) \
            - b.predict(ag[te].reshape(-1, 1))
    return g


def auc_cv(X, y, subs, seed=0):
    f = folds_of(subs, seed)
    o = np.zeros(len(y))
    for k in range(FOLDS):
        tr, te = f != k, f == k
        sc = StandardScaler().fit(X[tr])
        c = LogisticRegression(max_iter=4000).fit(sc.transform(X[tr]), y[tr])
        o[te] = c.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, o)


def one_pc(Zfit, Zall):
    sc = StandardScaler().fit(Zfit)
    p = PCA(n_components=1, random_state=0).fit(sc.transform(Zfit))
    return p.transform(sc.transform(Zall)), p.components_[0]


def main():
    M = load_pair()
    print(f"subjects with BOTH an image-derived and a true volume set: {len(M)}")
    print(M.DX.value_counts().to_string())
    print(f"age {M.age_visit.min():.0f}-{M.age_visit.max():.0f}, "
          f"median {M.age_visit.median():.0f}")

    print("\n" + "=" * 78)
    print("HOW GOOD IS THE BOTTLENECK? image-derived vs true, per region")
    print("=" * 78)
    for v in V:
        x, y = M[f"{v}_img"].values, M[f"{v}_true"].values
        r = np.corrcoef(x, y)[0, 1]
        mape = 100 * np.mean(np.abs(x - y) / np.abs(y))
        print(f"  {FA[v]:16s} r = {r:.3f}   mean abs error = {mape:5.1f}%")
    r_icv = np.corrcoef(M.ICV_img, M.ICV_true)[0, 1]
    print(f"  {'ICV':16s} r = {r_icv:.3f}")

    Zi, Zt = normative_z(M, "img"), normative_z(M, "true")
    gi, gt = brain_age_gap(M, "img"), brain_age_gap(M, "true")

    sel = M.DX.isin(["CN", "Dementia"]).values
    y = (M.DX.values[sel] == "Dementia").astype(int)
    subs = M.PTID.values[sel]
    ag = M.age_visit.values[sel].reshape(-1, 1)
    print(f"\n  AD vs CN task: n={len(y)} ({int(y.sum())} AD, "
          f"{int((1-y).sum())} CN)")

    print("\n" + "=" * 78)
    print("THE E48 COMPARISON, RUN TWICE")
    print("=" * 78)
    out = {}
    for src, Z, g in [("image-derived", Zi, gi), ("FreeSurfer", Zt, gt)]:
        Zs, cn_s = Z[sel], (M.DX.values[sel] == "CN")
        pc, load_ = one_pc(Zs[cn_s], Zs)
        a_gap = auc_cv(np.column_stack([g[sel], ag]), y, subs)
        a_pc = auc_cv(np.column_stack([pc, ag]), y, subs)
        a_six = auc_cv(np.column_stack([Zs, ag]), y, subs)
        out[src] = (a_gap, a_pc, a_six)
        print(f"\n  {src}")
        print(f"    brain-age gap      AUC = {a_gap:.3f}")
        print(f"    one component      AUC = {a_pc:.3f}")
        print(f"    six deviations     AUC = {a_six:.3f}")
        print("    loadings (controls-only rotation): " + "  ".join(
            f"{FA[v]} {w:+.2f}" for v, w in zip(V, load_)))

    print("\n" + "=" * 78)
    print("DOES THE COMPONENT STILL BEAT THE GAP?  (subject-level bootstrap,")
    print("rotation refitted inside every draw)")
    print("=" * 78)
    uq = np.unique(subs)
    for src, Z, g in [("image-derived", Zi, gi), ("FreeSurfer", Zt, gt)]:
        Zs, cn_s = Z[sel], (M.DX.values[sel] == "CN")
        rng = np.random.RandomState(0)
        d = []
        for _ in range(N_BOOT):
            pick = set(rng.choice(uq, len(uq), replace=True))
            m = np.array([s in pick for s in subs])
            if y[m].sum() < 20 or (1 - y[m]).sum() < 20 or cn_s[m].sum() < 30:
                continue
            try:
                pc, _ = one_pc(Zs[m][cn_s[m]], Zs[m])
                d.append(auc_cv(np.column_stack([pc, ag[m]]), y[m], subs[m])
                         - auc_cv(np.column_stack([g[sel][m], ag[m]]), y[m],
                                  subs[m]))
            except Exception:
                pass
        d = np.array(d)
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        print(f"\n  {src}: component minus gap")
        print(f"    {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  p={p:.3f}"
              f"   -> {'HOLDS' if p < 0.05 and d.mean() > 0 else 'does not hold'}")

    print("\n" + "=" * 78)
    print("DIAGNOSIS: DID THE NETWORK PRODUCE SIX NUMBERS, OR ONE NUMBER SIX")
    print("           TIMES?  A rotation has nothing to do if there is only one")
    print("           direction in the data to begin with.")
    print("=" * 78)
    for src, Z in [("image-derived", Zi), ("FreeSurfer", Zt)]:
        C = np.corrcoef(Z.T)
        ev = np.linalg.eigvalsh(C)[::-1]
        pr = (ev.sum() ** 2) / (ev ** 2).sum()      # participation ratio
        off = np.abs(C[np.triu_indices(len(V), 1)])
        print(f"\n  {src}")
        print(f"    mean |correlation| between regional deviations = "
              f"{off.mean():.3f}  (range {off.min():.3f}-{off.max():.3f})")
        print(f"    variance in the first direction = "
              f"{100*ev[0]/ev.sum():.1f}%")
        print(f"    EFFECTIVE NUMBER OF DIMENSIONS = {pr:.2f} out of 6")
    print("\n  If the image arm sits near 1.0 effective dimensions, this")
    print("  experiment did not test E48 - it tested the network, and the")
    print("  network collapsed six regions into one global atrophy score.")

    print("\n" + "=" * 78)
    print("WHAT THE IMAGE PIPELINE COSTS")
    print("=" * 78)
    for i, nm in enumerate(["brain-age gap", "one component", "six deviations"]):
        print(f"  {nm:18s} FreeSurfer {out['FreeSurfer'][i]:.3f}"
              f"   image {out['image-derived'][i]:.3f}"
              f"   cost {out['image-derived'][i]-out['FreeSurfer'][i]:+.3f}")


if __name__ == "__main__":
    main()
