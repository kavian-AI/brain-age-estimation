#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Do frozen pretrained features carry brain anatomy? (ADNI, n=1064 baselines)

The backbones are used as fixed feature extractors - no fine-tuning, no GPU.
That makes one comparison unusually clean:

    frozen pretrained features   vs   PCA on the same voxels

Neither is trained on the target, both are fixed linear-ish reductions of the
same scan, so a difference between them is attributable to the PRETRAINING and
not to capacity, optimisation, or sample size. Every earlier image arm confounded
those.

Routes, all starting from pixels:

  PATH A  features -> 7 FreeSurfer volumes -> age -> gap -> disease
  PATH B  features -> age -> gap -> disease
  PATH C  features + chronological age -> disease      (the fair baseline)
  CEIL    true FreeSurfer volumes + age -> disease     (not reachable at test time)

PATH C is what keeps the rest honest: the gap is a function of the image and age,
so a model given both should never do worse than the gap. Where it does, the gap
is discarding information rather than distilling it.

Out-of-fold by SUBJECT everywhere; the age model is fitted on CN subjects only.
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
OUT = os.path.join(BASE, "pilot")
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
TARGETS = V + ["ICV"]
FOLDS, SEEDS = 5, 5
ALPHAS = np.logspace(-1, 6, 40)


def folds_of(subs, seed):
    u = np.unique(subs)
    r = np.random.RandomState(seed)
    f = dict(zip(u, r.permutation(len(u)) % FOLDS))
    return np.array([f[s] for s in subs])


def oof_ridge(X, Y, subs, seed, fit_mask=None):
    """Out-of-fold ridge predictions, optionally fitted on a subset only."""
    P = np.zeros((len(Y), Y.shape[1]) if Y.ndim > 1 else (len(Y),))
    f = folds_of(subs, seed)
    for k in range(FOLDS):
        tr = np.where(f != k)[0]
        te = np.where(f == k)[0]
        if fit_mask is not None:
            tr = tr[fit_mask[tr]]
        if len(tr) < 25:
            continue
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[tr]), Y[tr])
        P[te] = m.predict(sc.transform(X[te]))
    return P


def gap_from(tab, age, y, subs, seed):
    """Two-stage: age model on CN training subjects -> bias-corrected gap."""
    g = np.zeros(len(y))
    f = folds_of(subs, seed)
    for k in range(FOLDS):
        tr, te = np.where(f != k)[0], np.where(f == k)[0]
        h = tr[y[tr] == 0]
        if len(h) < 25:
            continue
        sc = StandardScaler().fit(tab[h])
        m = RidgeCV(alphas=ALPHAS).fit(sc.transform(tab[h]), age[h])
        b = LinearRegression().fit(age[h].reshape(-1, 1),
                                   m.predict(sc.transform(tab[h])) - age[h])
        g[te] = (m.predict(sc.transform(tab[te])) - age[te]) \
            - b.predict(age[te].reshape(-1, 1))
    return g


def direct_auc(D, age, y, subs, seed, C=0.05):
    o = np.zeros(len(y))
    f = folds_of(subs, seed)
    Dz = np.column_stack([D, age])
    for k in range(FOLDS):
        tr, te = f != k, f == k
        sc = StandardScaler().fit(Dz[tr])
        c = LogisticRegression(max_iter=5000, C=C).fit(sc.transform(Dz[tr]), y[tr])
        o[te] = c.predict_proba(sc.transform(Dz[te]))[:, 1]
    return roc_auc_score(y, o)


def main():
    # ALLVISITS=1 uses every follow-up scan, so the frozen arms are measured at
    # the same sample size as the fine-tuned bottleneck (3,286 scans) instead of
    # 1,064 baselines - otherwise part of the fine-tuning gain is just more data.
    suf = "_allvisits" if os.environ.get("ALLVISITS", "0") == "1" else ""
    z = np.load(os.path.join(OUT, f"adni_frozen_feats{suf}.npz"))
    m = pd.read_csv(os.path.join(OUT, f"adni_frozen_meta{suf}.csv"))
    print(f"cache: adni_frozen_feats{suf}.npz")
    m["age_visit"] = m.AGE + m.Years_bl
    subs = m.PTID.values
    age = m.age_visit.values.astype(float)
    Yv = m[TARGETS].values.astype(float)
    cn = (m.DX == "CN").values
    print(f"n={len(m)} baselines, {m.PTID.nunique()} subjects, "
          f"{m.DX.value_counts().to_dict()}")

    reps = {k[2:]: z[k] for k in z.files}

    # voxel PCA on the same subjects, as the no-pretraining control
    vx = os.path.join(OUT, "adni_vox32.npz")
    if os.path.exists(vx):
        vz = np.load(vx)
        vm = pd.read_csv(os.path.join(OUT, "adni_vox32_meta.csv"))
        key = vm.PTID + "|" + vm.VISCODE
        want = m.PTID + "|" + m.VISCODE
        pos = {k: i for i, k in enumerate(key)}
        hit = np.array([pos.get(k, -1) for k in want])
        if (hit >= 0).mean() > 0.8:
            Xv = np.zeros((len(m), vz["X"].shape[1]), dtype=np.float32)
            ok = hit >= 0
            Xv[ok] = vz["X"][hit[ok]]
            v = Xv.std(0)
            Xv = Xv[:, v > np.percentile(v, 60)]
            reps["voxelPCA"] = PCA(n_components=100, random_state=0).fit_transform(
                StandardScaler().fit_transform(Xv))
            print(f"voxel PCA control: matched {ok.sum()}/{len(m)} scans")

    # ---------------------------------------------------- Stage A
    print("\n" + "=" * 74)
    print("STAGE A   features -> FreeSurfer volumes (out of fold, ridge)")
    print("=" * 74)
    print(f"  {'representation':16s} " + " ".join(f"{t[:9]:>9}" for t in TARGETS)
          + "   mean")
    predV = {}
    for nm, X in reps.items():
        P = np.mean([oof_ridge(X, Yv, subs, s) for s in range(SEEDS)], axis=0)
        predV[nm] = P
        rs = [np.corrcoef(P[:, j], Yv[:, j])[0, 1] for j in range(len(TARGETS))]
        print(f"  {nm:16s} " + " ".join(f"{r:9.3f}" for r in rs)
              + f"   {np.mean(rs):.3f}")
    print("\n  for reference, the FINE-TUNED ResNet-18 bottleneck reached")
    print("  Ventricles 0.920, Hippocampus 0.584, Entorhinal 0.443 "
          "(trained on 3,286 scans)")

    # ---------------------------------------------------- age model
    print("\n" + "=" * 74)
    print("PATH B   features -> age, fitted on CN only")
    print("=" * 74)
    predA = {}
    for nm, X in reps.items():
        P = np.mean([oof_ridge(X, age, subs, s, fit_mask=cn) for s in range(SEEDS)],
                    axis=0)
        predA[nm] = P
        print(f"  {nm:16s} CN MAE {np.abs(P[cn]-age[cn]).mean():5.2f} y   "
              f"r {np.corrcoef(P[cn], age[cn])[0,1]:.3f}")
    sc = StandardScaler().fit(np.column_stack([Yv, m.SEX.values])[cn])
    mm = RidgeCV(alphas=ALPHAS).fit(sc.transform(np.column_stack(
        [Yv, m.SEX.values])[cn]), age[cn])
    print(f"  {'TRUE volumes':16s} CN MAE "
          f"{np.abs(mm.predict(sc.transform(np.column_stack([Yv, m.SEX.values])[cn])) - age[cn]).mean():5.2f} y"
          "   (in-sample, ceiling reference)")

    # ---------------------------------------------------- disease
    rows = []
    for task, pos in [("AD vs CN", "Dementia"), ("MCI vs CN", "MCI")]:
        s = m.DX.isin(["CN", pos]).values
        y = (m.DX[s] == pos).astype(int).values
        a, sb = age[s], subs[s]
        print("\n" + "=" * 74)
        print(f"{task}   n={s.sum()} (CN {int((y==0).sum())}, {pos} {int(y.sum())})")
        print("=" * 74)
        print(f"  {'representation':16s} {'A gap':>8} {'A direct':>9} "
              f"{'B gap':>8} {'C feat+age':>11}")
        for nm, X in reps.items():
            tabA = np.column_stack([predV[nm][s], m.SEX.values[s]])
            ag = np.mean([roc_auc_score(y, gap_from(tabA, a, y, sb, q))
                          for q in range(SEEDS)])
            ad = np.mean([direct_auc(tabA, a, y, sb, q, C=1.0) for q in range(SEEDS)])
            gb = predA[nm][s] - a
            bb = LinearRegression().fit(a[y == 0].reshape(-1, 1), gb[y == 0])
            bg = roc_auc_score(y, gb - bb.predict(a.reshape(-1, 1)))
            cf = np.mean([direct_auc(X[s], a, y, sb, q) for q in range(SEEDS)])
            rows.append(dict(task=task, rep=nm, A_gap=ag, A_direct=ad,
                             B_gap=bg, C_feat=cf))
            print(f"  {nm:16s} {ag:8.3f} {ad:9.3f} {bg:8.3f} {cf:11.3f}")
        tabT = np.column_stack([Yv[s], m.SEX.values[s]])
        ct = np.mean([direct_auc(tabT, a, y, sb, q, C=1.0) for q in range(SEEDS)])
        cg = np.mean([roc_auc_score(y, gap_from(tabT, a, y, sb, q))
                      for q in range(SEEDS)])
        print(f"  {'CEILING true vol':16s} {cg:8.3f} {ct:9.3f}"
              "        (not reachable from pixels)")
        rows.append(dict(task=task, rep="CEILING", A_gap=cg, A_direct=ct,
                         B_gap=np.nan, C_feat=np.nan))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "exp29_frozen.csv"), index=False)
    print(f"\nsaved -> {OUT}\\exp29_frozen.csv")

    # the comparison this experiment exists for
    if "voxelPCA" in reps:
        print("\n" + "=" * 74)
        print("DOES PRETRAINING HELP?  (same slices, both representations fixed)")
        print("=" * 74)
        for task in df.task.unique():
            d = df[df.task == task].set_index("rep")
            ceil = d.loc["CEILING", "A_direct"]
            print(f"\n  {task}   ceiling {ceil:.3f}")
            for nm in reps:
                if nm == "voxelPCA":
                    continue
                a, b = d.loc[nm, "C_feat"], d.loc["voxelPCA", "C_feat"]
                print(f"    {nm:16s} {a:.3f}  vs voxel PCA {b:.3f}   "
                      f"{a-b:+.3f}   ({100*(a-.5)/(ceil-.5):.0f}% of ceiling)")


if __name__ == "__main__":
    main()
