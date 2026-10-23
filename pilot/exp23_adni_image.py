#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
THE MISSING EXPERIMENT: does the raw 3D image beat the volume numbers?

Every earlier experiment in this project used pre-computed volumetric features
(6 FreeSurfer regions + ICV). The standing objection is that an image contains
shape and texture that no handful of volumes can capture. This tests that
directly, on the SAME cohort (ADNI) and the SAME task, with the model class held
FIXED so only the input representation changes:

  TAB   6 FreeSurfer volumes + ICV + sex        <- what the report used
  IMG   32^3 voxels -> PCA                      <- the Franke-2010 recipe
  BOTH  concatenation

plus the fair baselines (features + chronological age, no brain-age transform),
because the gap is a function of the features and age, so a model handed both
should not do worse.

WHY THIS RUNS WHERE THE OASIS VERSION DID NOT
---------------------------------------------
The same recipe on OASIS gave AUC 0.53 against 0.71 from three tabular numbers.
The cause was not missing signal but missing registration: those slices are not
in a common space, so the leading components encoded head position (|r| up to
0.58 with head geometry, against at most 0.36 with age). These ADNI tensors ARE registered -
bounding boxes agree to 0.6-2.3 voxels on the 32 grid across subjects - so
voxel-wise comparison is meaningful here and the recipe gets a fair hearing.

INTENSITY
---------
The volumes arrive scaled to [0,1] per volume. They are used as-is. The OASIS
run showed that rescaling by a statistic which itself tracks atrophy (the
foreground mean) cancels the signal; here the foreground mean still predicts
diagnosis (AUC 0.399) and p95 likewise (0.374), i.e. the atrophy signal is
intact, and re-normalising would be the very mistake that broke OASIS.

SPLITS
------
By SUBJECT. The main analysis uses one scan per subject (baseline), but the
longitudinal arm uses several, and splitting by scan would leak a brain across
folds.
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
OUT = os.path.join(BASE, "pilot")
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
N_PCA = 100
N_REP = 8


def load():
    X = np.load(os.path.join(OUT, "adni_vox32.npz"))["X"]
    m = pd.read_csv(os.path.join(OUT, "adni_vox32_meta.csv"))
    return X, m


def design(m):
    return np.column_stack([m[V].values, m.ICV.values, m.SEX.values])


def brain_mask(A):
    """Voxels worth keeping, decided on the fitting set only.

    64% of the grid is background that is identically zero in every subject,
    and among the rest the smallest standard deviation is 3e-9. Dividing by
    those (per-voxel standardisation) multiplies dead voxels by up to 3e8 and
    the leading components then describe amplified noise. Images are not
    tabular features: they share one intensity scale already, so PCA should
    centre them and nothing more.
    """
    s = A.std(0)
    return s > 0.01 * s.max()


def build(tab, img, i_fit, sets, use_tab, use_img):
    """Assemble the design matrix for each index set, fitted on i_fit."""
    out = [[] for _ in sets]
    if use_tab:
        s = StandardScaler().fit(tab[i_fit])
        for j, ix in enumerate(sets):
            out[j].append(s.transform(tab[ix]))
    if use_img:
        msk = brain_mask(img[i_fit])
        p = PCA(n_components=min(N_PCA, len(i_fit) - 2),
                svd_solver="randomized", random_state=0).fit(img[i_fit][:, msk])
        for j, ix in enumerate(sets):
            # rescale components to unit variance so ridge penalises the image
            # and tabular blocks comparably in the BOTH arm
            z = p.transform(img[ix][:, msk])
            out[j].append(z / np.sqrt(p.explained_variance_))
    return [np.hstack(o) for o in out]


def gap_arm(tab, img, age, y, groups, use_tab, use_img):
    """Age model on healthy training subjects -> bias-corrected gap -> AUC.

    AUC is taken from the raw gap rather than from a fitted classifier: for a
    single monotonic score the two are identical, and it avoids training a
    second model whose fit would be contaminated by in-sample healthy gaps.
    """
    aucs, maes, raw = [], [], []
    for rep in range(N_REP):
        oof = np.full(len(y), np.nan)
        praw = np.full(len(y), np.nan)
        mae = []
        for tr, te in StratifiedGroupKFold(
                5, shuffle=True, random_state=rep).split(np.zeros(len(y)), y, groups):
            h = tr[y[tr] == 0]
            if len(h) < 20:
                continue
            Dh, Dte = build(tab, img, h, [h, te], use_tab, use_img)
            mdl = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(Dh, age[h])
            b = LinearRegression().fit(age[h].reshape(-1, 1),
                                       mdl.predict(Dh) - age[h])
            praw[te] = mdl.predict(Dte)
            oof[te] = ((mdl.predict(Dte) - age[te])
                       - b.predict(age[te].reshape(-1, 1)))
            ht = y[te] == 0
            if ht.sum():
                mae.append(np.abs(oof[te][ht]).mean())
        ok = ~np.isnan(oof)
        aucs.append(roc_auc_score(y[ok], oof[ok]))
        maes.append(np.mean(mae))
        # TRUE age accuracy on healthy test subjects, before bias correction.
        # The corrected gap is not a measure of accuracy: a model with no signal
        # predicts everyone at the mean age, the correction then regresses that
        # away, and its |gap| collapses towards zero - so a useless model scores
        # a flattering "MAE" on it. These two columns must be read separately.
        hh = ok & (y == 0)
        raw.append((np.abs(praw[hh] - age[hh]).mean(),
                    np.corrcoef(praw[hh], age[hh])[0, 1]))
    r = np.array(raw)
    return np.array(aucs), float(np.mean(maes)), r[:, 0].mean(), r[:, 1].mean()


def direct_arm(tab, img, extra, y, groups, use_tab, use_img):
    aucs = []
    for rep in range(N_REP):
        oof = np.full(len(y), np.nan)
        for tr, te in StratifiedGroupKFold(
                5, shuffle=True, random_state=rep).split(np.zeros(len(y)), y, groups):
            Dtr, Dte = build(tab, img, tr, [tr, te], use_tab, use_img)
            Dtr = np.hstack([Dtr, extra[tr]])
            Dte = np.hstack([Dte, extra[te]])
            s = StandardScaler().fit(Dtr)
            c = LogisticRegression(max_iter=4000, C=0.05).fit(s.transform(Dtr), y[tr])
            oof[te] = c.predict_proba(s.transform(Dte))[:, 1]
        ok = ~np.isnan(oof)
        aucs.append(roc_auc_score(y[ok], oof[ok]))
    return np.array(aucs)


def ci(a):
    return f"{a.mean():.3f} [{np.percentile(a, 2.5):.3f}, {np.percentile(a, 97.5):.3f}]"


def diff(a, b, label):
    d = a - b
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    print(f"  {label:38s} {d.mean():+.3f}   p={p:.3f}")
    return d


ARMS = [("TAB   volumes+ICV+sex", True, False),
        ("IMG   voxel PCA", False, True),
        ("BOTH", True, True)]


def task(name, m, X, keep, poslab):
    d = m[keep].reset_index(drop=True)
    img = X[np.where(keep)[0]]
    y = d.DX.isin(poslab).astype(int).values
    age = d.AGE.values.astype(float)
    tab = design(d)
    g = d.PTID.values
    print("\n" + "=" * 74)
    print(f"{name}   n={len(y)}  (neg {int((y == 0).sum())}, pos {int(y.sum())})")
    print("=" * 74)

    res = {}
    print("  (age MAE / r are on healthy test subjects, BEFORE bias correction)")
    for nm, ut, ui in ARMS:
        a, mae, rawmae, rr = gap_arm(tab, img, age, y, g, ut, ui)
        res["gap " + nm.split()[0]] = a
        print(f"  gap from {nm:24s} AUC = {ci(a)}   "
              f"age MAE {rawmae:.1f} y, r={rr:.2f}")
    print("\n  fair baselines - straight to disease, chronological age included:")
    for nm, ut, ui in ARMS:
        a = direct_arm(tab, img, age.reshape(-1, 1), y, g, ut, ui)
        res["dir " + nm.split()[0]] = a
        print(f"  {nm + ' + age':32s} AUC = {ci(a)}")

    print()
    diff(res["gap IMG"], res["gap TAB"], "image vs tabular, as a gap")
    diff(res["dir IMG"], res["dir TAB"], "image vs tabular, direct")
    diff(res["dir BOTH"], res["dir TAB"], "does the image ADD to volumes?")
    diff(res["gap TAB"], res["dir TAB"], "gap vs direct (tabular)")
    diff(res["gap IMG"], res["dir IMG"], "gap vs direct (image)")
    return res


def main():
    X, m = load()
    print(f"volumes {X.shape}, subjects {m.PTID.nunique()}")

    bl = (m.VISCODE == "bl") & m.DX.notna() & m.ICV.notna() & m[V].notna().all(1)
    bl &= m.AGE.notna()
    m2 = m[bl].reset_index(drop=True)
    X2 = X[np.where(bl.values)[0]]
    m2 = m2.drop_duplicates("PTID")
    X2 = X2[m2.index.values]
    m2 = m2.reset_index(drop=True)
    print(f"baseline, complete: n={len(m2)}   "
          f"{m2.DX.value_counts().to_dict()}")
    print(f"age {m2.AGE.min():.0f}-{m2.AGE.max():.0f}, mean {m2.AGE.mean():.1f}")

    out = {}
    out["AD"] = task("AD vs CN", m2, X2, m2.DX.isin(["CN", "Dementia"]).values,
                     ["Dementia"])
    out["MCI"] = task("MCI vs CN  (the harder, earlier question)",
                      m2, X2, m2.DX.isin(["CN", "MCI"]).values, ["MCI"])

    np.savez(os.path.join(OUT, "exp23_adni_image.npz"),
             **{f"{k}_{kk}".replace(" ", "_"): vv
                for k, v in out.items() for kk, vv in v.items()})
    m2[["PTID", "VISCODE", "AGE", "DX"]].to_csv(
        os.path.join(OUT, "exp23_subjects.csv"), index=False)
    print(f"\nsaved -> {OUT}\\exp23_adni_image.npz")


if __name__ == "__main__":
    main()
