#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Does training on CHANGE fix the attenuation?

E32 measured that image-predicted volumes report only a fraction of the change
FreeSurfer actually measures - 53% for hippocampus, 13% for entorhinal, and 94%
for ventricles. A rate built from such estimates is shrunk toward zero before any
survival model sees it, which is a sufficient explanation for why every
image-derived rate in this project failed (0.509, 0.521, 0.551) while
image-derived cross-sectional measures reached 0.836.

The suspicion is that nothing in the pipeline ever ASKED for sensitivity to
change. The model is fitted to absolute volume and is scored on absolute volume,
so it optimises the thing that is easy - a subject's stable anatomy - and change
is left as a residual.

THE TEST
--------
Same frozen features, same subjects, same folds. Only the target changes:

  ABSOLUTE   features(visit)          -> volume(visit)        then Δ is derived
  DELTA      features(v2) - features(v1) -> volume(v2) - volume(v1)

If attenuation is a consequence of the training target, DELTA should report a far
larger fraction of real change. If it does not, the attenuation is in the
representation itself and no change of target will fix it.

This runs on frozen features and needs no GPU - the Kaggle account with GPU has
exhausted its weekly quota and the one with quota is not phone-verified.
Frozen features understate absolute performance, but both arms are handicapped
identically, so the COMPARISON is fair.

Splits are by SUBJECT: a subject's pairs never straddle train and test.
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
OUT = os.path.join(BASE, "pilot")
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
TG = V + ["ICV"]
# E16: midtemp + hippocampus + ventricles = 98.8% of the full 6-region model
KEY = ["MidTemp", "Hippocampus", "Ventricles"]
FOLDS = 5
MIN_GAP = 0.75          # years between paired visits


def load():
    z = np.load(os.path.join(OUT, "adni_frozen_feats_allvisits.npz"))
    m = pd.read_csv(os.path.join(OUT, "adni_frozen_meta_allvisits.csv"))
    feats = {k[2:]: z[k] for k in z.files if k.startswith("F_")}
    return feats, m


def make_pairs(m):
    """First and last visit of every subject with a real follow-up interval."""
    rows = []
    for s, g in m.groupby("PTID"):
        g = g.sort_values("Years_bl")
        i, j = g.index[0], g.index[-1]
        dt = g.Years_bl.iloc[-1] - g.Years_bl.iloc[0]
        if dt >= MIN_GAP:
            rows.append(dict(PTID=s, i=i, j=j, dt=dt))
    return pd.DataFrame(rows)


def folds_of(subs, seed=0):
    u = np.unique(subs)
    f = dict(zip(u, np.random.RandomState(seed).permutation(len(u)) % FOLDS))
    return np.array([f[s] for s in subs])


def oof_ridge(X, y, fold):
    p = np.zeros(len(y))
    for k in range(FOLDS):
        tr, te = fold != k, fold == k
        sc = StandardScaler().fit(X[tr])
        mdl = RidgeCV(alphas=np.logspace(-2, 6, 40)).fit(sc.transform(X[tr]), y[tr])
        p[te] = mdl.predict(sc.transform(X[te]))
    return p


def main():
    feats, m = load()
    pairs = make_pairs(m)
    print(f"scans {len(m)}, subjects {m.PTID.nunique()}")
    print(f"usable visit pairs: {len(pairs)}  "
          f"(median interval {pairs.dt.median():.1f} y)\n")

    fold_pair = folds_of(pairs.PTID.values)
    fold_scan = pd.Series(folds_of(m.PTID.values), index=m.index)

    for bb in feats:
        F = feats[bb]
        print("=" * 76)
        print(f"BACKBONE  {bb}   ({F.shape[1]} frozen features)")
        print("=" * 76)
        print(f"  {'target':14s} {'ABSOLUTE':>22s} {'DELTA':>22s}")
        print(f"  {'':14s} {'sens':>8}{'atten':>8}{'  ':>6} "
              f"{'sens':>8}{'atten':>8}")

        summ = {}
        for t in TG:
            ytrue = m[t].values.astype(float)

            # ---- ABSOLUTE arm: fit volume, then derive the change
            pa = oof_ridge(F, ytrue, fold_scan.values)
            dp_abs = pa[pairs.j.values] - pa[pairs.i.values]

            # ---- DELTA arm: fit the change directly, from feature differences
            dX = F[pairs.j.values] - F[pairs.i.values]
            dY = ytrue[pairs.j.values] - ytrue[pairs.i.values]
            dp_del = oof_ridge(dX, dY, fold_pair)

            sa = np.corrcoef(dp_abs, dY)[0, 1]
            aa = np.polyfit(dY, dp_abs, 1)[0]
            sd = np.corrcoef(dp_del, dY)[0, 1]
            ad = np.polyfit(dY, dp_del, 1)[0]
            summ[t] = dict(sa=sa, aa=aa, sd=sd, ad=ad)
            star = "  <<" if t in KEY else ""
            print(f"  {t:14s} {sa:8.3f}{aa:8.3f}{'':6s} "
                  f"{sd:8.3f}{ad:8.3f}{star}")

        S = pd.DataFrame(summ).T
        print(f"  {'MEAN':14s} {S.sa.mean():8.3f}{S.aa.mean():8.3f}{'':6s} "
              f"{S.sd.mean():8.3f}{S.ad.mean():8.3f}")
        K = S.loc[KEY]
        print(f"  {'MEAN (key 3)':14s} {K.sa.mean():8.3f}{K.aa.mean():8.3f}{'':6s} "
              f"{K.sd.mean():8.3f}{K.ad.mean():8.3f}")
        print(f"\n  change in sensitivity: {S.sd.mean()-S.sa.mean():+.3f}   "
              f"change in attenuation: {S.ad.mean()-S.aa.mean():+.3f}")
        if S.sd.mean() > S.sa.mean() + 0.02:
            print("  -> training on CHANGE helps: the target was the problem.")
        elif S.sd.mean() < S.sa.mean() - 0.02:
            print("  -> training on CHANGE is WORSE. Differencing cancels the")
            print("     subject-level signal the features actually carry, and")
            print("     leaves a much noisier target.")
        else:
            print("  -> no material difference; attenuation is in the")
            print("     representation, not the training target.")
        print()


if __name__ == "__main__":
    main()
