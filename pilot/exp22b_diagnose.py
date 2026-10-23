#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Why did the image representation fail?

Q2 gave AUC 0.42 from the image gap and 0.50 from the image straight to disease,
on subjects whose enlarged ventricles are visible to the naked eye. A model that
cannot see an effect that obvious is broken, so before reporting anything we
find out which part is broken.

Three candidate explanations, tested in order:

  H1  the images are not spatially registered, so pixel (i,j) is different
      anatomy in different subjects and the top components encode head position
      and size rather than brain structure
  H2  the signal is there but a linear model cannot reach it
  H3  there is genuinely no signal at this resolution
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
OUT = os.path.join(BASE, "pilot")
NL, H, W = 13, 40, 80
BG = 0.25            # in units of the per-acquisition mean intensity


def load():
    c = np.load(os.path.join(OUT, "oasis_img_cache.npz"), allow_pickle=True)
    acq = pd.DataFrame(dict(subject=c["subject"]))
    X = c["X"]
    order = acq.groupby("subject").indices
    subs = sorted(order)
    Xs = np.stack([X[order[s]].mean(0) for s in subs])
    idx = pd.read_csv(os.path.join(OUT, "images_index.csv"))
    meta = idx.groupby("subject").first().reset_index()[
        ["subject", "Age", "M/F", "CDR", "nWBV", "eTIV"]]
    m = pd.DataFrame(dict(subject=subs)).merge(meta, on="subject", how="left")
    return Xs, m


def geometry(Xs):
    """Head centroid, extent and area per subject, from the mid-level slice."""
    V = Xs.reshape(-1, NL, H, W)
    mid = V[:, NL // 2]
    rows = []
    for im in mid:
        msk = im > BG
        if msk.sum() < 20:
            rows.append([np.nan] * 4)
            continue
        cy, cx = ndimage.center_of_mass(msk)
        ys, xs = np.where(msk)
        rows.append([cx, cy, msk.sum(), (xs.max() - xs.min()) / max(1, ys.max() - ys.min())])
    return pd.DataFrame(rows, columns=["cx", "cy", "area", "aspect"])


def auc_cv(X, y, model, n_rep=5):
    a = []
    for rep in range(n_rep):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y):
            s = StandardScaler().fit(X[tr])
            mdl = model().fit(s.transform(X[tr]), y[tr])
            oof[te] = mdl.predict_proba(s.transform(X[te]))[:, 1]
        a.append(roc_auc_score(y, oof))
    return np.mean(a)


def register(Xs):
    """Crude rigid alignment: put every head at a common centroid and scale.

    Not a real registration - no rotation, no nonlinear warp - but enough to
    tell whether misalignment is what destroyed the signal.
    """
    V = Xs.reshape(-1, NL, H, W).copy()
    ref_area = None
    out = np.empty_like(V)
    for i, vol in enumerate(V):
        msk = vol[NL // 2] > BG
        if msk.sum() < 20:
            out[i] = vol
            continue
        cy, cx = ndimage.center_of_mass(msk)
        area = msk.sum()
        if ref_area is None:
            ref_area = area
        s = np.sqrt(ref_area / area)
        for k in range(NL):
            z = ndimage.zoom(vol[k], s, order=1)
            zc = np.array(z.shape) / 2
            # place the scaled head so its centroid lands at the image centre
            sy, sx = zc[0] - (cy * s - H / 2) - H / 2, zc[1] - (cx * s - W / 2) - W / 2
            out[i, k] = ndimage.shift(
                _crop_pad(z, H, W, int(round(sy)), int(round(sx))), 0)
    return out.reshape(len(V), -1)


def _crop_pad(z, h, w, oy, ox):
    o = np.zeros((h, w), dtype=z.dtype)
    ys, xs = max(0, oy), max(0, ox)
    zy, zx = max(0, -oy), max(0, -ox)
    hh = min(h - ys, z.shape[0] - zy)
    ww = min(w - xs, z.shape[1] - zx)
    if hh > 0 and ww > 0:
        o[ys:ys + hh, xs:xs + ww] = z[zy:zy + hh, zx:zx + ww]
    return o


def main():
    Xs, m = load()
    healthy = (m.CDR.fillna(0) == 0).values
    age = m.Age.values.astype(float)
    lo = age[~healthy].min()
    sel = np.where(age >= lo)[0]
    y = (~healthy).astype(int)[sel]
    print(f"age-matched subset: n={len(sel)} (healthy {int((y==0).sum())}, "
          f"impaired {int(y.sum())})")

    # ---------------------------------------------------------------- H1
    print("\n" + "=" * 68)
    print("H1  do the top components encode head position and size?")
    print("=" * 68)
    g = geometry(Xs)
    print("  head geometry across all 347 subjects:")
    print(f"    centroid x: mean {g.cx.mean():.1f} px, SD {g.cx.std():.2f} px "
          f"(image is {W} px wide)")
    print(f"    centroid y: mean {g.cy.mean():.1f} px, SD {g.cy.std():.2f} px "
          f"(image is {H} px tall)")
    print(f"    head area:  mean {g.area.mean():.0f} px, "
          f"SD {100*g.area.std()/g.area.mean():.1f}% of mean")

    Z = PCA(n_components=10, random_state=0).fit_transform(
        StandardScaler().fit_transform(Xs))
    ev = PCA(n_components=10, random_state=0).fit(
        StandardScaler().fit_transform(Xs)).explained_variance_ratio_
    print("\n  correlation of each component with head geometry and with age:")
    print("    PC  var%   |r| cx   |r| cy   |r| area    |r| age   |r| CDR>0")
    cdr = (~healthy).astype(float)
    for i in range(6):
        c = [abs(np.corrcoef(Z[:, i], g[v].values)[0, 1]) for v in
             ("cx", "cy", "area")]
        ra = abs(np.corrcoef(Z[:, i], age)[0, 1])
        rd = abs(np.corrcoef(Z[sel, i], cdr[sel])[0, 1])
        print(f"    {i+1:2d}  {100*ev[i]:4.1f}   {c[0]:6.2f}   {c[1]:6.2f}   "
              f"{c[2]:6.2f}    {ra:6.2f}    {rd:6.2f}")

    # ---------------------------------------------------------------- H2
    print("\n" + "=" * 68)
    print("H2/H3  can any model find disease signal in these images?")
    print("=" * 68)
    Xp = PCA(n_components=80, random_state=0).fit_transform(
        StandardScaler().fit_transform(Xs))[sel]
    tab = np.column_stack([m.nWBV.values, m.eTIV.values,
                           (m["M/F"] == "M").astype(int).values])[sel]
    print(f"  linear  on image PCA          AUC = "
          f"{auc_cv(Xp, y, lambda: LogisticRegression(max_iter=3000, C=0.1)):.3f}")
    print(f"  forest  on image PCA          AUC = "
          f"{auc_cv(Xp, y, lambda: RandomForestClassifier(400, random_state=0)):.3f}")
    print(f"  linear  on nWBV+eTIV+sex      AUC = "
          f"{auc_cv(tab, y, lambda: LogisticRegression(max_iter=3000)):.3f}")

    # a single hand-picked measurement: how bright is the ventricle region?
    V = Xs.reshape(-1, NL, H, W)
    csf = np.array([(v[8:12] < BG).mean() for v in V])[sel]   # dark = CSF
    print(f"  one hand-made feature (dark fraction in upper slices) "
          f"AUC = {roc_auc_score(y, csf):.3f}")

    # ---------------------------------------------------------------- fix
    print("\n" + "=" * 68)
    print("does crude alignment rescue it?")
    print("=" * 68)
    Xr = register(Xs)
    gr = geometry(Xr)
    print(f"  centroid SD after alignment: x {gr.cx.std():.2f} px, "
          f"y {gr.cy.std():.2f} px  (was {g.cx.std():.2f}, {g.cy.std():.2f})")
    Xrp = PCA(n_components=80, random_state=0).fit_transform(
        StandardScaler().fit_transform(Xr))[sel]
    print(f"  linear  on aligned image PCA  AUC = "
          f"{auc_cv(Xrp, y, lambda: LogisticRegression(max_iter=3000, C=0.1)):.3f}")
    print(f"  forest  on aligned image PCA  AUC = "
          f"{auc_cv(Xrp, y, lambda: RandomForestClassifier(400, random_state=0)):.3f}")
    np.save(os.path.join(OUT, "oasis_registered.npy"), Xr.astype(np.float32))
    print(f"\n  saved aligned volumes -> {OUT}\\oasis_registered.npy")


if __name__ == "__main__":
    main()
