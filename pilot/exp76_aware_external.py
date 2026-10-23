#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DISEASE-AWARE BRAIN AGE WITH A PRIOR FROM ANOTHER STUDY.

E70b built a disease-aware brain age inside ADNI: relevance-weighted ridge, the
prior being ADNI's own Alzheimer effect map. The practical claim is stronger: the
prior is only a map of summary statistics, so it should work when it comes from a
DIFFERENT study. Here:
  - prior: ADNI (FreeSurfer) Alzheimer-vs-control effect map on the 87 common regions
  - age model: trained on OASIS stable controls only (FastSurfer), predicting age
  - scored in OASIS: CDR>0 vs 0 AUC and later MMSE decline, standard vs aware,
    over a grid of penalties so the standard model gets its best shot
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp62_rich_atlas as R  # noqa: E402
import exp66_oasis_external as O  # noqa: E402
from sklearn.linear_model import LinearRegression, Ridge  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def main():
    a, m, T, names = R.build("X")
    Ro, _, _ = O.load_fs()
    common = [n for n in names if n in Ro.columns]
    ia = [names.index(n) for n in common]
    cn, ad = m.DX.values == "CN", m.DX.values == "Dementia"
    Za, _ = R.normative(m, names, cn)
    Za = Za[:, ia]
    sd = np.sqrt(0.5 * (Za[cn].var(0, ddof=1) + Za[ad].var(0, ddof=1)))
    pi = np.abs((Za[ad].mean(0) - Za[cn].mean(0)) / sd)
    pi = np.r_[pi / pi.mean(), 1.0, 1.0]

    clin = pd.read_csv(os.path.join(O.BASE, "data", "oasis_longitudinal.csv")).rename(
        columns={"MRI ID": "session", "Subject ID": "subject"})
    M = Ro[Ro.mpr == "mpr-1"].merge(clin, on="session")
    M["SEX"] = (M["M/F"] == "M").astype(int)
    M = M.sort_values(["subject", "Visit"]).reset_index(drop=True)
    B = M.drop_duplicates("subject").reset_index(drop=True)
    ever = M.groupby("subject").CDR.max()
    ref = ((B.CDR == 0) & (B.subject.map(ever) == 0)).values
    y = (B.CDR > 0).astype(int).values
    ag = B.Age.values.astype(float)
    X = np.column_stack([B[common].values, B.eTIV.values, B.SEX.values]).astype(float)
    sl = {}
    for s, g in M.groupby("subject"):
        g = g.dropna(subset=["MMSE"])
        if len(g) >= 2 and g["MR Delay"].max() > 365:
            sl[s] = np.polyfit(g["MR Delay"].values / 365.25, g.MMSE.values, 1)[0]
    ms = B.subject.map(sl).values
    q = np.isfinite(ms)
    ids = B.subject.values.astype(str)
    f = O.folds_of(ids)

    def gap(alpha, scale):
        g = np.zeros(len(B))
        err = []
        for k in range(5):
            tr, te = np.where((f != k) & ref)[0], np.where(f == k)[0]
            sc = StandardScaler().fit(X[tr])
            Xt, Xe = sc.transform(X[tr]) * scale, sc.transform(X[te]) * scale
            md = Ridge(alpha=alpha).fit(Xt, ag[tr])
            p = md.predict(Xe)
            err.append(np.abs(p - ag[te])[ref[te]])
            bb = LinearRegression().fit(ag[tr].reshape(-1, 1), md.predict(Xt) - ag[tr])
            g[te] = (p - ag[te]) - bb.predict(ag[te].reshape(-1, 1))
        return g, np.concatenate(err).mean()

    def decline_r(x):
        C = np.column_stack([np.ones(q.sum()), ag[q], B.SEX.values[q]])
        rx = x[q] - C @ np.linalg.lstsq(C, x[q], rcond=None)[0]
        ry = -ms[q] - C @ np.linalg.lstsq(C, -ms[q], rcond=None)[0]
        return np.corrcoef(rx, ry)[0, 1]

    print(f"OASIS n={len(B)} (CDR>0 {y.sum()}, controls for the age model {ref.sum()})")
    print(f"  {'alpha':>7s} {'gamma':>6s} {'MAE':>6s} {'CDR AUC':>8s} {'MMSE-decline r':>15s}")
    res = []
    for al in [3, 10, 30, 100, 300]:
        for gm in [0, 2, 4]:
            g, mae = gap(al, pi ** (gm / 2))
            res.append((al, gm, mae, roc_auc_score(y, g), decline_r(g), g))
            print(f"  {al:7.0f} {gm:6d} {mae:6.2f} {res[-1][3]:8.3f} {res[-1][4]:+15.3f}")
    bs = max([r for r in res if r[1] == 0], key=lambda r: r[3])
    ba = max([r for r in res if r[1] > 0], key=lambda r: r[3])
    print(f"\n  best standard: alpha {bs[0]}, MAE {bs[2]:.2f}, AUC {bs[3]:.3f}, decline r {bs[4]:+.3f}")
    print(f"  best aware:    alpha {ba[0]}, gamma {ba[1]}, MAE {ba[2]:.2f}, AUC {ba[3]:.3f}, decline r {ba[4]:+.3f}")
    rng = np.random.RandomState(0)
    dd, dr = [], []
    for _ in range(2000):
        i = rng.randint(0, len(B), len(B))
        if y[i].min() == y[i].max():
            continue
        dd.append(roc_auc_score(y[i], ba[5][i]) - roc_auc_score(y[i], bs[5][i]))
    lo, hi = np.percentile(dd, [2.5, 97.5])
    p = 2 * min((np.array(dd) <= 0).mean(), (np.array(dd) >= 0).mean())
    print(f"  AUC aware minus standard {np.mean(dd):+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f}")


if __name__ == "__main__":
    main()
