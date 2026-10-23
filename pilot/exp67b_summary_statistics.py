#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CAN THE ALIGNMENT LAW RUN ON SUMMARY STATISTICS ALONE?

E67 predicted the disease AUC of 500 unseen brain summaries at r = 0.96. But it used
individual AD data to estimate the disease shift mu and the within-group covariance.
The practical question is whether it still works with what is PUBLISHED:

  - covariance from HEALTHY controls only (any normative dataset provides it)
  - the disease shift from a SMALL patient sample, or from a regional effect-size
    map of Cohen's d - exactly what ENIGMA publishes for a dozen disorders

If yes, the usefulness of brain age - or of any regional summary - for ANY disorder
can be predicted without touching individual patient data. Tested with shifts
estimated from n = 10, 20, 40, 80 patients (repeated draws), and with the shift
expressed as Cohen's d rounded to two decimals, as a paper would print it.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp62_rich_atlas as R  # noqa: E402
from scipy.stats import norm, pearsonr  # noqa: E402

BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"


def main():
    a, m, T, names = R.build("X")
    cn = (m.DX.values == "CN")
    Z, _ = R.normative(m, names, cn)
    D = pd.read_csv(os.path.join(BASE, "pilot", "exp67_subsets.csv"))
    # rebuild the weight vectors of the uniform subsets deterministically, as in E67
    rng = np.random.RandomState(1)
    W, obs = [], []
    S_sign = T.sign.values
    for i in range(400):
        k = rng.choice([3, 5, 8, 12, 20, 35, 60, 98])
        cols = sorted(rng.choice(len(names), k, replace=False))
        mask = np.zeros(len(names), bool)
        mask[cols] = True
        W.append(S_sign * mask)
        if i % 4 == 0:
            pass
    W = np.array(W)
    obs = D[D.summary == "uniform"].obs.values
    y = (m.DX.values == "Dementia").astype(int)
    sel = m.DX.isin(["CN", "Dementia"]).values
    half = np.random.RandomState(0).rand(len(m)) < 0.5
    A_, B_ = sel & half, sel & ~half
    Zc_all = Z[A_ & (y == 0)]
    Zd_all = Z[A_ & (y == 1)]
    S_h = np.cov(Zc_all.T)
    S_h = 0.9 * S_h + 0.1 * np.diag(np.diag(S_h))      # healthy-only covariance

    def predict(mu):
        num = (W @ mu) ** 2
        den = np.einsum("ij,jk,ik->i", W, S_h, W)
        return norm.cdf(np.sqrt(num / den) / np.sqrt(2))

    print("=" * 84)
    print("healthy-only covariance; disease shift from small samples; 400 unseen summaries")
    print("=" * 84)
    mu_full = Zd_all.mean(0) - Zc_all.mean(0)
    p = predict(mu_full)
    print(f"  all {len(Zd_all)} patients:       r = {pearsonr(p, obs)[0]:.3f}   "
          f"mean |error| {np.abs(p-obs).mean():.3f}")
    rs = np.random.RandomState(7)
    for n in [10, 20, 40, 80]:
        rr, ee = [], []
        for _ in range(50):
            idx = rs.choice(len(Zd_all), n, replace=False)
            mu = Zd_all[idx].mean(0) - Zc_all.mean(0)
            pp = predict(mu)
            rr.append(pearsonr(pp, obs)[0])
            ee.append(np.abs(pp - obs).mean())
        print(f"  {n:3d} patients (50 draws): r = {np.mean(rr):.3f} "
              f"[{np.percentile(rr,5):.3f}, {np.percentile(rr,95):.3f}]   mean |error| {np.mean(ee):.3f}")
    # Cohen's d map, rounded like a published table
    sd = np.sqrt(0.5 * (Zc_all.var(0, ddof=1) + Zd_all.var(0, ddof=1)))
    dmap = np.round(mu_full / sd, 2)
    p = predict(dmap * np.sqrt(np.diag(S_h)))
    print(f"\n  published-style Cohen's d map (2 decimals) + healthy covariance:"
          f" r = {pearsonr(p, obs)[0]:.3f}   mean |error| {np.abs(p-obs).mean():.3f}")
    # the most extreme version: d map + healthy covariance replaced by an identity
    num = (W @ dmap) ** 2
    den = (W ** 2).sum(1)
    p2 = norm.cdf(np.sqrt(num / den) / np.sqrt(2))
    print(f"  d map WITHOUT any covariance (regions assumed independent):"
          f" r = {pearsonr(p2, obs)[0]:.3f}   mean |error| {np.abs(p2-obs).mean():.3f}")
    print("  -> the gap between the last two lines is what the covariance structure buys")


if __name__ == "__main__":
    main()
