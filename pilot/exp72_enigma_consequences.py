#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
THREE CONSEQUENCES OF THE ALIGNMENT LAW, RUN ON EVERY ENIGMA DISORDER.

1  SAMPLE SIZE. A brain-age study of disorder D is powered by the brain-age
   effect d_BA = (w_age' mu)/sqrt(w_age' S w_age), not by how different the
   brains are (d*). n per group for 80% power at alpha 0.05 is ~15.7/d^2. How many
   participants does choosing brain age cost, disorder by disorder?

2  OPERATIONAL SIMILARITY BETWEEN DISORDERS. ENIGMA cross-disorder papers compare
   disorders by correlating their d maps across regions. That ignores that
   regions co-vary. The quantity that decides whether a marker built for A also
   detects B is the S-metric: transfer d_{A->B} = (w*_A' mu_B)/sqrt(w*_A' S w*_A),
   w*_A = S^-1 mu_A. Where do the two notions of "similar" disagree?

3  A DISORDER-AWARE BRAIN AGE FROM A PUBLIC MAP. E70 showed a relevance prior
   moves brain age past the standard accuracy/sensitivity frontier for Alzheimer's.
   Here the prior for each disorder is ITS OWN ENIGMA map; the age model is refit
   on ADNI controls (still predicting age, still in years) and the law gives the
   new brain-age effect for that disorder, plus the age-accuracy cost.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp53_units_and_dimension as E  # noqa: E402
import exp68_enigma_prediction as G  # noqa: E402
import tadpole_atlas as TA  # noqa: E402
from sklearn.linear_model import LinearRegression, Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

N80 = 2 * (1.959964 + 0.841621) ** 2          # n per group = N80 / d^2


def build():
    XP = TA.PIPE["X"]
    ct_names = [f"{s}_{nm}" for s in ("L", "R") for nm in TA.CORTICAL.values()]
    ct_codes = [(n + off, "TA") for off in (0, 59) for n in TA.CORTICAL.keys()]
    sv_names = list(G.SUBMAP.keys())
    cols = [f"ST{c}{k}_{XP}" for c, k in ct_codes] + [f"ST{c}{k}_{XP}" for c, k in G.SUBMAP.values()]
    names = ct_names + sv_names
    t = pd.read_csv(TA.TADPOLE, low_memory=False, usecols=["PTID", "VISCODE"] + cols + [f"ST10CV_{XP}"])
    t = t[t.VISCODE == "bl"].drop_duplicates("PTID")
    F = pd.DataFrame({"PTID": t.PTID.values, "ICV_T": pd.to_numeric(t[f"ST10CV_{XP}"], errors="coerce")})
    for nm, c in zip(names, cols):
        F[nm] = pd.to_numeric(t[c], errors="coerce").values
    a, b = E.load()
    m = b.merge(F, on="PTID").dropna(subset=names + ["ICV_T"]).reset_index(drop=True)
    C = m[m.DX.values == "CN"].reset_index(drop=True)
    ag = C.age_visit.values.astype(float)
    Xc = np.column_stack([ag, ag ** 2, C.SEX.values, C.ICV_T.values])
    Y = C[names].values.astype(float)
    sc = StandardScaler().fit(Xc)
    lr = LinearRegression().fit(sc.transform(Xc), Y)
    R_ = Y - lr.predict(sc.transform(Xc))
    Z = R_ / R_.std(0, ddof=1)
    S = np.cov(Z.T)
    S = 0.9 * S + 0.1 * np.diag(np.diag(S))
    return C, Z, S, names, sv_names, ag


def age_model_weights(C, Z, names, ag, scale=None, alpha=100.0):
    """Ridge brain age on controls (raw features + ICV + sex); returns z-space
    weights of the bias-corrected gap and the out-of-fold MAE."""
    X = np.column_stack([C[names].values, C.ICV_T.values, C.SEX.values])
    f = E.folds_of(C.PTID.values)
    g = np.zeros(len(C))
    err = []
    for k in range(5):
        tr, te = f != k, f == k
        sc = StandardScaler().fit(X[tr])
        Xt, Xe = sc.transform(X[tr]), sc.transform(X[te])
        if scale is not None:
            Xt, Xe = Xt * scale, Xe * scale
        md = Ridge(alpha=alpha).fit(Xt, ag[tr])
        p = md.predict(Xe)
        err.append(np.abs(p - ag[te]))
        bb = LinearRegression().fit(ag[tr].reshape(-1, 1), md.predict(Xt) - ag[tr])
        g[te] = (p - ag[te]) - bb.predict(ag[te].reshape(-1, 1))
    return LinearRegression().fit(Z, g).coef_, np.concatenate(err).mean()


def maps(names, sv_names):
    out = {}
    for dis, (ctf, svf, sgn) in G.DISORDERS.items():
        ct, sv = G.read_map(ctf), G.read_map(svf)
        v = []
        for nm in names:
            x = sv.get(nm, np.nan) if nm in sv_names else ct.get(nm, np.nan)
            if dis == "schizophrenia" and nm.endswith("LatVent"):
                x = np.nan
            v.append(sgn * x if np.isfinite(x) else np.nan)
        v = np.array(v)
        if np.isfinite(v).sum() >= 60:           # drop maps missing the cortex
            out[dis] = np.where(np.isfinite(v), v, 0.0)
    return out


def main():
    C, Z, S, names, sv_names, ag = build()
    w_age, mae0 = age_model_weights(C, Z, names, ag)
    hip = -np.array([n in ("Lhippo", "Rhippo") for n in names], float)
    M = maps(names, sv_names)
    d_of = lambda w, mu: (w @ mu) / np.sqrt(w @ S @ w)

    print("=" * 100)
    print("1  WHAT CHOOSING BRAIN AGE COSTS IN PARTICIPANTS  (n per group, 80% power, alpha .05)")
    print("=" * 100)
    print(f"  {'disorder':20s} {'d brain age':>11s} {'n brain age':>12s} {'d hippo':>8s} {'n hippo':>9s}"
          f" {'d* best':>8s} {'n best':>7s} {'brain age needs':>16s}")
    rows = []
    for dis, mu in M.items():
        dba, dh = d_of(w_age, mu), d_of(hip, mu)
        dst = np.sqrt(mu @ np.linalg.solve(S, mu))
        n = lambda d: N80 / d ** 2 if abs(d) > 1e-3 else np.inf
        rows.append((dis, dba, dh, dst))
        print(f"  {dis:20s} {dba:+11.2f} {n(dba):12,.0f} {dh:+8.2f} {n(dh):9,.0f} {dst:8.2f} {n(dst):7,.0f}"
              f" {n(dba)/n(dst):15.0f}x")

    print("\n" + "=" * 100)
    print("2  WHICH DISORDERS ARE 'SIMILAR'?  naive map correlation vs operational transfer")
    print("=" * 100)
    keys = list(M.keys())
    naive, oper = [], []
    print("  pair                                    naive r   transfer d(A->B)/d*(B)   verdict")
    for i in range(len(keys)):
        for j in range(len(keys)):
            if i == j:
                continue
            A, B = M[keys[i]], M[keys[j]]
            r = np.corrcoef(A, B)[0, 1]
            wA = np.linalg.solve(S, A)
            tr = d_of(wA, B) / np.sqrt(B @ np.linalg.solve(S, B))
            naive.append(r)
            oper.append(tr)
            if j > i and (abs(r - tr) > 0.35 or abs(r) > 0.6):
                print(f"  {keys[i][:18]:18s} -> {keys[j][:18]:18s} {r:+8.2f}   {tr:+14.2f}"
                      f"            {'DISAGREE' if abs(r - tr) > 0.35 else 'agree'}")
    print(f"\n  across all {len(naive)} ordered pairs: corr(naive r, operational transfer) = "
          f"{np.corrcoef(naive, oper)[0,1]:.3f}")

    print("\n" + "=" * 100)
    print("3  A DISORDER-AWARE BRAIN AGE, EACH TUNED BY ITS OWN PUBLIC ENIGMA MAP")
    print("=" * 100)
    print(f"  standard brain age MAE {mae0:.2f} y")
    print(f"  {'disorder':20s} {'standard d':>10s} {'aware d (g=2)':>14s} {'aware d (g=4)':>14s} {'MAE g=2':>8s} {'MAE g=4':>8s}")
    for dis, mu in M.items():
        pi = np.abs(mu)
        pi = np.r_[(pi + 0.02) / (pi + 0.02).mean(), 1.0, 1.0]
        res = []
        for gm in [2, 4]:
            w, mae = age_model_weights(C, Z, names, ag, scale=pi ** (gm / 2))
            res.append((d_of(w, mu), mae))
        print(f"  {dis:20s} {d_of(w_age, mu):+10.2f} {res[0][0]:+14.2f} {res[1][0]:+14.2f}"
              f" {res[0][1]:8.2f} {res[1][1]:8.2f}")


if __name__ == "__main__":
    main()
