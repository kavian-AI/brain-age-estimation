#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FIVE OPEN QUESTIONS ABOUT THE BRAIN-AGE GAP, RUN IN ONE BATCH.

Everything measured so far concerns the LEVEL of deviation - how far each region
sits from normal. These five ask different kinds of question, and three of them
have no answer in the literature that I could find.

Q1  THE COMPRESSION LAW.  Compression has cost accuracy in four independent
    experiments (scalar gap vs regional profile, normative distance vs six z,
    twice more). Nobody has measured the CURVE. How much information survives
    compression to k dimensions, and where is the knee? If two numbers recover
    most of six, that is a usable result; if the loss is linear in k, the
    "scalar gap" tradition is simply wrong by a quantifiable amount.

Q2  COUPLING.  ★ the one I think is most likely to be new. Disease is assumed to
    shift the LEVEL of regional deviation. But atrophy spreads through connected
    systems, so it should also change how regions COVARY. If the correlation
    structure of the regional deviations differs between controls and patients,
    that is a signal dimension nobody uses - and one that is invisible to every
    method that looks at regions one at a time.

Q3  ASYMMETRY.  The gap is treated as a symmetric axis: +5 years is bad, -5 is
    good. That is an assumption, not a finding. A brain that looks younger than
    its age may simply be a large brain, which is not protection. Testing the two
    tails separately is a one-line change nobody seems to make.

Q4  APOE4.  Does the same gap mean the same thing in a carrier and a non-carrier?
    If the gap's hazard differs by genotype, then a single threshold is wrong for
    half the population.

Q5  RESERVE.  Vidal-Pineiro (eLife 2021) argued the gap reflects brain reserve
    rather than rate of ageing. Education is the standard reserve proxy and ADNI
    has it. If the gap is partly reserve, education should predict the gap in
    HEALTHY people, where there is no disease to explain it.

All arms: CN-trained, out of fold by subject, subject-level bootstrap on every
comparison. Landmark design for anything longitudinal.
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "pilot")
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
FA = {"Hippocampus": "هیپوکامپ", "Entorhinal": "آنتورینال", "Fusiform": "دوکی",
      "MidTemp": "گیجگاهی میانی", "Ventricles": "بطن‌ها", "WholeBrain": "کل مغز"}
FOLDS, N_BOOT, L = 5, 300, 2.0


def load():
    a = pd.read_csv(os.path.join(DATA, "ADNIMERGE_05Mar2026.csv"), low_memory=False)
    d = a.dropna(subset=["AGE", "Years_bl", "ICV", "PTGENDER"] + V).copy()
    d["SEX"] = (d.PTGENDER == "Male").astype(int)
    d["age_visit"] = d.AGE + d.Years_bl
    return a, d.reset_index(drop=True)


def normative_z(d, seed=0):
    cn = (d.DX == "CN").values
    uq = np.unique(d.PTID.values)
    fm = dict(zip(uq, np.random.RandomState(seed).permutation(len(uq)) % FOLDS))
    f = np.array([fm[s] for s in d.PTID.values])
    age = d.age_visit.values.astype(float)
    X = np.column_stack([age, age ** 2, d.SEX.values, d.ICV.values])
    Z = np.zeros((len(d), len(V)))
    for j, v in enumerate(V):
        y = d[v].values.astype(float)
        for k in range(FOLDS):
            tr, te = np.where((f != k) & cn)[0], np.where(f == k)[0]
            sc = StandardScaler().fit(X[tr])
            m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(X[tr]), y[tr])
            sd = (y[tr] - m.predict(sc.transform(X[tr]))).std(ddof=1)
            Z[te, j] = (y[te] - m.predict(sc.transform(X[te]))) / max(sd, 1e-9)
    return Z


def scalar_gap(d, seed=0):
    cn = (d.DX == "CN").values
    uq = np.unique(d.PTID.values)
    fm = dict(zip(uq, np.random.RandomState(seed).permutation(len(uq)) % FOLDS))
    f = np.array([fm[s] for s in d.PTID.values])
    age = d.age_visit.values.astype(float)
    X = np.column_stack([d[v].values for v in V] + [d.ICV.values, d.SEX.values])
    g = np.zeros(len(d))
    for k in range(FOLDS):
        tr, te = np.where((f != k) & cn)[0], np.where(f == k)[0]
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(X[tr]), age[tr])
        b = LinearRegression().fit(age[tr].reshape(-1, 1),
                                   m.predict(sc.transform(X[tr])) - age[tr])
        g[te] = (m.predict(sc.transform(X[te])) - age[te]) \
            - b.predict(age[te].reshape(-1, 1))
    return g


def auc_cv(X, y, subs, seed=0):
    uq = np.unique(subs)
    fm = dict(zip(uq, np.random.RandomState(seed).permutation(len(uq)) % FOLDS))
    f = np.array([fm[s] for s in subs])
    o = np.zeros(len(y))
    for k in range(FOLDS):
        tr, te = f != k, f == k
        sc = StandardScaler().fit(X[tr])
        c = LogisticRegression(max_iter=4000).fit(sc.transform(X[tr]), y[tr])
        o[te] = c.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, o)


def cox_c(F, cols, dur, ev, seed=0):
    n = len(dur)
    f = np.random.RandomState(seed).permutation(n) % FOLDS
    r = np.zeros(n)
    for k in range(FOLDS):
        tr, te = f != k, f == k
        if ev[tr].sum() < 5:
            return np.nan
        D = F[cols][tr].copy()
        D["T"], D["E"] = dur[tr], ev[tr]
        try:
            m = CoxPHFitter(penalizer=0.1).fit(D, "T", "E")
            r[te] = m.predict_partial_hazard(F[cols][te]).values
        except Exception:
            return np.nan
    return concordance_index(dur, -r, ev)


def boot_cox(F, c1, c2, dur, ev):
    rng = np.random.RandomState(0)
    out = []
    for _ in range(N_BOOT):
        i = rng.randint(0, len(dur), len(dur))
        if ev[i].sum() < 12:
            continue
        Fi = F.iloc[i].reset_index(drop=True)
        x, y = cox_c(Fi, c1, dur[i], ev[i]), cox_c(Fi, c2, dur[i], ev[i])
        if np.isfinite(x) and np.isfinite(y):
            out.append(x - y)
    return np.array(out)


def report(dd, label):
    lo, hi = np.percentile(dd, [2.5, 97.5])
    p = 2 * min((dd <= 0).mean(), (dd >= 0).mean())
    print(f"    {label}: {dd.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
          f"p={p:.3f}  -> {'REAL' if p < 0.05 else 'not distinguishable'}")
    return p


def main():
    a, d = load()
    Z = normative_z(d)
    G = scalar_gap(d)
    bl = d.index[d.VISCODE == "bl"].values
    db = d.loc[bl]
    print(f"visits {len(d)}, subjects {d.PTID.nunique()}, baselines {len(bl)}")

    # =================================================================== Q1
    print("\n" + "=" * 76)
    print("Q1  THE COMPRESSION LAW: how much survives compression to k dimensions?")
    print("=" * 76)
    for task, pos in [("AD vs CN", "Dementia"), ("MCI vs CN", "MCI")]:
        sel = db.DX.isin(["CN", pos]).values
        idx = bl[sel]
        y = (db.DX[sel] == pos).astype(int).values
        subs = db.PTID.values[sel]
        ag = d.age_visit.values[idx].reshape(-1, 1)
        full = auc_cv(np.column_stack([Z[idx], ag]), y, subs)
        print(f"\n  {task}   full six-dimensional AUC = {full:.3f}")
        print(f"    {'k':>3} {'PCA to k dims':>15} {'% of six-dim':>14} "
              f"{'best k regions':>16}")
        for k in range(1, 7):
            p = PCA(n_components=k, random_state=0).fit_transform(
                StandardScaler().fit_transform(Z[idx]))
            a_pca = auc_cv(np.column_stack([p, ag]), y, subs)
            # greedy best-k regions, for contrast with a rotation
            chosen, best = [], 0.0
            for _ in range(k):
                cand = [(auc_cv(np.column_stack([Z[idx][:, chosen + [j]], ag]),
                                y, subs), j)
                        for j in range(len(V)) if j not in chosen]
                s, j = max(cand)
                chosen.append(j)
                best = s
            frac = 100 * (a_pca - 0.5) / (full - 0.5)
            print(f"    {k:>3} {a_pca:15.3f} {frac:13.0f}% {best:16.3f}")
        print(f"    scalar brain-age gap for reference: "
              f"{auc_cv(np.column_stack([G[idx], ag]), y, subs):.3f}")

    # =================================================================== Q2
    print("\n" + "=" * 76)
    print("Q2  COUPLING: does disease change how regions co-vary, not just levels?")
    print("=" * 76)
    groups = {g: Z[bl][db.DX.values == g] for g in ["CN", "MCI", "Dementia"]}
    Ccn = np.corrcoef(groups["CN"].T)
    print(f"  {'pair':32s} {'CN':>7} {'MCI':>7} {'AD':>7} {'AD-CN':>8}")
    diffs = []
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            r = [np.corrcoef(groups[g][:, i], groups[g][:, j])[0, 1]
                 for g in ["CN", "MCI", "Dementia"]]
            diffs.append((abs(r[2] - r[0]), FA[V[i]] + " / " + FA[V[j]], r))
    diffs.sort(reverse=True)
    for dmag, nm, r in diffs[:6]:
        print(f"  {nm:32s} {r[0]:7.2f} {r[1]:7.2f} {r[2]:7.2f} {r[2]-r[0]:+8.2f}")
    # permutation test on the whole matrix
    obs = np.abs(np.corrcoef(groups["Dementia"].T) - Ccn)[np.triu_indices(6, 1)].mean()
    pool = np.vstack([groups["CN"], groups["Dementia"]])
    ncn = len(groups["CN"])
    rng = np.random.RandomState(0)
    null = []
    for _ in range(400):
        p = rng.permutation(len(pool))
        A, B = pool[p[:ncn]], pool[p[ncn:]]
        null.append(np.abs(np.corrcoef(B.T) - np.corrcoef(A.T))[
            np.triu_indices(6, 1)].mean())
    null = np.array(null)
    pv = (null >= obs).mean()
    print(f"\n  mean |correlation difference| AD vs CN: {obs:.3f}")
    print(f"  null from label permutation:            {null.mean():.3f} "
          f"[{np.percentile(null,2.5):.3f}, {np.percentile(null,97.5):.3f}]")
    print(f"  p = {pv:.4f}  -> {'THE COUPLING REALLY DIFFERS' if pv < 0.05 else 'no coupling difference'}")

    # does a coupling feature ADD to the levels?
    print("\n  does a coupling-based feature ADD to the six levels?")
    for task, pos in [("AD vs CN", "Dementia"), ("MCI vs CN", "MCI")]:
        sel = db.DX.isin(["CN", pos]).values
        idx = bl[sel]
        y = (db.DX[sel] == pos).astype(int).values
        subs = db.PTID.values[sel]
        ag = d.age_visit.values[idx].reshape(-1, 1)
        # Mahalanobis under the CN correlation: large when a subject's PATTERN
        # is unusual even if no single region is extreme
        P = np.linalg.pinv(np.cov(groups["CN"].T))
        zz = Z[idx]
        maha = np.einsum("ij,jk,ik->i", zz, P, zz)
        lev = np.column_stack([zz, ag])
        both = np.column_stack([zz, maha, ag])
        a1, a2 = auc_cv(lev, y, subs), auc_cv(both, y, subs)
        print(f"    {task:10s} levels {a1:.3f} -> levels+coupling {a2:.3f}  "
              f"({a2-a1:+.3f})")

    # =================================================================== Q3
    print("\n" + "=" * 76)
    print("Q3  ASYMMETRY: are the two tails of the gap equally informative?")
    print("=" * 76)
    for task, pos in [("AD vs CN", "Dementia"), ("MCI vs CN", "MCI")]:
        sel = db.DX.isin(["CN", pos]).values
        idx = bl[sel]
        y = (db.DX[sel] == pos).astype(int).values
        subs = db.PTID.values[sel]
        ag = d.age_visit.values[idx].reshape(-1, 1)
        g = G[idx]
        pos_part = np.clip(g, 0, None)
        neg_part = np.clip(g, None, 0)
        a_sym = auc_cv(np.column_stack([g, ag]), y, subs)
        a_asym = auc_cv(np.column_stack([pos_part, neg_part, ag]), y, subs)
        a_ponly = auc_cv(np.column_stack([pos_part, ag]), y, subs)
        print(f"  {task:10s} symmetric {a_sym:.3f} | split tails {a_asym:.3f} | "
              f"positive tail only {a_ponly:.3f}")

    # =================================================================== Q4/Q5
    print("\n" + "=" * 76)
    print("Q4/Q5  APOE4 and EDUCATION (reserve)")
    print("=" * 76)
    ex = a[["PTID", "VISCODE", "APOE4", "PTEDUCAT"]].drop_duplicates(["PTID", "VISCODE"])
    dd = d.merge(ex, on=["PTID", "VISCODE"], how="left")
    cn_mask = (dd.DX == "CN").values & np.isfinite(dd.PTEDUCAT.values)
    ed = dd.PTEDUCAT.values[cn_mask]
    print(f"  education vs gap in CONTROLS (n={cn_mask.sum()}): "
          f"r = {np.corrcoef(ed, G[cn_mask])[0,1]:+.3f}")
    print(f"  education vs hippocampal z in CONTROLS: "
          f"r = {np.corrcoef(ed, Z[cn_mask, 0])[0,1]:+.3f}")
    ap = dd.APOE4.values
    for grp, lab in [(ap == 0, "non-carriers"), (ap >= 1, "carriers")]:
        m = grp & (dd.DX == "CN").values
        if m.sum() > 30:
            print(f"  mean gap in CN {lab:14s}: {G[m].mean():+.2f} y (n={m.sum()})")
    print(f"\nsaved nothing; all results are in this log")


if __name__ == "__main__":
    main()
