#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TWO PROPERTIES OF THE NUMBER ITSELF: ITS UNIT, AND ITS DIMENSION.

E48 said the gap is the wrong single number. E52 said it is a level, not a speed.
Both are about what the gap MEANS. Two properties of the quantity remain untested.

PART A — IS THE UNIT HONEST?
Every paper in the field writes "this brain is 5 years older". That sentence makes
a quantitative claim nobody checks: that a gap of +5 is worth 5 years of age. The
operational meaning of "worth" is risk, so put both in one hazard model and
compare the coefficients. If a year of gap carries the same hazard as a year of
age, the unit is honest. If not, "years" is an arbitrary scale wearing the costume
of time, and every paper that reports it is mistranslating its own result.

PART B — IS THE DEVIATION ONE-DIMENSIONAL?
E48 found a better single number. The deeper question is why a single number at
all. Take k principal components of the regional deviations, k = 1..6, fitted on
CONTROLS ONLY so nothing is supervised, and ask where the increments stop.
    if k=2 beats k=1  -> the one-number paradigm is wrong in principle, and the
                         right output is a short vector whose length we measured
    if it does not    -> the disease deviation really IS one-dimensional, and
                         brain age's only sin is picking the wrong dimension
Informative either way, which is the point.

Both endpoints are used throughout: AD vs CN at baseline (AUC) and MCI -> AD
conversion (Cox C-index). Every comparison carries a subject-level bootstrap, and
the rotation is refitted inside each bootstrap draw so nothing is optimistic.
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

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project root (local or Kaggle)
DATA = os.path.join(BASE, "data")
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
FA = {"Hippocampus": "هیپوکامپ", "Entorhinal": "آنتورینال", "Fusiform": "دوکی",
      "MidTemp": "گیجگاهی میانی", "Ventricles": "بطن‌ها", "WholeBrain": "کل مغز"}
FOLDS, N_AUC_BOOT, N_COX_BOOT = 5, 300, 200


def load():
    a = pd.read_csv(os.path.join(DATA, "ADNIMERGE_05Mar2026.csv"), low_memory=False)
    d = a.dropna(subset=["AGE", "Years_bl", "ICV", "PTGENDER", "DX"] + V).copy()
    d["SEX"] = (d.PTGENDER == "Male").astype(int)
    d["age_visit"] = d.AGE + d.Years_bl
    b = d[d.VISCODE == "bl"].drop_duplicates("PTID").reset_index(drop=True)
    return a, b


def folds_of(ids, seed=0):
    uq = np.unique(ids)
    fm = dict(zip(uq, np.random.RandomState(seed).permutation(len(uq)) % FOLDS))
    return np.array([fm[s] for s in ids])


def normative_z(b):
    """CN-trained regional deviation at baseline, out of fold by subject."""
    f = folds_of(b.PTID.values)
    cn = (b.DX == "CN").values
    ag = b.age_visit.values.astype(float)
    X = np.column_stack([ag, ag ** 2, b.SEX.values, b.ICV.values])
    Z = np.zeros((len(b), len(V)))
    for j, v in enumerate(V):
        y = b[v].values.astype(float)
        for k in range(FOLDS):
            tr, te = np.where((f != k) & cn)[0], np.where(f == k)[0]
            sc = StandardScaler().fit(X[tr])
            m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(X[tr]), y[tr])
            s = (y[tr] - m.predict(sc.transform(X[tr]))).std(ddof=1)
            Z[te, j] = (y[te] - m.predict(sc.transform(X[te]))) / max(s, 1e-9)
    return Z


def brain_age_gap(b):
    f = folds_of(b.PTID.values)
    cn = (b.DX == "CN").values
    ag = b.age_visit.values.astype(float)
    X = np.column_stack([b[v].values for v in V] + [b.ICV.values, b.SEX.values])
    g = np.zeros(len(b))
    for k in range(FOLDS):
        tr, te = np.where((f != k) & cn)[0], np.where(f == k)[0]
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(X[tr]), ag[tr])
        bb = LinearRegression().fit(ag[tr].reshape(-1, 1),
                                    m.predict(sc.transform(X[tr])) - ag[tr])
        g[te] = (m.predict(sc.transform(X[te])) - ag[te]) \
            - bb.predict(ag[te].reshape(-1, 1))
    return g


def survival(a, ids, targets):
    """Time to first visit with a DX in `targets`, strictly after baseline."""
    h = a[["PTID", "Years_bl", "DX"]].dropna(subset=["DX"])
    ev_t = h[h.DX.isin(targets)].groupby("PTID").Years_bl.min()
    last = h.groupby("PTID").Years_bl.max()
    keep, dur, ev = [], [], []
    for s in ids:
        if s not in last.index:
            continue
        t = ev_t[s] if s in ev_t.index else last[s]
        if t > 0:
            keep.append(s)
            dur.append(t)
            ev.append(1 if s in ev_t.index else 0)
    return keep, np.array(dur), np.array(ev)


# =============================================================== PART A
def part_a(a, b, gap, Z):
    print("=" * 78)
    print("PART A  IS ONE YEAR OF GAP WORTH ONE YEAR OF AGE?")
    print("=" * 78)
    print("  Both go into the SAME hazard model. If the unit is honest the two")
    print("  coefficients match. Hippocampal deviation is included as a third")
    print("  yardstick, to translate all three into each other.\n")

    for label, pool, targets in [
            ("MCI -> dementia", b[b.DX == "MCI"], ["Dementia"]),
            ("CN -> MCI or dementia", b[b.DX == "CN"], ["MCI", "Dementia"])]:
        keep, dur, ev = survival(a, pool.PTID.values, targets)
        idx = pool.index[pool.PTID.isin(set(keep))].values
        o = {s: i for i, s in enumerate(keep)}
        ii = np.array([o[s] for s in b.PTID.values[idx]])
        dur_, ev_ = dur[ii], ev[ii]
        if ev_.sum() < 30:
            print(f"  {label}: only {int(ev_.sum())} events, skipped")
            continue
        D = pd.DataFrame(dict(age=b.age_visit.values[idx], gap=gap[idx],
                              hipp=-Z[idx, 0], T=dur_, E=ev_))
        m = CoxPHFitter().fit(D, "T", "E")
        ba, bg, bh = (m.params_["age"], m.params_["gap"], m.params_["hipp"])
        print(f"  {label}   n={len(D)}  events={int(ev_.sum())}  "
              f"age range {D.age.min():.0f}-{D.age.max():.0f}")
        for nm, coef in [("chronological age", ba), ("brain-age gap", bg),
                         ("hippocampal deviation (1 SD)", bh)]:
            ci = m.confidence_intervals_.loc[
                {"chronological age": "age", "brain-age gap": "gap",
                 "hippocampal deviation (1 SD)": "hipp"}[nm]]
            print(f"    {nm:32s} beta = {coef:+.4f}   "
                  f"HR per unit = {np.exp(coef):.3f} "
                  f"[{np.exp(ci.iloc[0]):.3f}, {np.exp(ci.iloc[1]):.3f}]")
        # bootstrap the ratio
        rng = np.random.RandomState(0)
        rr = []
        for _ in range(1000):
            i = rng.randint(0, len(D), len(D))
            if D.E.values[i].sum() < 20:
                continue
            try:
                mm = CoxPHFitter().fit(D.iloc[i].reset_index(drop=True), "T", "E")
                if abs(mm.params_["age"]) > 1e-4:
                    rr.append(mm.params_["gap"] / mm.params_["age"])
            except Exception:
                pass
        rr = np.array(rr)
        lo, hi = np.percentile(rr, [2.5, 97.5])
        print(f"    -> ONE YEAR OF GAP = {bg/ba:.1f} YEARS OF AGE"
              f"   95% CI [{lo:.1f}, {hi:.1f}]")
        print(f"    -> one SD of hippocampal loss = {bh/bg:.1f} years of gap")
        print()
    print("  CAVEAT, stated not hidden: ADNI's age range is narrow and its")
    print("  subjects are a memory-clinic sample, so chronological age is a")
    print("  weak predictor here. That inflates the ratio. The direction of the")
    print("  conclusion is safe; the exact multiplier is cohort-dependent.\n")


# =============================================================== PART B
def auc_cv(X, y, subs, seed=0):
    f = folds_of(subs, seed)
    o = np.zeros(len(y))
    for k in range(FOLDS):
        tr, te = f != k, f == k
        sc = StandardScaler().fit(X[tr])
        c = LogisticRegression(max_iter=4000).fit(sc.transform(X[tr]), y[tr])
        o[te] = c.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, o)


def cox_c(F, cols, dur, ev, seed=0):
    f = np.random.RandomState(seed).permutation(len(dur)) % FOLDS
    r = np.zeros(len(dur))
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


def rotate(Zfit, Zall, k):
    """PCA fitted on CONTROLS ONLY, top-k scores for everyone. No supervision."""
    sc = StandardScaler().fit(Zfit)
    p = PCA(n_components=k, random_state=0).fit(sc.transform(Zfit))
    return p.transform(sc.transform(Zall)), p


def part_b(a, b, gap, Z):
    print("=" * 78)
    print("PART B  HOW MANY DIMENSIONS DOES THE DEVIATION ACTUALLY NEED?")
    print("=" * 78)
    print("  Rotation fitted on CONTROLS ONLY, so k components is an unsupervised")
    print("  compression of the same six numbers. k=6 is the six themselves.\n")

    # ---------------------------------------------------- endpoint 1: AD vs CN
    sel = b.DX.isin(["CN", "Dementia"]).values
    y = (b.DX.values[sel] == "Dementia").astype(int)
    subs = b.PTID.values[sel]
    ag = b.age_visit.values[sel].reshape(-1, 1)
    Zs, cn_s = Z[sel], (b.DX.values[sel] == "CN")
    print(f"  ENDPOINT 1: AD vs CN at baseline   n={len(y)} "
          f"({int(y.sum())} AD, {int((1-y).sum())} CN)")
    base = auc_cv(np.column_stack([gap[sel], ag]), y, subs)
    print(f"    brain-age gap (1 number)        AUC = {base:.3f}")
    for k in range(1, 7):
        P, _ = rotate(Zs[cn_s], Zs, k)
        print(f"    k = {k} component{'s' if k>1 else ' '}                  "
              f"AUC = {auc_cv(np.column_stack([P, ag]), y, subs):.3f}")

    print("\n    increments, subject-level bootstrap with the rotation REFITTED:")
    rng = np.random.RandomState(0)
    uq = np.unique(subs)
    acc = {k: [] for k in range(1, 7)}
    for _ in range(N_AUC_BOOT):
        pick = set(rng.choice(uq, len(uq), replace=True))
        m = np.array([s in pick for s in subs])
        if y[m].sum() < 25 or (1 - y[m]).sum() < 25:
            continue
        Zm, cm = Zs[m], cn_s[m]
        try:
            prev = None
            for k in range(1, 7):
                P, _ = rotate(Zm[cm], Zm, k)
                cur = auc_cv(np.column_stack([P, ag[m]]), y[m], subs[m])
                if prev is not None:
                    acc[k].append(cur - prev)
                else:
                    acc[1].append(cur - auc_cv(
                        np.column_stack([gap[sel][m], ag[m]]), y[m], subs[m]))
                prev = cur
        except Exception:
            pass
    for k in range(1, 7):
        d = np.array(acc[k])
        if len(d) < 20:
            continue
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        lab = "k=1 minus the gap" if k == 1 else f"k={k} minus k={k-1}"
        print(f"      {lab:22s} {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"  p={p:.3f}  {'REAL' if p<0.05 else '-'}")

    # ------------------------------------------- endpoint 2: MCI -> conversion
    pool = b[b.DX == "MCI"]
    keep, dur, ev = survival(a, pool.PTID.values, ["Dementia"])
    idx = pool.index[pool.PTID.isin(set(keep))].values
    o = {s: i for i, s in enumerate(keep)}
    ii = np.array([o[s] for s in b.PTID.values[idx]])
    dur, ev = dur[ii], ev[ii]
    Zm_all, cn_all = Z, (b.DX.values == "CN")
    print(f"\n  ENDPOINT 2: MCI -> AD conversion   n={len(idx)}, "
          f"{int(ev.sum())} events")
    F = pd.DataFrame(dict(age=b.age_visit.values[idx], gap=gap[idx]))
    print(f"    brain-age gap (1 number)        C = "
          f"{np.nanmean([cox_c(F, ['gap','age'], dur, ev, s) for s in range(6)]):.3f}")
    for k in range(1, 7):
        P, _ = rotate(Z[cn_all], Z[idx], k)
        G = F.copy()
        for j in range(k):
            G[f"c{j}"] = P[:, j]
        cols = [f"c{j}" for j in range(k)] + ["age"]
        print(f"    k = {k} component{'s' if k>1 else ' '}                  "
              f"C = {np.nanmean([cox_c(G, cols, dur, ev, s) for s in range(6)]):.3f}")

    print("\n    increments, subject-level bootstrap:")
    rng = np.random.RandomState(1)
    accC = {k: [] for k in range(2, 7)}
    for _ in range(N_COX_BOOT):
        i = rng.randint(0, len(dur), len(dur))
        if ev[i].sum() < 25:
            continue
        P6, _ = rotate(Z[cn_all], Z[idx][i], 6)
        G = pd.DataFrame(dict(age=F.age.values[i]))
        for j in range(6):
            G[f"c{j}"] = P6[:, j]
        prev = None
        for k in range(1, 7):
            c = cox_c(G, [f"c{j}" for j in range(k)] + ["age"], dur[i], ev[i], 0)
            if not np.isfinite(c):
                prev = None
                break
            if prev is not None:
                accC[k].append(c - prev)
            prev = c
    for k in range(2, 7):
        d = np.array(accC[k])
        if len(d) < 20:
            continue
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        print(f"      k={k} minus k={k-1}        {d.mean():+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]  p={p:.3f}  "
              f"{'REAL' if p<0.05 else '-'}")

    # ---------------------------------------------------- what ARE they?
    print("\n  what the first three components are, region by region:")
    _, p = rotate(Z[cn_all], Z, 3)
    for j in range(3):
        print(f"    component {j+1} ({100*p.explained_variance_ratio_[j]:4.1f}% of "
              f"control variance): " +
              "  ".join(f"{FA[v]} {w:+.2f}" for v, w in zip(V, p.components_[j])))


def main():
    a, b = load()
    Z = normative_z(b)
    gap = brain_age_gap(b)
    print(f"baseline scans: {len(b)}  "
          f"(CN {int((b.DX=='CN').sum())}, MCI {int((b.DX=='MCI').sum())}, "
          f"AD {int((b.DX=='Dementia').sum())})\n")
    part_a(a, b, gap, Z)
    part_b(a, b, gap, Z)


if __name__ == "__main__":
    main()
