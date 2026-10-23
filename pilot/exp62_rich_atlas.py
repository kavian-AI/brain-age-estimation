#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EVERYTHING AGAIN, ON 98 REGIONS INSTEAD OF SIX.

The biggest hole in the proposal was the atlas. ADNIMERGE ships six regions, so
every claim so far - the uniform average beats the gap on prognosis, prognosis is
one-dimensional, the age model misallocates its weight - was made in a space where
the published outlier count is degenerate (86% of controls score zero) and where
"uniform" means averaging six numbers. TADPOLE gives the same ADNI subjects with 98
FreeSurfer regions. Here every headline is re-run in that space.

    PART A  the headline, three endpoints: gap vs uniform vs hippocampus vs outlier
            count vs mean|z| vs first PC vs a supervised ceiling
    PART B  is prognosis still one-dimensional with 98 regions to choose from?
    PART C  the mechanism: where does a 98-region age model put its weight,
            compared with a 98-region DISEASE model?
    PART D  software: the same test on the LONGITUDINAL FreeSurfer pipeline, and a
            transfer from FreeSurfer 4.3 subjects to FreeSurfer 5.1 subjects

Predictions written before running, so the result can contradict them:
  1. uniform still beats the gap on conversion
  2. outlier count is no longer degenerate and closes some of its gap to uniform
  3. the age model's weight sits outside the medial temporal lobe
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp53_units_and_dimension as E  # noqa: E402
import tadpole_atlas as TA  # noqa: E402
from lifelines import CoxPHFitter  # noqa: E402
from lifelines.utils import concordance_index  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import (LinearRegression, LogisticRegression,  # noqa: E402
                                  RidgeCV)
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

FOLDS, N_AUC, N_COX = 5, 300, 200


# ---------------------------------------------------------------- building
def build(pipe="X"):
    a, b = E.load()
    R, T = TA.load_regions(pipe)
    names = list(T.name)
    m = b.merge(R, on="PTID", how="inner")
    m = m.dropna(subset=names + ["ICV_T"]).reset_index(drop=True)
    return a, m, T, names


def normative(m, names, cn):
    """Deviation of every region from controls, out of fold by subject.
    Also the bilateral hippocampus as its own region, so 'hippocampus alone'
    is scored on exactly the same footing as everything else."""
    f = E.folds_of(m.PTID.values)
    ag = m.age_visit.values.astype(float)
    X = np.column_stack([ag, ag ** 2, m.SEX.values, m.ICV_T.values])
    Y = np.column_stack([m[n].values for n in names]
                        + [m["L_hippocampus"].values + m["R_hippocampus"].values])
    Z = np.zeros_like(Y, dtype=float)
    for k in range(FOLDS):
        tr, te = np.where((f != k) & cn)[0], np.where(f == k)[0]
        sc = StandardScaler().fit(X[tr])
        lr = LinearRegression().fit(sc.transform(X[tr]), Y[tr])
        res = Y[tr] - lr.predict(sc.transform(X[tr]))
        sd = res.std(0, ddof=1)
        sd[sd < 1e-9] = 1e-9
        Z[te] = (Y[te] - lr.predict(sc.transform(X[te]))) / sd
    return Z[:, :-1], Z[:, -1]


def rich_gap(m, names, cn):
    f = E.folds_of(m.PTID.values)
    ag = m.age_visit.values.astype(float)
    X = np.column_stack([m[n].values for n in names] + [m.ICV_T.values, m.SEX.values])
    g = np.zeros(len(m))
    for k in range(FOLDS):
        tr, te = np.where((f != k) & cn)[0], np.where(f == k)[0]
        sc = StandardScaler().fit(X[tr])
        md = RidgeCV(alphas=np.logspace(-2, 5, 40)).fit(sc.transform(X[tr]), ag[tr])
        bb = LinearRegression().fit(ag[tr].reshape(-1, 1),
                                    md.predict(sc.transform(X[tr])) - ag[tr])
        g[te] = (md.predict(sc.transform(X[te])) - ag[te]) \
            - bb.predict(ag[te].reshape(-1, 1))
    return g


def pc_scores(Z, cn, k):
    sc = StandardScaler().fit(Z[cn])
    p = PCA(n_components=k, random_state=0).fit(sc.transform(Z[cn]))
    return p.transform(sc.transform(Z))


def cidx(X, d_, e_, seed=0, pen=1.0):
    f = np.random.RandomState(seed).permutation(len(d_)) % 3
    r = np.zeros(len(d_))
    nm = [f"x{j}" for j in range(X.shape[1])]
    for k in range(3):
        tr, te = f != k, f == k
        if e_[tr].sum() < 5:
            return np.nan
        D = pd.DataFrame(X[tr], columns=nm)
        D["T"], D["E"] = d_[tr], e_[tr]
        try:
            mm = CoxPHFitter(penalizer=pen).fit(D, "T", "E")
            r[te] = mm.predict_partial_hazard(pd.DataFrame(X[te], columns=nm)).values
        except Exception:
            return np.nan
    return concordance_index(d_, -r, e_)


def auc_ridge(X, y, subs, C=0.05):
    """Penalised logistic for the high-dimensional ceiling arm."""
    f = E.folds_of(subs)
    o = np.zeros(len(y))
    for k in range(FOLDS):
        tr, te = f != k, f == k
        sc = StandardScaler().fit(X[tr])
        c = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X[tr]), y[tr])
        o[te] = c.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, o)


def show(lab, d):
    d = np.asarray(d)
    if len(d) < 20:
        print(f"      {lab:40s} too few draws ({len(d)})")
        return
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    print(f"      {lab:40s} {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.3f}"
          f"  {'REAL' if p < 0.05 else '-'}")


def endpoints(a, m, arms, cn, label, ceiling_Z=None, boot=True):
    """Score every arm on the three endpoints, bootstrap the key contrasts."""
    print(f"\n  ---- {label} ----")
    pairs = [("uniform", "gap"), ("uniform", "hippocampus"), ("uniform", "outlier count")]
    for pos, nm in [("Dementia", "AD vs CN"), ("MCI", "MCI vs CN")]:
        sel = m.DX.isin(["CN", pos]).values
        y = (m.DX.values[sel] == pos).astype(int)
        subs = m.PTID.values[sel]
        ag = m.age_visit.values[sel].reshape(-1, 1)
        line = "  ".join(f"{k} {E.auc_cv(np.column_stack([v[sel], ag]), y, subs):.3f}"
                         for k, v in arms.items())
        ceil = ""
        if ceiling_Z is not None:
            ceil = f"  | supervised ceiling {auc_ridge(np.column_stack([ceiling_Z[sel], ag]), y, subs):.3f}"
        print(f"    {nm} (n={len(y)}, {int(y.sum())} cases)")
        print(f"      {line}{ceil}")
        if boot:
            rng = np.random.RandomState(0)
            uq = np.unique(subs)
            acc = {p: [] for p in pairs}
            for _ in range(N_AUC):
                pick = set(rng.choice(uq, len(uq), replace=True))
                mm = np.array([s in pick for s in subs])
                if y[mm].sum() < 20 or (1 - y[mm]).sum() < 20:
                    continue
                try:
                    v = {k: E.auc_cv(np.column_stack([arms[k][sel][mm], ag[mm]]),
                                     y[mm], subs[mm]) for k in {x for p in pairs for x in p}}
                    for p in pairs:
                        acc[p].append(v[p[0]] - v[p[1]])
                except Exception:
                    pass
            for p in pairs:
                show(f"{p[0]} minus {p[1]}", acc[p])

    pool = m[m.DX == "MCI"]
    keep, dur, ev = E.survival(a, pool.PTID.values, ["Dementia"])
    idx = pool.index[pool.PTID.isin(set(keep))].values
    o = {s: i for i, s in enumerate(keep)}
    ii = np.array([o[s] for s in m.PTID.values[idx]])
    dur, ev = dur[ii], ev[ii]
    agC = m.age_visit.values[idx]
    line = "  ".join(f"{k} {np.nanmean([cidx(np.column_stack([v[idx], agC]), dur, ev, s) for s in range(3)]):.3f}"
                     for k, v in arms.items())
    ceil = ""
    if ceiling_Z is not None:
        cz = np.nanmean([cidx(np.column_stack([ceiling_Z[idx], agC]), dur, ev, s, pen=5.0)
                         for s in range(3)])
        ceil = f"  | supervised ceiling {cz:.3f}"
    print(f"    MCI -> AD conversion (n={len(idx)}, {int(ev.sum())} events)")
    print(f"      {line}{ceil}")
    if boot:
        rng = np.random.RandomState(1)
        acc = {p: [] for p in pairs}
        for _ in range(N_COX):
            i = rng.randint(0, len(dur), len(dur))
            if ev[i].sum() < 20:
                continue
            v = {k: cidx(np.column_stack([arms[k][idx][i], agC[i]]), dur[i], ev[i], 0)
                 for k in {x for p in pairs for x in p}}
            if any(not np.isfinite(x) for x in v.values()):
                continue
            for p in pairs:
                acc[p].append(v[p[0]] - v[p[1]])
        for p in pairs:
            show(f"{p[0]} minus {p[1]}", acc[p])
    return idx, dur, ev, agC


def main():
    # ================================================================ A
    a, m, T, names = build("X")
    cn = (m.DX.values == "CN")
    print("=" * 88)
    print(f"RICH ATLAS: {len(names)} FreeSurfer regions (cross-sectional pipeline)")
    print(f"subjects with every region present: {len(m)}  "
          f"(CN {int(cn.sum())}, MCI {int((m.DX=='MCI').sum())}, "
          f"AD {int((m.DX=='Dementia').sum())})")
    print("=" * 88)
    Z, zh = normative(m, names, cn)
    S = T.sign.values
    Zs = Z * S
    gap = rich_gap(m, names, cn)
    uni = Zs.mean(1)
    oc = (Zs > 1.96).sum(1).astype(float)
    pc1 = pc_scores(Z, cn, 1)[:, 0]
    if np.corrcoef(pc1, uni)[0, 1] < 0:
        pc1 = -pc1
    arms = {"gap": gap, "uniform": uni, "hippocampus": -zh,
            "outlier count": oc, "mean|z|": np.abs(Z).mean(1), "PC1": pc1}

    print("\n  is the outlier count still degenerate?")
    for g in ["CN", "MCI", "Dementia"]:
        q = m.DX.values == g
        print(f"    {g:9s} mean {oc[q].mean():5.2f} of {len(names)} flagged,"
              f"  {100*(oc[q]==0).mean():3.0f}% of subjects score zero")
    print(f"\n  corr(uniform, gap) = {np.corrcoef(uni, gap)[0,1]:+.3f}   "
          f"corr(uniform, PC1) = {np.corrcoef(uni, pc1)[0,1]:+.3f}   "
          f"corr(uniform, hippocampus) = {np.corrcoef(uni, -zh)[0,1]:+.3f}")

    print("\n" + "=" * 88)
    print("PART A  THE HEADLINE ON 98 REGIONS")
    print("=" * 88)
    idx, dur, ev, agC = endpoints(a, m, arms, cn, "cross-sectional FreeSurfer, 98 regions",
                                  ceiling_Z=Z)

    # ================================================================ B
    print("\n" + "=" * 88)
    print("PART B  IS PROGNOSIS STILL ONE-DIMENSIONAL WITH 98 REGIONS?")
    print("=" * 88)
    P = pc_scores(Z, cn, 20)
    P[:, 0] = P[:, 0] * np.sign(np.corrcoef(P[:, 0], uni)[0, 1])
    for k in [1, 2, 3, 5, 10, 20]:
        c = np.nanmean([cidx(np.column_stack([P[idx, :k], agC]), dur, ev, s, pen=2.0)
                        for s in range(3)])
        print(f"    k = {k:2d} components   C = {c:.3f}")
    rng = np.random.RandomState(2)
    acc = {(5, 1): [], (20, 1): []}
    for _ in range(N_COX):
        i = rng.randint(0, len(dur), len(dur))
        if ev[i].sum() < 20:
            continue
        v = {k: cidx(np.column_stack([P[idx][i, :k], agC[i]]), dur[i], ev[i], 0, pen=2.0)
             for k in [1, 5, 20]}
        if any(not np.isfinite(x) for x in v.values()):
            continue
        for (hi_, lo_) in acc:
            acc[(hi_, lo_)].append(v[hi_] - v[lo_])
    show("k=5 minus k=1", acc[(5, 1)])
    show("k=20 minus k=1", acc[(20, 1)])

    # ================================================================ C
    print("\n" + "=" * 88)
    print("PART C  WHERE DOES EACH MODEL PUT ITS WEIGHT?")
    print("=" * 88)
    ag = m.age_visit.values.astype(float)
    Xv = np.column_stack([m[n].values for n in names] + [m.ICV_T.values, m.SEX.values])
    sc = StandardScaler().fit(Xv[cn])
    age_model = RidgeCV(alphas=np.logspace(-2, 5, 40)).fit(sc.transform(Xv[cn]), ag[cn])
    wa = np.abs(age_model.coef_[:len(names)])
    selD = m.DX.isin(["CN", "Dementia"]).values
    Xd = np.column_stack([Z[selD], ag[selD]])
    dis_model = LogisticRegression(C=0.05, max_iter=5000).fit(
        StandardScaler().fit_transform(Xd), (m.DX.values[selD] == "Dementia").astype(int))
    wd = np.abs(dis_model.coef_[0][:len(names)])
    cls = T.cls.values
    print(f"    {'region class':14s} {'n':>3s} {'age model':>11s} {'disease model':>15s}")
    for c in ["mtl", "cortex", "subcortical", "cerebellum", "ventricle", "callosum",
              "brainstem"]:
        q = cls == c
        print(f"    {c:14s} {q.sum():3d} {100*wa[q].sum()/wa.sum():10.1f}% "
              f"{100*wd[q].sum()/wd.sum():14.1f}%")
    print(f"    ICV weight in the age model (standardised): "
          f"{age_model.coef_[len(names)]:+.3f}  vs the largest region "
          f"{wa.max():.3f}")
    top = np.argsort(-wa)[:12]
    print("\n    the age model's twelve heaviest regions:")
    for j in top:
        print(f"      {names[j]:30s} {age_model.coef_[j]:+.3f}   ({cls[j]})")
    topd = np.argsort(-wd)[:12]
    print("\n    the disease model's twelve heaviest regions:")
    for j in topd:
        print(f"      {names[j]:30s} {dis_model.coef_[0][j]:+.3f}   ({cls[j]})")
    rank_corr = pd.Series(wa).rank().corr(pd.Series(wd).rank())
    print(f"\n    rank correlation between the two weightings: {rank_corr:+.3f}")

    # ================================================================ D
    print("\n" + "=" * 88)
    print("PART D  SOFTWARE: LONGITUDINAL PIPELINE, AND FreeSurfer 4.3 -> 5.1")
    print("=" * 88)
    aL, mL, TL, namesL = build("L")
    cnL = (mL.DX.values == "CN")
    ZL, zhL = normative(mL, namesL, cnL)
    armsL = {"gap": rich_gap(mL, namesL, cnL), "uniform": (ZL * TL.sign.values).mean(1),
             "hippocampus": -zhL,
             "outlier count": ((ZL * TL.sign.values) > 1.96).sum(1).astype(float)}
    endpoints(aL, mL, armsL, cnL, f"LONGITUDINAL pipeline, n={len(mL)}")

    v43 = m.FSVERSION_T.str.contains("4.3", na=False).values
    v51 = m.FSVERSION_T.str.contains("5.1", na=False).values
    print(f"\n  FreeSurfer 4.3 subjects {v43.sum()}, 5.1 subjects {v51.sum()}")
    if v43.sum() > 200 and v51.sum() > 200:
        tr, te = m[v43].reset_index(drop=True), m[v51].reset_index(drop=True)
        cn_tr = tr.DX.values == "CN"
        agT, agE = tr.age_visit.values, te.age_visit.values
        XT = np.column_stack([agT, agT ** 2, tr.SEX.values, tr.ICV_T.values])
        XE = np.column_stack([agE, agE ** 2, te.SEX.values, te.ICV_T.values])
        s1 = StandardScaler().fit(XT[cn_tr])
        YT = np.column_stack([tr[n].values for n in names])
        YE = np.column_stack([te[n].values for n in names])
        lr = LinearRegression().fit(s1.transform(XT[cn_tr]), YT[cn_tr])
        sd = (YT[cn_tr] - lr.predict(s1.transform(XT[cn_tr]))).std(0, ddof=1)
        ZE = (YE - lr.predict(s1.transform(XE))) / sd
        uE = (ZE * S).mean(1)
        XaT = np.column_stack([YT, tr.ICV_T.values, tr.SEX.values])
        XaE = np.column_stack([YE, te.ICV_T.values, te.SEX.values])
        s2 = StandardScaler().fit(XaT[cn_tr])
        am = RidgeCV(alphas=np.logspace(-2, 5, 40)).fit(s2.transform(XaT[cn_tr]), agT[cn_tr])
        bb = LinearRegression().fit(agT[cn_tr].reshape(-1, 1),
                                    am.predict(s2.transform(XaT[cn_tr])) - agT[cn_tr])
        gE = (am.predict(s2.transform(XaE)) - agE) - bb.predict(agE.reshape(-1, 1))
        sel = te.DX.isin(["CN", "Dementia"]).values
        y = (te.DX.values[sel] == "Dementia").astype(int)
        agS = agE[sel].reshape(-1, 1)

        def auc_in(x, yy, aa):
            s_ = StandardScaler().fit(np.column_stack([x, aa]))
            c_ = LogisticRegression(max_iter=4000).fit(s_.transform(np.column_stack([x, aa])), yy)
            return roc_auc_score(yy, c_.predict_proba(s_.transform(np.column_stack([x, aa])))[:, 1])
        print(f"  trained on 4.3 ({int(cn_tr.sum())} CN) -> tested on 5.1 AD vs CN "
              f"(n={len(y)}, {int(y.sum())} AD)")
        print(f"    gap {auc_in(gE[sel], y, agS):.3f}   uniform {auc_in(uE[sel], y, agS):.3f}")
        rng = np.random.RandomState(0)
        dd = []
        for _ in range(400):
            i = rng.randint(0, len(y), len(y))
            if y[i].sum() < 15 or (1 - y[i]).sum() < 15:
                continue
            dd.append(auc_in(uE[sel][i], y[i], agS[i]) - auc_in(gE[sel][i], y[i], agS[i]))
        show("uniform minus gap, across software versions", dd)


if __name__ == "__main__":
    main()
