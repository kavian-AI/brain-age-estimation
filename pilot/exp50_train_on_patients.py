#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WHAT IF THE AGE MODEL IS TRAINED ON PATIENTS?

Every brain-age paper fits the age model in healthy people. The justification is
circular: the gap is defined as a deviation from healthy aging, so the reference
must be healthy. But nothing forces that. A model fitted in AD patients learns
"what a brain looks like at a given age GIVEN disease", and its residual on an
MCI subject is a different quantity - closer to "how far along the disease is
this brain for its age" than to "how old does this brain look".

E49 part 3 asked whether such a model reads AGE well. It does not. But age
accuracy is not the point - E48 already showed that optimising for age is what
breaks brain age in the first place. The question that matters is PROGNOSIS:

    does a gap from a PATIENT-trained age model predict MCI -> AD conversion
    better than the gap from the standard control-trained one?

If yes, the field has been leaving signal on the table by convention alone. If
no, the convention is at least harmless, and E48's story tightens: the damage is
in the TARGET (age), not in the POPULATION the target is learned from.

FAIRNESS, which is most of the work here:
  - every training set is n=200, drawn to the SAME age strata, so arms differ
    only in the DIAGNOSIS of the people trained on
  - each arm's gap is bias-corrected against ITS OWN training group, which is
    the internally consistent definition
  - the MCI-trained arm is out-of-fold by subject; it never sees a test subject
  - each gap is averaged over 20 matched draws, so an arm is not judged on the
    luck of one draw
  - the test set is IDENTICAL for every arm
  - subject-level bootstrap on every comparison
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler

BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
DATA = os.path.join(BASE, "data")
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
FA = {"Hippocampus": "هیپوکامپ", "Entorhinal": "آنتورینال", "Fusiform": "دوکی",
      "MidTemp": "گیجگاهی میانی", "Ventricles": "بطن‌ها", "WholeBrain": "کل مغز"}
N_TR, N_DRAW, FOLDS, N_BOOT = 200, 20, 5, 400


def load():
    a = pd.read_csv(os.path.join(DATA, "ADNIMERGE_05Mar2026.csv"), low_memory=False)
    d = a.dropna(subset=["AGE", "Years_bl", "ICV", "PTGENDER", "DX"] + V).copy()
    d["SEX"] = (d.PTGENDER == "Male").astype(int)
    d["age_visit"] = d.AGE + d.Years_bl
    b = d[d.VISCODE == "bl"].drop_duplicates("PTID").reset_index(drop=True)
    return a, b


def feats(f):
    return np.column_stack([f[v].values for v in V] + [f.ICV.values, f.SEX.values])


def fit_gap_model(tr):
    """Age model plus its OWN bias correction, both from the same training set."""
    sc = StandardScaler().fit(feats(tr))
    m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(feats(tr)),
                                                   tr.age_visit.values)
    at = tr.age_visit.values
    bc = LinearRegression().fit(at.reshape(-1, 1),
                                m.predict(sc.transform(feats(tr))) - at)

    def gap(te):
        ae = te.age_visit.values
        return (m.predict(sc.transform(feats(te))) - ae) - bc.predict(ae.reshape(-1, 1))
    return gap


def matched_draw(src, edges, r):
    """n=N_TR drawn to a fixed age profile, so arms differ only by diagnosis."""
    take = []
    for k in range(len(edges) - 1):
        c = src[(src.age_visit >= edges[k]) & (src.age_visit < edges[k + 1])]
        if len(c) == 0:
            continue
        take.append(c.sample(n=N_TR // (len(edges) - 1),
                             replace=len(c) < N_TR // (len(edges) - 1),
                             random_state=r))
    return pd.concat(take)


def outcome(a, ids):
    """Conversion to dementia strictly AFTER the baseline visit."""
    h = a[["PTID", "Years_bl", "DX"]].dropna(subset=["DX"])
    conv = h[h.DX == "Dementia"].groupby("PTID").Years_bl.min()
    last = h.groupby("PTID").Years_bl.max()
    keep, dur, ev = [], [], []
    for s in ids:
        if s not in last.index:
            continue
        t = conv[s] if s in conv.index else last[s]
        if t > 0:
            keep.append(s)
            dur.append(t)
            ev.append(1 if s in conv.index else 0)
    return keep, np.array(dur), np.array(ev)


def cox_c(F, cols, dur, ev, seed):
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


def boot(F, c1, c2, dur, ev):
    rng = np.random.RandomState(0)
    out = []
    for _ in range(N_BOOT):
        i = rng.randint(0, len(dur), len(dur))
        if ev[i].sum() < 12:
            continue
        Fi = F.iloc[i].reset_index(drop=True)
        x, y = cox_c(Fi, c1, dur[i], ev[i], 0), cox_c(Fi, c2, dur[i], ev[i], 0)
        if np.isfinite(x) and np.isfinite(y):
            out.append(x - y)
    return np.array(out)


def main():
    a, b = load()
    edges = np.percentile(b.age_visit.values, [5, 24, 43, 62, 81, 95])

    # ---- the test set: MCI at baseline, identical for every arm
    mci_all = b[b.DX == "MCI"]
    keep, dur, ev = outcome(a, mci_all.PTID.values)
    T = mci_all[mci_all.PTID.isin(set(keep))].reset_index(drop=True)
    order = {s: i for i, s in enumerate(keep)}
    ii = np.array([order[s] for s in T.PTID.values])
    dur, ev = dur[ii], ev[ii]
    print(f"test set: {len(T)} MCI at baseline, {int(ev.sum())} converted "
          f"({100*ev.mean():.0f}%), median follow-up {np.median(dur):.1f} y")

    # ---- fold map for the MCI-trained arm, so it never sees a test subject
    fold = np.random.RandomState(0).permutation(len(T)) % FOLDS

    pools = {"CN": b[b.DX == "CN"], "AD": b[b.DX == "Dementia"],
             "POOL": b[b.DX.isin(["CN", "MCI", "Dementia"])]}
    GA = {"CN": "سالم", "MCI": "MCI (out-of-fold)", "AD": "آلزایمر",
          "POOL": "هر سه گروه"}

    gaps = {}
    for name in ["CN", "AD", "POOL"]:
        acc = np.zeros(len(T))
        for r in range(N_DRAW):
            src = pools[name]
            if name == "POOL":          # POOL must not contain test subjects
                src = src[~src.PTID.isin(set(T.PTID.values))]
            acc += fit_gap_model(matched_draw(src, edges, r))(T)
        gaps[name] = acc / N_DRAW

    # MCI-trained, out of fold
    acc = np.zeros(len(T))
    for r in range(N_DRAW):
        for k in range(FOLDS):
            tr_pool = T[fold != k]
            g = fit_gap_model(matched_draw(tr_pool, edges, r * 10 + k))
            acc[fold == k] += g(T[fold == k])
    gaps["MCI"] = acc / N_DRAW

    # ---- reference ceiling: the six normative deviations (E48's better number)
    cn = b[b.DX == "CN"]
    ag = cn.age_visit.values
    Xn = np.column_stack([ag, ag ** 2, cn.SEX.values, cn.ICV.values])
    agT = T.age_visit.values
    XnT = np.column_stack([agT, agT ** 2, T.SEX.values, T.ICV.values])
    Z = np.zeros((len(T), len(V)))
    sc = StandardScaler().fit(Xn)
    for j, v in enumerate(V):
        y = cn[v].values.astype(float)
        m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(Xn), y)
        s = (y - m.predict(sc.transform(Xn))).std(ddof=1)
        Z[:, j] = (T[v].values - m.predict(sc.transform(XnT))) / s
    pc = PCA(n_components=1, random_state=0).fit_transform(
        StandardScaler().fit_transform(Z))[:, 0]

    # ---- assemble
    F = pd.DataFrame(dict(age=agT))
    for k, g in gaps.items():
        F["gap_" + k] = g
    for j, v in enumerate(V):
        F["z_" + v] = Z[:, j]
    F["pc"] = pc

    print("\n" + "=" * 78)
    print("WHAT DOES EACH TRAINING GROUP PRODUCE?")
    print("=" * 78)
    print(f"  {'trained on':20s} {'mean gap':>10s} {'SD':>8s} "
          f"{'converters':>11s} {'stayers':>9s} {'diff':>8s}")
    for k in ["CN", "MCI", "AD", "POOL"]:
        g = gaps[k]
        print(f"  {GA[k]:20s} {g.mean():>+10.2f} {g.std():>8.2f} "
              f"{g[ev==1].mean():>+11.2f} {g[ev==0].mean():>+9.2f} "
              f"{g[ev==1].mean()-g[ev==0].mean():>+8.2f}")

    print("\n" + "=" * 78)
    print("PROGNOSIS: MCI -> AD CONVERSION, C-index")
    print("=" * 78)
    arms = [("age only", ["age"]),
            ("gap | trained سالم  (the standard)", ["gap_CN", "age"]),
            ("gap | trained MCI", ["gap_MCI", "age"]),
            ("gap | trained آلزایمر", ["gap_AD", "age"]),
            ("gap | trained هر سه", ["gap_POOL", "age"]),
            ("all four gaps at once", ["gap_CN", "gap_MCI", "gap_AD",
                                       "gap_POOL", "age"]),
            ("one PCA component (E48)", ["pc", "age"]),
            ("six deviations (ceiling)", [f"z_{v}" for v in V] + ["age"])]
    res = {}
    for nm, cols in arms:
        c = np.nanmean([cox_c(F, cols, dur, ev, s) for s in range(8)])
        res[nm] = cols
        print(f"  {nm:38s} C = {c:.3f}")

    print("\n" + "=" * 78)
    print("DOES TRAINING ON PATIENTS HELP?  (subject-level bootstrap vs standard)")
    print("=" * 78)
    std = "gap | trained سالم  (the standard)"
    for nm in ["gap | trained MCI", "gap | trained آلزایمر",
               "gap | trained هر سه", "all four gaps at once",
               "one PCA component (E48)"]:
        dd = boot(F, res[nm], res[std], dur, ev)
        lo, hi = np.percentile(dd, [2.5, 97.5])
        p = 2 * min((dd <= 0).mean(), (dd >= 0).mean())
        print(f"\n  {nm} minus the standard")
        print(f"    {dd.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  p={p:.3f}"
              f"   -> {'REAL' if p < 0.05 else 'not distinguishable'}")

    print("\n" + "=" * 78)
    print("ARE THESE GAPS EVEN DIFFERENT NUMBERS?")
    print("=" * 78)
    ks = ["CN", "MCI", "AD", "POOL"]
    print("       " + "".join(f"{GA[k][:8]:>10s}" for k in ks))
    for i in ks:
        print(f"  {GA[i][:6]:6s}" + "".join(
            f"{np.corrcoef(gaps[i], gaps[j])[0,1]:>10.3f}" for j in ks))


if __name__ == "__main__":
    main()
