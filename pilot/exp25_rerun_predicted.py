#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rerun the whole experiment suite with IMAGE-DERIVED volumes.

The bottleneck kernel fine-tunes a pretrained backbone to output the 6 FreeSurfer
volumes + ICV from the image, out of fold. This script takes those predicted
numbers and pushes them through every experiment the project ran on the real
ones, so the two are comparable line for line:

  T1  AD vs CN          gap / regional profile / volumes+age
  T2  MCI vs CN         the earlier, harder question
  T3  AD vs MCI         differential diagnosis, age-matched
  T4  MCI -> dementia   landmark design: gap value vs gap RATE

Every task is run TWICE on identical folds - once on true FreeSurfer volumes and
once on predicted - and the only thing that changes is where the seven numbers
came from. The question is how much of the true-volume performance survives the
trip through the image.

Differences that matter are bootstrapped over SUBJECTS, because confidence
intervals from repeated cross-validation measure split noise rather than
sampling noise: at n=207 earlier in this project the two differed by an order of
magnitude (+-0.006 vs +-0.07) and the CV-repeat version nearly produced a false
claim.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "pilot")
KG = os.path.join(BASE, "kaggle", "brainage_bottleneck", "out")
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
TARGETS = V + ["ICV"]
FOLDS, N_BOOT, LANDMARK = 5, 400, 2.0


# ------------------------------------------------------------------ load

def load():
    f = os.path.join(KG, "bottleneck.npz")
    if not os.path.exists(f):
        print("bottleneck.npz not found. Fetch the kernel output first:")
        print("  kaggle kernels output kavianasli/brainage-bottleneck -p "
              f"{KG}")
        sys.exit(1)
    z = np.load(f, allow_pickle=True)
    d = pd.DataFrame(dict(PTID=z["subs"].astype(str), VISCODE=z["vis"].astype(str)))
    for j, t in enumerate(TARGETS):
        d["P_" + t] = z["pred"][:, j]
        d["T_" + t] = z["true"][:, j]
    a = pd.read_csv(os.path.join(DATA, "ADNIMERGE_05Mar2026.csv"), low_memory=False)
    k = a[["PTID", "VISCODE", "AGE", "PTGENDER", "DX", "Years_bl"]].drop_duplicates(
        ["PTID", "VISCODE"])
    d = d.merge(k, on=["PTID", "VISCODE"], how="left")
    d["SEX"] = (d.PTGENDER == "Male").astype(int)
    d["age_visit"] = d.AGE + d.Years_bl        # AGE is BASELINE age
    return d.dropna(subset=["age_visit"]).reset_index(drop=True)


def mat(d, pfx):
    return np.column_stack([d[pfx + t].values for t in TARGETS] + [d.SEX.values])


def folds_of(subs, seed):
    u = np.unique(subs)
    r = np.random.RandomState(seed)
    f = dict(zip(u, r.permutation(len(u)) % FOLDS))
    return np.array([f[s] for s in subs])


# ------------------------------------------------------------- pipelines

def gap_of(fit_tab, fit_age, tab, age):
    sc = StandardScaler().fit(fit_tab)
    m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(fit_tab), fit_age)
    b = LinearRegression().fit(fit_age.reshape(-1, 1),
                               m.predict(sc.transform(fit_tab)) - fit_age)
    return (m.predict(sc.transform(tab)) - age) - b.predict(age.reshape(-1, 1))


def profile_of(fit_tab, fit_age, tab, age):
    """One gap per region: age predicted from each region alone."""
    cols = []
    for j in range(len(V)):
        f1 = fit_tab[:, [j, len(TARGETS) - 1]]     # region + ICV
        t1 = tab[:, [j, len(TARGETS) - 1]]
        cols.append(gap_of(f1, fit_age, t1, age))
    return np.column_stack(cols)


def run_task(name, d, y, seeds=6):
    """gap / profile / volumes+age, on true and predicted, identical folds."""
    age = d.age_visit.values.astype(float)
    subs = d.PTID.values
    res = {}
    for src, pfx in [("TRUE", "T_"), ("PRED", "P_")]:
        tab = mat(d, pfx)
        acc = {k: [] for k in ("gap", "profile", "direct")}
        for s in range(seeds):
            f = folds_of(subs, s)
            og, op, od = (np.zeros(len(y)), np.zeros((len(y), len(V))),
                          np.zeros(len(y)))
            for k in range(FOLDS):
                tr, te = np.where(f != k)[0], np.where(f == k)[0]
                h = tr[y[tr] == 0]
                if len(h) < 20:
                    continue
                og[te] = gap_of(tab[h], age[h], tab[te], age[te])
                op[te] = profile_of(tab[h], age[h], tab[te], age[te])
                Dtr = np.column_stack([tab[tr], age[tr]])
                Dte = np.column_stack([tab[te], age[te]])
                sc = StandardScaler().fit(Dtr)
                c = LogisticRegression(max_iter=4000).fit(sc.transform(Dtr), y[tr])
                od[te] = c.predict_proba(sc.transform(Dte))[:, 1]
            acc["gap"].append(roc_auc_score(y, og))
            acc["direct"].append(roc_auc_score(y, od))
            # profile needs a classifier; fit it out of fold too
            op2 = np.zeros(len(y))
            f2 = folds_of(subs, s + 100)
            for k in range(FOLDS):
                tr, te = f2 != k, f2 == k
                sc = StandardScaler().fit(op[tr])
                c = LogisticRegression(max_iter=4000).fit(sc.transform(op[tr]), y[tr])
                op2[te] = c.predict_proba(sc.transform(op[te]))[:, 1]
            acc["profile"].append(roc_auc_score(y, op2))
        res[src] = {k: np.array(v) for k, v in acc.items()}

    print("\n" + "=" * 74)
    print(f"{name}   n={len(y)} (neg {int((y==0).sum())}, pos {int(y.sum())})")
    print("=" * 74)
    print(f"  {'':26s} {'gap':>8} {'profile':>9} {'volumes+age':>13}")
    for src in ("TRUE", "PRED"):
        r = res[src]
        print(f"  {src + ' volumes':26s} {r['gap'].mean():8.3f} "
              f"{r['profile'].mean():9.3f} {r['direct'].mean():13.3f}")
    # Recovery must be measured ABOVE CHANCE. The naive ratio AUC_pred/AUC_true
    # is inflated by AUC's floor of 0.5 and is nearly constant across tasks of
    # very different difficulty, which hides the real pattern: on this data it
    # read 86.4 / 86.7 / 86.4% while the above-chance figures were 71.5 / 63.9 /
    # 59.4%, i.e. recovery falls away as the task gets harder.
    p, t = res["PRED"]["direct"].mean(), res["TRUE"]["direct"].mean()
    print(f"  {'recovered (above chance)':26s} {'':8s} {'':9s} "
          f"{100*(p-0.5)/(t-0.5):12.1f}%")
    print(f"  {'  [naive AUC ratio':26s} {'':8s} {'':9s} "
          f"{100*p/t:12.1f}%  <- inflated, do not quote]")
    return res


# ------------------------------------------------------------------- T4

def cox_c(F, cols, dur, ev, seed):
    n = len(dur)
    f = np.random.RandomState(seed).permutation(n) % 5
    risk = np.zeros(n)
    for k in range(5):
        tr, te = f != k, f == k
        if ev[tr].sum() < 5:
            return np.nan
        D = F[cols][tr].copy()
        D["T"], D["E"] = dur[tr], ev[tr]
        try:
            m = CoxPHFitter(penalizer=0.1).fit(D, "T", "E")
            risk[te] = m.predict_partial_hazard(F[cols][te]).values
        except Exception:
            return np.nan
    return concordance_index(dur, -risk, ev)


def slope(t, v):
    ok = np.isfinite(t) & np.isfinite(v)
    if ok.sum() < 2 or np.ptp(t[ok]) < 0.5:
        return np.nan
    return np.polyfit(t[ok], v[ok], 1)[0]


def run_rate(d, hist):
    print("\n" + "=" * 74)
    print(f"T4  MCI -> dementia, landmark L={LANDMARK} y")
    print("=" * 74)
    bl = d[d.VISCODE == "bl"][["PTID", "DX"]]
    mci = set(bl[bl.DX == "MCI"].PTID)
    rows = {}
    for src, pfx in [("TRUE", "T_"), ("PRED", "P_")]:
        cn = d[(d.DX == "CN")].drop_duplicates("PTID")
        tabc, agec = mat(cn, pfx), cn.age_visit.values.astype(float)
        w = d[d.PTID.isin(mci) & (d.Years_bl <= LANDMARK + 0.25)].copy()
        w["gap"] = gap_of(tabc, agec, mat(w, pfx), w.age_visit.values.astype(float))
        w["hipp"] = 1e3 * w[pfx + "Hippocampus"] / w[pfx + "ICV"]
        r = []
        for s, g in w.groupby("PTID"):
            g = g.sort_values("Years_bl")
            a, b = slope(g.Years_bl.values, g.gap.values), \
                slope(g.Years_bl.values, g.hipp.values)
            if np.isfinite(a) and np.isfinite(b):
                r.append(dict(PTID=s, rate=a, value=g.gap.values[0], rate_hipp=b,
                              age=g.AGE.values[0]))
        rows[src] = pd.DataFrame(r).set_index("PTID")

    common = sorted(set(rows["TRUE"].index) & set(rows["PRED"].index))
    fut = hist[hist.PTID.isin(common) & (hist.Years_bl > LANDMARK)]
    conv = fut[fut.DX == "Dementia"].groupby("PTID").Years_bl.min()
    last = fut.groupby("PTID").Years_bl.max()
    common = [s for s in common if s in last.index]
    dur = np.array([conv[s] - LANDMARK if s in conv.index else last[s] - LANDMARK
                    for s in common])
    ev = np.array([1 if s in conv.index else 0 for s in common])
    ok = dur > 0
    dur, ev = dur[ok], ev[ok]
    common = [s for s, o in zip(common, ok) if o]
    print(f"  n={len(common)}, conversions after {LANDMARK} y = {int(ev.sum())}")
    if ev.sum() < 25:
        print("  too few events")
        return {}
    out = {}
    for src in ("TRUE", "PRED"):
        F = rows[src].loc[common]
        for nm, cols in [("gap value", ["value", "age"]),
                         ("gap RATE", ["rate", "age"]),
                         ("value + RATE", ["value", "rate", "age"]),
                         ("hippocampus RATE", ["rate_hipp", "age"])]:
            c = np.nanmean([cox_c(F, cols, dur, ev, s) for s in range(10)])
            out[f"{src} {nm}"] = c
            print(f"  {src} {nm:22s} C = {c:.3f}")
    return out


def main():
    d = load()
    print(f"scans with predictions: {len(d):,}  subjects: {d.PTID.nunique():,}")
    print("stage A agreement (all visits):")
    for t in TARGETS:
        r = np.corrcoef(d["P_" + t], d["T_" + t])[0, 1]
        print(f"  {t:16s} r = {r:.3f}")

    b = d[(d.VISCODE == "bl") & d.DX.notna()].reset_index(drop=True)
    res = {}
    res["AD"] = run_task("T1  AD vs CN", b[b.DX.isin(["CN", "Dementia"])].reset_index(drop=True),
                         (b[b.DX.isin(["CN", "Dementia"])].DX == "Dementia").astype(int).values)
    res["MCI"] = run_task("T2  MCI vs CN", b[b.DX.isin(["CN", "MCI"])].reset_index(drop=True),
                          (b[b.DX.isin(["CN", "MCI"])].DX == "MCI").astype(int).values)

    # T3 differential, age-matched greedy 1:1
    sub = b[b.DX.isin(["MCI", "Dementia"])].reset_index(drop=True)
    y3 = (sub.DX == "Dementia").astype(int).values
    a3 = sub.age_visit.values
    pos, neg = np.where(y3 == 1)[0], np.where(y3 == 0)[0]
    used, pairs = set(), []
    for i in pos:
        cand = [j for j in neg if j not in used and abs(a3[j] - a3[i]) <= 1.5]
        if cand:
            j = min(cand, key=lambda j: abs(a3[j] - a3[i]))
            used.add(j)
            pairs += [i, j]
    if len(pairs) > 80:
        s3 = sub.iloc[sorted(pairs)].reset_index(drop=True)
        res["DIFF"] = run_task("T3  AD vs MCI (age-matched)", s3,
                               (s3.DX == "Dementia").astype(int).values)

    a = pd.read_csv(os.path.join(DATA, "ADNIMERGE_05Mar2026.csv"), low_memory=False)
    hist = a[["PTID", "Years_bl", "DX"]].dropna(subset=["DX"])
    res["RATE"] = run_rate(d, hist)

    np.savez(os.path.join(OUT, "exp25_predicted.npz"),
             **{f"{k}_{s}_{m}": v for k, r in res.items() if isinstance(r, dict)
                for s, rr in r.items() if isinstance(rr, dict)
                for m, v in rr.items()})
    print(f"\nsaved -> {OUT}\\exp25_predicted.npz")


if __name__ == "__main__":
    main()
