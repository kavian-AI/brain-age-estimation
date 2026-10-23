#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DOES THE RAW IMAGE CONTAIN MORE THAN THE VOLUME NUMBERS?

This closes the main limitation of the report: every earlier experiment used
pre-computed volumetric features, never the image itself.

The design keeps the MODEL CLASS FIXED and varies only the INPUT, so any
difference is attributable to the representation and not to model capacity:

  TAB   nWBV, eTIV, sex          <- three numbers, what the report used
  IMG   13 axial slices -> PCA   <- the Franke-2010 recipe
  BOTH  concatenation

Three questions, in order:

  Q1  age prediction on healthy subjects (n=266, ages 18-96)
      Reported on the full range AND on 60+ only. The full range flatters any
      model enormously - telling a 20-year-old from an 80-year-old is not the
      task we care about - so the restricted number is the honest one.

  Q2  disease classification through the brain-age gap, age-matched,
      against the fair baselines (features + chronological age, no gap).

  Q3  test-retest. Every subject has 3-4 acquisitions from the SAME session.
      Same brain, same day, minutes apart, so any difference in the gap is
      measurement noise. This puts a number on the gap's noise floor.

PCA and scaling are fitted inside each training fold. Splits are by SUBJECT.
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
OUT = os.path.join(BASE, "pilot")
N_PCA = 80
N_REP = 10


# ----------------------------------------------------------------- data

def load():
    c = np.load(os.path.join(OUT, "oasis_img_cache.npz"), allow_pickle=True)
    acq = pd.DataFrame(dict(subject=c["subject"], mr=c["mr"], mpr=c["mpr"]))
    X = c["X"]

    idx = pd.read_csv(os.path.join(OUT, "images_index.csv"))
    meta = idx.groupby("subject").first().reset_index()[
        ["subject", "Age", "M/F", "CDR", "nWBV", "eTIV"]]

    # one row per subject: average the repeat acquisitions
    order = acq.groupby("subject").indices
    subs = sorted(order)
    Xs = np.stack([X[order[s]].mean(0) for s in subs])
    m = pd.DataFrame(dict(subject=subs)).merge(meta, on="subject", how="left")
    return X, acq, Xs, m


def features(m):
    return np.column_stack([m.nWBV.values, m.eTIV.values,
                            (m["M/F"] == "M").astype(int).values])


def pca_fold(Xtr, Xte, k):
    sc = StandardScaler().fit(Xtr)
    p = PCA(n_components=min(k, Xtr.shape[0] - 2), svd_solver="randomized",
            random_state=0).fit(sc.transform(Xtr))
    return p.transform(sc.transform(Xtr)), p.transform(sc.transform(Xte))


# ----------------------------------------------------- Q1: age on healthy

def age_cv(Ximg, tab, age, groups, use_img, use_tab, n_rep=N_REP):
    """Out-of-fold age predictions for healthy subjects."""
    preds = np.zeros((n_rep, len(age)))
    for rep in range(n_rep):
        cv = GroupKFold(5) if n_rep == 1 else _shuffled_group_kfold(groups, rep)
        for tr, te in cv:
            parts_tr, parts_te = [], []
            if use_tab:
                s = StandardScaler().fit(tab[tr])
                parts_tr.append(s.transform(tab[tr]))
                parts_te.append(s.transform(tab[te]))
            if use_img:
                a, b = pca_fold(Ximg[tr], Ximg[te], N_PCA)
                parts_tr.append(a)
                parts_te.append(b)
            Xtr = np.hstack(parts_tr)
            Xte = np.hstack(parts_te)
            mdl = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(Xtr, age[tr])
            preds[rep, te] = mdl.predict(Xte)
    return preds


def _shuffled_group_kfold(groups, seed):
    """GroupKFold has no shuffle; permute the group labels instead."""
    rng = np.random.RandomState(seed)
    uq = np.unique(groups)
    fold = dict(zip(uq, rng.permutation(len(uq)) % 5))
    f = np.array([fold[g] for g in groups])
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(5)]


# ------------------------------------------------- Q2: disease through gap

def gap_and_disease(Ximg, tab, age, y, groups, use_img, use_tab, n_rep=N_REP):
    aucs, gaps = [], np.zeros((n_rep, len(y)))
    for rep in range(n_rep):
        oof = np.zeros(len(y))
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=rep)
        for tr, te in cv.split(np.zeros(len(y)), y, groups):
            h = tr[y[tr] == 0]
            if len(h) < 15:
                continue
            parts = []
            for mask in (h, tr, te):
                pass
            # build the design matrix once for the three index sets
            if use_tab:
                s = StandardScaler().fit(tab[h])
                Th, Ttr, Tte = (s.transform(tab[i]) for i in (h, tr, te))
            if use_img:
                sc = StandardScaler().fit(Ximg[h])
                p = PCA(n_components=min(N_PCA, len(h) - 2),
                        svd_solver="randomized", random_state=0
                        ).fit(sc.transform(Ximg[h]))
                Ih, Itr, Ite = (p.transform(sc.transform(Ximg[i]))
                                for i in (h, tr, te))
            if use_tab and use_img:
                Dh, Dtr, Dte = (np.hstack([a, b]) for a, b in
                                ((Th, Ih), (Ttr, Itr), (Tte, Ite)))
            elif use_tab:
                Dh, Dtr, Dte = Th, Ttr, Tte
            else:
                Dh, Dtr, Dte = Ih, Itr, Ite

            # age model on healthy training subjects, then bias correction
            mdl = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(Dh, age[h])
            resid = mdl.predict(Dh) - age[h]
            bias = LinearRegression().fit(age[h].reshape(-1, 1), resid)
            g = lambda D, a: (mdl.predict(D) - a) - bias.predict(a.reshape(-1, 1))
            gtr, gte = g(Dtr, age[tr]), g(Dte, age[te])
            gaps[rep, te] = gte

            s2 = StandardScaler().fit(gtr.reshape(-1, 1))
            clf = LogisticRegression(max_iter=3000).fit(
                s2.transform(gtr.reshape(-1, 1)), y[tr])
            oof[te] = clf.predict_proba(s2.transform(gte.reshape(-1, 1)))[:, 1]
        aucs.append(roc_auc_score(y, oof))
    return np.array(aucs), gaps.mean(0)


def direct(Ximg, tab, extra, y, groups, use_img, use_tab, n_rep=N_REP):
    """Straight to disease, no brain-age transform: the fair baseline."""
    aucs = []
    for rep in range(n_rep):
        oof = np.zeros(len(y))
        for tr, te in StratifiedGroupKFold(
                5, shuffle=True, random_state=rep).split(np.zeros(len(y)), y, groups):
            parts_tr, parts_te = [extra[tr]], [extra[te]]
            if use_tab:
                s = StandardScaler().fit(tab[tr])
                parts_tr.append(s.transform(tab[tr]))
                parts_te.append(s.transform(tab[te]))
            if use_img:
                a, b = pca_fold(Ximg[tr], Ximg[te], N_PCA)
                parts_tr.append(a)
                parts_te.append(b)
            Xtr, Xte = np.hstack(parts_tr), np.hstack(parts_te)
            s3 = StandardScaler().fit(Xtr)
            c = LogisticRegression(max_iter=3000, C=0.1).fit(s3.transform(Xtr), y[tr])
            oof[te] = c.predict_proba(s3.transform(Xte))[:, 1]
        aucs.append(roc_auc_score(y, oof))
    return np.array(aucs)


# --------------------------------------------------------------- report

def pct(a):
    return f"{a.mean():.3f} [{np.percentile(a, 2.5):.3f}, {np.percentile(a, 97.5):.3f}]"


def main():
    X, acq, Xs, m = load()
    print(f"subjects: {len(m)}, acquisitions: {len(acq)}")

    healthy = (m.CDR.fillna(0) == 0).values
    ok = np.isfinite(m.Age.values) & np.isfinite(features(m)).all(1)
    print(f"healthy: {healthy.sum()}  impaired: {(~healthy).sum()}  usable: {ok.sum()}")

    tab, age = features(m), m.Age.values.astype(float)

    # ============================================================= Q1
    print("\n" + "=" * 72)
    print("Q1  PREDICTING AGE IN HEALTHY SUBJECTS")
    print("=" * 72)
    hs = np.where(healthy & ok)[0]
    print(f"n = {len(hs)}, age {age[hs].min():.0f}-{age[hs].max():.0f}, "
          f"mean {age[hs].mean():.1f}")
    old = age[hs] >= 60
    print(f"of which 60+: {old.sum()}\n")

    q1 = {}
    for nm, (ui, ut) in [("TAB   nWBV+eTIV+sex", (0, 1)),
                         ("IMG   image PCA", (1, 0)),
                         ("BOTH", (1, 1))]:
        P = age_cv(Xs[hs], tab[hs], age[hs], m.subject.values[hs], ui, ut)
        err = np.abs(P - age[hs])
        mae_all = err.mean(1)
        mae_old = err[:, old].mean(1)
        r = np.array([np.corrcoef(p, age[hs])[0, 1] for p in P])
        q1[nm] = dict(mae_all=mae_all, mae_old=mae_old, r=r, pred=P.mean(0))
        print(f"  {nm:22s} MAE(18-96) {mae_all.mean():5.2f} y   "
              f"MAE(60+) {mae_old.mean():5.2f} y   r = {r.mean():.3f}")

    d = q1["TAB   nWBV+eTIV+sex"]["mae_old"] - q1["IMG   image PCA"]["mae_old"]
    print(f"\n  image vs tabular, MAE on 60+: {d.mean():+.2f} y "
          f"(positive = image better)  p={2*min((d<=0).mean(),(d>=0).mean()):.3f}")

    # ============================================================= Q2
    print("\n" + "=" * 72)
    print("Q2  DISEASE CLASSIFICATION (CDR>0 vs CDR=0), AGE-MATCHED")
    print("=" * 72)
    y_all = (~healthy).astype(int)
    lo = age[(~healthy) & ok].min()
    sel = np.where(ok & (age >= lo))[0]
    y, g = y_all[sel], m.subject.values[sel]
    print(f"age >= {lo:.0f}: n = {len(sel)}  "
          f"(healthy {int((y == 0).sum())}, impaired {int(y.sum())})\n")

    q2 = {}
    for nm, (ui, ut) in [("TAB   nWBV+eTIV+sex", (0, 1)),
                         ("IMG   image PCA", (1, 0)),
                         ("BOTH", (1, 1))]:
        a, gp = gap_and_disease(Xs[sel], tab[sel], age[sel], y, g, ui, ut)
        q2["gap " + nm] = a
        q2["_gapvals_" + nm] = gp
        print(f"  gap from {nm:22s} AUC = {pct(a)}")

    print("\n  fair baselines - straight to disease, no brain-age transform:")
    for nm, (ui, ut) in [("TAB + age", (0, 1)),
                         ("IMG + age", (1, 0)),
                         ("BOTH + age", (1, 1))]:
        a = direct(Xs[sel], tab[sel], age[sel].reshape(-1, 1), y, g, ui, ut)
        q2["dir " + nm] = a
        print(f"  {nm:27s} AUC = {pct(a)}")

    for lbl, k1, k2 in [
            ("image vs tabular gap",
             "gap IMG   image PCA", "gap TAB   nWBV+eTIV+sex"),
            ("gap vs direct model (image)",
             "gap IMG   image PCA", "dir IMG + age"),
            ("gap vs direct model (tabular)",
             "gap TAB   nWBV+eTIV+sex", "dir TAB + age")]:
        dd = q2[k1] - q2[k2]
        print(f"\n  {lbl:32s} {dd.mean():+.3f}  "
              f"p={2*min((dd<=0).mean(),(dd>=0).mean()):.3f}")

    # ============================================================= Q3
    print("\n" + "=" * 72)
    print("Q3  TEST-RETEST: HOW MUCH OF THE GAP IS NOISE?")
    print("=" * 72)
    rel = retest(X, acq, m, tab, age, healthy & ok)

    np.savez(os.path.join(OUT, "exp22_image.npz"),
             **{k: v for k, v in q2.items() if not k.startswith("_")},
             **{f"q1_{k.split()[0]}_{s}": v[s]
                for k, v in q1.items() for s in ("mae_all", "mae_old", "r")},
             retest=rel)
    m.assign(sel=np.isin(np.arange(len(m)), sel)).to_csv(
        os.path.join(OUT, "exp22_subjects.csv"), index=False)
    print(f"\nsaved -> {OUT}\\exp22_image.npz")


def retest(X, acq, m, tab, age, healthy_ok):
    """Two acquisitions, same brain, same session: the gap's noise floor."""
    info = acq.merge(m[["subject", "Age", "CDR"]], on="subject", how="left")
    hs = np.where(healthy_ok)[0]
    hsub = set(m.subject.values[hs])

    # train the age model on ONE acquisition per healthy subject, out of fold,
    # then score every acquisition of the held-out subjects
    groups = m.subject.values[hs]
    preds = np.full(len(acq), np.nan)
    for tr, te in _shuffled_group_kfold(groups, 0):
        tr_sub = set(groups[tr])
        te_sub = set(groups[te])
        itr = np.where(info.subject.isin(tr_sub))[0]
        ite = np.where(info.subject.isin(te_sub))[0]
        sc = StandardScaler().fit(X[itr])
        p = PCA(n_components=min(N_PCA, len(itr) - 2), svd_solver="randomized",
                random_state=0).fit(sc.transform(X[itr]))
        mdl = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(
            p.transform(sc.transform(X[itr])), info.Age.values[itr])
        preds[ite] = mdl.predict(p.transform(sc.transform(X[ite])))

    info["pred"] = preds
    d = info.dropna(subset=["pred"])
    d = d[d.subject.isin(hsub)].copy()

    # Reliability of the GAP, not of predicted age. Predicted age is dominated
    # by true age, so its reliability is high no matter how noisy the model is
    # and means nothing. The gap is what the whole literature interprets, and it
    # is a small residual, so this is where noise actually bites.
    d["gap"] = d.pred - d.Age
    b = LinearRegression().fit(d.Age.values.reshape(-1, 1), d.gap.values)
    d["gap"] -= b.predict(d.Age.values.reshape(-1, 1))

    per = d.groupby("subject").gap.agg(["mean", "std", "count"])
    per = per[per["count"] >= 2]
    within = per["std"].mean()
    between = per["mean"].std()
    icc = (between**2 - within**2 / per["count"].mean()) / between**2
    print(f"  healthy subjects with >=2 acquisitions: {len(per)} "
          f"({per['count'].mean():.1f} acquisitions each, same session)")
    print(f"  within-subject SD of the gap (same brain, minutes apart): "
          f"{within:.2f} y")
    print(f"  between-subject SD of the gap:                            "
          f"{between:.2f} y")
    print(f"  reliability of the gap:                                   {icc:.3f}")
    print(f"\n  -> rescanning the same person minutes later moves their gap by "
          f"{within:.1f} y,")
    print(f"     against a between-person spread of {between:.1f} y. "
          f"{100*(1-icc):.0f}% of a single")
    print("     measurement's variance is noise rather than biology.")
    return np.array([within, between, icc])


if __name__ == "__main__":
    main()
