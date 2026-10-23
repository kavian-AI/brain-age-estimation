#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
THE EXTERNAL COHORT: OASIS-2, SEGMENTED WITH A DIFFERENT PROGRAM.

Everything so far is ADNI, and ADNI's own FreeSurfer tables. OASIS-2 is a separate
study (Washington University), a different scanner, different recruitment, and
here a different segmentation program as well (FastSurfer, run on raw T1 on a
Kaggle GPU). 150 people aged 60-96, 373 sessions, Clinical Dementia Rating and
MMSE at every visit, plus a same-session REPEAT scan at every first visit.

The claims carried over from ADNI, each of which can fail here:

  1  DIAGNOSIS. The hippocampus alone matches or beats the brain-age gap; an
     average over the medial temporal regions beats it; an average over ALL
     regions does not; and a brain-age model built on more regions reads age
     better and dementia worse.
  2  MECHANISM. The age model's regional weights are unrelated to a disease
     model's (ADNI rank r = -0.03); the MTL gets a much smaller share of the age
     model's weight.
  3  PROGNOSIS. Baseline summaries against later MMSE decline (continuous, so
     all 150 people contribute, not only the 14 who converted).
  4  SEX. The age-only bias correction leaves a sex bias in controls; age+sex
     correction removes it.
  5  NEW HERE: RELIABILITY. The repeat scan gives test-retest for every summary.
     Averages of many regions should be more reliable than one; the question is
     where brain age sits.

n is small (72 controls) - every result carries a bootstrap interval and is
reported as replicated, not replicated, or underpowered.
"""

import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project root (local or Kaggle)
FS_DIR = os.path.join(BASE, "kaggle", "oasis_fs_results")
FOLDS, N_BOOT = 5, 1000

# FreeSurfer LUT labels in FastSurfer's aparc.DKTatlas+aseg
SUB = {"hippocampus": (17, 53), "amygdala": (18, 54), "thalamus": (10, 49),
       "caudate": (11, 50), "putamen": (12, 51), "pallidum": (13, 52),
       "accumbens": (26, 58), "ventraldc": (28, 60), "cerebellumcortex": (8, 47),
       "cerebellumwm": (7, 46), "cerebralwm": (2, 41)}
VENT = {"latvent": (4, 43), "inflatvent": (5, 44)}
MID = {"thirdventricle": (14,), "fourthventricle": (15,), "brainstem": (16,)}
DKT = {1002: "caudalanteriorcingulate", 1003: "caudalmiddlefrontal", 1005: "cuneus",
       1006: "entorhinal", 1007: "fusiform", 1008: "inferiorparietal", 1009: "inferiortemporal",
       1010: "isthmuscingulate", 1011: "lateraloccipital", 1012: "lateralorbitofrontal",
       1013: "lingual", 1014: "medialorbitofrontal", 1015: "middletemporal",
       1016: "parahippocampal", 1017: "paracentral", 1018: "parsopercularis",
       1019: "parsorbitalis", 1020: "parstriangularis", 1021: "pericalcarine",
       1022: "postcentral", 1023: "posteriorcingulate", 1024: "precentral", 1025: "precuneus",
       1026: "rostralanteriorcingulate", 1027: "rostralmiddlefrontal", 1028: "superiorfrontal",
       1029: "superiorparietal", 1030: "superiortemporal", 1031: "supramarginal",
       1034: "transversetemporal", 1035: "insula"}
MTL = {"entorhinal", "parahippocampal", "fusiform", "inferiortemporal", "middletemporal",
       "hippocampus", "amygdala"}


def load_fs():
    rows = []
    for f in glob.glob(os.path.join(FS_DIR, "*.jsonl")):
        for line in open(f):
            rows.append(json.loads(line))
    F = pd.DataFrame(rows).fillna(0.0)
    regions, signs, cls = {}, {}, {}
    for nm, (l, r) in SUB.items():
        for side, lab in (("L", l), ("R", r)):
            regions[f"{side}_{nm}"] = F.get(f"L{lab}", 0.0)
            signs[f"{side}_{nm}"] = -1
            cls[f"{side}_{nm}"] = "mtl" if nm in MTL else "subcortical"
    for nm, (l, r) in VENT.items():
        for side, lab in (("L", l), ("R", r)):
            regions[f"{side}_{nm}"] = F.get(f"L{lab}", 0.0)
            signs[f"{side}_{nm}"] = +1
            cls[f"{side}_{nm}"] = "ventricle"
    for nm, labs in MID.items():
        regions[nm] = sum(F.get(f"L{lab}", 0.0) for lab in labs)
        signs[nm] = +1 if "ventricle" in nm else -1
        cls[nm] = "ventricle" if "ventricle" in nm else "brainstem"
    for code, nm in DKT.items():
        for side, off in (("L", 0), ("R", 1000)):
            regions[f"{side}_{nm}"] = F.get(f"L{code+off}", 0.0)
            signs[f"{side}_{nm}"] = -1
            cls[f"{side}_{nm}"] = "mtl" if nm in MTL else "cortex"
    R = pd.DataFrame(regions)
    R["session"], R["mpr"] = F.session.values, F.mpr.values
    tissue = [c for c in R.columns if c not in ("session", "mpr") and signs[c] < 0]
    R["wholebrain"] = R[tissue].sum(1)
    return R, signs, cls


def six_like(R):
    """The ADNIMERGE six, rebuilt from FastSurfer labels."""
    return pd.DataFrame({
        "Hippocampus": R.L_hippocampus + R.R_hippocampus,
        "Entorhinal": R.L_entorhinal + R.R_entorhinal,
        "Fusiform": R.L_fusiform + R.R_fusiform,
        "MidTemp": R.L_middletemporal + R.R_middletemporal,
        "Ventricles": R.L_latvent + R.R_latvent + R.L_inflatvent + R.R_inflatvent,
        "WholeBrain": R.wholebrain})


def folds_of(ids, seed=0):
    uq = np.unique(ids)
    fm = dict(zip(uq, np.random.RandomState(seed).permutation(len(uq)) % FOLDS))
    return np.array([fm[s] for s in ids])


def normative(Y, age, sex, icv, ref, ids):
    f = folds_of(ids)
    X = np.column_stack([age, age ** 2, sex, icv])
    Z = np.zeros_like(Y, dtype=float)
    for k in range(FOLDS):
        tr, te = np.where((f != k) & ref)[0], np.where(f == k)[0]
        sc = StandardScaler().fit(X[tr])
        lr = LinearRegression().fit(sc.transform(X[tr]), Y[tr])
        sd = (Y[tr] - lr.predict(sc.transform(X[tr]))).std(0, ddof=1)
        sd[sd < 1e-9] = 1e-9
        Z[te] = (Y[te] - lr.predict(sc.transform(X[te]))) / sd
    return Z


def gap_of(Y, age, sex, icv, ref, ids, correct="age"):
    f = folds_of(ids)
    X = np.column_stack([Y, icv, sex])
    g = np.zeros(len(age))
    mae = []
    for k in range(FOLDS):
        tr, te = np.where((f != k) & ref)[0], np.where(f == k)[0]
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=np.logspace(-2, 5, 40)).fit(sc.transform(X[tr]), age[tr])
        p_tr, p_te = m.predict(sc.transform(X[tr])), m.predict(sc.transform(X[te]))
        mae.append(np.abs(p_te - age[te])[ref[te]])
        C_tr = np.column_stack([age[tr]] + ([sex[tr]] if correct == "age+sex" else []))
        C_te = np.column_stack([age[te]] + ([sex[te]] if correct == "age+sex" else []))
        bb = LinearRegression().fit(C_tr, p_tr - age[tr])
        g[te] = (p_te - age[te]) - bb.predict(C_te)
    return g, np.concatenate(mae).mean()


def auc_cv(x, y, age, ids):
    f = folds_of(ids)
    X = np.column_stack([x, age])
    o = np.zeros(len(y))
    for k in range(FOLDS):
        tr, te = f != k, f == k
        sc = StandardScaler().fit(X[tr])
        c = LogisticRegression(max_iter=4000).fit(sc.transform(X[tr]), y[tr])
        o[te] = c.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, o)


def boot_diff(a1, a2, y, age, ids, n=N_BOOT):
    rng = np.random.RandomState(0)
    out = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].sum() < 10 or (1 - y[i]).sum() < 10:
            continue
        try:
            # duplicated subjects keep their id, so folds_of keeps every copy of a
            # person in the same fold - no one is ever in train and test at once
            out.append(auc_cv(a1[i], y[i], age[i], ids[i]) - auc_cv(a2[i], y[i], age[i], ids[i]))
        except Exception:
            pass
    out = np.array(out)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return out.mean(), lo, hi, 2 * min((out <= 0).mean(), (out >= 0).mean())


def icc31(a, b):
    """Two-way mixed, consistency, single measure: repeat scan agreement."""
    Y = np.column_stack([a, b])
    n, k = Y.shape
    gm = Y.mean()
    msr = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1)
    mse = ((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse)


def main():
    R, signs, cls = load_fs()
    clin = pd.read_csv(os.path.join(BASE, "data", "oasis_longitudinal.csv"))
    clin = clin.rename(columns={"MRI ID": "session", "Subject ID": "subject"})
    print(f"segmented scans: {len(R)}  (mpr-1 {int((R.mpr=='mpr-1').sum())}, "
          f"mpr-2 {int((R.mpr=='mpr-2').sum())})")
    thal = R[R.mpr == "mpr-1"]
    print(f"sanity: L vs R thalamus identical in "
          f"{100*(thal.L_thalamus == thal.R_thalamus).mean():.0f}% of scans; "
          f"median hippocampus L/R {thal.L_hippocampus.median():.0f}/{thal.R_hippocampus.median():.0f} mm3")

    M = R[R.mpr == "mpr-1"].merge(clin, on="session")
    M["SEX"] = (M["M/F"] == "M").astype(int)
    M = M.sort_values(["subject", "Visit"]).reset_index(drop=True)
    B = M.drop_duplicates("subject").reset_index(drop=True)          # baseline
    # reference = CDR 0 at baseline AND never demented during follow-up
    ever = M.groupby("subject").CDR.max()
    B["ref"] = (B.CDR == 0) & (B.subject.map(ever) == 0)
    B["case"] = (B.CDR > 0).astype(int)
    print(f"baseline: n={len(B)}, CDR 0 {int((B.CDR==0).sum())} (stable reference "
          f"{int(B.ref.sum())}), CDR>0 {int(B.case.sum())}")

    names = [c for c in R.columns if c in signs]
    Yall = B[names].values.astype(float)
    S = np.array([signs[c] for c in names])
    C = np.array([cls[c] for c in names])
    age, sex, icv = B.Age.values.astype(float), B.SEX.values, B.eTIV.values.astype(float)
    ref, ids = B.ref.values, B.subject.values.astype(str)
    Z = normative(Yall, age, sex, icv, ref, ids)
    Y6 = six_like(B).values.astype(float)
    Z6 = normative(Y6, age, sex, icv, ref, ids)
    g6, mae6 = gap_of(Y6, age, sex, icv, ref, ids)
    gA, maeA = gap_of(Yall, age, sex, icv, ref, ids)
    hz = normative((B.L_hippocampus + B.R_hippocampus).values.reshape(-1, 1), age, sex, icv, ref, ids)[:, 0]
    mtl = C == "mtl"
    arms = {"gap (6-like)": g6, f"gap ({len(names)} regions)": gA,
            "uniform (6-like)": (Z6 * np.array([-1, -1, -1, -1, 1, -1])).mean(1),
            f"uniform ({len(names)} regions)": (Z * S).mean(1),
            f"uniform ({mtl.sum()} MTL)": (Z[:, mtl] * S[mtl]).mean(1),
            "hippocampus": -hz}

    print("\n" + "=" * 86)
    print("1  DIAGNOSIS: CDR>0 vs CDR 0 at baseline")
    print("=" * 86)
    y = B.case.values
    print(f"  age MAE in the reference group: 6-like {mae6:.2f} y, {len(names)}-region {maeA:.2f} y")
    for nm, x in arms.items():
        print(f"    {nm:26s} AUC = {auc_cv(x, y, age, ids):.3f}")
    for A, Bb in [("hippocampus", "gap (6-like)"), (f"uniform ({mtl.sum()} MTL)", f"gap ({len(names)} regions)"),
                  (f"uniform ({len(names)} regions)", f"gap ({len(names)} regions)"),
                  ("uniform (6-like)", "gap (6-like)"), (f"gap ({len(names)} regions)", "gap (6-like)")]:
        d, lo, hi, p = boot_diff(arms[A], arms[Bb], y, age, ids, n=500)
        print(f"      {A} minus {Bb}: {d:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.3f}")

    print("\n" + "=" * 86)
    print("2  MECHANISM: where does each model put its weight?")
    print("=" * 86)
    sc = StandardScaler().fit(np.column_stack([Yall, icv, sex])[ref])
    am = RidgeCV(alphas=np.logspace(-2, 5, 40)).fit(sc.transform(np.column_stack([Yall, icv, sex])[ref]), age[ref])
    wa = np.abs(am.coef_[:len(names)])
    dm = LogisticRegression(C=0.05, max_iter=5000).fit(
        StandardScaler().fit_transform(np.column_stack([Z, age])), y)
    wd = np.abs(dm.coef_[0][:len(names)])
    for c in ["mtl", "cortex", "subcortical", "ventricle", "brainstem"]:
        q = C == c
        print(f"    {c:12s} n={q.sum():3d}  age model {100*wa[q].sum()/wa.sum():5.1f}%   "
              f"disease model {100*wd[q].sum()/wd.sum():5.1f}%")
    print(f"    rank correlation of the two weightings: "
          f"{pd.Series(wa).rank().corr(pd.Series(wd).rank()):+.3f}   (ADNI: -0.031)")

    print("\n" + "=" * 86)
    print("3  PROGNOSIS: baseline summary vs later MMSE decline (>=2 visits)")
    print("=" * 86)
    sl = {}
    for s, g in M.groupby("subject"):
        g = g.dropna(subset=["MMSE"])
        if len(g) >= 2 and g["MR Delay"].max() > 365:
            sl[s] = np.polyfit(g["MR Delay"].values / 365.25, g.MMSE.values, 1)[0]
    B["mmse_slope"] = B.subject.map(sl)
    q = B.mmse_slope.notna().values
    print(f"  n={q.sum()} with follow-up")
    for nm, x in arms.items():
        Cv = np.column_stack([np.ones(q.sum()), age[q], sex[q]])
        rx = x[q] - Cv @ np.linalg.lstsq(Cv, x[q], rcond=None)[0]
        ry = -B.mmse_slope.values[q] - Cv @ np.linalg.lstsq(Cv, -B.mmse_slope.values[q], rcond=None)[0]
        r = np.corrcoef(rx, ry)[0, 1]
        rng = np.random.RandomState(0)
        bs = []
        for _ in range(N_BOOT):
            i = rng.randint(0, len(rx), len(rx))
            bs.append(np.corrcoef(rx[i], ry[i])[0, 1])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"    {nm:26s} partial r with MMSE decline {r:+.3f} [{lo:+.3f},{hi:+.3f}]")

    print("\n" + "=" * 86)
    print("4  SEX BIAS IN THE REFERENCE GROUP")
    print("=" * 86)
    for corr in ["age", "age+sex"]:
        g, _ = gap_of(Y6, age, sex, icv, ref, ids, correct=corr)
        gm_, gw_ = g[ref & (sex == 1)], g[ref & (sex == 0)]
        rng = np.random.RandomState(0)
        dd = [rng.choice(gm_, len(gm_)).mean() - rng.choice(gw_, len(gw_)).mean() for _ in range(N_BOOT)]
        print(f"    correction on {corr:8s} men {gm_.mean():+.2f}  women {gw_.mean():+.2f}  "
              f"difference {np.mean(dd):+.2f} [{np.percentile(dd,2.5):+.2f},{np.percentile(dd,97.5):+.2f}]"
              f"   (n men {len(gm_)}, women {len(gw_)})")

    print("\n" + "=" * 86)
    print("5  TEST-RETEST: same session, second MPRAGE, same pipeline")
    print("=" * 86)
    R2 = R[R.mpr == "mpr-2"].set_index("session")
    P = B[B.session.isin(R2.index)].copy()
    print(f"  pairs: {len(P)}")
    if len(P) >= 20:
        Y2 = R2.loc[P.session, names].values.astype(float)
        Y1 = P[names].values.astype(float)
        idx = B.index[B.session.isin(R2.index)].values
        # apply the SAME fitted normative model to both scans: fit on all baseline refs
        X = np.column_stack([age, age ** 2, sex, icv])
        s1 = StandardScaler().fit(X[ref])
        lr = LinearRegression().fit(s1.transform(X[ref]), Yall[ref])
        sd = (Yall[ref] - lr.predict(s1.transform(X[ref]))).std(0, ddof=1)
        Za = (Y1 - lr.predict(s1.transform(X[idx]))) / sd
        Zb = (Y2 - lr.predict(s1.transform(X[idx]))) / sd
        Y6b = six_like(R2.loc[P.session].reset_index()).values.astype(float)
        sa = StandardScaler().fit(np.column_stack([Y6, icv, sex])[ref])
        am6 = RidgeCV(alphas=np.logspace(-2, 5, 40)).fit(sa.transform(np.column_stack([Y6, icv, sex])[ref]), age[ref])
        ga = am6.predict(sa.transform(np.column_stack([Y6[idx], icv[idx], sex[idx]]))) - age[idx]
        gb = am6.predict(sa.transform(np.column_stack([Y6b, icv[idx], sex[idx]]))) - age[idx]
        summ = {"brain-age gap (6-like)": (ga, gb),
                "hippocampus": ((P.L_hippocampus + P.R_hippocampus).values,
                                (R2.loc[P.session, "L_hippocampus"] + R2.loc[P.session, "R_hippocampus"]).values),
                f"uniform ({len(names)} regions)": ((Za * S).mean(1), (Zb * S).mean(1)),
                f"uniform ({mtl.sum()} MTL)": ((Za[:, mtl] * S[mtl]).mean(1), (Zb[:, mtl] * S[mtl]).mean(1))}
        for nm, (u, v) in summ.items():
            rng = np.random.RandomState(0)
            bs = [icc31(u[i], v[i]) for i in [rng.randint(0, len(u), len(u)) for _ in range(N_BOOT)]]
            print(f"    {nm:26s} ICC {icc31(u, v):.3f} [{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]")
        per = np.array([icc31(Y1[:, j], Y2[:, j]) for j in range(len(names))])
        print(f"    single regions: median ICC {np.median(per):.3f}, range {per.min():.3f}-{per.max():.3f}")
        worst = np.argsort(per)[:5]
        print("    least reliable: " + ", ".join(f"{names[j]} {per[j]:.2f}" for j in worst))


if __name__ == "__main__":
    main()
