#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DIAGNOSIS vs PROGRESSION: which baseline marker wins on which task?

Two questions, same subjects, same cross-validation:

  DIAGNOSIS  : is this person AD-like now?          (classification, AUC)
  PROGRESSION: how fast will they decline?          (regression, predict
               future ADAS-Cog13 slope from baseline scan)

Markers, all from the BASELINE scan, all out-of-fold:
  AGE        : chronological age only          (a fair floor)
  DIRECT     : P(dementia) from a CN-vs-Dementia classifier
  GAP        : brain-age gap (age model on CN only)
  DIRECT+GAP : both

The hypothesis worth testing: the direct model may win DIAGNOSIS but the
brain-age gap may hold its own (or win) on PROGRESSION, especially in the
cognitively-normal, where diagnosis has little to grab.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
DATA = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\data"
OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]


def build():
    d = pd.read_csv(f"{DATA}/ADNIMERGE_05Mar2026.csv", low_memory=False)
    d = d.sort_values(["RID", "Years_bl"])
    rows = []
    for rid, g in d.groupby("RID"):
        gg = g.sort_values("Years_bl")
        b = gg.iloc[0]
        if b[V + ["ICV", "AGE", "PTGENDER", "DX"]].isna().any():
            continue
        # future cognitive decline slope (ADAS13 over time), needs >=3 points
        gc = gg.dropna(subset=["ADAS13", "Years_bl"])
        slope = np.nan
        if gc.Years_bl.nunique() >= 3:
            slope = np.polyfit(gc.Years_bl, gc.ADAS13, 1)[0]
        # ever convert to dementia?
        conv = gg[gg["DX"] == "Dementia"]
        rows.append(dict(rid=rid, bdx=b["DX"], age=b["AGE"],
                         sex=1 if b["PTGENDER"] == "Male" else 0, icv=b["ICV"],
                         slope=slope, ever_dem=int(len(conv) > 0),
                         **{v: b[v] for v in V}))
    return pd.DataFrame(rows)


def oof_markers(df):
    """Out-of-fold direct score and brain-age gap for every subject."""
    n = len(df)
    vol = df[V].values
    age = df.age.values
    icv = df.icv.values
    sex = df.sex.values
    # target for the direct classifier: AD-like brain (baseline Dementia vs CN)
    is_dem = (df.bdx == "Dementia").astype(int).values
    is_cn = (df.bdx == "CN").astype(int).values

    direct = np.full(n, np.nan)
    gap = np.full(n, np.nan)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(vol):
        # direct model trained on CN + Dementia in the training fold
        lab_tr = tr[(is_dem[tr] == 1) | (is_cn[tr] == 1)]
        ytr = is_dem[lab_tr]
        feat = np.column_stack([vol, icv, sex, age])
        sc = StandardScaler().fit(feat[lab_tr])
        clf = LogisticRegression(max_iter=3000).fit(sc.transform(feat[lab_tr]), ytr)
        direct[te] = clf.predict_proba(sc.transform(feat[te]))[:, 1]

        # age model trained on CN in the training fold
        h = tr[is_cn[tr] == 1]
        Xh = np.column_stack([vol[h], icv[h], sex[h]])
        sca = StandardScaler().fit(Xh)
        m = LinearRegression().fit(sca.transform(Xh), age[h])
        bias = LinearRegression().fit(age[h].reshape(-1, 1),
                                      m.predict(sca.transform(Xh)) - age[h])
        Xte = np.column_stack([vol[te], icv[te], sex[te]])
        raw = m.predict(sca.transform(Xte)) - age[te]
        gap[te] = raw - bias.predict(age[te].reshape(-1, 1))
    df = df.copy()
    df["direct"] = direct
    df["gap"] = gap
    return df


def progression_cv(sub, feats, target="slope", n_rep=20):
    """Predict future decline slope from baseline markers; return CV Spearman."""
    s = sub.dropna(subset=[target] + feats)
    X = s[feats].values
    y = s[target].values
    scores = []
    for rep in range(n_rep):
        pred = np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=rep).split(X):
            sc = StandardScaler().fit(X[tr])
            m = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
            pred[te] = m.predict(sc.transform(X[te]))
        scores.append(spearmanr(pred, y).correlation)
    return np.array(scores), len(s)


def diagnosis_auc(sub, feats):
    s = sub.dropna(subset=feats)
    y = (s.bdx == "Dementia").astype(int).values
    if y.sum() < 5 or (1 - y).sum() < 5:
        return np.nan
    X = s[feats].values
    aucs = []
    for rep in range(20):
        pred = np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=rep).split(X):
            sc = StandardScaler().fit(X[tr])
            c = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
            pred[te] = c.predict_proba(sc.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y, pred))
    return np.mean(aucs)


def main():
    df = oof_markers(build())
    print(f"total n={len(df)}  (CN={sum(df.bdx=='CN')}, MCI={sum(df.bdx=='MCI')}, "
          f"Dem={sum(df.bdx=='Dementia')})")
    print(f"with decline slope: {df.slope.notna().sum()}")
    print(f"corr(direct, gap) = {np.corrcoef(df.direct, df.gap)[0,1]:.3f}\n")

    markers = {"age": ["age"], "direct": ["direct"], "gap": ["gap"],
               "direct+gap": ["direct", "gap"]}

    print("=" * 70)
    print("PROGRESSION: predicting future cognitive decline rate (CV Spearman)")
    print("=" * 70)
    prog = {}
    for grp, sub in [("whole cohort", df),
                     ("MCI only", df[df.bdx == "MCI"]),
                     ("CN only", df[df.bdx == "CN"])]:
        print(f"\n  --- {grp} ---")
        prog[grp] = {}
        for name, f in markers.items():
            sc, n = progression_cv(sub, f)
            prog[grp][name] = sc
            print(f"    {name:12s} rho = {sc.mean():+.3f}  "
                  f"[{np.percentile(sc,2.5):+.3f}, {np.percentile(sc,97.5):+.3f}]  (n={n})")

    print("\n" + "=" * 70)
    print("DIAGNOSIS: AD vs CN right now (AUC), same markers")
    print("=" * 70)
    diag_sub = df[df.bdx.isin(["CN", "Dementia"])]
    diag = {}
    for name, f in markers.items():
        diag[name] = diagnosis_auc(diag_sub, f)
        print(f"    {name:12s} AUC = {diag[name]:.3f}")

    np.savez(f"{OUT}/exp7_progression.npz",
             **{f"prog_{g}_{m}": v for g, mm in prog.items() for m, v in mm.items()},
             **{f"diag_{m}": np.array([v]) for m, v in diag.items()})
    df.to_csv(f"{OUT}/exp7_progression.csv", index=False)
    print(f"\nsaved -> {OUT}/exp7_progression.npz")


if __name__ == "__main__":
    main()
