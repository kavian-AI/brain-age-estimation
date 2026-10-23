#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
THE CONTROL EXPERIMENT NOBODY RUNS.

Regional brain age is computed FROM regional volumes plus chronological age.
So it cannot contain information those inputs lack - it can only re-express it.
The question is whether that re-expression helps a downstream classifier.

The fair baseline is therefore  [volumes + age],  NOT volumes alone.
Many papers compare a brain-age profile against volumes WITHOUT age, which
hands brain age a free advantage, because the gap smuggles age in.

Representations compared (identical subjects, folds, and classifier):
  R1  global brain age gap            (1 number)
  R2  regional brain age profile      (R numbers)
  R3  regional profile, SHAPE only    (unit-norm; global severity removed)
  B1  volumes only                    (no age)      <- the unfair baseline
  B2  volumes + chronological age                   <- the FAIR baseline
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
DATA = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\data"
OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"
rng = np.random.default_rng(7)


def regional_gaps(vol_tr, age_tr, icv_tr, sex_tr, vol_te, age_te, icv_te, sex_te):
    """Per-region brain age, trained on healthy only. Returns gap matrix."""
    n_te, R = vol_te.shape
    gaps = np.zeros((n_te, R))
    for r in range(R):
        Xtr = np.column_stack([vol_tr[:, r], icv_tr, sex_tr])
        Xte = np.column_stack([vol_te[:, r], icv_te, sex_te])
        sc = StandardScaler().fit(Xtr)
        m = LinearRegression().fit(sc.transform(Xtr), age_tr)
        raw_tr = m.predict(sc.transform(Xtr)) - age_tr
        bias = LinearRegression().fit(age_tr.reshape(-1, 1), raw_tr)
        raw_te = m.predict(sc.transform(Xte)) - age_te
        gaps[:, r] = raw_te - bias.predict(age_te.reshape(-1, 1))
    return gaps


def evaluate(vol, age, icv, sex, y, name, n_splits=5, n_rep=20):
    """y: 1 = patient, 0 = healthy control."""
    reps = {k: [] for k in ("R1", "R2", "R3", "B1", "B2")}
    for rep in range(n_rep):
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=rep)
        oof = {k: np.zeros(len(y)) for k in reps}
        for tr, te in skf.split(vol, y):
            h = tr[y[tr] == 0]                       # healthy reference
            G_te = regional_gaps(vol[h], age[h], icv[h], sex[h],
                                 vol[te], age[te], icv[te], sex[te])
            G_tr = regional_gaps(vol[h], age[h], icv[h], sex[h],
                                 vol[tr], age[tr], icv[tr], sex[tr])

            def fit_pred(Ftr, Fte):
                s = StandardScaler().fit(Ftr)
                clf = LogisticRegression(max_iter=4000, C=1.0)
                clf.fit(s.transform(Ftr), y[tr])
                return clf.predict_proba(s.transform(Fte))[:, 1]

            oof["R1"][te] = fit_pred(G_tr.mean(1, keepdims=True),
                                     G_te.mean(1, keepdims=True))
            oof["R2"][te] = fit_pred(G_tr, G_te)
            nrm_tr = G_tr / (np.linalg.norm(G_tr, axis=1, keepdims=True) + 1e-9)
            nrm_te = G_te / (np.linalg.norm(G_te, axis=1, keepdims=True) + 1e-9)
            oof["R3"][te] = fit_pred(nrm_tr, nrm_te)

            Vtr = vol[tr] / icv[tr][:, None]
            Vte = vol[te] / icv[te][:, None]
            oof["B1"][te] = fit_pred(np.column_stack([Vtr, sex[tr]]),
                                     np.column_stack([Vte, sex[te]]))
            oof["B2"][te] = fit_pred(np.column_stack([Vtr, sex[tr], age[tr]]),
                                     np.column_stack([Vte, sex[te], age[te]]))
        for k in reps:
            reps[k].append(roc_auc_score(y, oof[k]))

    print(f"\n{'='*70}\n{name}   (n={len(y)}, patients={int(y.sum())}, "
          f"regions={vol.shape[1]})\n{'='*70}")
    lbl = {"R1": "R1  global brain age gap (scalar)",
           "R2": "R2  regional brain age profile",
           "R3": "R3  profile SHAPE only (severity removed)",
           "B1": "B1  volumes only            [unfair baseline]",
           "B2": "B2  volumes + age           [FAIR baseline]"}
    res = {}
    for k in ("R1", "R2", "R3", "B1", "B2"):
        a = np.array(reps[k])
        res[k] = a
        print(f"  {lbl[k]:46s} AUC = {a.mean():.3f}  "
              f"[{np.percentile(a,2.5):.3f}, {np.percentile(a,97.5):.3f}]")

    d = res["R2"] - res["B2"]
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    print(f"\n  regional profile - fair baseline = {d.mean():+.3f}  p={p:.3f}")
    print("  ->", "brain-age transform ADDS value" if p < .05 and d.mean() > 0
          else "brain-age transform adds NOTHING over volumes+age")
    return res


# ------------------------------------------------------------------ ADNI
adni = pd.read_csv(f"{DATA}/ADNIMERGE_05Mar2026.csv", low_memory=False)
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
a = adni.dropna(subset=V + ["ICV", "AGE", "DX", "PTGENDER"]).sort_values("Years_bl")
a = a.groupby("RID").first().reset_index()          # baseline visit per subject
a = a[a.DX.isin(["CN", "Dementia"])]
res_adni = evaluate(a[V].values, a.AGE.values, a.ICV.values,
                    (a.PTGENDER == "Male").astype(int).values,
                    (a.DX == "Dementia").astype(int).values,
                    "ADNI  -  Alzheimer's dementia vs cognitively normal")

# ------------------------------------------------------------------ ABIDE
ab = pd.read_csv(f"{DATA}/abide_merged.csv")
roi = [c for c in ab.columns if c.startswith(("Left-", "Right-"))]
roi = [c for c in roi if ab[c].notna().all() and ab[c].std() > 0]
ab = ab.dropna(subset=roi + ["IntraCranialVol", "AGE_AT_SCAN", "DX_GROUP", "SEX"])
res_abide = evaluate(ab[roi].values, ab.AGE_AT_SCAN.values,
                     ab.IntraCranialVol.values, (ab.SEX == 1).astype(int).values,
                     (ab.DX_GROUP == 1).astype(int).values,
                     "ABIDE  -  autism vs control")

np.savez(f"{OUT}/exp2_regional.npz",
         **{f"adni_{k}": v for k, v in res_adni.items()},
         **{f"abide_{k}": v for k, v in res_abide.items()})
print(f"\nsaved -> {OUT}/exp2_regional.npz")
