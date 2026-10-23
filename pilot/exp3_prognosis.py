#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extension: does the same conclusion hold for PROGNOSIS (predicting the future),
not just diagnosis (describing the present)?

Task: among people who are MCI at baseline, who converts to dementia?
This is the clinically valuable question and the hardest one.
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
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]


def regional_gaps(vtr, atr, itr, str_, vte, ate, ite, ste):
    R = vte.shape[1]
    g = np.zeros((vte.shape[0], R))
    for r in range(R):
        Xtr = np.column_stack([vtr[:, r], itr, str_])
        Xte = np.column_stack([vte[:, r], ite, ste])
        sc = StandardScaler().fit(Xtr)
        m = LinearRegression().fit(sc.transform(Xtr), atr)
        b = LinearRegression().fit(atr.reshape(-1, 1),
                                   m.predict(sc.transform(Xtr)) - atr)
        g[:, r] = (m.predict(sc.transform(Xte)) - ate) - b.predict(ate.reshape(-1, 1))
    return g


def main():
    d = pd.read_csv(f"{DATA}/ADNIMERGE_05Mar2026.csv", low_memory=False)
    d = d.dropna(subset=["DX"]).sort_values(["RID", "Years_bl"])

    # label MCI subjects by whether they ever convert to dementia
    lab = {}
    for rid, g in d.groupby("RID"):
        dx = g.DX.tolist()
        if dx[0] == "MCI":
            lab[rid] = int("Dementia" in dx)

    base = d.dropna(subset=V + ["ICV", "AGE", "PTGENDER"]).groupby("RID").first()
    base = base[base.index.isin(lab)]
    y = np.array([lab[r] for r in base.index])
    vol, age = base[V].values, base.AGE.values
    icv = base.ICV.values
    sex = (base.PTGENDER == "Male").astype(int).values

    # healthy reference for the age model: the stable MCI group is NOT healthy,
    # so borrow the CN group from the same cohort instead.
    cn = d.dropna(subset=V + ["ICV", "AGE", "PTGENDER"])
    cn = cn[cn.DX == "CN"].groupby("RID").first()
    cnv, cna = cn[V].values, cn.AGE.values
    cni, cns = cn.ICV.values, (cn.PTGENDER == "Male").astype(int).values

    print(f"MCI at baseline: n={len(y)}, converters={int(y.sum())}, "
          f"stable={int((1-y).sum())}   |  CN reference n={len(cn)}")

    reps = {k: [] for k in ("R1", "R2", "R3", "B1", "B2")}
    for rep in range(20):
        oof = {k: np.zeros(len(y)) for k in reps}
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(vol, y):
            Gtr = regional_gaps(cnv, cna, cni, cns, vol[tr], age[tr], icv[tr], sex[tr])
            Gte = regional_gaps(cnv, cna, cni, cns, vol[te], age[te], icv[te], sex[te])

            def fp(Ftr, Fte):
                s = StandardScaler().fit(Ftr)
                c = LogisticRegression(max_iter=4000).fit(s.transform(Ftr), y[tr])
                return c.predict_proba(s.transform(Fte))[:, 1]

            oof["R1"][te] = fp(Gtr.mean(1, keepdims=True), Gte.mean(1, keepdims=True))
            oof["R2"][te] = fp(Gtr, Gte)
            ntr = Gtr / (np.linalg.norm(Gtr, axis=1, keepdims=True) + 1e-9)
            nte = Gte / (np.linalg.norm(Gte, axis=1, keepdims=True) + 1e-9)
            oof["R3"][te] = fp(ntr, nte)
            Vtr, Vte = vol[tr] / icv[tr][:, None], vol[te] / icv[te][:, None]
            oof["B1"][te] = fp(np.column_stack([Vtr, sex[tr]]),
                               np.column_stack([Vte, sex[te]]))
            oof["B2"][te] = fp(np.column_stack([Vtr, sex[tr], age[tr]]),
                               np.column_stack([Vte, sex[te], age[te]]))
        for k in reps:
            reps[k].append(roc_auc_score(y, oof[k]))

    lbl = {"R1": "R1  global brain age gap (scalar)",
           "R2": "R2  regional brain age profile",
           "R3": "R3  profile SHAPE only",
           "B1": "B1  volumes only          [unfair baseline]",
           "B2": "B2  volumes + age         [FAIR baseline]"}
    print(f"\n{'='*70}\nPROGNOSIS: MCI -> dementia conversion\n{'='*70}")
    res = {}
    for k in ("R1", "R2", "R3", "B1", "B2"):
        a = np.array(reps[k]); res[k] = a
        print(f"  {lbl[k]:44s} AUC = {a.mean():.3f}  "
              f"[{np.percentile(a,2.5):.3f}, {np.percentile(a,97.5):.3f}]")
    dd = res["R2"] - res["B2"]
    p = 2 * min((dd <= 0).mean(), (dd >= 0).mean())
    print(f"\n  regional profile - fair baseline = {dd.mean():+.3f}  p={p:.3f}")
    np.savez(f"{OUT}/exp3_prognosis.npz", **res)
    print(f"saved -> {OUT}/exp3_prognosis.npz")


if __name__ == "__main__":
    main()
