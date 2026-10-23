#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The user's question: is [volumes + brain-age profile] better than [volumes] alone?
If yes, brain age IS contributing something, even if it can't stand alone.

We test the full ladder so nothing is ambiguous:
  A  volumes only            (no age at all)
  B  volumes + age           (the fair baseline)
  C  brain-age profile only
  D  volumes + profile       <- does adding the gap help over A?
  E  volumes + age + profile <- does it help over B?

Tasks: AD vs CN, CN vs MCI, and MCI->dementia conversion. ADNI.
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


def gaps(ref_v, ref_a, ref_i, ref_s, v, a, i, s):
    G = np.zeros((v.shape[0], len(V)))
    for r in range(len(V)):
        Xh = np.column_stack([ref_v[:, r], ref_i, ref_s])
        sc = StandardScaler().fit(Xh)
        m = LinearRegression().fit(sc.transform(Xh), ref_a)
        b = LinearRegression().fit(ref_a.reshape(-1, 1),
                                   m.predict(sc.transform(Xh)) - ref_a)
        X = sc.transform(np.column_stack([v[:, r], i, s]))
        G[:, r] = (m.predict(X) - a) - b.predict(a.reshape(-1, 1))
    return G


def run(vol, age, icv, sex, y, title, n_rep=20):
    print(f"\n{'='*66}\n{title}   n={len(y)}, positives={int(y.sum())}\n{'='*66}")
    res = {k: [] for k in "ABCDE"}
    for rep in range(n_rep):
        oof = {k: np.zeros(len(y)) for k in res}
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(vol, y):
            h = tr[y[tr] == 0]
            Gtr = gaps(vol[h], age[h], icv[h], sex[h], vol[tr], age[tr], icv[tr], sex[tr])
            Gte = gaps(vol[h], age[h], icv[h], sex[h], vol[te], age[te], icv[te], sex[te])
            Vtr = vol[tr] / icv[tr][:, None]
            Vte = vol[te] / icv[te][:, None]

            def fp(Ftr, Fte):
                sc = StandardScaler().fit(Ftr)
                c = LogisticRegression(max_iter=5000).fit(sc.transform(Ftr), y[tr])
                return c.predict_proba(sc.transform(Fte))[:, 1]

            oof["A"][te] = fp(np.column_stack([Vtr, sex[tr]]),
                              np.column_stack([Vte, sex[te]]))
            oof["B"][te] = fp(np.column_stack([Vtr, sex[tr], age[tr]]),
                              np.column_stack([Vte, sex[te], age[te]]))
            oof["C"][te] = fp(Gtr, Gte)
            oof["D"][te] = fp(np.column_stack([Vtr, sex[tr], Gtr]),
                              np.column_stack([Vte, sex[te], Gte]))
            oof["E"][te] = fp(np.column_stack([Vtr, sex[tr], age[tr], Gtr]),
                              np.column_stack([Vte, sex[te], age[te], Gte]))
        for k in res:
            res[k].append(roc_auc_score(y, oof[k]))

    lbl = {"A": "A  volumes only (no age)",
           "B": "B  volumes + age          [fair baseline]",
           "C": "C  brain-age profile only",
           "D": "D  volumes + profile",
           "E": "E  volumes + age + profile"}
    for k in "ABCDE":
        a = np.array(res[k])
        print(f"  {lbl[k]:44s} AUC = {a.mean():.4f}  "
              f"[{np.percentile(a,2.5):.4f}, {np.percentile(a,97.5):.4f}]")

    def cmp(x, yk, msg):
        d = np.array(res[x]) - np.array(res[yk])
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        verdict = "HELPS" if d.mean() > 0 and p < .05 else \
                  ("hurts" if d.mean() < 0 and p < .05 else "no difference")
        print(f"    {msg}: {d.mean():+.4f}  p={p:.3f}  -> {verdict}")
    print("  --- does adding the profile help? ---")
    cmp("D", "A", "profile added to volumes-only     ")
    cmp("E", "B", "profile added to volumes+age      ")
    cmp("B", "A", "age added to volumes (reference)  ")
    return res


def main():
    d = pd.read_csv(f"{DATA}/ADNIMERGE_05Mar2026.csv", low_memory=False)
    a = d.dropna(subset=V + ["ICV", "AGE", "DX", "PTGENDER"]).sort_values("Years_bl")
    base = a.groupby("RID").first().reset_index()

    def arrs(df):
        return (df[V].values, df.AGE.values, df.ICV.values,
                (df.PTGENDER == "Male").astype(int).values)

    s = base[base.DX.isin(["CN", "Dementia"])]
    v, ag, ic, sx = arrs(s)
    run(v, ag, ic, sx, (s.DX == "Dementia").astype(int).values, "AD vs CN")

    s = base[base.DX.isin(["CN", "MCI"])]
    v, ag, ic, sx = arrs(s)
    run(v, ag, ic, sx, (s.DX == "MCI").astype(int).values, "CN vs MCI (early detection)")

    # conversion
    dd = d.dropna(subset=["DX"]).sort_values(["RID", "Years_bl"])
    lab = {}
    for rid, g in dd.groupby("RID"):
        dx = g.DX.tolist()
        if dx[0] == "MCI":
            lab[rid] = int("Dementia" in dx)
    m = base[base.RID.isin(lab)]
    v, ag, ic, sx = arrs(m)
    run(v, ag, ic, sx, np.array([lab[r] for r in m.RID]),
        "MCI -> dementia conversion (prognosis)")


if __name__ == "__main__":
    main()
