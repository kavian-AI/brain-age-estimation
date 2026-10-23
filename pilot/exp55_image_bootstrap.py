#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
THE TWO IMAGE RUNS DISAGREE. SETTLE IT WITH A BOOTSTRAP.

E54, scoring the OLD bottleneck network's volumes, found the component and the
gap exactly tied (-0.0002, p=0.924) and concluded the image pipeline had
collapsed too far for E48 to be testable.

The new tri-head kernel reports the opposite: component minus gap of +0.023,
+0.021 and +0.020 across its three heads, against +0.026 for FreeSurfer. If that
holds it would mean E48 survives the image pipeline after all.

Two reasons not to believe it yet. The kernel scored with an IN-SAMPLE logistic
fit, and it reported no confidence interval. Both are fixed here: cross-validated
AUC by subject, and a subject-level bootstrap with the rotation refitted inside
every draw, exactly as every other E48 test has been run.
"""
import numpy as np, pandas as pd, warnings, os
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
CSV = os.path.join(BASE, "kaggle", "brainage_whiten", "out", "deviations.csv")
V = ["Hippocampus","Entorhinal","Fusiform","MidTemp","Ventricles","WholeBrain"]
FA = {"zV":"سر V: حجم خام","zZ":"سر Z: انحراف نورماتیو","zW":"سر W: سفیدشده","zT":"FreeSurfer"}
FOLDS, N_BOOT = 5, 500

def folds_of(ids, seed=0):
    uq = np.unique(ids)
    fm = dict(zip(uq, np.random.RandomState(seed).permutation(len(uq)) % FOLDS))
    return np.array([fm[s] for s in ids])

def auc_cv(X, y, subs, seed=0):
    f = folds_of(subs, seed); o = np.zeros(len(y))
    for k in range(FOLDS):
        tr, te = f != k, f == k
        if y[tr].sum() < 5 or (1-y[tr]).sum() < 5: return np.nan
        sc = StandardScaler().fit(X[tr])
        c = LogisticRegression(max_iter=4000).fit(sc.transform(X[tr]), y[tr])
        o[te] = c.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, o)

def one_pc(Zfit, Zall):
    sc = StandardScaler().fit(Zfit)
    p = PCA(n_components=1, random_state=0).fit(sc.transform(Zfit))
    return p.transform(sc.transform(Zall))

d = pd.read_csv(CSV)
d = d.sort_values(["PTID", "age_visit"]).drop_duplicates("PTID").reset_index(drop=True)
d = d[d.DX.isin(["CN", "Dementia"])].reset_index(drop=True)
y = (d.DX == "Dementia").astype(int).values
subs = d.PTID.values
ag = d.age_visit.values.reshape(-1, 1)
cn = (d.DX == "CN").values
print(f"AD vs CN, one visit per subject: n={len(y)} ({int(y.sum())} AD, {int((1-y).sum())} CN)\n")

print(f"  {'arm':26s} {'gap':>7s} {'1 comp':>8s} {'six':>7s}   component minus gap")
print("  " + "-"*74)
rng_master = 0
for pre in ["zV", "zZ", "zW", "zT"]:
    Z = d[[f"{pre}_{v}" for v in V]].values
    g = d.gap_true.values if pre == "zT" else d.gap_img.values
    a_gap = auc_cv(np.column_stack([g, ag]), y, subs)
    a_pc  = auc_cv(np.column_stack([one_pc(Z[cn], Z), ag]), y, subs)
    a_six = auc_cv(np.column_stack([Z, ag]), y, subs)
    rng = np.random.RandomState(rng_master); uq = np.unique(subs); diff = []
    for _ in range(N_BOOT):
        pick = set(rng.choice(uq, len(uq), replace=True))
        m = np.array([s in pick for s in subs])
        if y[m].sum() < 25 or (1-y[m]).sum() < 25 or cn[m].sum() < 30: continue
        try:
            pc = one_pc(Z[m][cn[m]], Z[m])
            x1 = auc_cv(np.column_stack([pc, ag[m]]), y[m], subs[m])
            x2 = auc_cv(np.column_stack([g[m], ag[m]]), y[m], subs[m])
            if np.isfinite(x1) and np.isfinite(x2): diff.append(x1 - x2)
        except Exception: pass
    diff = np.array(diff); lo, hi = np.percentile(diff, [2.5, 97.5])
    p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
    print(f"  {FA[pre]:26s} {a_gap:7.3f} {a_pc:8.3f} {a_six:7.3f}   "
          f"{diff.mean():+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.3f} "
          f"{'HOLDS' if p<0.05 and diff.mean()>0 else '-'}")
