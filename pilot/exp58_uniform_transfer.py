#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DOES THE UNIFORM AVERAGE SURVIVE SCANNER AND COHORT SHIFT?

E48's fitted rotation was put through three domain shifts and held. The uniform
average has nothing fitted, so it cannot overfit a cohort by construction - but
the NORMATIVE MODEL underneath it still is fitted, on training controls, and that
can transfer badly. The advantage over the gap therefore still has to be measured
on held-out cohorts rather than assumed.

Three shifts, all harder than a random split:
    STUDY PHASE      ADNI1 -> ADNI2/GO/3     different recruitment waves
    FIELD STRENGTH   1.5 T -> 3 T            different physics
    SOFTWARE         FreeSurfer 4.3/5.1 -> 6.0

Everything fitted on the training cohort only: the normative model on its CN
subjects, the brain-age model on its CN subjects, and its bias correction. The
uniform average has no parameters at all. Scored on the test cohort with a
subject-level bootstrap.
"""
import os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
V = ["Hippocampus","Entorhinal","Fusiform","MidTemp","Ventricles","WholeBrain"]
SIGN = np.array([-1.,-1.,-1.,-1.,+1.,-1.])
N_BOOT = 400

a = pd.read_csv(os.path.join(BASE,"data","ADNIMERGE_05Mar2026.csv"), low_memory=False)
keep = ["PTID","VISCODE","AGE","PTGENDER","DX","ICV","Years_bl","ORIGPROT",
        "FLDSTRENG","FSVERSION"] + V
d = a[keep].dropna(subset=["AGE","Years_bl","ICV","PTGENDER"]+V)
d = d[d.VISCODE=="bl"].drop_duplicates("PTID").copy()
d["SEX"] = (d.PTGENDER=="Male").astype(int)
d["age_visit"] = d.AGE + d.Years_bl
d = d.reset_index(drop=True)

def fit_apply(tr, te):
    cn = (tr.DX=="CN").values
    agT, agE = tr.age_visit.values.astype(float), te.age_visit.values.astype(float)
    XnT = np.column_stack([agT, agT**2, tr.SEX.values, tr.ICV.values])
    XnE = np.column_stack([agE, agE**2, te.SEX.values, te.ICV.values])
    sc = StandardScaler().fit(XnT[cn]); Ze = np.zeros((len(te), len(V)))
    for j, v in enumerate(V):
        y = tr[v].values.astype(float)
        m = RidgeCV(alphas=np.logspace(-2,5,30)).fit(sc.transform(XnT[cn]), y[cn])
        s = max((y[cn]-m.predict(sc.transform(XnT[cn]))).std(ddof=1), 1e-9)
        Ze[:, j] = (te[v].values - m.predict(sc.transform(XnE))) / s
    XaT = np.column_stack([tr[v].values for v in V]+[tr.ICV.values, tr.SEX.values])
    XaE = np.column_stack([te[v].values for v in V]+[te.ICV.values, te.SEX.values])
    sa = StandardScaler().fit(XaT[cn])
    ma = RidgeCV(alphas=np.logspace(-2,5,30)).fit(sa.transform(XaT[cn]), agT[cn])
    bb = LinearRegression().fit(agT[cn].reshape(-1,1),
                                ma.predict(sa.transform(XaT[cn]))-agT[cn])
    gap = (ma.predict(sa.transform(XaE))-agE) - bb.predict(agE.reshape(-1,1))
    return Ze, gap

def auc_of(X, y):
    sc = StandardScaler().fit(X)
    c = LogisticRegression(max_iter=4000).fit(sc.transform(X), y)
    return roc_auc_score(y, c.predict_proba(sc.transform(X))[:,1])

def run(tr, te, label):
    T = te[te.DX.isin(["CN","Dementia"])].reset_index(drop=True)
    y = (T.DX=="Dementia").astype(int).values
    if y.sum()<20 or (1-y).sum()<20:
        print(f"\n{label}: too few in one class"); return
    Ze, gap = fit_apply(tr, T)
    uni = (Ze*SIGN).mean(1); ag = T.age_visit.values.reshape(-1,1)
    a_g = auc_of(np.column_stack([gap, ag]), y)
    a_u = auc_of(np.column_stack([uni, ag]), y)
    rng = np.random.RandomState(0); dd=[]
    for _ in range(N_BOOT):
        i = rng.randint(0,len(y),len(y))
        if y[i].sum()<15 or (1-y[i]).sum()<15: continue
        try:
            dd.append(auc_of(np.column_stack([uni[i], ag[i]]), y[i])
                      - auc_of(np.column_stack([gap[i], ag[i]]), y[i]))
        except Exception: pass
    dd = np.array(dd); lo,hi = np.percentile(dd,[2.5,97.5])
    p = 2*min((dd<=0).mean(), (dd>=0).mean())
    print(f"\n{label}")
    print(f"  train n={len(tr)} ({int((tr.DX=='CN').sum())} CN) -> "
          f"test n={len(T)} ({int((1-y).sum())} CN, {int(y.sum())} AD)")
    print(f"  gap {a_g:.3f} | uniform average {a_u:.3f}")
    print(f"  uniform minus gap {dd.mean():+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.3f}"
          f"  -> {'HOLDS' if p<0.05 and dd.mean()>0 else 'does not hold'}")

rng = np.random.RandomState(0); m = rng.rand(len(d))<0.5
run(d[m], d[~m], "0. RANDOM half/half (reference)")
tr, te = d[d.ORIGPROT=="ADNI1"], d[d.ORIGPROT!="ADNI1"]
if len(tr)>100 and len(te)>100: run(tr, te, "1. STUDY PHASE  ADNI1 -> ADNI2/GO/3")
f15 = d.FLDSTRENG.astype(str).str.contains("1.5", na=False)
f30 = d.FLDSTRENG.astype(str).str.contains("^3 ", na=False, regex=True)
if f15.sum()>100 and f30.sum()>100: run(d[f15], d[f30], "2. FIELD STRENGTH  1.5T -> 3T")
old = d.FSVERSION.astype(str).str.contains("4.3|5.1", na=False)
new = d.FSVERSION.astype(str).str.contains("6.0", na=False)
if old.sum()>100 and new.sum()>100: run(d[old], d[new], "3. SOFTWARE  FS 4.3/5.1 -> 6.0")
