#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"YOUR BRAIN-AGE MODEL WAS JUST WEAK." AND "YOUR NORMATIVE MODEL WAS ARBITRARY."

Two specification attacks, both fair, both run here.

PART A — WAS THE BRAIN-AGE MODEL A STRAW MAN?
The gap we compare against uses ridge regression plus the standard regression-
dilution bias correction. A reviewer can reasonably say a stronger age model would
close the +0.016. So: build seven brain-age models spanning weak to strong, report
each one's AGE ACCURACY alongside its disease AUC, and see whether ANY of them
beats the uniform average.

This also puts a number on something the field has only asserted. Schulz et al.
(PLOS Biology 2025) report that brain-age models with LOWER age accuracy have
HIGHER disease sensitivity. If that holds here, age accuracy and disease utility
trade off directly, and the best possible brain-age model is beaten by an average
that never looked at age at all.

PART B — DOES THE NORMATIVE MODEL'S FORM MATTER?
The deviations come from ridge on [age, age^2, sex, ICV]. Five alternatives, from
linear-in-age to cubic to dropping ICV to normalising by ICV instead. If the
advantage survives all of them it is not an artefact of one specification.
"""
import os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp53_units_and_dimension as E
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

V = E.V
SIGN = np.array([-1.,-1.,-1.,-1.,+1.,-1.])
FOLDS, N_BOOT = 5, 400

a, b = E.load()
cn_all = (b.DX.values=="CN")
sel = b.DX.isin(["CN","Dementia"]).values
y = (b.DX.values[sel]=="Dementia").astype(int)
subs = b.PTID.values[sel]; ag = b.age_visit.values[sel].reshape(-1,1)

# ======================================================================= A
print("="*86)
print("A  SEVEN BRAIN-AGE MODELS, WEAK TO STRONG. DOES ANY OF THEM WIN?")
print("="*86)
X_all = np.column_stack([b[v].values for v in V]+[b.ICV.values, b.SEX.values])
age = b.age_visit.values.astype(float)
f = E.folds_of(b.PTID.values)

def build(make, correct=True):
    g = np.zeros(len(b)); pred = np.zeros(len(b))
    for k in range(FOLDS):
        tr, te = np.where((f!=k)&cn_all)[0], np.where(f==k)[0]
        sc = StandardScaler().fit(X_all[tr])
        m = make().fit(sc.transform(X_all[tr]), age[tr])
        pred[te] = m.predict(sc.transform(X_all[te]))
        raw = pred[te]-age[te]
        if correct:
            bb = LinearRegression().fit(age[tr].reshape(-1,1),
                                        m.predict(sc.transform(X_all[tr]))-age[tr])
            g[te] = raw - bb.predict(age[te].reshape(-1,1))
        else:
            g[te] = raw
    return g, np.abs(pred[cn_all]-age[cn_all]).mean()

models = [
    ("ridge + bias correction (ours)", lambda: RidgeCV(alphas=np.logspace(-2,5,30)), True),
    ("ridge, NO bias correction",      lambda: RidgeCV(alphas=np.logspace(-2,5,30)), False),
    ("plain linear regression",        lambda: LinearRegression(), True),
    ("random forest",                  lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1), True),
    ("gradient boosting",              lambda: GradientBoostingRegressor(random_state=0), True),
    ("gradient boosting, deeper",      lambda: GradientBoostingRegressor(n_estimators=500, max_depth=4, learning_rate=.05, random_state=0), True),
    ("support vector regression",      lambda: SVR(C=10, epsilon=1.0), True),
]
Z = E.normative_z(b); uni = (Z*SIGN).mean(1)
a_uni = E.auc_cv(np.column_stack([uni[sel], ag]), y, subs)
print(f"  {'brain-age model':34s} {'age MAE (CN)':>13s} {'AD vs CN AUC':>14s}"
      f" {'uniform beats it by':>21s}")
best = None
rows = []
for nm, mk, corr in models:
    g, mae = build(mk, corr)
    au = E.auc_cv(np.column_stack([g[sel], ag]), y, subs)
    rows.append((nm, mae, au, g))
    print(f"  {nm:34s} {mae:13.3f} {au:14.3f} {a_uni-au:21.3f}")
    if best is None or au > best[2]: best = (nm, mae, au, g)
print(f"\n  uniform average (no age model at all)          {'-':>13s} {a_uni:14.3f}")
print(f"\n  strongest brain-age model was: {best[0]} (AUC {best[2]:.3f})")

rng = np.random.RandomState(0); uq = np.unique(subs); dd=[]
gb = best[3]
for _ in range(N_BOOT):
    pick=set(rng.choice(uq,len(uq),replace=True))
    m=np.array([s in pick for s in subs])
    if y[m].sum()<25 or (1-y[m]).sum()<25: continue
    try:
        dd.append(E.auc_cv(np.column_stack([uni[sel][m],ag[m]]),y[m],subs[m])
                  - E.auc_cv(np.column_stack([gb[sel][m],ag[m]]),y[m],subs[m]))
    except Exception: pass
dd=np.array(dd); lo,hi=np.percentile(dd,[2.5,97.5]); p=2*min((dd<=0).mean(),(dd>=0).mean())
print(f"  uniform minus the BEST of them: {dd.mean():+.4f} [{lo:+.4f},{hi:+.4f}]"
      f" p={p:.3f} {'HOLDS' if p<0.05 and dd.mean()>0 else 'does not hold'}")

r = np.corrcoef([x[1] for x in rows], [x[2] for x in rows])[0,1]
print(f"\n  across the seven models, corr(age MAE, disease AUC) = {r:+.3f}")
print("  positive means: the WORSE a model reads age, the BETTER it detects disease")

# ======================================================================= B
print("\n"+"="*86)
print("B  DOES THE NORMATIVE MODEL'S FORM MATTER?")
print("="*86)
gap_ours = rows[0][3]
def z_spec(spec):
    agv = b.age_visit.values.astype(float)
    if spec=="linear":       X = np.column_stack([agv, b.SEX.values, b.ICV.values])
    elif spec=="quad":       X = np.column_stack([agv, agv**2, b.SEX.values, b.ICV.values])
    elif spec=="cubic":      X = np.column_stack([agv, agv**2, agv**3, b.SEX.values, b.ICV.values])
    elif spec=="no_icv":     X = np.column_stack([agv, agv**2, b.SEX.values])
    elif spec=="icv_norm":   X = np.column_stack([agv, agv**2, b.SEX.values])
    Zc = np.zeros((len(b), len(V)))
    for j, v in enumerate(V):
        yv = b[v].values.astype(float)
        if spec=="icv_norm": yv = yv / b.ICV.values
        for k in range(FOLDS):
            tr, te = np.where((f!=k)&cn_all)[0], np.where(f==k)[0]
            sc = StandardScaler().fit(X[tr])
            m = RidgeCV(alphas=np.logspace(-2,5,30)).fit(sc.transform(X[tr]), yv[tr])
            s = max((yv[tr]-m.predict(sc.transform(X[tr]))).std(ddof=1), 1e-9)
            Zc[te, j] = (yv[te]-m.predict(sc.transform(X[te])))/s
    return Zc

for spec in ["linear","quad","cubic","no_icv","icv_norm"]:
    Zc = z_spec(spec); u = (Zc*SIGN).mean(1)
    au = E.auc_cv(np.column_stack([u[sel], ag]), y, subs)
    rng = np.random.RandomState(0); d2=[]
    for _ in range(250):
        pick=set(rng.choice(uq,len(uq),replace=True))
        m=np.array([s in pick for s in subs])
        if y[m].sum()<25 or (1-y[m]).sum()<25: continue
        try:
            d2.append(E.auc_cv(np.column_stack([u[sel][m],ag[m]]),y[m],subs[m])
                      - E.auc_cv(np.column_stack([gap_ours[sel][m],ag[m]]),y[m],subs[m]))
        except Exception: pass
    d2=np.array(d2); lo,hi=np.percentile(d2,[2.5,97.5]); p=2*min((d2<=0).mean(),(d2>=0).mean())
    print(f"  {spec:10s} uniform AUC {au:.3f}   minus our gap {d2.mean():+.4f}"
          f" [{lo:+.4f},{hi:+.4f}] p={p:.3f} {'HOLDS' if p<0.05 and d2.mean()>0 else '-'}")
