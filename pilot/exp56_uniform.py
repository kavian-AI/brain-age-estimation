#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DO YOU EVEN NEED THE ROTATION? UNIFORM WEIGHTS AGAINST EVERYTHING.

Every controls-only rotation we have fitted, on FreeSurfer volumes and on all
three image heads, came back near-uniform: roughly +0.43 on each atrophy region
and about -0.27 on the ventricles. If a plain sign-corrected AVERAGE of the six
deviations does what the fitted component does, then E48 stops being "fit a PCA
on your controls" and becomes something far simpler and far more deployable:

    stop weighting the regions by how well they predict chronological age.
    Weight them equally. That is the whole intervention.

No rotation to fit, nothing to transfer, no controls needed at deployment beyond
the normative model that a reference dataset already supplies. Tested on both
endpoints against the gap AND against the fitted component, so the claim can fail
in either direction: uniform might lose to the rotation, which would mean the
rotation earns its complexity.
"""
import os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import exp53_units_and_dimension as E
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

V = E.V
N_BOOT = 500

a, b = E.load(); Z = E.normative_z(b); gap = E.brain_age_gap(b)
cn_all = (b.DX.values == "CN")

# "badness": atrophy regions count negatively, ventricles positively.
SIGN = np.array([-1., -1., -1., -1., +1., -1.])   # order = V
uniform = (Z * SIGN).mean(1)

print("sanity: correlation of the uniform average with...")
pc_all = E.rotate(Z[cn_all], Z, 1)[0][:, 0]
print(f"  fitted component  r = {abs(np.corrcoef(uniform, pc_all)[0,1]):.4f}")
print(f"  brain-age gap     r = {abs(np.corrcoef(uniform, gap)[0,1]):.4f}\n")

# ---------------------------------------------------------- endpoint 1
sel = b.DX.isin(["CN", "Dementia"]).values
y = (b.DX.values[sel] == "Dementia").astype(int)
subs = b.PTID.values[sel]; ag = b.age_visit.values[sel].reshape(-1, 1)
Zs, cns = Z[sel], cn_all[sel]
print("=" * 74); print("ENDPOINT 1: AD vs CN at baseline  n=%d" % len(y)); print("=" * 74)
arms = {"brain-age gap": gap[sel], "uniform average": uniform[sel],
        "fitted component": E.rotate(Zs[cns], Zs, 1)[0][:, 0]}
for nm, x in arms.items():
    print(f"  {nm:20s} AUC = {E.auc_cv(np.column_stack([x, ag]), y, subs):.3f}")

rng = np.random.RandomState(0); uq = np.unique(subs)
res = {("uniform average","brain-age gap"): [], ("uniform average","fitted component"): []}
for _ in range(N_BOOT):
    pick = set(rng.choice(uq, len(uq), replace=True))
    m = np.array([s in pick for s in subs])
    if y[m].sum() < 25 or (1-y[m]).sum() < 25 or cns[m].sum() < 30: continue
    try:
        v = {"brain-age gap": gap[sel][m], "uniform average": uniform[sel][m],
             "fitted component": E.rotate(Zs[m][cns[m]], Zs[m], 1)[0][:, 0]}
        s = {k: E.auc_cv(np.column_stack([x, ag[m]]), y[m], subs[m]) for k, x in v.items()}
        for p in res: res[p].append(s[p[0]] - s[p[1]])
    except Exception: pass
for (A, B), dd in res.items():
    dd = np.array(dd); lo, hi = np.percentile(dd, [2.5, 97.5])
    p = 2*min((dd<=0).mean(), (dd>=0).mean())
    print(f"    {A} minus {B:18s} {dd.mean():+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.3f}"
          f"  {'REAL' if p<0.05 else 'not distinguishable'}")

# ---------------------------------------------------------- endpoint 2
pool = b[b.DX == "MCI"]
keep, dur, ev = E.survival(a, pool.PTID.values, ["Dementia"])
idx = pool.index[pool.PTID.isin(set(keep))].values
o = {s: i for i, s in enumerate(keep)}
ii = np.array([o[s] for s in b.PTID.values[idx]]); dur, ev = dur[ii], ev[ii]
age_m = b.age_visit.values[idx]
print("\n" + "=" * 74); print(f"ENDPOINT 2: MCI -> AD conversion  n={len(idx)}, {int(ev.sum())} events"); print("=" * 74)

def cidx(X, d_, e_, seed=0):
    f = np.random.RandomState(seed).permutation(len(d_)) % 3
    r = np.zeros(len(d_)); nm = [f"x{j}" for j in range(X.shape[1])]
    for k in range(3):
        tr, te = f != k, f == k
        if e_[tr].sum() < 5: return np.nan
        D = pd.DataFrame(X[tr], columns=nm); D["T"], D["E"] = d_[tr], e_[tr]
        try:
            mm = CoxPHFitter(penalizer=1.0).fit(D, "T", "E")
            r[te] = mm.predict_partial_hazard(pd.DataFrame(X[te], columns=nm)).values
        except Exception: return np.nan
    return concordance_index(d_, -r, e_)

M = {"brain-age gap": np.column_stack([gap[idx], age_m]),
     "uniform average": np.column_stack([uniform[idx], age_m]),
     "fitted component": np.column_stack([E.rotate(Z[cn_all], Z[idx], 1)[0], age_m])}
for nm, X in M.items():
    print(f"  {nm:20s} C = {np.nanmean([cidx(X, dur, ev, s) for s in range(5)]):.3f}")
rng = np.random.RandomState(1)
res2 = {("uniform average","brain-age gap"): [], ("uniform average","fitted component"): []}
ok = 0
for _ in range(250):
    i = rng.randint(0, len(dur), len(dur))
    if ev[i].sum() < 25: continue
    s = {k: cidx(X[i], dur[i], ev[i], 0) for k, X in M.items()}
    if any(not np.isfinite(x) for x in s.values()): continue
    ok += 1
    for p in res2: res2[p].append(s[p[0]] - s[p[1]])
print(f"  converged draws: {ok}/250")
for (A, B), dd in res2.items():
    dd = np.array(dd); lo, hi = np.percentile(dd, [2.5, 97.5])
    p = 2*min((dd<=0).mean(), (dd>=0).mean())
    print(f"    {A} minus {B:18s} {dd.mean():+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.3f}"
          f"  {'REAL' if p<0.05 else 'not distinguishable'}")
