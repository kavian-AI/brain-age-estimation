#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
THE UNIFORM AVERAGE AGAINST THE SUMMARY THE LITERATURE ALREADY USES.

E56 showed a sign-corrected average of the six normative deviations beats the
brain-age gap by +0.016 AUC and +0.018 C-index, and ties or beats the fitted
rotation. Before that is called new, it has to be run against what normative
modelling already reports.

The established summary is not the mean - it is the OUTLIER COUNT: how many
regions fall beyond z = +/-1.96. Neuroanatomical normative modelling papers use
it, and it is reported to predict MCI-to-AD conversion. A search found no
head-to-head against the brain-age gap, and none against a mean, so both
comparisons are run here on both endpoints.

Four summaries of the SAME six deviations, so nothing differs except how the six
numbers are collapsed into one:

    brain-age gap        weights from predicting chronological age
    outlier count        threshold at 1.96, count the regions   <- the literature
    mean |z|             magnitude only, direction discarded
    uniform average      sign-corrected mean                    <- ours
    fitted component     controls-only PCA                      <- E48

If the outlier count matches the uniform average, our contribution is the
head-to-head, not the score. If the average wins, thresholding throws away
information and that is worth stating.
"""
import os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp53_units_and_dimension as E
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

V = E.V
SIGN = np.array([-1., -1., -1., -1., +1., -1.])   # ventricles enlarge, rest shrink
N_BOOT, N_COX = 500, 250

a, b = E.load(); Z = E.normative_z(b); gap = E.brain_age_gap(b)
cn_all = (b.DX.values == "CN")
Zs_signed = Z * SIGN                      # high = worse, every region

def summaries(Zm, gm, fit_cn):
    return {
        "brain-age gap":     gm,
        "outlier count":     (Zm * SIGN > 1.96).sum(1).astype(float),
        "mean |z|":          np.abs(Zm).mean(1),
        "uniform average":   (Zm * SIGN).mean(1),
        "fitted component":  E.rotate(Zm[fit_cn], Zm, 1)[0][:, 0],
    }

print("how many regions are flagged as outliers at all?")
oc = (Zs_signed > 1.96).sum(1)
for g in ["CN", "MCI", "Dementia"]:
    m = b.DX.values == g
    print(f"  {g:10s} mean count {oc[m].mean():.2f} of 6   "
          f"({100*(oc[m]==0).mean():.0f}% of subjects have ZERO flagged)")

# ------------------------------------------------------------- endpoint 1
sel = b.DX.isin(["CN", "Dementia"]).values
y = (b.DX.values[sel] == "Dementia").astype(int)
subs = b.PTID.values[sel]; ag = b.age_visit.values[sel].reshape(-1, 1)
Zsel, cns = Z[sel], cn_all[sel]
print("\n" + "="*76); print(f"ENDPOINT 1  AD vs CN, n={len(y)}"); print("="*76)
S = summaries(Zsel, gap[sel], cns)
for nm, x in S.items():
    print(f"  {nm:20s} AUC = {E.auc_cv(np.column_stack([x, ag]), y, subs):.3f}")

rng = np.random.RandomState(0); uq = np.unique(subs)
pairs = [("uniform average","outlier count"), ("uniform average","mean |z|"),
         ("outlier count","brain-age gap")]
acc = {p: [] for p in pairs}
for _ in range(N_BOOT):
    pick = set(rng.choice(uq, len(uq), replace=True))
    m = np.array([s in pick for s in subs])
    if y[m].sum() < 25 or (1-y[m]).sum() < 25 or cns[m].sum() < 30: continue
    try:
        Sm = summaries(Zsel[m], gap[sel][m], cns[m])
        v = {k: E.auc_cv(np.column_stack([x, ag[m]]), y[m], subs[m]) for k, x in Sm.items()}
        for p in pairs: acc[p].append(v[p[0]] - v[p[1]])
    except Exception: pass
for (A,B) in pairs:
    d = np.array(acc[(A,B)]); lo,hi = np.percentile(d,[2.5,97.5])
    pv = 2*min((d<=0).mean(), (d>=0).mean())
    print(f"    {A:17s} minus {B:17s} {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]"
          f" p={pv:.3f} {'REAL' if pv<0.05 else '-'}")

# ------------------------------------------------------------- endpoint 2
pool = b[b.DX == "MCI"]
keep, dur, ev = E.survival(a, pool.PTID.values, ["Dementia"])
idx = pool.index[pool.PTID.isin(set(keep))].values
o = {s:i for i,s in enumerate(keep)}
ii = np.array([o[s] for s in b.PTID.values[idx]]); dur, ev = dur[ii], ev[ii]
age_m = b.age_visit.values[idx]
print("\n" + "="*76); print(f"ENDPOINT 2  MCI -> AD, n={len(idx)}, {int(ev.sum())} events"); print("="*76)

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

Sm = summaries(Z[idx], gap[idx], None) if False else None
S2 = {"brain-age gap": gap[idx],
      "outlier count": (Z[idx]*SIGN > 1.96).sum(1).astype(float),
      "mean |z|": np.abs(Z[idx]).mean(1),
      "uniform average": (Z[idx]*SIGN).mean(1),
      "fitted component": E.rotate(Z[cn_all], Z[idx], 1)[0][:,0]}
M = {k: np.column_stack([x, age_m]) for k, x in S2.items()}
for nm, X in M.items():
    print(f"  {nm:20s} C = {np.nanmean([cidx(X, dur, ev, s) for s in range(5)]):.3f}")
rng = np.random.RandomState(1); acc2 = {p: [] for p in pairs}; ok=0
for _ in range(N_COX):
    i = rng.randint(0, len(dur), len(dur))
    if ev[i].sum() < 25: continue
    v = {k: cidx(X[i], dur[i], ev[i], 0) for k, X in M.items()}
    if any(not np.isfinite(x) for x in v.values()): continue
    ok += 1
    for p in pairs: acc2[p].append(v[p[0]] - v[p[1]])
print(f"  converged draws: {ok}/{N_COX}")
for (A,B) in pairs:
    d = np.array(acc2[(A,B)]); lo,hi = np.percentile(d,[2.5,97.5])
    pv = 2*min((d<=0).mean(), (d>=0).mean())
    print(f"    {A:17s} minus {B:17s} {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]"
          f" p={pv:.3f} {'REAL' if pv<0.05 else '-'}")
