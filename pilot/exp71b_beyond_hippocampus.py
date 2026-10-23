# E71b: is the composition effect (E71) just hippocampal atrophy in disguise?
# Same atrophied-MCI sample; composition tested ADJUSTED for bilateral hippocampal z
# (and for the MTL average), for both amyloid status and conversion.
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, "pilot")
import exp53_units_and_dimension as E, exp62_rich_atlas as R, exp64_what_it_measures as W, exp69_decomposed_brain_age as D
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
a, m, T, names = R.build("X"); cn = m.DX.values == "CN"
Z, zh = R.normative(m, names, cn); gap = R.rich_gap(m, names, cn); gd, go, s = D.decompose(m, Z, gap)
mtl = T.cls.values == "mtl"
m = m.assign(gap=gap, gd=gd, go=go, hip=-zh, mtl=(Z[:, mtl] * T.sign.values[mtl]).mean(1))
bl = a[a.VISCODE == "bl"].drop_duplicates("PTID").set_index("PTID")
for c in ["AV45", "ABETA"]: m[c] = W.num(bl.reindex(m.PTID)[c]).values
m["A"] = np.where(np.isfinite(m.AV45), m.AV45 > 1.11, np.where(np.isfinite(m.ABETA), m.ABETA < 980, np.nan)).astype(float)
M = m[m.DX.values == "MCI"].copy()
keep, dur, ev = E.survival(a, M.PTID.values, ["Dementia"]); o = {x: i for i, x in enumerate(keep)}
M["T"] = [dur[o[x]] if x in o else np.nan for x in M.PTID]; M["E"] = [ev[o[x]] if x in o else np.nan for x in M.PTID]
H = M[M.gap >= np.percentile(M.gap, 66.7)].copy(); H["comp"] = H.go - H.gd
z = lambda x: (x - x.mean()) / x.std()
print("ATROPHIED MCI (top third of gap)")
Ha = H.dropna(subset=["A"])
for cols in [["hip"], ["mtl"], ["comp"], ["hip", "comp"], ["mtl", "comp"]]:
    X = np.column_stack([z(Ha[c]) for c in cols]); y = (1 - Ha.A).values
    p = cross_val_predict(LogisticRegression(), X, y, cv=10, method="predict_proba")[:, 1]
    print(f"  amyloid-NEGATIVE, CV AUC from {'+'.join(cols):10s} {roc_auc_score(y, p):.3f}  (n={len(Ha)})")
lr = LogisticRegression(C=1e6).fit(np.column_stack([z(Ha.hip), z(Ha.comp)]), 1 - Ha.A)
import statsmodels.api as sm
res = sm.Logit((1 - Ha.A).values, sm.add_constant(np.column_stack([z(Ha.hip), z(Ha.comp)]))).fit(disp=0)
print(f"  logistic, both terms: hippocampus OR/SD {np.exp(res.params[1]):.2f} p={res.pvalues[1]:.4f}   composition OR/SD {np.exp(res.params[2]):.2f} p={res.pvalues[2]:.4f}")
Hc = H.dropna(subset=["T", "E"])
for cols in [["comp", "gap", "age_visit"], ["comp", "gap", "hip", "age_visit"], ["comp", "gap", "mtl", "age_visit"]]:
    Dd = Hc[cols + ["T", "E"]].copy()
    for c in cols: Dd[c] = z(Dd[c])
    cf = CoxPHFitter().fit(Dd, "T", "E")
    print(f"  conversion, Cox with {'+'.join(cols):28s} composition HR/SD {np.exp(cf.params_['comp']):.2f} p={cf.summary.loc['comp','p']:.4f}"
          + (f"   hippocampus HR/SD {np.exp(cf.params_['hip']):.2f} p={cf.summary.loc['hip','p']:.4f}" if 'hip' in cols else "")
          + (f"   MTL HR/SD {np.exp(cf.params_['mtl']):.2f} p={cf.summary.loc['mtl','p']:.4f}" if 'mtl' in cols else ""))
