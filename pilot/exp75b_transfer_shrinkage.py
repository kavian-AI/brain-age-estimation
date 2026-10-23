# E75b: which disease direction survives a change of cohort AND segmentation program?
# w(lam) = (S + lam*diag(S))^-1 mu, from ADNI; lam=0 is the discriminant, lam->inf is the
# effect map itself used as weights. Scored in OASIS (FastSurfer). Also within ADNI (held-out half)
# so the price of shrinkage at home is visible next to its benefit abroad.
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, "pilot")
import exp62_rich_atlas as R, exp66_oasis_external as O
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
a, m, T, names = R.build("X"); Ro, _, _ = O.load_fs()
common = [n for n in names if n in Ro.columns]; ia = [names.index(n) for n in common]
cn = m.DX.values == "CN"; ad = m.DX.values == "Dementia"
Za, _ = R.normative(m, names, cn); Za = Za[:, ia]
half = np.random.RandomState(0).rand(len(m)) < 0.5
A_ = half & (cn | ad); B_ = ~half & (cn | ad)
S = LedoitWolf().fit(np.vstack([Za[A_ & cn] - Za[A_ & cn].mean(0), Za[A_ & ad] - Za[A_ & ad].mean(0)])).covariance_
mu = Za[A_ & ad].mean(0) - Za[A_ & cn].mean(0)
clin = pd.read_csv(O.BASE + r"\data\oasis_longitudinal.csv").rename(columns={"MRI ID": "session", "Subject ID": "subject"})
M = Ro[Ro.mpr == "mpr-1"].merge(clin, on="session"); M["SEX"] = (M["M/F"] == "M").astype(int)
M = M.sort_values(["subject", "Visit"]).reset_index(drop=True); B = M.drop_duplicates("subject").reset_index(drop=True)
ever = M.groupby("subject").CDR.max(); ref = ((B.CDR == 0) & (B.subject.map(ever) == 0)).values; y = (B.CDR > 0).astype(int).values
Zo = O.normative(B[common].values.astype(float), B.Age.values.astype(float), B.SEX.values, B.eTIV.values.astype(float), ref, B.subject.values.astype(str))
print(f"OASIS n={len(B)}  |  {'shrinkage':>10s} {'ADNI half-B AUC':>16s} {'OASIS AUC':>10s}")
for lam in [0, 0.1, 0.3, 1, 3, 10, 30, 1e9]:
    w = np.linalg.solve(S + lam * np.diag(np.diag(S)), mu)
    print(f"  {'map as weights' if lam > 1e6 else lam:>26} {roc_auc_score(ad[B_].astype(int), Za[B_] @ w):16.3f} {roc_auc_score(y, Zo @ w):10.3f}")
