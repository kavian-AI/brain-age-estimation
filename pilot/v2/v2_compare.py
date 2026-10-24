"""Final head-to-head on OASIS: published atlas vs atlas v2 vs PCN braincharts (identical people and folds)."""
import os, numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _midrank(x):
    J = np.argsort(x); Zs = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Zs[j] == Zs[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    out = np.empty(N); out[J] = T; return out
def delong(pos, neg):
    m, n = pos.shape[1], neg.shape[1]
    tx = np.array([_midrank(p) for p in pos]); ty = np.array([_midrank(q) for q in neg])
    tz = np.array([_midrank(np.r_[p, q]) for p, q in zip(pos, neg)])
    auc = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    return auc, np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
def auc_ci(p, q):
    a, S = delong(p[None], q[None]); se = np.sqrt(S[0, 0]); return a[0], a[0] - 1.96 * se, a[0] + 1.96 * se
def paired(p1, q1, p2, q2):
    a, S = delong(np.vstack([p1, p2]), np.vstack([q1, q2])); d = a[0] - a[1]
    se = np.sqrt(max(S[0, 0] + S[1, 1] - 2 * S[0, 1], 1e-12)); return d, d - 1.96 * se, d + 1.96 * se, 2 * stats.norm.sf(abs(d) / se)
def sens95(p, q): return float(np.mean(p > np.quantile(q, 0.95)))

key = ["subject", "group"]
V2 = pd.read_csv(os.path.join(HERE, "v2_oasis_scores.csv"))
pub = pd.read_csv(os.path.join(REPO, "pilot", "p6_out", "scores.csv"))
pcn = pd.read_csv(os.path.join(HERE, "pcn", "cmp_scores_all.csv"))
pcn_raw = [c for c in pcn.columns if c[:5] in ("Left-", "Right") or c in ("3rd-Ventricle", "4th-Ventricle", "Brain-Stem")]
D = V2.merge(pub[key + [c for c in pub.columns if c.startswith("z_")] + ["learned"]], on=key) \
      .merge(pcn[key + pcn_raw + ["pcn_learned_hemi15", "pcn_learned_all_refit"]], on=key)
assert len(D) == 357

# ------------------------------------------------ derived scores
SIG = [("hippocampus", "hippocampus", "Hippocampus"), ("amygdala", "amygdala", "Amygdala"),
       ("inf. lat. ventricles", "inf_lat_ventricles", "Inf-Lat-Vent")]
D["pub_sig3"] = D[["z_" + b for b, _, _ in SIG]].mean(axis=1)
D["v2_sig3"] = D[[b for b, _, _ in SIG]].mean(axis=1)
D["v2_sig3_worse"] = np.mean([np.maximum(D["L_" + s], D["R_" + s]) for _, s, _ in SIG], axis=0)
psign = lambda name: 1.0 if "Vent" in name else -1.0
D["pcn_sig3_worse"] = np.mean([np.maximum(psign(p) * D["Left-" + p], psign(p) * D["Right-" + p]) for _, _, p in SIG], axis=0)
D["pcn_sig3"] = np.mean([psign(p) * (D["Left-" + p] + D["Right-" + p]) / 2 for _, _, p in SIG], axis=0)

SUB_PAIRS = ["hippocampus", "amygdala", "lateral_ventricles", "inf_lat_ventricles", "thalamus", "putamen",
             "cerebellum_cortex", "caudate", "pallidum", "accumbens", "ventral_dc", "cerebellum_wm"]
SHARED7 = ["hippocampus", "amygdala", "lateral_ventricles", "inf_lat_ventricles", "thalamus", "putamen", "cerebellum_cortex"]
F16 = [f"{s}_{p}" for p in SHARED7 for s in "LR"] + ["third ventricle", "brainstem"]
F27 = [f"{s}_{p}" for p in SUB_PAIRS for s in "LR"] + ["third ventricle", "fourth ventricle", "brainstem"]
CORT = [f"{s}_{p}" for p in ("entorhinal", "precuneus", "superiorfrontal", "lateraloccipital", "precentral") for s in "LR"]
def learned(cols, name):
    D[name] = np.nan
    X = D[cols].fillna(0.0)                      # a few lower-brain values are QC-blanked: treat as 'typical'
    for k in range(5):
        tr, te = D.fold != k, D.fold == k
        clf = LogisticRegression(C=0.3, max_iter=5000).fit(X[tr], D.is_patient[tr])
        D.loc[te, name] = clf.decision_function(X[te])
learned(F16, "v2_learned16"); learned(F27, "v2_learned27"); learned(F27 + ["entorhinal"], "v2_learned27+ent")

CMP = [("OASIS-1 CDR 0.5", "OASIS-1 controls"), ("OASIS-1 CDR>=1", "OASIS-1 controls"),
       ("OASIS-2 demented", "OASIS-2 nondemented"), ("OASIS-2 converted", "OASIS-2 nondemented")]
SCORES = [("published: AD signature", "pub_sig3"), ("v2: AD signature", "v2_sig3"), ("v2: AD signature, worse side", "v2_sig3_worse"),
          ("PCN: AD signature, worse side", "pcn_sig3_worse"),
          ("published: learned 17", "learned"), ("v2: learned 16 L/R", "v2_learned16"), ("PCN: learned 16 L/R", "pcn_learned_hemi15"),
          ("v2: learned 27 subcortical", "v2_learned27"), ("PCN: learned 27 subcortical", "pcn_learned_all_refit"),
          ("v2: learned 27 + entorhinal", "v2_learned27+ent")]
print("================ AUC [95% CI] | sensitivity at 95% specificity ================")
tab = []
for pat, ctl in CMP:
    P_, Q_ = D[D.group == pat], D[D.group == ctl]
    print(f"\n--- {pat} ({len(P_)}) vs {ctl} ({len(Q_)}) ---")
    for lab, s in SCORES:
        a, lo, hi = auc_ci(P_[s].values, Q_[s].values); se = sens95(P_[s].values, Q_[s].values)
        tab.append({"comparison": pat, "score": lab, "AUC": a, "lo": lo, "hi": hi, "sens95": se})
        print(f"  {lab:<34} {a:.3f} [{lo:.2f},{hi:.2f}]   sens@95%spec {se:5.0%}")
pd.DataFrame(tab).to_csv(os.path.join(HERE, "v2_final_auc.csv"), index=False)

PAIRS = [("v2: AD signature", "v2_sig3", "published: AD signature", "pub_sig3"),
         ("v2: AD sig. worse side", "v2_sig3_worse", "v2: AD signature", "v2_sig3"),
         ("v2: AD sig. worse side", "v2_sig3_worse", "PCN: AD sig. worse side", "pcn_sig3_worse"),
         ("v2: learned 16 L/R", "v2_learned16", "PCN: learned 16 L/R", "pcn_learned_hemi15"),
         ("v2: learned 27", "v2_learned27", "PCN: learned 27", "pcn_learned_all_refit"),
         ("v2: learned 27", "v2_learned27", "published: learned 17", "learned")]
print("\n================ paired DeLong tests ================")
pt = []
for pat, ctl in CMP[:3]:
    P_, Q_ = D[D.group == pat], D[D.group == ctl]
    print(f"--- {pat} vs {ctl} ---")
    for la, a, lb, b in PAIRS:
        d, lo, hi, p = paired(P_[a].values, Q_[a].values, P_[b].values, Q_[b].values)
        pt.append({"comparison": pat, "A": la, "B": lb, "diff": d, "lo": lo, "hi": hi, "p": p})
        print(f"  {la:<24} minus {lb:<26} {d:+.3f} [{lo:+.2f},{hi:+.2f}]  p={p:.3f}{' *' if p < .05 else ''}")
pd.DataFrame(pt).to_csv(os.path.join(HERE, "v2_final_paired.csv"), index=False)

print("\n================ calibration on held-out OASIS controls (target SD 1, 95% within 1.96, r_ICV 0) ================")
rows = []
for ctl in ("OASIS-1 controls", "OASIS-2 nondemented"):
    c = D[D.group == ctl]; licv = np.log(c.icv)
    for lab, cols in (("published (9 shared, bilateral)", ["z_hippocampus", "z_amygdala", "z_lateral ventricles", "z_inf. lat. ventricles",
                                                             "z_third ventricle", "z_thalamus", "z_putamen", "z_brainstem", "z_cerebellum cortex"]),
                      ("v2 (27 subcortical, L/R)", F27), ("PCN (27 subcortical, L/R)", pcn_raw)):
        Zc = c[cols]
        r = [abs(np.corrcoef(Zc[k].values[Zc[k].notna()], licv.values[Zc[k].notna()])[0, 1]) for k in cols]
        v = Zc.values.ravel(); v = v[np.isfinite(v)]
        rows.append({"controls": ctl, "atlas": lab, "median SD": np.median(Zc.std()), "within 1.96": np.mean(np.abs(v) < 1.96),
                     "median |r ICV|": np.median(r)})
print(pd.DataFrame(rows).round(3).to_string(index=False))
D.to_csv(os.path.join(HERE, "v2_final_scores_all.csv"), index=False)
