"""Step C: head-to-head disease detection, our atlas vs PCN braincharts, on identical people/folds."""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

SHARED = {"hippocampus": ["Left-Hippocampus", "Right-Hippocampus"],
          "amygdala": ["Left-Amygdala", "Right-Amygdala"],
          "lateral ventricles": ["Left-Lateral-Ventricle", "Right-Lateral-Ventricle"],
          "inf. lat. ventricles": ["Left-Inf-Lat-Vent", "Right-Inf-Lat-Vent"],
          "third ventricle": ["3rd-Ventricle"],
          "thalamus": ["Left-Thalamus-Proper", "Right-Thalamus-Proper"],
          "putamen": ["Left-Putamen", "Right-Putamen"],
          "brainstem": ["Brain-Stem"],
          "cerebellum cortex": ["Left-Cerebellum-Cortex", "Right-Cerebellum-Cortex"]}
FLUID = {"lateral ventricles", "inf. lat. ventricles", "third ventricle"}
SIG3 = ["hippocampus", "amygdala", "inf. lat. ventricles"]
COMPARISONS = [("OASIS-1 CDR 0.5", "OASIS-1 controls"), ("OASIS-1 CDR>=1", "OASIS-1 controls"),
               ("OASIS-2 demented", "OASIS-2 nondemented"), ("OASIS-2 converted", "OASIS-2 nondemented")]

# ---------------------------------------------------------------- DeLong (Sun & Xu 2014)
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    out = np.empty(N); out[J] = T; return out

def delong(pos, neg):
    """pos, neg: (k scores) x n arrays. Returns AUCs (k,) and covariance (k,k)."""
    m, n = pos.shape[1], neg.shape[1]
    tx = np.array([_midrank(p) for p in pos]); ty = np.array([_midrank(q) for q in neg])
    tz = np.array([_midrank(np.r_[p, q]) for p, q in zip(pos, neg)])
    auc = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    S = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return auc, S

def auc_ci(p, q):
    a, S = delong(p[None], q[None]); se = np.sqrt(S[0, 0])
    return a[0], a[0] - 1.96 * se, a[0] + 1.96 * se

def paired(p1, q1, p2, q2):
    a, S = delong(np.vstack([p1, p2]), np.vstack([q1, q2]))
    d = a[0] - a[1]; se = np.sqrt(max(S[0, 0] + S[1, 1] - 2 * S[0, 1], 1e-12))
    return d, d - 1.96 * se, d + 1.96 * se, 2 * stats.norm.sf(abs(d) / se)

def sens95(p, q):
    return float(np.mean(p > np.quantile(q, 0.95)))

# ---------------------------------------------------------------- data
ours = pd.read_csv(os.path.join(REPO, "pilot", "p6_out", "scores.csv"))
pcn = pd.read_csv(os.path.join(HERE, "pcn_z.csv"))
ppl = pd.read_csv(os.path.join(HERE, "people.csv"))
key = ["subject", "group"]
D = ours.merge(pcn.drop(columns=["cohort", "fold"]), on=key).merge(ppl[key + ["icv"]], on=key)
assert len(D) == len(ours) == len(pcn), (len(D), len(ours), len(pcn))
D["licv"] = np.log(D.icv)
print(f"merged {len(D)} people; folds identical by construction")

for r, cols in SHARED.items():                       # PCN regional z: mean of hemispheres, + = abnormal
    sgn = 1.0 if r in FLUID else -1.0
    D["pcn_" + r] = sgn * D[cols].mean(axis=1)
    D["our_" + r] = D["z_" + r]                         # already sign-flipped in p6
for a in ("our", "pcn"):
    D[f"{a}_sig3"] = D[[f"{a}_{r}" for r in SIG3]].mean(axis=1)

def learned(prefix, feats, name):
    D[name] = np.nan
    for k in range(5):
        tr, te = D[D.fold != k], D[D.fold == k]
        X = [f"{prefix}{f}" for f in feats]
        clf = LogisticRegression(C=0.3, max_iter=4000).fit(tr[X], tr.is_patient)
        D.loc[te.index, name] = clf.decision_function(te[X])
learned("our_", list(SHARED), "our_learned9")
learned("pcn_", list(SHARED), "pcn_learned9")

# PCN at full strength: logistic on every subcortical phenotype we can feed it
PHEN = [c for c in pcn.columns if c not in ("subject", "group", "cohort", "fold")]
for c in PHEN:
    D["pcnall_" + c] = (1.0 if ("Vent" in c) else -1.0) * D[c]
learned("pcnall_", PHEN, "pcn_learned_all")

# PCN + ICV: cross-fitted removal of the head-size slope learned on other folds' controls
for r in list(SHARED) + ["sig3"]:
    col, new = f"pcn_{r}", f"pcnicv_{r}"
    D[new] = np.nan
    for coh in D.cohort.unique():
        for k in range(5):
            tr = D[(D.cohort == coh) & (D.fold != k) & ~D.is_patient]
            te = D[(D.cohort == coh) & (D.fold == k)]
            b, a = np.polyfit(tr.licv, tr[col], 1)
            D.loc[te.index, new] = te[col] - (a + b * te.licv)

# ---------------------------------------------------------------- 1. calibration on held-out controls
print("\n================ 1. HEALTHY CALIBRATION on held-out OASIS controls ================")
print("(z should be ~N(0,1); corr with log ICV should be ~0)")
rows = []
for coh, ctl in (("OASIS-1", "OASIS-1 controls"), ("OASIS-2", "OASIS-2 nondemented")):
    c = D[D.group == ctl]
    for r, cols in SHARED.items():
        zo = c["z_" + r]
        zp = c[cols].values.ravel(); lp = np.repeat(c.licv.values, len(cols))
        rows.append({"cohort": coh, "region": r,
                     "ours_mean": zo.mean(), "ours_sd": zo.std(), "ours_in1.96": np.mean(np.abs(zo) < 1.96),
                     "ours_r_icv": np.corrcoef(zo, c.licv)[0, 1],
                     "pcn_mean": np.mean(zp), "pcn_sd": np.std(zp), "pcn_in1.96": np.mean(np.abs(zp) < 1.96),
                     "pcn_r_icv": np.corrcoef(zp, lp)[0, 1]})
cal = pd.DataFrame(rows)
print(cal.round(3).to_string(index=False))
print("\nmedian over regions & cohorts:")
print(cal.drop(columns=["cohort", "region"]).abs().median().round(3).to_string())
cal.to_csv(os.path.join(HERE, "cmp_calibration.csv"), index=False)

# ---------------------------------------------------------------- 2. disease detection
print("\n================ 2. DISEASE DETECTION: AUC [95% CI], sensitivity @95% specificity ================")
scores = [(r, f"our_{r}", f"pcn_{r}") for r in SHARED] + [("AD signature (hip+amy+temporal horn)", "our_sig3", "pcn_sig3"),
          ("learned, 9 shared regions", "our_learned9", "pcn_learned9")]
res = []
for pat, ctl in COMPARISONS:
    P_, Q_ = D[D.group == pat], D[D.group == ctl]
    for lab, so, sp in scores:
        ao, lo_o, hi_o = auc_ci(P_[so].values, Q_[so].values)
        ap, lo_p, hi_p = auc_ci(P_[sp].values, Q_[sp].values)
        d, dlo, dhi, p = paired(P_[so].values, Q_[so].values, P_[sp].values, Q_[sp].values)
        row = {"comparison": f"{pat} ({len(P_)}) vs {ctl} ({len(Q_)})", "score": lab,
               "AUC_ours": ao, "ours_lo": lo_o, "ours_hi": hi_o, "AUC_pcn": ap, "pcn_lo": lo_p, "pcn_hi": hi_p,
               "diff": d, "diff_lo": dlo, "diff_hi": dhi, "p": p,
               "sens95_ours": sens95(P_[so].values, Q_[so].values), "sens95_pcn": sens95(P_[sp].values, Q_[sp].values)}
        icv = {"hippocampus": "pcnicv_hippocampus", "AD signature (hip+amy+temporal horn)": "pcnicv_sig3"}.get(lab)
        if icv:
            row["AUC_pcn+icv"] = auc_ci(P_[icv].values, Q_[icv].values)[0]
        res.append(row)
R = pd.DataFrame(res)
R.to_csv(os.path.join(HERE, "cmp_auc.csv"), index=False)
for comp, g in R.groupby("comparison", sort=False):
    print(f"\n--- {comp} ---")
    for _, x in g.iterrows():
        extra = f"   PCN+ICV {x['AUC_pcn+icv']:.3f}" if pd.notna(x.get("AUC_pcn+icv", np.nan)) else ""
        star = " *" if x.p < 0.05 else ""
        print(f"  {x.score:<38} ours {x.AUC_ours:.3f} [{x.ours_lo:.2f},{x.ours_hi:.2f}]  "
              f"PCN {x.AUC_pcn:.3f} [{x.pcn_lo:.2f},{x.pcn_hi:.2f}]  "
              f"diff {x['diff']:+.3f} [{x.diff_lo:+.2f},{x.diff_hi:+.2f}] p={x.p:.3f}{star}  "
              f"sens95 {x.sens95_ours:.2f} vs {x.sens95_pcn:.2f}{extra}")

# ---------------------------------------------------------------- 3. each atlas at full available strength
print("\n================ 3. EACH ATLAS AT ITS FULL AVAILABLE STRENGTH (not like-for-like) ================")
full = [("ours: published learned, 17 regions", "learned"), ("ours: published AD signature (4, incl. entorhinal)", "ad_mean"),
        ("PCN: learned, all %d subcortical phenotypes" % len(PHEN), "pcn_learned_all")]
for pat, ctl in COMPARISONS:
    P_, Q_ = D[D.group == pat], D[D.group == ctl]
    print(f"--- {pat} vs {ctl} ---")
    for lab, s in full:
        a, lo, hi = auc_ci(P_[s].values, Q_[s].values)
        print(f"  {lab:<48} AUC {a:.3f} [{lo:.2f},{hi:.2f}]  sens95 {sens95(P_[s].values, Q_[s].values):.2f}")

# ---------------------------------------------------------------- 4. how many like-for-like contests each won
lk = R[~R.comparison.str.contains("converted")]
print("\n================ 4. SCOREBOARD (dementia comparisons, %d contests) ================" % len(lk))
print(f"  ours higher AUC: {(lk['diff'] > 0).sum()}   PCN higher: {(lk['diff'] < 0).sum()}   "
      f"significant (p<.05): ours {((lk['diff'] > 0) & (lk.p < .05)).sum()}, PCN {((lk['diff'] < 0) & (lk.p < .05)).sum()}")
print(f"  median AUC difference (ours - PCN): {lk['diff'].median():+.3f}")
D.to_csv(os.path.join(HERE, "cmp_scores_all.csv"), index=False)
