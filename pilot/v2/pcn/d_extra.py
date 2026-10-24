"""Follow-ups: (a) paired tests + ablations for the full-strength result,
(b) clinical flag rates at |z|>1.96, (c) residual head-size dependence of our CSF z by age."""
import os, sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from c_compare import SHARED, COMPARISONS, auc_ci, paired, sens95   # reuse DeLong etc. (module reruns, quiet below)
D = pd.read_csv(os.path.join(HERE, "cmp_scores_all.csv"))

def learned(feats, name):
    D[name] = np.nan
    for k in range(5):
        tr, te = D[D.fold != k], D[D.fold == k]
        clf = LogisticRegression(C=0.3, max_iter=4000).fit(tr[feats], tr.is_patient)
        D.loc[te.index, name] = clf.decision_function(te[feats])

hemi15 = ["pcnall_" + c for r in SHARED for c in SHARED[r]]
allp = [c for c in D.columns if c.startswith("pcnall_")]
learned(hemi15, "pcn_learned_hemi15")
learned(allp, "pcn_learned_all_refit")      # identical recipe to c_compare, re-fitted for the paired tests

print("\n================ (a) FULL STRENGTH: paired DeLong tests + ablation ================")
print("PCN ablation: 9 bilateral regions -> same 9 as 15 left/right models -> all 27 subcortical structures")
for pat, ctl in COMPARISONS[:3]:
    P_, Q_ = D[D.group == pat], D[D.group == ctl]
    a = lambda s: auc_ci(P_[s].values, Q_[s].values)[0]
    print(f"\n--- {pat} vs {ctl} ---")
    print(f"  PCN: 9 bilateral {a('pcn_learned9'):.3f} | 15 hemispheric {a('pcn_learned_hemi15'):.3f} | 27 structures {a('pcn_learned_all_refit'):.3f}")
    for la, sa, lb, sb in [("ours 17 (published)", "learned", "PCN 27", "pcn_learned_all_refit"),
                           ("ours 9", "our_learned9", "PCN 27", "pcn_learned_all_refit"),
                           ("PCN 9 bilateral", "pcn_learned9", "PCN 15 hemispheric", "pcn_learned_hemi15"),
                           ("PCN 15 hemispheric", "pcn_learned_hemi15", "PCN 27", "pcn_learned_all_refit")]:
        d, lo, hi, p = paired(P_[sa].values, Q_[sa].values, P_[sb].values, Q_[sb].values)
        print(f"  {la:<20} minus {lb:<20} {d:+.3f} [{lo:+.2f},{hi:+.2f}]  p={p:.3f}{' *' if p < .05 else ''}")

print("\n================ (b) CLINICAL FLAG RATES: share beyond z=1.96 in the disease direction ================")
print("(controls should be ~2.5%; higher in patients = more cases caught at the conventional cut-off)")
rows = []
for r in ["hippocampus", "amygdala", "inf. lat. ventricles", "lateral ventricles"]:
    for g in ["OASIS-1 controls", "OASIS-1 CDR 0.5", "OASIS-1 CDR>=1", "OASIS-2 nondemented", "OASIS-2 demented"]:
        x = D[D.group == g]
        # PCN: flag if either hemisphere crosses (that is how a clinician would read two separate charts)
        cols = SHARED[r]; sgn = 1.0 if "entric" in r else -1.0
        pcn_flag = (sgn * x[cols] > 1.96).any(axis=1).mean()
        rows.append({"region": r, "group": g, "n": len(x), "ours": float((x["our_" + r] > 1.96).mean()),
                     "PCN (either side)": float(pcn_flag)})
F = pd.DataFrame(rows)
print(F.pivot_table(index=["region", "group"], values=["ours", "PCN (either side)"], sort=False).round(3).to_string())

print("\n================ (c) OUR ATLAS: residual head-size dependence of z, by age (whole held-out pool) ================")
Z = pd.read_csv(os.path.join(os.path.dirname(HERE), "published_heldout_z.csv.gz"))
Z["licv"] = np.log(Z.icv)
Z["band"] = pd.cut(Z.age, [0, 18, 40, 60, 75, 100], labels=["<18", "18-40", "40-60", "60-75", "75+"])
out = []
for r in ["hippocampus", "amygdala", "thalamus", "lateral ventricles", "inf. lat. ventricles", "third ventricle",
          "brainstem", "cerebellum cortex", "cortex", "total brain"]:
    g = Z[Z.region == r]
    row = {"region": r}
    for b, gb in g.groupby("band", observed=True):
        row[f"{b} (n={len(gb)})" if r == "hippocampus" else str(b)] = np.corrcoef(gb.z, gb.licv)[0, 1]
    out.append(row)
T = pd.DataFrame(out); T.columns = [c.split(" (")[0] for c in T.columns]
print(T.set_index("region").round(3).to_string())
n_by = Z[Z.region == "lateral ventricles"].groupby("band", observed=True).size()
print("people per band:", n_by.to_dict(), "  (|r| > ~%.2f is beyond chance at 60-75, > ~%.2f at 75+)"
      % (1.96 / np.sqrt(n_by.get("60-75", 1)), 1.96 / np.sqrt(n_by.get("75+", 1))))
D.to_csv(os.path.join(HERE, "cmp_scores_all.csv"), index=False)
