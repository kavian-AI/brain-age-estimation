"""Held-out 5-fold calibration of the SHASH atlas, broken down by SEX and HEAD SIZE.

Replicates p5_full_pool.calibration() exactly (same rng(0) folds, same fit_region, same
SHASH), but carries sex and icv through so z can be grouped by them.

Output: pilot/v2/published_heldout_z.csv.gz and published_calibration_by_{sexlab,icv_q,band}.csv
(used by audit.py and v2_build.py as the "published atlas" reference). pilot/p5_out is untouched.
"""
import os, sys, numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "pilot"))

import p3_pooled as P
import p5_full_pool as F
F.OUT = os.path.join(SCRATCH, "_tmp"); os.makedirs(F.OUT, exist_ok=True)   # QC side files, never p5_out
print(f"MODEL = {F.MODEL}", flush=True)

allp, D, O, N = F.full_pool()
assert (allp.dataset != "ADNI").all(), "ADNI leaked into the pool"
print(f"pool: {len(allp)} people, {allp.site.nunique()} sites, {allp.dataset.nunique()} datasets, "
      f"ages {allp.age.min():.0f}-{allp.age.max():.0f}, "
      f"sex0={int((allp.sex==0).sum())} sex1={int((allp.sex==1).sum())}", flush=True)

subj = allp.subject.unique()
fold_of = dict(zip(subj, np.random.default_rng(0).permutation(len(subj)) % 5))

out = []
for i, r in enumerate(P.REGIONS, 1):
    d, ly = P.region_data(allp, r)
    f = d.subject.map(fold_of).values
    z = np.full(len(d), np.nan)
    for k in range(5):
        tr, te = f != k, f == k
        m, _ = F.fit_region(d[tr].reset_index(drop=True), ly[tr], r)
        keep = d[te].site.isin(m.sites).values
        dte = d[te].reset_index(drop=True)
        dte = dte[dte.site.isin(m.sites)]
        z[np.where(te)[0][keep]] = m.z(dte, np.log(dte[r].values.astype(float)))
    out.append(pd.DataFrame({"region": r, "site": d.site.values, "dataset": d.dataset.values,
                             "age": d.age.values, "sex": d.sex.values, "icv": d.icv.values,
                             "z": z}))
    print(f"  [{i}/17] {r}", flush=True)

Z = pd.concat(out, ignore_index=True).dropna(subset=["z"])
Z["licv"] = np.log(Z.icv)
Z["band"] = pd.cut(Z.age, [0, 18, 40, 60, 75, 100], labels=["<18", "18-40", "40-60", "60-75", "75+"])
Z["sexlab"] = np.where(Z.sex == 1, "men", "women")
# head-size quintiles WITHIN sex: a global split would mostly reproduce the sex split
Z["icv_q"] = Z.groupby("sexlab").licv.transform(
    lambda s: pd.qcut(s, 5, labels=["Q1 smallest", "Q2", "Q3", "Q4", "Q5 largest"]))
try:
    Z.to_csv(os.path.join(SCRATCH, "published_heldout_z.csv.gz"), index=False)
    print(f"checkpoint saved: {len(Z)} rows", flush=True)
except Exception as _e:
    print(f"(checkpoint save skipped: {_e})", flush=True)

def summ(g):
    return pd.Series({"n": len(g), "mean_z": g.z.mean(), "sd_z": g.z.std(),
                      "within_1.96": float(np.mean(np.abs(g.z) < 1.96)),
                      "within_1": float(np.mean(np.abs(g.z) < 1))})

def table(key, name):
    t = pd.DataFrame({k: summ(g) for k, g in Z.groupby(key, observed=True)}).T
    t["n"] = t["n"].astype(int)
    print(f"\n===== CALIBRATION BY {name} =====")
    print(t.round(4).to_string())
    t.to_csv(os.path.join(SCRATCH, f"published_calibration_by_{key}.csv"))

print(f"\n===== OVERALL (n={len(Z)}) =====")
print(summ(Z).round(4).to_string())
table("sexlab", "SEX")
table("icv_q", "HEAD SIZE (quintile within sex)")
table("band", "AGE BAND  [reproduces published run]")

print("\n===== RESIDUAL DEPENDENCE OF z  (all should be ~0) =====")
print(f"{'region':<22}{'corr(z,logICV)':>16}{'corr(z,age)':>13}{'mean z men-women':>19}")
for r in P.REGIONS:
    g = Z[Z.region == r]
    ci = np.corrcoef(g.z, g.licv)[0, 1]
    ca = np.corrcoef(g.z, g.age)[0, 1]
    dm = g[g.sexlab == "men"].z.mean() - g[g.sexlab == "women"].z.mean()
    flag = "   <-- CHECK" if (abs(ci) > 0.10 or abs(dm) > 0.20) else ""
    print(f"{r:<22}{ci:>+16.3f}{ca:>+13.3f}{dm:>+19.3f}{flag}")
print(f"\nPOOLED: corr(z,logICV) = {np.corrcoef(Z.z, Z.licv)[0,1]:+.4f}   "
      f"corr(z,age) = {np.corrcoef(Z.z, Z.age)[0,1]:+.4f}")
print(f"\nwrote: {SCRATCH}")
