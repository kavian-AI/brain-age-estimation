"""Which scale / site options earn their place?  base = age-x-sex curve everywhere (+ age x head size
for temporal horn and third ventricle). Variants add: spread by sex, spread by head size, scanner shrinkage."""
import os, sys, time, numpy as np, pandas as pd
from scipy import stats
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "pilot")); sys.path.insert(0, HERE)
import p3_pooled as P
import p5_full_pool as F
F.OUT = os.path.join(HERE, "_tmp"); os.makedirs(F.OUT, exist_ok=True)
from atlas_v2_model import AtlasModel
T0 = time.time()
FLUID = {"lateral ventricles", "inf. lat. ventricles", "third ventricle"}
AGEICV = {"inf. lat. ventricles", "third ventricle"}
VARIANTS = {"base": {}, "+spread by sex": {"scale_sex": True}, "+spread by head size": {"scale_icv": True},
            "+scanner shrinkage": {"shrink": True},
            "+all three": {"scale_sex": True, "scale_icv": True, "shrink": True}}
def fit(d, ly, r, opts):
    young = d.age.min() < 15
    kw = dict(df_mu=7 if young else 5, df_sd=4 if young else 3, lo=5.0 if young else 18.0, hi=98.0,
              icv_groups=[g for g in F.ICV_GROUPS if (d.icv_method == g).any()], sexcurve=True, ageicv=r in AGEICV)
    kw.update(opts); return AtlasModel(d, ly, **kw)

allp, *_ = F.full_pool()
site_n = allp.site.value_counts()
SMALL = set(site_n[site_n < 40].index)
print(f"pool {len(allp)}; {len(site_n)} sites, {len(SMALL)} with n < 40 ({int(site_n[site_n < 40].sum())} people)", flush=True)

lr = []
for r in P.REGIONS:
    d, ly = P.region_data(allp, r)
    m0 = fit(d, ly, r, {})
    for name in ("+spread by sex", "+spread by head size"):
        m = fit(d, ly, r, VARIANTS[name]); g = m.loglik - m0.loglik; df = m.k_params - m0.k_params
        lr.append({"region": r, "variant": name, "loglik_gain": g, "df": df, "p": stats.chi2.sf(2 * max(g, 0), df)})
print(f"[{(time.time()-T0)/60:.1f} min] full fits done", flush=True)
LR = pd.DataFrame(lr); LR.to_csv(os.path.join(HERE, "v2_spec_lr.csv"), index=False)

subj = allp.subject.unique()
fold_of = dict(zip(subj, np.random.default_rng(0).permutation(len(subj)) % 5))
out = []
for r in P.REGIONS:
    d, ly = P.region_data(allp, r)
    f = d.subject.map(fold_of).values
    for name, opts in VARIANTS.items():
        z = np.full(len(d), np.nan); lp = np.full(len(d), np.nan)
        for k in range(5):
            tr, te = f != k, f == k
            m = fit(d[tr].reset_index(drop=True), ly[tr], r, opts)
            keep = d[te].site.isin(m.sites).values
            dte = d[te].reset_index(drop=True); dte = dte[dte.site.isin(m.sites)]
            yte = np.log(dte[r].values.astype(float)); idx = np.where(te)[0][keep]
            z[idx] = m.z(dte, yte); lp[idx] = m.logpdf_np(dte, yte)
        out.append(pd.DataFrame({"region": r, "variant": name, "site": d.site.values, "age": d.age.values,
                                 "sex": d.sex.values, "licv": d.licv.values, "z": z, "logp": lp}))
    print(f"[{(time.time()-T0)/60:.1f} min] held-out done: {r}", flush=True)
Z = pd.concat(out, ignore_index=True).dropna(subset=["z"])
Z.to_csv(os.path.join(HERE, "v2_spec_heldout.csv.gz"), index=False)

# ------------------------------------------------------------ summaries
Z["small_site"] = Z.site.isin(SMALL)
Z["in_tail"] = np.where(Z.region.isin(FLUID), Z.z > 1.96, Z.z < -1.96)
Z["icv_q"] = Z.groupby(["region", "variant", "sex"]).licv.transform(lambda s: pd.qcut(s, 5, labels=False))
rows = []
for (r, v), g in Z.groupby(["region", "variant"], sort=False):
    old = g[g.age >= 60]; sm = g[g.small_site]
    site_means = sm.groupby("site").z.mean()
    rows.append({"region": r, "variant": v,
                 "heldout logp (all)": g.logp.mean(), "heldout logp (small sites)": sm.logp.mean(),
                 "SD z men/women 60+": old[old.sex == 1].z.std() / old[old.sex == 0].z.std(),
                 "60+ tail women": old[old.sex == 0].in_tail.mean(), "60+ tail men": old[old.sex == 1].in_tail.mean(),
                 "SD z Q1": g[g.icv_q == 0].z.std(), "SD z Q3": g[g.icv_q == 2].z.std(), "SD z Q5": g[g.icv_q == 4].z.std(),
                 "small sites SD z": sm.z.std(), "small sites in1.96": np.mean(np.abs(sm.z) < 1.96),
                 "small sites |site mean z|": site_means.abs().mean()})
S = pd.DataFrame(rows); S.to_csv(os.path.join(HERE, "v2_spec_summary.csv"), index=False)
base = S[S.variant == "base"].set_index("region")
print("\n=========== LR tests (full data, vs base) ===========")
print(LR.pivot(index="region", columns="variant", values="p").map(lambda p: f"{p:.1e}").to_string())
print("\n=========== median across 17 regions ===========")
agg = S.drop(columns="region").groupby("variant", sort=False).median()
agg.insert(0, "gain in heldout logp vs base (x1000, all)",
           1000 * (S.set_index("region").groupby("variant", sort=False)["heldout logp (all)"].apply(lambda s: (s - base["heldout logp (all)"]).median())))
agg.insert(1, "gain (x1000, small sites)",
           1000 * (S.set_index("region").groupby("variant", sort=False)["heldout logp (small sites)"].apply(lambda s: (s - base["heldout logp (small sites)"]).median())))
pd.set_option("display.width", 250)
print(agg.round(3).T.to_string())
print("\nregions where each variant improves held-out logp (all people):")
for v in VARIANTS:
    if v == "base": continue
    gain = S[S.variant == v].set_index("region")["heldout logp (all)"] - base["heldout logp (all)"]
    gsm = S[S.variant == v].set_index("region")["heldout logp (small sites)"] - base["heldout logp (small sites)"]
    print(f"  {v:<22} all: {int((gain > 0).sum())}/17 better (total {1000*gain.sum():+.1f} x1e-3)   "
          f"small sites: {int((gsm > 0).sum())}/17 better (total {1000*gsm.sum():+.1f} x1e-3)")
print(f"\n[{(time.time()-T0)/60:.1f} min] done")
