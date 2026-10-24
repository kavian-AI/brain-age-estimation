import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
FLUID = {"lateral ventricles", "inf. lat. ventricles", "third ventricle"}
Z = pd.read_csv(os.path.join(HERE, "v2_spec_heldout.csv.gz")); LR = pd.read_csv(os.path.join(HERE, "v2_spec_lr.csv"))
site_n = Z[Z.variant == "base"].groupby("site").size() / Z.region.nunique()
SMALL = set(site_n[site_n < 40].index)
Z["small_site"] = Z.site.isin(SMALL)
Z["in_tail"] = np.where(Z.region.isin(FLUID), Z.z > 1.96, Z.z < -1.96)
Z["icv_q"] = Z.groupby(["region", "variant", "sex"]).licv.transform(lambda s: pd.qcut(s, 5, labels=False))
rows = []
for (r, v), g in Z.groupby(["region", "variant"], sort=False):
    old = g[g.age >= 60]; sm = g[g.small_site]
    rows.append({"region": r, "variant": v, "logp_all": g.logp.mean(), "logp_small": sm.logp.mean(),
                 "SDratio_men_women_60+": old[old.sex == 1].z.std() / old[old.sex == 0].z.std(),
                 "tail60_women": old[old.sex == 0].in_tail.mean(), "tail60_men": old[old.sex == 1].in_tail.mean(),
                 "SD_Q1": g[g.icv_q == 0].z.std(), "SD_Q3": g[g.icv_q == 2].z.std(), "SD_Q5": g[g.icv_q == 4].z.std(),
                 "small_SD": sm.z.std(), "small_in1.96": np.mean(np.abs(sm.z) < 1.96),
                 "small_abs_site_mean": sm.groupby("site").z.mean().abs().mean()})
S = pd.DataFrame(rows); S.to_csv(os.path.join(HERE, "v2_spec_summary.csv"), index=False)
base = S[S.variant == "base"].set_index("region")
print(f"small sites: {len(SMALL)} (n<40)")
print("\n=========== LR tests vs base (full data) ===========")
print(LR.pivot(index="region", columns="variant", values="p").map(lambda p: f"{p:.1e}").to_string())
print("\n=========== held-out, median over 17 regions ===========")
pd.set_option("display.width", 250)
med = S.drop(columns="region").groupby("variant", sort=False).median()
print(med.round(3).T.to_string())
print("\n=========== held-out log-likelihood gain vs base (sum over 17 regions, per person x 1000) ===========")
for v in S.variant.unique():
    if v == "base": continue
    s = S[S.variant == v].set_index("region")
    ga, gs = s.logp_all - base.logp_all, s.logp_small - base.logp_small
    print(f"  {v:<22} all people: {1000*ga.sum():+7.1f} ({int((ga>0).sum())}/17 regions better)   "
          f"small sites: {1000*gs.sum():+7.1f} ({int((gs>0).sum())}/17 better)")
print("\n=========== per region: 60+ tail rate women/men (target 2.5%) and men/women SD ratio ===========")
for r in S.region.unique():
    g = S[S.region == r].set_index("variant")
    print(f"  {r:<22}" + "  ".join(f"{v.replace('+','')[:14]:>14}: {g.loc[v,'tail60_women']:.1%}/{g.loc[v,'tail60_men']:.1%} sd{g.loc[v,'SDratio_men_women_60+']:.2f}"
                                   for v in ["base", "+spread by sex", "+all three"]))
