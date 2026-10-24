"""Fit the 59 final atlas-v2 models on the full healthy pool and write a PORTABLE atlas (plain tables that
need only numpy/pandas to score anyone), then verify the tables reproduce the model's z exactly."""
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "pilot")); sys.path.insert(0, HERE)
import v2_build as B
import p3_pooled as P
OUT = os.path.join(HERE, "atlas_v2"); os.makedirs(OUT, exist_ok=True)
allp, _ = B.load()
LICV_MEAN = float(np.log(allp.icv).mean())
AGES = np.round(np.arange(5.0, 95.0001, 0.25), 2)
grid, coefs, sites, checks = [], [], [], []
for i, f in enumerate(B.FEATS, 1):
    d, ly = P.region_data(allp, f); m = B.fit(d, ly, f)
    for sx in (0, 1):
        g = pd.DataFrame({"age": AGES, "sex": float(sx), "licv": 0.0, "site": m.sites[0], "icv_method": "SynthStrip"})
        loc, sc = m._loc_scale(g, zero_site=True)
        grid.append(pd.DataFrame({"feature": f, "sex": sx, "age": AGES, "loc0": loc, "logsd0": np.log(sc)}))
    nB, nBs, K = len(m.Bm.column_names), len(m.Bs.column_names), len(m.sites)
    base_m = nB + 1 + m.G + (K - 1) + (1 if m.agesex else 0)              # parent _Xm columns
    c = float(m.bm[base_m]) if m.ageicv else 0.0
    base_s = nBs + (K - 1 if m.site_sd else 0)
    d1, d2 = ((float(m.bs[base_s]) / 2, float(m.bs[base_s + 1]) / 2) if m.scale_icv else (0.0, 0.0))  # log-sd scale
    tm, ts = B.eb_taus(m, d)
    coefs.append({"feature": f, "fluid": B.struct(f) in B.FLUID, "b_head": float(m.bm[nB + 1]), "c_age_head": c,
                  "d1_head_logsd": d1, "d2_head2_logsd": d2, "nu": m.nu, "tau": m.tau,
                  "tau_site_shift_z": tm, "tau_site_logsd": ts, "n_fit": len(d)})
    sm, ss = m._site_slices()
    offm = np.r_[-m.bm[sm].sum(), m.bm[sm]]; offs = np.r_[-m.bs[ss].sum(), m.bs[ss]] / 2.0
    n = d.site.value_counts().reindex(m.sites).fillna(0).astype(int).values
    sites.append(pd.DataFrame({"feature": f, "site": m.sites, "n": n, "shift_log": offm, "logsd_offset": offs}))
    # verification: portable formula vs the fitted model on every person
    G = pd.concat(grid[-2:]); 
    l0 = np.where(d.sex.values == 1, np.interp(d.age, AGES, G[G.sex == 1].loc0), np.interp(d.age, AGES, G[G.sex == 0].loc0))
    s0 = np.where(d.sex.values == 1, np.interp(d.age, AGES, G[G.sex == 1].logsd0), np.interp(d.age, AGES, G[G.sex == 0].logsd0))
    so = dict(zip(m.sites, offm)); sso = dict(zip(m.sites, offs)); x = d.licv.values
    loc = l0 + coefs[-1]["b_head"] * x + c * (d.age.values - 60) / 10 * x + d.site.map(so).values
    lsd = s0 + d.site.map(sso).values + d1 * x + d2 * x ** 2
    z0 = (ly - loc) / np.exp(lsd); z = np.sinh(m.tau * np.arcsinh(z0) - m.nu)
    checks.append({"feature": f, "max_abs_diff_z": float(np.max(np.abs(z - m.z(d, ly))))})
    if i % 10 == 0 or i == len(B.FEATS): print(f"  portable {i}/{len(B.FEATS)}", flush=True)
pd.concat(grid).to_csv(os.path.join(OUT, "grid_v2.csv.gz"), index=False)
pd.DataFrame(coefs).to_csv(os.path.join(OUT, "coefs_v2.csv"), index=False)
pd.concat(sites).to_csv(os.path.join(OUT, "sites_v2.csv"), index=False)
C = pd.DataFrame(checks); C.to_csv(os.path.join(OUT, "portable_check.csv"), index=False)
meta = json.load(open(os.path.join(OUT, "atlas_v2_meta.json"))) if os.path.exists(os.path.join(OUT, "atlas_v2_meta.json")) else {}
meta.update({"log_icv_mean": LICV_MEAN, "reference_icv_mm3": float(np.exp(LICV_MEAN)), "age_range": [5.0, 95.0],
             "fitted_age_range_data": [float(allp.age.min()), float(allp.age.max())], "pipeline": "FastSurfer 2.4.2 aseg+DKT, SynthStrip ICV"})
json.dump(meta, open(os.path.join(OUT, "atlas_v2_meta.json"), "w"), indent=1)
print(f"portable atlas written; reproduction check: max |z difference| over all 59 models = {C.max_abs_diff_z.max():.2e}")
