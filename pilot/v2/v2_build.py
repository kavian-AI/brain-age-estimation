"""Atlas v2 final build + evaluation.   python v2_build.py [calib|transfer|disease|export|all]
Run from the repo root: python pilot/v2/v2_build.py all. Each stage caches its output in pilot/v2/."""
import os, sys, time, json, numpy as np, pandas as pd
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "pilot")); sys.path.insert(0, HERE)
import p3_pooled as P
import p5_full_pool as F
F.OUT = os.path.join(HERE, "_tmp"); os.makedirs(F.OUT, exist_ok=True)     # QC side files
import p6_disease_test as S6
from atlas_v2_model import AtlasModel
T0 = time.time()
OUT = os.environ.get("V2_OUT", HERE); os.makedirs(OUT, exist_ok=True)
SMOKE = os.environ.get("SMOKE") == "1"
def log(m): print(f"[{(time.time()-T0)/60:5.1f} min] {m}", flush=True)

# ============================================================ configuration (set from the spec tests)
CONFIG = json.load(open(os.path.join(HERE, "v2_config.json")))
SCALE_SEX, SHRINK = CONFIG["scale_sex"], CONFIG["shrink"]
ICV_SPREAD = set(CONFIG["icv_in_spread"])

NEW_BIL = {"caudate": "caudate", "pallidum": "pallidum", "accumbens": "accumbens", "ventral DC": "ventral_dc",
           "cerebellum WM": "cerebellum_wm"}
PAIRS = {"hippocampus": "hippocampus", "amygdala": "amygdala", "lateral ventricles": "lateral_ventricles",
         "inf. lat. ventricles": "inf_lat_ventricles", "thalamus": "thalamus", "putamen": "putamen",
         "cerebellum cortex": "cerebellum_cortex", "cerebral WM": "cerebral_wm", "entorhinal": "entorhinal",
         "precuneus": "precuneus", "superior frontal": "superiorfrontal", "lateral occipital": "lateraloccipital",
         "precentral": "precentral", **NEW_BIL}
BIL = list(P.REGIONS) + list(NEW_BIL) + ["fourth ventricle"]
HEMI = {f"{s}_{stem}": reg for reg, stem in PAIRS.items() for s in ("L", "R")}
FEATS = BIL + list(HEMI)
if SMOKE:
    FEATS = ["hippocampus", "inf. lat. ventricles", "caudate", "fourth ventricle", "L_cerebellum_wm", "R_inf_lat_ventricles"]
    BIL = ["hippocampus", "inf. lat. ventricles", "caudate", "fourth ventricle"]
FLUID = {"lateral ventricles", "inf. lat. ventricles", "third ventricle", "fourth ventricle"}
AGEICV = set(CONFIG["age_x_icv_mean"])
def struct(f): return HEMI.get(f, f)
def sign(f): return 1.0 if struct(f) in FLUID else -1.0

def add_features(df):
    df = df.copy()
    for name, stem in NEW_BIL.items():
        df[name] = df["L_" + stem] + df["R_" + stem]
    df["fourth ventricle"] = df["fourth_ventricle"] if "fourth_ventricle" in df else np.nan
    low = df["cerebellum cortex"].isna() | df["brainstem"].isna()          # FOV-QC'd lower brain
    for c in ["cerebellum WM", "L_cerebellum_wm", "R_cerebellum_wm", "fourth ventricle"]:
        df.loc[low, c] = np.nan
    return df

def fit(d, ly, f):
    young = d.age.min() < 15
    kw = dict(df_mu=7 if young else 5, df_sd=4 if young else 3, lo=5.0 if young else 18.0, hi=98.0,
              icv_groups=[g for g in F.ICV_GROUPS if (d.icv_method == g).any()],
              sexcurve=True, ageicv=struct(f) in AGEICV, scale_sex=SCALE_SEX, scale_icv=struct(f) in ICV_SPREAD, shrink=SHRINK)
    return AtlasModel(d, ly, **kw)

def load():
    allp, o2, people, licv_mean = S6.build()
    allp = add_features(allp); people = add_features(people)
    sc = pd.read_csv(os.path.join(REPO, "pilot", "p6_out", "scores.csv"))
    people = people.drop(columns=["fold"]).merge(sc[["subject", "group", "fold"]], on=["subject", "group"])
    assert len(people) == len(sc)
    return allp, people

BANDS = [0, 18, 40, 60, 75, 100]; BLAB = ["<18", "18-40", "40-60", "60-75", "75+"]

# ============================================================ stage 1: held-out calibration
def stage_calib(allp):
    out_path = os.path.join(OUT, "v2_heldout_z.csv.gz")
    if not os.path.exists(out_path):
        subj = allp.subject.unique()
        fold_of = dict(zip(subj, np.random.default_rng(0).permutation(len(subj)) % 5))
        out = []
        for i, f in enumerate(FEATS, 1):
            d, ly = P.region_data(allp, f)
            fo = d.subject.map(fold_of).values
            z = np.full(len(d), np.nan)
            for k in range(5):
                tr, te = fo != k, fo == k
                m = fit(d[tr].reset_index(drop=True), ly[tr], f)
                keep = d[te].site.isin(m.sites).values
                dte = d[te].reset_index(drop=True); dte = dte[dte.site.isin(m.sites)]
                z[np.where(te)[0][keep]] = m.z(dte, np.log(dte[f].values.astype(float)))
            out.append(pd.DataFrame({"feature": f, "subject": d.subject.values, "dataset": d.dataset.values,
                                     "site": d.site.values, "age": d.age.values, "sex": d.sex.values,
                                     "licv": d.licv.values, "z": z}))
            if i % 10 == 0 or i == len(FEATS): log(f"calib {i}/{len(FEATS)}")
        pd.concat(out, ignore_index=True).dropna(subset=["z"]).to_csv(out_path, index=False)
    Z = pd.read_csv(out_path)
    site_n = allp.site.value_counts(); small = set(site_n[site_n < 40].index)
    Z["band"] = pd.cut(Z.age, BANDS, labels=BLAB)
    rows = []
    for f, g in Z.groupby("feature", sort=False):
        gaps = []
        for _, gb in g.groupby("band", observed=True):
            m_, w_ = gb[gb.sex == 1].z, gb[gb.sex == 0].z
            gaps.append(((m_.mean() - w_.mean()), np.sqrt(m_.var() / len(m_) + w_.var() / len(w_))))
        ci = [abs(np.corrcoef(gs.z, gs.licv)[0, 1]) for _, gs in g.groupby(["sex", "band"], observed=True) if len(gs) > 30]
        q = g.groupby("sex").licv.transform(lambda s: pd.qcut(s, 5, labels=False))
        ds = g.groupby("dataset").z.apply(lambda s: np.mean(np.abs(s) < 1.96))
        sm = g[g.site.isin(small)]
        old = g[g.age >= 60]; tail = (old.z > 1.96) if struct(f) in FLUID else (old.z < -1.96)
        rows.append({"feature": f, "n": len(g), "mean": g.z.mean(), "SD": g.z.std(), "in1.96": np.mean(np.abs(g.z) < 1.96),
                     "max|sex gap|": max(abs(a) for a, _ in gaps), "bands >2SE": sum(abs(a) > 2 * s for a, s in gaps),
                     "max|r ICV| (sex x age)": max(ci), "SD Q1/Q3/Q5": f"{g[q==0].z.std():.2f}/{g[q==2].z.std():.2f}/{g[q==4].z.std():.2f}",
                     "datasets in1.96 min-max": f"{ds.min():.3f}-{ds.max():.3f}", "small sites in1.96": np.mean(np.abs(sm.z) < 1.96),
                     "60+ tail women/men": f"{tail[old.sex==0].mean():.1%}/{tail[old.sex==1].mean():.1%}"})
    C = pd.DataFrame(rows); C.to_csv(os.path.join(OUT, "v2_calibration.csv"), index=False)
    # same metrics for the PUBLISHED atlas (17 bilateral) from the earlier held-out run
    pub_path = os.path.join(HERE, "published_heldout_z.csv.gz")
    if not os.path.exists(pub_path):
        log("published_heldout_z.csv.gz not found (run published_heldout.py first): skipping the published comparison")
        return C, None
    Pub = pd.read_csv(pub_path).rename(columns={"region": "feature"})
    Pub["band"] = pd.cut(Pub.age, BANDS, labels=BLAB)
    pub_rows = []
    for f, g in Pub.groupby("feature", sort=False):
        gaps = [abs(gb[gb.sex == 1].z.mean() - gb[gb.sex == 0].z.mean()) for _, gb in g.groupby("band", observed=True)]
        old = g[g.age >= 60]; tail = (old.z > 1.96) if f in FLUID else (old.z < -1.96)
        pub_rows.append({"feature": f, "pub max|sex gap|": max(gaps), "pub 60+ tail women/men": f"{tail[old.sex==0].mean():.1%}/{tail[old.sex==1].mean():.1%}"})
    log("calibration summary written")
    return C, pd.DataFrame(pub_rows)

# ============================================================ stage 2: new-scanner transfer (leave one site out)
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(80); GH_W = GH_W / GH_W.sum()
def shash_moments(nu, tau):
    z0 = np.sinh((np.arcsinh(GH_X) + nu) / tau); m = (GH_W * z0).sum(); return m, np.sqrt((GH_W * z0 ** 2).sum() - m ** 2)

def eb_taus(m, d):
    """typical scanner shift (z0 units) and log-spread variation, from sites with n >= 100 (as in shrinkage)."""
    sm, ss = m._site_slices(); n = d.site.value_counts().reindex(m.sites).fillna(0).values; big = n >= 100
    offm = np.r_[-m.bm[sm].sum(), m.bm[sm]]; offs = np.r_[-m.bs[ss].sum(), m.bs[ss]] / 2.0
    sc = float(np.median(np.sqrt(np.exp(np.clip(m._Xs(d) @ m.bs, -16, 4)))))
    tm = np.sqrt(max(np.var(offm[big], ddof=1) - np.mean(sc ** 2 / n[big]), 1e-6)) / sc
    ts = np.sqrt(max(np.var(offs[big], ddof=1) - np.mean(1 / (2 * n[big])), 1e-6))
    return tm, ts

def adapt(u_cal, E0, S0, tm, ts):
    n = len(u_cal)
    if n < 2: return 0.0, 1.0
    b_raw = np.std(u_cal, ddof=1) / S0
    ws = ts ** 2 / (ts ** 2 + 1 / (2 * (n - 1))); b = np.exp(ws * np.log(b_raw))
    a_raw = np.mean(u_cal) - b * E0
    wm = tm ** 2 / (tm ** 2 + b ** 2 / n); return wm * a_raw, b

def stage_transfer(allp):
    out_path = os.path.join(OUT, "v2_transfer.csv")
    if not os.path.exists(out_path):
        sites = (allp.groupby("site").agg(n=("subject", "size"), ds=("dataset", "first"), lo=("age", "min"), hi=("age", "max"))
                 .query("n >= 100").sort_values("n", ascending=False))
        sites = sites.drop_duplicates("ds").head(2 if SMOKE else 10)
        log("transfer sites: " + ", ".join(f"{s} (n={r.n})" for s, r in sites.iterrows()))
        rng = np.random.default_rng(7); rows = []
        for s, info in sites.iterrows():
            train_all, test_all = allp[allp.site != s], allp[allp.site == s]
            for f in BIL:
                d, ly = P.region_data(train_all, f)
                te = test_all[test_all[f].notna() & (test_all[f] > 0)]
                if len(te) < 60: continue
                m = fit(d, ly, f); E0, S0 = shash_moments(m.nu, m.tau); tm, ts = eb_taus(m, d)
                u, _ = m.z_unseen_site(te, np.log(te[f].values.astype(float)))
                for ncal in (0, 5, 10, 20, 40, 80):
                    if ncal >= len(te) - 20: continue
                    for rep in range(20 if ncal else 1):
                        idx = rng.permutation(len(te)); cal, ev = idx[:ncal], idx[ncal:]
                        a, b = adapt(u[cal], E0, S0, tm, ts)
                        z = np.sinh(m.tau * np.arcsinh((u[ev] - a) / b) - m.nu)
                        rows.append({"site": s, "feature": f, "ncal": ncal, "rep": rep, "mean": z.mean(), "sd": z.std(),
                                     "in1.96": np.mean(np.abs(z) < 1.96), "tau_shift_z": tm, "tau_logsd": ts})
            log(f"transfer done: {s}")
        pd.DataFrame(rows).to_csv(out_path, index=False)
    return pd.read_csv(out_path)

# ============================================================ stage 3: OASIS disease scoring (p6 protocol)
def stage_disease(allp, people):
    out_path = os.path.join(OUT, "v2_oasis_scores.csv")
    if not os.path.exists(out_path):
        Zs = pd.DataFrame(np.nan, index=people.index, columns=FEATS)
        for k in (range(1) if SMOKE else range(5)):
            held = people[people.fold == k]
            train = allp.drop(index=held.pool_index[held.pool_index >= 0].values)
            for f in FEATS:
                d, ly = P.region_data(train, f)
                m = fit(d, ly, f)
                ok = held[f].notna() & (held[f] > 0)
                Zs.loc[held.index[ok], f] = sign(f) * m.z(held[ok], np.log(held.loc[ok, f].values.astype(float)))
            log(f"disease fold {k} scored")
        out = pd.concat([people[["subject", "group", "cohort", "is_patient", "fold", "age", "sex", "icv"]], Zs], axis=1)
        out.to_csv(out_path, index=False)
    return pd.read_csv(out_path)

# ============================================================ stage 4: export the fitted atlas
def stage_export(allp):
    exp_dir = os.path.join(OUT, "atlas_v2"); os.makedirs(exp_dir, exist_ok=True)
    ages = np.arange(7, 90, 1.0); cent = [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]
    ref_icv = float(np.exp(np.log(allp.icv).mean()))
    rows, srows = [], []
    for i, f in enumerate(FEATS, 1):
        d, ly = P.region_data(allp, f); m = fit(d, ly, f)
        for sx in (0, 1):
            g = pd.DataFrame({"age": ages, "sex": float(sx), "licv": 0.0, "site": m.sites[0], "icv_method": "SynthStrip"})
            loc, scale = m._loc_scale(g, zero_site=True)
            med = loc + scale * np.sinh(m.nu / m.tau)
            sd_pct = 100 * scale * m.c_sd
            slope = 100 * np.gradient(med, ages)
            q = {f"c{100*p:g}": np.exp(m.quantile_from(loc, scale, p)) for p in cent}
            rows.append(pd.DataFrame({"feature": f, "sex": "men" if sx else "women", "age": ages, "median_mm3": np.exp(med),
                                      "spread_pct": sd_pct, "slope_pct_per_yr": slope,
                                      "R_years": sd_pct / np.maximum(np.abs(slope), 1e-6), **q}))
        sm, ss = m._site_slices()
        offm = np.r_[-m.bm[sm].sum(), m.bm[sm]]; offs = np.r_[-m.bs[ss].sum(), m.bs[ss]] / 2.0
        n = d.site.value_counts().reindex(m.sites).fillna(0).astype(int).values
        nB = len(m.Bm.column_names)
        srows.append(pd.DataFrame({"feature": f, "site": m.sites, "n": n, "mean_shift_pct": 100 * (np.exp(offm) - 1),
                                   "spread_factor": np.exp(offs), "head_exponent_b": m.bm[nB + 1], "nu": m.nu, "tau": m.tau}))
        if i % 10 == 0 or i == len(FEATS): log(f"export {i}/{len(FEATS)}")
    pd.concat(rows).to_csv(os.path.join(exp_dir, "everyage_table_v2.csv"), index=False)
    pd.concat(srows).to_csv(os.path.join(exp_dir, "site_params_v2.csv"), index=False)
    json.dump({"reference_icv_mm3": ref_icv, "n_people": int(len(allp)), "n_sites": int(allp.site.nunique()),
               "config": CONFIG, "features": FEATS, "fluid": sorted(FLUID), "age_x_icv": sorted(AGEICV)},
              open(os.path.join(exp_dir, "atlas_v2_meta.json"), "w"), indent=1)
    log("export written")

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    allp, people = load(); log(f"pool {len(allp)}, people {len(people)}, features {len(FEATS)}")
    if which in ("calib", "all"): stage_calib(allp)
    if which in ("transfer", "all"): stage_transfer(allp)
    if which in ("disease", "all"): stage_disease(allp, people)
    if which in ("export", "all"): stage_export(allp)
    log("ALL DONE")
