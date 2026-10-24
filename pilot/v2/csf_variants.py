"""Which interaction do the CSF regions need? Compares, for lateral / inferior-lateral / third
ventricle, the published SHASH model against versions with age x sex and/or age x head size.
Same pool, same SHASH, same 5 held-out folds (rng 0) as the published calibration.
Writes only to pilot/v2/.   usage: python pilot/v2/csf_variants.py [test]"""
import os, sys, time, numpy as np, pandas as pd, patsy
from scipy import stats
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "pilot"))
import p3_pooled as P
import p5_full_pool as F
F.OUT = os.path.join(HERE, "_tmp"); os.makedirs(F.OUT, exist_ok=True)
import shash

class SHASHX(shash.SHASH):
    """Published SHASH plus optional mean terms:
       ageicv : (age-60)/10 x log ICV            -> head-size exponent changes linearly with age
       sexcurve: sex x natural spline(age, 3df)   -> male-female gap follows a free curve over age
                 (first basis column dropped: cr() spans the constant, which the sex main effect has)"""
    def __init__(self, d, ly, ageicv=False, sexcurve=False, **kw):
        self.ageicv, self.sexcurve = ageicv, sexcurve
        if sexcurve:
            self._Bsx = patsy.dmatrix(f"cr(age, df=3, lower_bound={kw.get('lo', 18.0)}, upper_bound={kw.get('hi', 98.0)}) - 1",
                                      d, return_type="dataframe").design_info
        super().__init__(d, ly, **kw)
    def _Xm(self, d, zero_site=False):
        cols = [super()._Xm(d, zero_site=zero_site)]
        if self.ageicv:
            cols.append((((d.age.values - 60) / 10) * d.licv.values)[:, None])
        if self.sexcurve:
            B = patsy.build_design_matrices([self._Bsx], d, return_type="dataframe")[0].values
            cols.append(B[:, 1:] * (d.sex.values - self.sx)[:, None])
        return np.column_stack(cols)

VARIANTS = {"published": {}, "+age x sex (linear)": {"agesex": True}, "+age x sex (curve)": {"sexcurve": True},
            "+age x head size": {"ageicv": True}, "+age x sex (curve) +age x head size": {"sexcurve": True, "ageicv": True}}
REGIONS = ["lateral ventricles", "inf. lat. ventricles", "third ventricle"]

def fit(d, ly, opts):
    young = d.age.min() < 15
    kw = dict(agesex=False, df_mu=7 if young else 5, df_sd=4 if young else 3, lo=5.0 if young else 18.0, hi=98.0,
              icv_groups=[g for g in F.ICV_GROUPS if (d.icv_method == g).any()])
    kw.update(opts)
    return SHASHX(d, ly, **kw)

allp, *_ = F.full_pool()
if len(sys.argv) > 1 and sys.argv[1] == "test":
    d, ly = P.region_data(allp, "third ventricle")
    t = time.time(); m0 = fit(d, ly, {}); m1 = fit(d, ly, VARIANTS["+age x sex (curve) +age x head size"])
    X = m1._Xm(d)
    print(f"fit time {time.time()-t:.1f}s for two fits; design cols {m0._Xm(d).shape[1]} -> {X.shape[1]}, "
          f"rank {np.linalg.matrix_rank(X)} (full rank = {X.shape[1]}); k_params {m0.k_params} -> {m1.k_params}; "
          f"loglik gain {m1.loglik - m0.loglik:.1f}")
    sys.exit()

# ---------------- 1. full-data fits: likelihood-ratio tests vs the published model
rows = []
for r in REGIONS:
    d, ly = P.region_data(allp, r)
    base = None
    for name, opts in VARIANTS.items():
        m = fit(d, ly, opts)
        if base is None: base = m
        lr = 2 * (m.loglik - base.loglik); df = m.k_params - base.k_params
        rows.append({"region": r, "variant": name, "extra_params": df, "loglik_gain": m.loglik - base.loglik,
                     "LR_p": stats.chi2.sf(lr, df) if df > 0 else np.nan,
                     "dAIC": -2 * (m.loglik - base.loglik) + 2 * df, "dBIC": -2 * (m.loglik - base.loglik) + df * np.log(len(d))})
    print(f"  full fits done: {r}", flush=True)
LR = pd.DataFrame(rows); LR.to_csv(os.path.join(HERE, "csf_lr_tests.csv"), index=False)
print("\n========== 1. LIKELIHOOD-RATIO TESTS vs the published model (dAIC/dBIC < 0 = better) ==========")
print(LR.round({"loglik_gain": 1, "dAIC": 1, "dBIC": 1}).to_string(index=False, formatters={"LR_p": lambda v: "" if pd.isna(v) else f"{v:.1e}"}))

# ---------------- 2. held-out z for every variant, same folds as the published calibration
subj = allp.subject.unique()
fold_of = dict(zip(subj, np.random.default_rng(0).permutation(len(subj)) % 5))
out = []
for r in REGIONS:
    d, ly = P.region_data(allp, r)
    f = d.subject.map(fold_of).values
    for name, opts in VARIANTS.items():
        z = np.full(len(d), np.nan)
        for k in range(5):
            tr, te = f != k, f == k
            m = fit(d[tr].reset_index(drop=True), ly[tr], opts)
            keep = d[te].site.isin(m.sites).values
            dte = d[te].reset_index(drop=True); dte = dte[dte.site.isin(m.sites)]
            z[np.where(te)[0][keep]] = m.z(dte, np.log(dte[r].values.astype(float)))
        out.append(pd.DataFrame({"region": r, "variant": name, "age": d.age, "sex": d.sex, "licv": d.licv, "z": z}))
        print(f"  held-out done: {r} | {name}", flush=True)
Z = pd.concat(out, ignore_index=True).dropna(subset=["z"])
Z["band"] = pd.cut(Z.age, [0, 18, 40, 60, 75, 100], labels=["<18", "18-40", "40-60", "60-75", "75+"])
Z.to_csv(os.path.join(HERE, "csf_heldout_z.csv.gz"), index=False)

print("\n========== 2. HELD-OUT CHECKS (all ages; target: 0 / 0 / 1.00 / 95%) ==========")
for r in REGIONS:
    print(f"\n--- {r} ---")
    print(f"{'variant':<38}{'men-women mean z by age band  <18 18-40 40-60 60-75 75+':<62}{'max|corr ICV| within sex':>25}{'SD z':>7}{'in 1.96':>8}"
          f"{'  60+ flagged women / men':>26}")
    for name in VARIANTS:
        g = Z[(Z.region == r) & (Z.variant == name)]
        gap = [gb[gb.sex == 1].z.mean() - gb[gb.sex == 0].z.mean() for _, gb in g.groupby("band", observed=True)]
        ci = [abs(np.corrcoef(gs.z, gs.licv)[0, 1]) for (_, _), gs in g.groupby(["sex", "band"], observed=True) if len(gs) > 30]
        old = g[g.age >= 60]
        fw, fm = (old[old.sex == 0].z > 1.96).mean(), (old[old.sex == 1].z > 1.96).mean()
        print(f"{name:<38}{'  '.join(f'{x:+.2f}' for x in gap):<62}{max(ci):>25.2f}{g.z.std():>7.3f}{np.mean(np.abs(g.z) < 1.96):>8.3f}"
              f"{fw:>15.1%} / {fm:.1%}")
