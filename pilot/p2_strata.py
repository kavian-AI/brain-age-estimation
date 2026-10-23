# Is aging the same in every group of healthy people?
#
# The atlas quantities (aging rate, spread, sensitivity, resolution) are computed
# separately inside groups defined by sex x scanner (vendor + field strength), and
# the groups are compared. Everything is on the LOG scale, so
#     slope  = 100 * d log(y) / d age      -> % per year, comparable across groups
#     spread = SD of log residuals x 100   -> % (coefficient of variation)
#     R      = spread / |slope|            -> years
# and head size enters as log(ICV): log y = f(age) + b log ICV + ..., i.e. the
# allometric (power-law) normalisation y / ICV^b. b = 1 would be the proportion
# method, b = 0 no correction.
#
# Two pools, because they use different software:
#   ADNI CN   FreeSurfer 4.3/5.1 tables, real eTIV; groups: sex x 1.5T/3T
#   FastSurfer pool  IXI (Philips 1.5T / Philips 3T / GE 1.5T), DLBS (Philips 3T),
#             OASIS-2 (Siemens 1.5T); groups: sex x scanner. Head size = SynthStrip ICV.
#
# Outputs (p2_out/): strata_atlas.csv, strata_tests.csv, headsize_methods.csv
import os
import sys
import warnings

import numpy as np
import pandas as pd
import patsy

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p1_definitions as P1  # noqa: E402
import tadpole_atlas as TA  # noqa: E402
from p2_lifespan_data import load_dlbs, load_ixi, load_oasis2  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p2_out")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(2)
NB = 200
SCANNER = {"IXI-Guys": "Philips 1.5T", "IXI-HH": "Philips 3T", "IXI-IOP": "GE 1.5T",
           "DLBS": "Philips 3T (DLBS)", "OASIS2": "Siemens 1.5T"}
FS_REG = {"hippocampus": "hippocampus", "entorhinal": "entorhinal", "amygdala": "amygdala",
          "lateral ventricles": "lateral_ventricles", "inf. lat. ventricles": "inf_lat_ventricles",
          "third ventricle": "third_ventricle", "thalamus": "thalamus", "putamen": "putamen",
          "brainstem": "brainstem", "cerebellum cortex": "cerebellum_cortex",
          "precuneus": "precuneus", "superior frontal": "superiorfrontal",
          "lateral occipital": "lateraloccipital", "precentral": "precentral",
          "cortex": "cortex", "cerebral WM": "cerebral_wm", "total brain": "total_brain"}
ADNI_REG = ["hippocampus", "entorhinal", "amygdala", "lateral ventricles", "inf. lat. ventricles",
            "third ventricle", "thalamus", "putamen", "brainstem", "cerebellum cortex", "precuneus",
            "superior frontal", "lateral occipital", "precentral"]


# ------------------------------------------------------------------ data
def adni_pool():
    m, T, names, comp = P1.load()
    t = pd.read_csv(TA.TADPOLE, low_memory=False, usecols=["PTID", "VISCODE", "FLDSTRENG_bl"])
    t = t[t.VISCODE == "bl"].drop_duplicates("PTID")
    fs = dict(zip(t.PTID, t.FLDSTRENG_bl.astype(str)))
    d = pd.DataFrame({"subject": m.PTID.values, "age": m.age_visit.values.astype(float),
                      "sex": m.SEX.values.astype(float), "icv": m.ICV_T.values.astype(float),
                      "dx": m.DX.values})
    d["scanner"] = ["1.5T" if "1.5" in fs.get(p, "") else ("3T" if "3 " in fs.get(p, "") else "?")
                    for p in m.PTID]
    for r in ADNI_REG:
        d[r] = comp[r]
    d = d[(d.dx == "CN") & (d.scanner != "?")].reset_index(drop=True)
    d["dataset"] = "ADNI"
    return d, ADNI_REG


def fs_pool():
    parts = []
    I = load_ixi()
    I["scanner"] = I.site.map(SCANNER)
    parts.append(I)
    D = load_dlbs()
    D = D[(D.wave == 1) & (D.run == 1)].copy()
    D["scanner"] = SCANNER["DLBS"]
    parts.append(D)
    O = load_oasis2()
    O = O[O.wave == 1].copy()
    O["scanner"] = SCANNER["OASIS2"]
    parts.append(O)
    cols = ["subject", "dataset", "scanner", "age", "sex", "icv"] + list(FS_REG.values())
    d = pd.concat([p[cols] for p in parts], ignore_index=True)
    d = d.rename(columns={v: k for k, v in FS_REG.items()})
    # one head-size measure (SynthStrip ICV) for every dataset -> centre globally; any
    # scanner scaling (gradient calibration) affects ICV and volumes alike and cancels
    d["licv"] = np.log(d.icv) - np.log(d.icv).mean()
    return d.dropna(subset=["age", "icv"]).reset_index(drop=True), list(FS_REG)


# ------------------------------------------------------------------ models
def fit_curve(d, y, ages, licv=True, df=3):
    """log y = spline(age) + b*log ICV (+ sex if mixed). Returns slope %/yr, spread %, b."""
    lo, hi = d.age.min(), d.age.max()
    f = f"cr(age, df={df}, lower_bound={lo}, upper_bound={hi})"
    B = patsy.dmatrix(f, d, return_type="dataframe")
    cols = [B.values]
    if licv:
        cols.append(d.licv.values[:, None])
    if d.sex.nunique() > 1:
        cols.append(d.sex.values[:, None] - d.sex.mean())
    X = np.column_stack(cols)
    ly = np.log(y)
    b, *_ = np.linalg.lstsq(X, ly, rcond=None)
    e = ly - X @ b
    nb = B.shape[1]
    G = patsy.build_design_matrices([B.design_info], pd.DataFrame({"age": ages}), return_type="dataframe")[0].values
    Gp = patsy.build_design_matrices([B.design_info], pd.DataFrame({"age": np.minimum(ages + 0.5, hi)}), return_type="dataframe")[0].values
    Gm = patsy.build_design_matrices([B.design_info], pd.DataFrame({"age": np.maximum(ages - 0.5, lo)}), return_type="dataframe")[0].values
    width = np.minimum(ages + 0.5, hi) - np.maximum(ages - 0.5, lo)
    slope = 100 * ((Gp - Gm) @ b[:nb]) / width
    # spread: |residual| linear in age, x sqrt(pi/2), as %
    A = np.column_stack([np.ones(len(d)), d.age.values - 70])
    c, *_ = np.linalg.lstsq(A, np.abs(e), rcond=None)
    spread = 100 * np.sqrt(np.pi / 2) * (c[0] + c[1] * (ages - 70))
    spread = np.maximum(spread, 25 * np.sqrt(np.pi / 2) * np.abs(e).mean())
    bicv = b[nb] if licv else np.nan
    return slope, spread, bicv


def strata_atlas(d, regions, group_cols, ages_by_pool, label):
    rows = []
    for key, g in d.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        g = g.reset_index(drop=True)
        ages = np.array([a for a in ages_by_pool if g.age.min() + 3 <= a <= g.age.max() - 3], float)
        if len(g) < 30 or len(ages) == 0:
            continue
        for r in regions:
            y = g[r].values.astype(float)
            if (y <= 0).any():
                continue
            s, sp, bi = fit_curve(g, y, ages)
            boot = []
            for _ in range(NB):
                i = RNG.integers(0, len(g), len(g))
                try:
                    boot.append(fit_curve(g.iloc[i].reset_index(drop=True), y[i], ages)[:2])
                except Exception:
                    pass
            boot = np.array(boot)
            for j, a in enumerate(ages):
                Rb = boot[:, 1, j] / np.maximum(np.abs(boot[:, 0, j]), 1e-6)
                rows.append({"pool": label, **dict(zip(group_cols, key)), "n": len(g), "region": r,
                             "age": a, "slope": s[j], "slope_lo": np.percentile(boot[:, 0, j], 2.5),
                             "slope_hi": np.percentile(boot[:, 0, j], 97.5), "spread": sp[j],
                             "spread_lo": np.percentile(boot[:, 1, j], 2.5),
                             "spread_hi": np.percentile(boot[:, 1, j], 97.5),
                             "R": min(sp[j] / max(abs(s[j]), 1e-6), 999),
                             "R_lo": min(np.percentile(Rb, 2.5), 999), "R_hi": min(np.percentile(Rb, 97.5), 999),
                             "b_icv": bi})
    return pd.DataFrame(rows)


def interaction_tests(d, regions, factors, label, licv=True):
    """Does the aging slope (and the spread) depend on sex / scanner / head size?
    Nested OLS on log volume; F-test for age x factor terms; Brown-Forsythe-type test
    for spread (|residual| ~ factor). Slope differences reported in %/yr."""
    from scipy import stats
    out = []
    lo, hi = d.age.min(), d.age.max()
    sp = f"cr(age, df=4, lower_bound={lo}, upper_bound={hi})"
    base_terms = [sp, "sex"] + (["licv"] if licv else []) + ([f"C(scanner)"] if d.scanner.nunique() > 1 else [])
    for r in regions:
        dd = d.assign(ly=np.log(d[r].astype(float)), agec=(d.age - 70) / 10)
        if not np.isfinite(dd.ly).all():
            continue
        base = "ly ~ " + " + ".join(base_terms)
        y0, X0 = patsy.dmatrices(base, dd, return_type="dataframe")
        b0, *_ = np.linalg.lstsq(X0.values, y0.values.ravel(), rcond=None)
        e0 = y0.values.ravel() - X0.values @ b0
        for fac, term in factors.items():
            if fac == "scanner" and d.scanner.nunique() < 2:
                continue
            y1, X1 = patsy.dmatrices(base + f" + agec:{term}", dd, return_type="dataframe")
            b1, *_ = np.linalg.lstsq(X1.values, y1.values.ravel(), rcond=None)
            e1 = y1.values.ravel() - X1.values @ b1
            q = np.linalg.matrix_rank(X1.values) - np.linalg.matrix_rank(X0.values)
            dfe = len(dd) - np.linalg.matrix_rank(X1.values)
            F = ((e0 @ e0 - e1 @ e1) / q) / (e1 @ e1 / dfe)
            p = stats.f.sf(F, q, dfe)
            inter = [c for c in X1.columns if c.startswith("agec:")]
            raw = {c.replace("agec:", ""): 10 * b1[list(X1.columns).index(c)] for c in inter}  # %/yr
            if fac == "scanner":
                # full dummy coding (the spline carries the common slope): only differences
                # between scanners are identified -> report them around the scanner mean
                mbar = np.mean(list(raw.values()))
                coefs = {k.replace("C(scanner)[", "").rstrip("]"): v - mbar for k, v in raw.items()}
            elif fac == "head size":
                coefs = {"per SD of log ICV": raw["licv"] * dd.licv.std()}
            else:
                coefs = {"men - women": raw["sex"]}
            # spread: does |residual| differ by factor level?
            if fac == "head size":
                g = pd.qcut(dd.licv, 3, labels=["small", "mid", "large"])
            elif fac == "sex":
                g = dd.sex
            else:
                g = dd.scanner
            groups = [np.abs(e0[g.values == v]) for v in pd.unique(g)]
            pS = stats.f_oneway(*groups).pvalue if len(groups) > 1 else np.nan
            ratio = max(np.mean(x) for x in groups) / min(np.mean(x) for x in groups)
            out.append({"pool": label, "region": r, "factor": fac, "n": len(dd), "p_slope": p,
                        "slope_diff_pct_per_yr": "; ".join(f"{k}: {v:+.3f}" for k, v in coefs.items()),
                        "max_abs_slope_diff": (max(coefs.values()) - min(coefs.values())) if len(coefs) > 1 else abs(list(coefs.values())[0]),
                        "p_spread": pS, "spread_ratio_max_min": ratio})
    return pd.DataFrame(out)


def headsize_methods(d, regions, label):
    """None / proportion / residual (linear covariate) / power-law, judged on healthy data."""
    from scipy import stats
    rows = []
    for r in regions:
        y = d[r].values.astype(float)
        icv = d.icv.values
        ly, lic = np.log(y), d.licv.values
        agec = d.age.values - d.age.mean()
        Z = np.column_stack([np.ones(len(d)), agec, agec ** 2])
        if d.scanner.nunique() > 1:
            Z = np.column_stack([Z] + [(d.scanner == s).astype(float) for s in sorted(d.scanner.unique())[1:]])
        # allometric exponent b from log y ~ age + scanner + log ICV (sex NOT included: b should explain it)
        bb, *_ = np.linalg.lstsq(np.column_stack([Z, lic]), ly, rcond=None)
        b = bb[-1]
        bl, *_ = np.linalg.lstsq(np.column_stack([Z, icv - icv.mean()]), y, rcond=None)
        variants = {"none": ly, "proportion (b=1)": ly - lic, f"power-law (b={b:.2f})": ly - b * lic,
                    "residual (linear)": np.log(np.maximum(y - bl[-1] * (icv - icv.mean()), 1e-3))}
        for name, v in variants.items():
            c, *_ = np.linalg.lstsq(Z, v, rcond=None)
            res = v - Z @ c
            rows.append({"pool": label, "region": r, "method": name.split(" (")[0] if "power" not in name else "power-law",
                         "b": b, "r_with_icv": stats.pearsonr(res, lic)[0],
                         "sex_d": (res[d.sex == 1].mean() - res[d.sex == 0].mean()) / res.std(),
                         "spread_pct": 100 * res.std(), "slope_pct_yr": 100 * c[1]})
    return pd.DataFrame(rows)


def main():
    A, ra = adni_pool()
    F, rf = fs_pool()
    print("ADNI CN:", len(A), A.groupby(["sex", "scanner"]).size().to_dict())
    print("FastSurfer pool:", len(F), F.groupby(["scanner", "sex"]).size().to_dict())
    A["licv"] = np.log(A.icv) - np.log(A.icv).mean()
    A["sex_lbl"] = np.where(A.sex == 1, "men", "women")
    F["sex_lbl"] = np.where(F.sex == 1, "men", "women")

    H = pd.concat([headsize_methods(A, ra, "ADNI (eTIV)"), headsize_methods(F, rf, "FastSurfer pool (SynthStrip ICV)")])
    H.to_csv(os.path.join(OUT, "headsize_methods.csv"), index=False)

    Fy = F[F.dataset != "OASIS2"].reset_index(drop=True)          # IXI + DLBS, ages 20-89
    Fo = F[F.age >= 60].reset_index(drop=True)                      # all five scanners, ages 60+
    T = pd.concat([interaction_tests(A, ra, {"sex": "sex", "scanner": "C(scanner)", "head size": "licv"}, "ADNI"),
                   interaction_tests(F, rf, {"sex": "sex", "head size": "licv"}, "FastSurfer"),
                   interaction_tests(Fy, rf, {"scanner": "C(scanner)"}, "FastSurfer IXI+DLBS 20-89"),
                   interaction_tests(Fo, rf, {"scanner": "C(scanner)"}, "FastSurfer age>=60")])
    T.to_csv(os.path.join(OUT, "strata_tests.csv"), index=False)

    S = pd.concat([strata_atlas(A, ra, ["sex_lbl", "scanner"], [65, 70, 75, 80, 85], "ADNI"),
                   strata_atlas(F, rf, ["sex_lbl", "scanner"], [30, 40, 50, 60, 70, 75, 80], "FastSurfer")])
    # head-size tertiles (within sex, so the tertile is not just sex)
    for pool, d, regs in (("ADNI", A, ra), ("FastSurfer", F, rf)):
        d = d.copy()
        d["head"] = d.groupby("sex_lbl").licv.transform(lambda s: pd.qcut(s, 3, labels=["small head", "mid head", "large head"]))
        S = pd.concat([S, strata_atlas(d, regs, ["head"], [30, 40, 50, 60, 65, 70, 75, 80, 85], pool + " head")])
    S.to_csv(os.path.join(OUT, "strata_atlas.csv"), index=False)
    print("written", len(S), "atlas rows,", len(T), "tests,", len(H), "head-size rows")


if __name__ == "__main__":
    main()
