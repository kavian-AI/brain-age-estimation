# Phase 2: ONE pooled healthy normative model per region, all datasets together.
#
# People (cross-sectional, one scan each):
#   IXI (3 scanners), DLBS wave 1, OASIS-2 baseline nondemented   FastSurfer, SynthStrip ICV
#   ADNI CN (1.5T, 3T)                                            FreeSurfer, eTIV
# Every scanner / software combination is a "site".
#
# Location-scale model on log volume (Gaussian, fitted by iterated weighted LS):
#   mean:   log y = f(age) + b_sex*sex + b*log ICV + site offset   (+ age x sex if it earns it)
#   spread: log sigma = g(age) + site offset
# f, g natural cubic splines; site effects are sum-to-zero, so the reported curve is
# the "average scanner". b is the power-law head-size exponent (estimated).
#
# From the model, per region and age:
#   slope_cross  = 100 d f / d a                         %/yr (cross-sectional)
#   sigma_ref    = exp(g(a)) x 100                        % (average scanner); per-site too
#   S = |slope|/sigma, R = sigma/|slope|                 SD per year, years
# plus WITHIN-PERSON slopes (DLBS 3 waves + OASIS-2 visits, ICV-rescaled for scanner
# drift) -> slope_long, R_long, and measurement resolution M from OASIS-2 same-session
# repeats. Held-out (5-fold) z-scores test calibration by site, sex and age.
# Whole-brain precision floor R_brain(a) from the pooled z-scores (FastSurfer, 91 features).
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import patsy
from scipy import stats
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2_lifespan_atlas import HEMI  # noqa: E402
from p2_lifespan_data import (KAG, _read_jsonl, _synthstrip, load_dlbs, load_ixi,  # noqa: E402
                              load_oasis2, regions_from_labels)
from p2_strata import ADNI_REG, FS_REG, adni_pool  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p3_out")
os.makedirs(OUT, exist_ok=True)
AGES = np.arange(20, 91, 5).astype(float)
NB = 200
RNG = np.random.default_rng(3)
LOG_CHI = 1.2704                     # -E[log chi2_1]: log e^2 = 2 log sigma - 1.2704 + noise
SITE_NAME = {"IXI-Guys": "IXI Philips 1.5T", "IXI-HH": "IXI Philips 3T", "IXI-IOP": "IXI GE 1.5T",
             "DLBS": "DLBS Philips 3T", "OASIS2": "OASIS-2 Siemens 1.5T"}
REGIONS = list(FS_REG)               # 17 display names; 14 of them also in ADNI
# Model choice (exact Gaussian ML, p3_spread_check.py): a scanner-specific spread is
# significant in 13/17 regions (LR test; 6/17 after Bonferroni; spread ratio up to 1.5
# at age 70) and makes held-out z-scores calibrated on every scanner (SD 1.02-1.05,
# 94-95 % within +-1.96; a common spread gives 0.95-1.11). BIC prefers the common
# spread only because its log(n) penalty is harsh. An age x sex slope term earns its
# place for hippocampus and thalamus.
SITE_SD = True
AGESEX = {"hippocampus", "thalamus"}


# ------------------------------------------------------------------ data
def assemble():
    I = load_ixi()
    D = load_dlbs()
    O = load_oasis2()
    D1 = D[(D.wave == 1) & (D.run == 1)].copy()
    O1 = O[O.wave == 1].copy()
    I["site"] = I.site.map(SITE_NAME)
    D1["site"] = SITE_NAME["DLBS"]
    O1["site"] = SITE_NAME["OASIS2"]
    fs = pd.concat([I, D1, O1], ignore_index=True)
    for k, v in FS_REG.items():          # display-name copies; hemisphere columns stay
        fs[k] = fs[v]
    try:
        A, _ = adni_pool()
    except FileNotFoundError:            # ADNI/TADPOLE tables are not distributed with the repo (DUA);
        # p5_full_pool drops ADNI from the healthy atlas anyway, so continue without it
        A = pd.DataFrame({c: pd.Series(dtype="object" if c in ("subject", "scanner", "dx") else "float64")
                          for c in ["subject", "age", "sex", "icv", "dx", "scanner"] + list(ADNI_REG)})
    A["site"] = "ADNI FreeSurfer " + A.scanner
    A["dataset"] = "ADNI"
    keep = ["subject", "dataset", "site", "age", "sex", "icv"]
    cross = pd.concat([fs[keep + REGIONS + [h for h in HEMI if h not in REGIONS]], A[keep + ADNI_REG]],
                      ignore_index=True)
    cross["licv"] = np.log(cross.icv) - np.log(cross.icv).mean()
    return cross, D, O


# ------------------------------------------------------------------ model
class LS:
    """Gaussian location-scale model on log volume, iterated WLS."""

    def __init__(self, d, ly, df_mu=5, df_sd=3, agesex=False, site_sd=None, lo=18.0, hi=98.0, icv_groups=None):
        site_sd = SITE_SD if site_sd is None else site_sd
        self.lo, self.hi = lo, hi
        # head-size exponent per ICV-measuring method (a noisy method would dilute a shared b)
        self.icv_groups = icv_groups if icv_groups and "icv_method" in d else None
        self.G = len(self.icv_groups) if self.icv_groups else 1
        self.sites = sorted(d.site.unique())
        self.sx = d.sex.mean()
        self.agesex = agesex
        self.site_sd = site_sd
        self.Bm = patsy.dmatrix(f"cr(age, df={df_mu}, lower_bound={self.lo}, upper_bound={self.hi}) - 1",
                                d, return_type="dataframe").design_info
        self.Bs = patsy.dmatrix(f"cr(age, df={df_sd}, lower_bound={self.lo}, upper_bound={self.hi}) - 1",
                                d, return_type="dataframe").design_info
        Xm, Xs = self._Xm(d), self._Xs(d)
        # Gaussian maximum likelihood by alternating: weighted LS for the mean, and for
        # log sigma^2 = Xs g Fisher scoring of a gamma GLM (log link) on e^2, which is
        # exactly the Gaussian ML for the variance given the mean.
        w = np.ones(len(d))
        e = ly.copy()
        self.bs = np.zeros(Xs.shape[1])
        for _ in range(30):
            sw = np.sqrt(w)
            self.bm, *_ = np.linalg.lstsq(Xm * sw[:, None], ly * sw, rcond=None)
            e = ly - Xm @ self.bm
            if not self.bs.any():
                self.bs, *_ = np.linalg.lstsq(Xs, np.log(e ** 2 + 1e-12) + LOG_CHI, rcond=None)
            for _ in range(5):
                # log-variance bounded (sigma 0.03 %..740 %): a scanner group with a single person
                # would otherwise get variance -> 0, weight -> inf and break the least squares
                eta = np.clip(Xs @ self.bs, -16.0, 4.0)
                zz = eta + (e ** 2 / np.exp(eta) - 1.0)
                self.bs, *_ = np.linalg.lstsq(Xs, zz, rcond=None)
            w_new = 1.0 / np.exp(np.clip(Xs @ self.bs, -16.0, 4.0))
            if np.max(np.abs(w_new / w - 1)) < 1e-6:
                w = w_new
                break
            w = w_new
        sig = np.sqrt(1 / w)
        self.loglik = float(np.sum(stats.norm.logpdf(e, 0, sig)))
        self.k_params = Xm.shape[1] + Xs.shape[1]
        self.nB = len(self.Bm.column_names)
        self.site_start = self.nB + 1 + self.G

    def _site(self, d, zero=False):
        if len(self.sites) < 2:
            return np.zeros((len(d), 0))
        if zero:
            return np.zeros((len(d), len(self.sites) - 1))
        S = np.column_stack([(d.site.values == s).astype(float) for s in self.sites[1:]])
        S[d.site.values == self.sites[0]] = -1.0            # sum-to-zero (deviation) coding
        return S

    def _Xm(self, d, zero_site=False):
        B = patsy.build_design_matrices([self.Bm], d, return_type="dataframe")[0].values
        if self.icv_groups:
            L = np.column_stack([d.licv.values * (d.icv_method.values == g) for g in self.icv_groups])
        else:
            L = d.licv.values[:, None]
        cols = [B, (d.sex.values - self.sx)[:, None], L, self._site(d, zero_site)]
        if self.agesex:
            cols.append((((d.age.values - 60) / 10) * (d.sex.values - self.sx))[:, None])
        return np.column_stack(cols)

    def _Xs(self, d, zero_site=False):
        B = patsy.build_design_matrices([self.Bs], d, return_type="dataframe")[0].values
        if not self.site_sd:
            return B
        return np.column_stack([B, self._site(d, zero_site)])

    def _grid(self, ages, site=None):
        g = pd.DataFrame({"age": np.asarray(ages, float), "sex": self.sx, "licv": 0.0,
                          "site": site if site else self.sites[0]})
        if self.icv_groups:
            g["icv_method"] = self.icv_groups[0]
        return g

    def mu(self, ages):
        return self._Xm(self._grid(ages), zero_site=True) @ self.bm

    def slope(self, ages, h=0.5):            # %/yr, average scanner
        return 100 * (self.mu(np.asarray(ages) + h) - self.mu(np.asarray(ages) - h)) / (2 * h)

    def sigma(self, ages, site=None):        # %, average scanner unless site given
        g = self._grid(ages, site)
        Xs = self._Xs(g, zero_site=site is None)
        return 100 * np.sqrt(np.exp(Xs @ self.bs))

    def z(self, d, ly):
        sig = np.sqrt(np.exp(self._Xs(d) @ self.bs))
        return (ly - self._Xm(d) @ self.bm) / sig

    @property
    def b_icv(self):
        return self.bm[self.nB + 1]

    @property
    def b_sex(self):
        return 100 * self.bm[self.nB]


def region_data(cross, r):
    d = cross[cross[r].notna() & (cross[r] > 0) & cross.icv.notna()].reset_index(drop=True)
    return d, np.log(d[r].values.astype(float))


def model_choice(cross):
    rows = []
    for r in REGIONS:
        d, ly = region_data(cross, r)
        n = len(d)
        fits = {"common spread": LS(d, ly, site_sd=False), "site spread": LS(d, ly, site_sd=True),
                "site spread + age x sex": LS(d, ly, agesex=True, site_sd=True),
                "common spread + age x sex": LS(d, ly, agesex=True, site_sd=False)}
        for k, m in fits.items():
            rows.append({"region": r, "model": k, "n": n, "loglik": m.loglik, "k": m.k_params,
                         "BIC": -2 * m.loglik + m.k_params * np.log(n)})
    M = pd.DataFrame(rows)
    M["dBIC_vs_site_spread"] = M.BIC - M.groupby("region").BIC.transform(
        lambda s: s[M.loc[s.index, "model"] == "site spread"].iloc[0])
    # likelihood-ratio test: does a scanner-specific spread improve on a common one?
    lr = []
    for r, g in M.groupby("region"):
        a = g[g.model == "common spread"].iloc[0]
        b = g[g.model == "site spread"].iloc[0]
        stat = max(2 * (b.loglik - a.loglik), 0)
        lr.append({"region": r, "LR_site_spread": stat, "df": int(b.k - a.k),
                   "p_site_spread": stats.chi2.sf(stat, int(b.k - a.k))})
    return M.merge(pd.DataFrame(lr), on="region")


# ------------------------------------------------------------------ longitudinal
def within_person(D, O, regions):
    """Per-person slopes of log volume (%/yr): DLBS waves + OASIS-2 nondemented visits.
    Each scan is rescaled by person-mean ICV / scan ICV (scanner drift; the skull
    does not change)."""
    D = D[(D.run == 1) & D.icv.notna()].assign(src="DLBS")
    O = O[O.icv.notna()].assign(src="OASIS2")
    L = pd.concat([D, O], ignore_index=True)
    scale = L.groupby("subject").icv.transform("mean") / L.icv
    per = {r: [] for r in regions}
    resid = {r: [] for r in regions}
    for s, g in L.groupby("subject"):
        if len(g) < 2 or np.ptp(g.age) < 2:
            continue
        a = g.age.values.astype(float)
        A = np.column_stack([np.ones_like(a), a - a.mean()])
        for r in regions:
            if r not in g or g[r].isna().any() or (g[r] <= 0).any():
                continue
            ly = np.log(g[r].values * scale[g.index].values)
            b, *_ = np.linalg.lstsq(A, ly, rcond=None)
            per[r].append((s, a.mean(), 100 * b[1], np.ptp(a)))
            if len(a) >= 3:
                resid[r] += list(100 * (ly - A @ b))
    return per, resid


def long_curve(P, ages, boot=True):
    P = pd.DataFrame(P, columns=["subject", "mid", "slope", "span"])
    info = patsy.dmatrix("cr(mid, df=3, lower_bound=18, upper_bound=98) - 1", P).design_info

    def fit(Q):
        X = patsy.build_design_matrices([info], Q, return_type="dataframe")[0].values
        X = np.column_stack([np.ones(len(Q)), X])
        w = np.sqrt(Q.span.values ** 2)
        b, *_ = np.linalg.lstsq(X * w[:, None], Q.slope.values * w, rcond=None)
        G = patsy.build_design_matrices([info], pd.DataFrame({"mid": ages}), return_type="dataframe")[0].values
        return np.column_stack([np.ones(len(ages)), G]) @ b

    k = fit(P)
    if not boot:
        return k, None, None, len(P)
    B = np.array([fit(P.iloc[RNG.integers(0, len(P), len(P))]) for _ in range(NB)])
    return k, np.percentile(B, 2.5, axis=0), np.percentile(B, 97.5, axis=0), len(P)


def repeat_noise(regions_map):
    """OASIS-2 same-session repeats (mpr-1 vs mpr-2 of visit 1): eps in % (log units)."""
    rows = _read_jsonl(os.path.join(KAG, "oasis_fs_results", "oasis_fs_part*.jsonl"))
    by = {}
    for r in rows:
        if r["session"].endswith("_MR1"):
            by.setdefault(r["session"], {})[r["mpr"]] = regions_from_labels(r, "L")
    pairs = [(v["mpr-1"], v["mpr-2"]) for v in by.values() if "mpr-1" in v and "mpr-2" in v]
    eps = {}
    for disp, col in regions_map.items():
        dif = [np.log(b[col]) - np.log(a[col]) for a, b in pairs if a[col] > 0 and b[col] > 0]
        eps[disp] = 100 * np.std(dif, ddof=1) / np.sqrt(2)
    return eps, len(pairs)


# ------------------------------------------------------------------ atlas
def atlas(cross, D, O):
    per, resid = within_person(D.rename(columns={v: k for k, v in FS_REG.items()}),
                               O.rename(columns={v: k for k, v in FS_REG.items()}), REGIONS)
    eps_rep, npairs = repeat_noise(FS_REG)
    rows = []
    for r in REGIONS:
        d, ly = region_data(cross, r)
        m = LS(d, ly, agesex=(r in AGESEX))
        sl, sg = m.slope(AGES), m.sigma(AGES)
        bs = []
        groups = [np.where(d.site.values == s)[0] for s in m.sites]
        for _ in range(NB):
            ii = np.concatenate([g[RNG.integers(0, len(g), len(g))] for g in groups])
            try:
                mb = LS(d.iloc[ii].reset_index(drop=True), ly[ii], agesex=(r in AGESEX))
                bs.append((mb.slope(AGES), mb.sigma(AGES)))
            except Exception:
                pass
        bs = np.array(bs)
        Rb = bs[:, 1] / np.maximum(np.abs(bs[:, 0]), 1e-6)
        kl, kl_lo, kl_hi, npl = long_curve(per[r], AGES)
        eps_long = np.sqrt(np.mean(np.square(resid[r]))) if resid[r] else np.nan
        ms = LS(d, ly, agesex=(r in AGESEX), site_sd=True)        # sensitivity: per-scanner spread
        per_site = {s: ms.sigma(AGES, site=s) for s in ms.sites}
        for j, a in enumerate(AGES):
            lo_ok = a >= d.age.min() + 2 and a <= d.age.max() - 2
            row = {"region": r, "age": a, "n": len(d), "n_sites": len(m.sites), "in_range": lo_ok,
                   "b_icv": m.b_icv, "sex_offset_pct": m.b_sex,
                   "slope_cross": sl[j], "slope_cross_lo": np.percentile(bs[:, 0, j], 2.5),
                   "slope_cross_hi": np.percentile(bs[:, 0, j], 97.5),
                   "sigma": sg[j], "sigma_lo": np.percentile(bs[:, 1, j], 2.5),
                   "sigma_hi": np.percentile(bs[:, 1, j], 97.5),
                   "S_cross": abs(sl[j]) / sg[j], "R_cross": min(sg[j] / max(abs(sl[j]), 1e-6), 999),
                   "R_cross_lo": min(np.percentile(Rb[:, j], 2.5), 999),
                   "R_cross_hi": min(np.percentile(Rb[:, j], 97.5), 999),
                   "slope_long": kl[j], "slope_long_lo": kl_lo[j], "slope_long_hi": kl_hi[j], "n_long": npl,
                   "R_long": min(sg[j] / max(abs(kl[j]), 1e-6), 999),
                   "R_long_lo": min(sg[j] / max(abs(kl_lo[j]), abs(kl_hi[j]), 1e-6), 999),
                   "R_long_hi": (min(sg[j] / max(min(abs(kl_lo[j]), abs(kl_hi[j])), 1e-6), 999)
                                 if np.sign(kl_lo[j]) == np.sign(kl_hi[j]) else 999),
                   "eps_repeat": eps_rep.get(r, np.nan), "eps_long": eps_long,
                   "M_repeat": min(eps_rep.get(r, np.nan) / max(abs(kl[j]), 1e-6), 999)}
            for s in ms.sites:
                row[f"sigma[{s}]"] = per_site[s][j]
            rows.append(row)
        j70 = list(AGES).index(70.0)
        print(f"  {r:22s} n={len(d)} sites={len(m.sites)} b={m.b_icv:.2f} | at 70: slope {sl[j70]:+.2f} "
              f"long {kl[j70]:+.2f} %/yr, sigma {sg[j70]:.1f} %, R_cross {sg[j70]/abs(sl[j70]):.1f} "
              f"R_long {sg[j70]/abs(kl[j70]):.1f} y", flush=True)
    return pd.DataFrame(rows), npairs


def calibration(cross):
    out = []
    folds = pd.Series(cross.subject.unique()).sample(frac=1, random_state=0).reset_index(drop=True)
    fold_of = {s: i % 5 for i, s in enumerate(folds)}
    for r in REGIONS:
        d, ly = region_data(cross, r)
        f = d.subject.map(fold_of).values
        z = np.full(len(d), np.nan)
        for k in range(5):
            tr, te = f != k, f == k
            m = LS(d[tr].reset_index(drop=True), ly[tr], agesex=(r in AGESEX))
            z[te] = m.z(d[te].reset_index(drop=True), ly[te])
        d = d.assign(z=z, region=r)
        out.append(d[["region", "site", "sex", "age", "z"]])
    Z = pd.concat(out)
    Z["band"] = pd.cut(Z.age, [17, 40, 60, 75, 100], labels=["18-40", "40-60", "60-75", "75+"])
    summ = lambda g: pd.Series({"n": len(g), "mean_z": g.z.mean(), "sd_z": g.z.std(),
                                "within_1": np.mean(np.abs(g.z) < 1), "within_1.96": np.mean(np.abs(g.z) < 1.96),
                                "skew": stats.skew(g.z), "excess_kurt": stats.kurtosis(g.z)})
    res = {"all": summ(Z)}
    tabs = {k: Z.groupby(k).apply(summ) for k in ("site", "sex", "band", "region")}
    return res, tabs


def floor(cross, D, O):
    """Whole-brain precision floor from pooled z-scores (FastSurfer sites, 91 features)."""
    fsx = cross[cross.dataset != "ADNI"].reset_index(drop=True)
    feats = [h for h in HEMI if h in fsx and (fsx[h] > 0).all()]
    per, _ = within_person(D, O, feats)
    Z = np.zeros((len(fsx), len(feats)))
    S_cross, S_long = [], []
    for j, h in enumerate(feats):
        ly = np.log(fsx[h].values.astype(float))
        m = LS(fsx, ly)
        Z[:, j] = m.z(fsx, ly)
        sg = m.sigma(AGES) / 100
        S_cross.append(m.slope(AGES) / 100 / sg)
        kl, *_ = long_curve(per[h], AGES, boot=False)
        S_long.append(kl / 100 / sg)
    S_cross, S_long = np.array(S_cross), np.array(S_long)       # feats x ages
    rows = []
    for j, a in enumerate(AGES):
        w = np.abs(fsx.age.values - a) <= 7.5
        if w.sum() < 60:
            continue
        Zw = Z[w]
        Si = np.linalg.pinv(LedoitWolf().fit(Zw).covariance_)
        row = {"age": a, "n_window": int(w.sum()), "p": len(feats)}
        for nm, s in (("cross", S_cross[:, j]), ("long", S_long[:, j])):
            info = s @ Si @ s
            fold = RNG.permutation(len(Zw)) % 5
            g = np.zeros(len(Zw))
            for f in range(5):
                Sif = np.linalg.pinv(LedoitWolf().fit(Zw[fold != f]).covariance_)
                g[fold == f] = (Zw[fold == f] @ Sif @ s) / (s @ Sif @ s)
            row[f"R_brain_{nm}"] = 1 / np.sqrt(info)
            row[f"SD_g_heldout_{nm}"] = float(np.std(g))
            row[f"best_single_R_{nm}"] = float(np.min(1 / np.abs(s)))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    cross, D, O = assemble()
    print("pooled healthy people:", len(cross))
    print(cross.groupby("site").agg(n=("age", "size"), age_min=("age", "min"), age_max=("age", "max"),
                                    men=("sex", "mean")).round(2).to_string())
    M = model_choice(cross)
    M.to_csv(os.path.join(OUT, "model_choice.csv"), index=False)
    print("\nmedian dBIC vs 'site spread' (negative = better):")
    print(M.groupby("model").dBIC_vs_site_spread.median().round(1).to_string())
    A, npairs = atlas(cross, D, O)
    A.to_csv(os.path.join(OUT, "pooled_atlas.csv"), index=False)
    res, tabs = calibration(cross)
    for k, t in tabs.items():
        t.to_csv(os.path.join(OUT, f"calibration_by_{k}.csv"))
    print("\nheld-out calibration, all:", res["all"].round(3).to_dict())
    print(tabs["site"].round(3).to_string())
    print(tabs["band"].round(3).to_string())
    F = floor(cross, D, O)
    F.to_csv(os.path.join(OUT, "pooled_floor.csv"), index=False)
    print("\nfloor:\n", F.round(2).to_string(index=False))
    json.dump({"n_people": len(cross), "sites": cross.site.value_counts().to_dict(), "oasis_repeat_pairs": npairs},
              open(os.path.join(OUT, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
