#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PHASE 1 — every definition computed on real data (no rules / thresholds yet).

Healthy reference: ADNI/TADPOLE cognitively normal adults (98 FreeSurfer regions).
Measurement noise: OASIS-2 FastSurfer same-session repeat scans.

Per region r, at age a (healthy only; covariates sex + ICV held at their mean):
  D1  mu_r(a), sigma_r(a)          normative mean / spread (quadratic mean, linear spread)
  D2  z_r                          (y - mu)/sigma
  D3  k_r(a) = d mu / da            healthy aging rate
  D4  S_r(a) = |k| / sigma          aging sensitivity, SD per year
  D5  R_r(a) = sigma / |k|          age resolution, years
  D6  M_r = eps / |k|, ICC          measurement resolution, reliability
  D7  G_r = z * sigma / k           regional age gap (years)
Whole brain (out of fold):
  D9  s(a) = k/sigma                aging direction
  D10 Sigma                         healthy covariance of z (Ledoit-Wolf)
  D11 g = s'S^-1 z / s'S^-1 s       aging-like deviation (years)
  D12 T2 = z'S^-1 z - g^2 s'S^-1 s  atypicality
  D13 R_brain = 1/sqrt(s'S^-1 s)    precision floor
Checks:
  P2 identity, P3 floor + shrinkage slope beta = V/(V+R^2), P4 predictability
  artefact of per-region predicted age with one shared correction, P6 classic
  bias-corrected brain-age gap ~ g.
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp53_units_and_dimension as E  # noqa: E402
import exp62_rich_atlas as R  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402
from sklearn.linear_model import LinearRegression, RidgeCV  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p1_out")
os.makedirs(OUT, exist_ok=True)
AGES = np.array([65.0, 70.0, 75.0, 80.0, 85.0])
C0 = 74.0          # centring age
NBOOT = 300
FOLDS = 5


# --------------------------------------------------------------- normative fit
class Norm:
    """y = b0 + b1 c + b2 c^2 + b3 sex + b4 icv + e ;  |e| = d0 + d1 c ;  c = age - C0"""

    def __init__(self, y, age, sex, icv):
        c = age - C0
        self.sx, self.ic = sex.mean(), icv.mean()
        X = np.column_stack([np.ones_like(c), c, c ** 2, sex - self.sx, icv - self.ic])
        self.b, *_ = np.linalg.lstsq(X, y, rcond=None)
        e = y - X @ self.b
        Xs = np.column_stack([np.ones_like(c), c])
        self.d, *_ = np.linalg.lstsq(Xs, np.abs(e), rcond=None)
        self.floor = 0.25 * np.sqrt(np.pi / 2) * np.abs(e).mean()

    def mu(self, age, sex=None, icv=None):
        c = np.asarray(age, float) - C0
        v = self.b[0] + self.b[1] * c + self.b[2] * c ** 2
        if sex is not None:
            v = v + self.b[3] * (sex - self.sx) + self.b[4] * (icv - self.ic)
        return v

    def sigma(self, age):
        c = np.asarray(age, float) - C0
        return np.maximum(np.sqrt(np.pi / 2) * (self.d[0] + self.d[1] * c), self.floor)

    def k(self, age):
        c = np.asarray(age, float) - C0
        return self.b[1] + 2 * self.b[2] * c

    def z(self, y, age, sex, icv):
        return (y - self.mu(age, sex, icv)) / self.sigma(age)


def fit_all(Y, age, sex, icv):
    return [Norm(Y[:, j], age, sex, icv) for j in range(Y.shape[1])]


# --------------------------------------------------------------- data
def load():
    a, m, T, names = R.build("X")
    m = m.copy()
    feats = dict((n, m[n].values.astype(float)) for n in names)
    composites = {
        "hippocampus": ["L_hippocampus", "R_hippocampus"],
        "entorhinal": ["L_entorhinal", "R_entorhinal"],
        "amygdala": ["L_amygdala", "R_amygdala"],
        "lateral ventricles": ["L_latvent", "R_latvent"],
        "inf. lat. ventricles": ["L_inflatvent", "R_inflatvent"],
        "third ventricle": ["thirdventricle"],
        "brainstem": ["brainstem"],
        "cerebellum cortex": ["L_cerebellumcortex", "R_cerebellumcortex"],
        "thalamus": ["L_thalamus", "R_thalamus"],
        "putamen": ["L_putamen", "R_putamen"],
        "precentral": ["L_precentral", "R_precentral"],
        "superior frontal": ["L_superiorfrontal", "R_superiorfrontal"],
        "lateral occipital": ["L_lateraloccipital", "R_lateraloccipital"],
        "precuneus": ["L_precuneus", "R_precuneus"],
        "cc central": ["cc_central"],
    }
    comp = {k: sum(feats[n] for n in v) for k, v in composites.items()}
    return m, T, names, comp


def table_regions(Yd, age, sex, icv, cn, label):
    """D1, D3, D4, D5 at each age in AGES, with bootstrap CIs."""
    rng = np.random.RandomState(0)
    idx = np.where(cn)[0]
    rows = []
    for name, y in Yd.items():
        nm = Norm(y[cn], age[cn], sex[cn], icv[cn])
        boots = []
        for _ in range(NBOOT):
            bi = rng.choice(idx, len(idx), replace=True)
            nb = Norm(y[bi], age[bi], sex[bi], icv[bi])
            boots.append([nb.k(AGES), nb.sigma(AGES), nb.mu(AGES)])
        boots = np.array(boots)  # B x 3 x A
        kb, sb, mb = boots[:, 0], boots[:, 1], boots[:, 2]
        Rb = sb / np.maximum(np.abs(kb), 1e-12)
        for j, A in enumerate(AGES):
            k, s, mu = nm.k(A), nm.sigma(A), nm.mu(A)
            rows.append(dict(
                region=name, age=A, mu=mu, sigma=s, k=k,
                pct_per_year=100 * k / mu, S=abs(k) / s, R=s / max(abs(k), 1e-12),
                R_lo=np.percentile(Rb[:, j], 2.5), R_hi=np.percentile(Rb[:, j], 97.5),
                pct_lo=np.percentile(100 * kb[:, j] / mb[:, j], 2.5),
                pct_hi=np.percentile(100 * kb[:, j] / mb[:, j], 97.5)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"region_table_{label}.csv"), index=False)
    return df


# --------------------------------------------------------------- OASIS repeats
def oasis_repeat_noise():
    import exp66_oasis_external as O
    Ro, _, _ = O.load_fs()
    p = Ro[Ro.mpr == "mpr-1"].merge(Ro[Ro.mpr == "mpr-2"], on="session", suffixes=("_1", "_2"))
    cols = [c for c in Ro.columns if c not in ("session", "mpr")]
    eps = {}
    for c in cols:
        d = p[c + "_1"].values - p[c + "_2"].values
        eps[c] = np.std(d, ddof=1) / np.sqrt(2)
    return eps, len(p)


# --------------------------------------------------------------- whole brain
def whole_brain(m, names, cn_mask):
    Y = np.column_stack([m[n].values.astype(float) for n in names])
    age = m.age_visit.values.astype(float)
    sex = m.SEX.values.astype(float)
    icv = m.ICV_T.values.astype(float)
    f = E.folds_of(m.PTID.values)
    n, p = Y.shape
    g = np.full(n, np.nan)
    T2 = np.full(n, np.nan)
    D2 = np.full(n, np.nan)
    Rb = np.full(n, np.nan)
    gap = np.full(n, np.nan)
    raw_pred = np.full(n, np.nan)
    beta_obs, beta_pred, Vtr = [], [], []
    for k in range(FOLDS):
        tr = np.where((f != k) & cn_mask)[0]
        te = np.where(f == k)[0]
        norms = fit_all(Y[tr], age[tr], sex[tr], icv[tr])
        Ztr = np.column_stack([nm.z(Y[tr, j], age[tr], sex[tr], icv[tr]) for j, nm in enumerate(norms)])
        Zte = np.column_stack([nm.z(Y[te, j], age[te], sex[te], icv[te]) for j, nm in enumerate(norms)])
        lw = LedoitWolf().fit(Ztr)
        Si = np.linalg.inv(lw.covariance_)
        S_te = np.column_stack([nm.k(age[te]) / nm.sigma(age[te]) for nm in norms])  # n_te x p
        for i, ii in enumerate(te):
            s = S_te[i]
            q = s @ Si @ s
            zz = Zte[i]
            gi = (s @ Si @ zz) / q
            g[ii] = gi
            Rb[ii] = 1 / np.sqrt(q)
            D2[ii] = zz @ Si @ zz
            T2[ii] = D2[ii] - gi ** 2 * q
        # classic brain age: ridge on raw features + icv + sex, healthy-trained
        X = np.column_stack([Y, icv, sex])
        sc = StandardScaler().fit(X[tr])
        md = RidgeCV(alphas=np.logspace(-2, 5, 40)).fit(sc.transform(X[tr]), age[tr])
        ptr = md.predict(sc.transform(X[tr]))
        lb = LinearRegression().fit(age[tr].reshape(-1, 1), ptr)
        b1, b0 = lb.coef_[0], lb.intercept_
        pte = md.predict(sc.transform(X[te]))
        raw_pred[te] = pte
        gap[te] = (pte - b0) / b1 - age[te]          # standard correction (stretch)
        beta_obs.append(b1)
        Vtr.append(np.var(age[tr]))
    return dict(g=g, T2=T2, D2=D2, Rb=Rb, gap=gap, raw_pred=raw_pred, age=age,
                beta_obs=np.array(beta_obs), V=np.array(Vtr), p=p)


# --------------------------------------------------------------- P4 artefact
def regional_predicted_age(m, names, cn_mask):
    """Per-region age predictor (healthy-trained), one shared 'semiglobal' correction."""
    Y = np.column_stack([m[n].values.astype(float) for n in names])
    age = m.age_visit.values.astype(float)
    sex = m.SEX.values.astype(float)
    icv = m.ICV_T.values.astype(float)
    f = E.folds_of(m.PTID.values)
    n, p = Y.shape
    app = np.full((n, p), np.nan)      # apparent regional gap after shared correction
    betas = np.zeros((FOLDS, p))
    mtr_all = []
    for k in range(FOLDS):
        tr = np.where((f != k) & cn_mask)[0]
        te = np.where(f == k)[0]
        mtr = age[tr].mean()
        mtr_all.append(mtr)
        preds_te = np.zeros((len(te), p))
        for j in range(p):
            Xtr = np.column_stack([Y[tr, j], sex[tr], icv[tr]])
            lr = LinearRegression().fit(Xtr, age[tr])
            ptr = lr.predict(Xtr)
            betas[k, j] = np.polyfit(age[tr], ptr, 1)[0]
            preds_te[:, j] = lr.predict(np.column_stack([Y[te, j], sex[te], icv[te]]))
        bbar = np.median(betas[k])
        app[te] = mtr + (preds_te - mtr) / bbar - age[te][:, None]
    return app, betas.mean(0), float(np.mean(mtr_all))


def main():
    m, T, names, comp = load()
    age = m.age_visit.values.astype(float)
    sex = m.SEX.values.astype(float)
    icv = m.ICV_T.values.astype(float)
    cn = (m.DX == "CN").values
    res = {}

    # ---- D1-D5 for featured regions (bootstrap) and all 98 regions (point)
    feat = table_regions(comp, age, sex, icv, cn, "featured")
    allr = table_regions({n: m[n].values.astype(float) for n in names}, age, sex, icv, cn, "all98") \
        if NBOOT <= 0 else None
    # all 98 regions without bootstrap (fast)
    rows = []
    for n in names:
        nm = Norm(m[n].values[cn].astype(float), age[cn], sex[cn], icv[cn])
        for A in AGES:
            k, s, mu = nm.k(A), nm.sigma(A), nm.mu(A)
            rows.append(dict(region=n, cls=T.set_index("name").cls[n], age=A, mu=mu, sigma=s,
                             k=k, pct_per_year=100 * k / mu, S=abs(k) / s, R=s / max(abs(k), 1e-12)))
    all98 = pd.DataFrame(rows)
    all98.to_csv(os.path.join(OUT, "region_table_all98.csv"), index=False)

    # ---- D6 measurement noise (OASIS same-session repeats)
    eps, npairs = oasis_repeat_noise()
    a75 = all98[all98.age == 75].set_index("region")
    mrows = []
    for r_, e_ in eps.items():
        if r_ in a75.index:
            row = a75.loc[r_]
            mrows.append(dict(region=r_, eps=e_, sigma=row.sigma, k=row.k,
                              M=e_ / max(abs(row.k), 1e-12), R=row.R,
                              ICC=1 - e_ ** 2 / row.sigma ** 2))
    meas = pd.DataFrame(mrows)
    meas.to_csv(os.path.join(OUT, "measurement_resolution.csv"), index=False)
    res["oasis_pairs"] = npairs

    # ---- whole brain, out of fold
    wb = whole_brain(m, names, cn)
    np.savez(os.path.join(OUT, "whole_brain.npz"), **{k: v for k, v in wb.items() if isinstance(v, np.ndarray)},
             DX=m.DX.values, PTID=m.PTID.values)
    gcn, gapcn, Rcn = wb["g"][cn], wb["gap"][cn], wb["Rb"][cn]
    res["R_brain_mean_CN"] = float(np.sqrt(np.mean(Rcn ** 2)))
    res["R_brain_by_age"] = {f"{lo}-{hi}": float(np.sqrt(np.mean(Rcn[(age[cn] >= lo) & (age[cn] < hi)] ** 2)))
                             for lo, hi in [(60, 70), (70, 75), (75, 80), (80, 91)]}
    res["SD_g_CN_empirical"] = float(np.std(gcn, ddof=1))
    res["SD_classic_gap_CN"] = float(np.std(gapcn, ddof=1))
    res["MAE_raw_ridge_CN"] = float(np.mean(np.abs(wb["raw_pred"][cn] - age[cn])))
    res["MAE_floor_0.798R"] = float(0.798 * res["R_brain_mean_CN"])
    Vbar = float(np.mean(wb["V"]))
    res["beta_obs_ridge"] = float(np.mean(wb["beta_obs"]))
    res["beta_pred_V/(V+R2)_floor"] = float(Vbar / (Vbar + res["R_brain_mean_CN"] ** 2))
    res["beta_pred_V/(V+SDg2)_empirical"] = float(Vbar / (Vbar + res["SD_g_CN_empirical"] ** 2))
    res["corr_g_classicgap_CN"] = float(np.corrcoef(gcn, gapcn)[0, 1])
    res["corr_g_classicgap_all"] = float(np.corrcoef(wb["g"], wb["gap"])[0, 1])
    # P2 identity check
    q = 1 / wb["Rb"] ** 2
    res["P2_max_abs_error"] = float(np.nanmax(np.abs(wb["D2"] - (wb["g"] ** 2 * q + wb["T2"]))))
    # by diagnosis
    from scipy.stats import chi2
    p = wb["p"]
    dx = {}
    for d in ["CN", "MCI", "Dementia"]:
        s_ = (m.DX == d).values
        frac_aging = (wb["g"][s_] ** 2 * q[s_]) / wb["D2"][s_]
        dx[d] = dict(n=int(s_.sum()), g_mean=float(np.mean(wb["g"][s_])),
                     classic_gap_mean=float(np.mean(wb["gap"][s_])),
                     T2_centile_median=float(np.median(chi2.cdf(wb["T2"][s_], p - 1))),
                     D2_centile_median=float(np.median(chi2.cdf(wb["D2"][s_], p))),
                     aging_share_of_D2_median=float(np.median(frac_aging)))
    res["by_dx"] = dx
    sel = m.DX.isin(["CN", "Dementia"]).values
    y = (m.DX.values[sel] == "Dementia").astype(int)
    res["AUC_AD_vs_CN"] = {k: float(roc_auc_score(y, v[sel])) for k, v in
                           [("g", wb["g"]), ("T2", wb["T2"]), ("D2", wb["D2"]), ("classic_gap", wb["gap"])]}

    # ---- P4 predictability artefact
    app, betas, mtr = regional_predicted_age(m, names, cn)
    old = cn & (age > mtr)
    young = cn & (age <= mtr)
    art = pd.DataFrame(dict(region=names, beta=betas,
                            app_gap_old=np.nanmean(app[old], 0),
                            app_gap_young=np.nanmean(app[young], 0)))
    a74 = all98[all98.age == 75].set_index("region").loc[names]
    art["R75"] = a74.R.values
    art.to_csv(os.path.join(OUT, "p4_artefact.csv"), index=False)
    res["P4_train_mean_age"] = mtr
    res["P4_corr_beta_vs_appgap_old"] = float(np.corrcoef(art.beta, art.app_gap_old)[0, 1])
    res["P4_corr_beta_vs_appgap_young"] = float(np.corrcoef(art.beta, art.app_gap_young)[0, 1])
    res["P4_mean_age_old"] = float(age[old].mean())
    res["P4_mean_age_young"] = float(age[young].mean())
    Vcn = float(np.var(age[cn]))
    art["beta_pred"] = Vcn / (Vcn + art.R75 ** 2)
    res["P4_corr_beta_obs_vs_pred"] = float(np.corrcoef(art.beta, art.beta_pred)[0, 1])

    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))
    print("\nFEATURED REGIONS (age 75):")
    print(feat[feat.age == 75][["region", "mu", "sigma", "k", "pct_per_year", "S", "R", "R_lo", "R_hi"]]
          .round(3).to_string(index=False))
    print("\nFEATURED REGIONS R by age:")
    print(feat.pivot(index="region", columns="age", values="R").round(1).to_string())
    print("\nMEASUREMENT (featured-ish, sorted by M):")
    print(meas.sort_values("M").round(3).head(15).to_string(index=False))
    print(meas.sort_values("M").round(3).tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
