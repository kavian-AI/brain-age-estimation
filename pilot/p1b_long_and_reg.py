#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PHASE 1b.
(a) Within-person aging rates from repeated scans of stably healthy ADNI adults
    (TADPOLE longitudinal FreeSurfer pipeline), compared with the cross-sectional
    rates of p1_definitions.py; between-session noise from residuals around each
    person's line.
(b) Regularised whole-brain aging direction: Sigma_lam = (Sigma + lam I)/(1+lam),
    lam swept; empirical (held-out healthy) calibration of atypicality T2.
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
import p1_definitions as P  # noqa: E402
import tadpole_atlas as TA  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

OUT = P.OUT
AGES = P.AGES
CONSISTENT = True


# ------------------------------------------------------------------ (a)
def load_long(pipe="L"):
    T = TA.region_table("CV")
    suf = TA.PIPE[pipe]
    need = [f"ST{c}{k}_{suf}" for c, k in zip(T.code, T.kind)] + [f"ST10CV_{TA.PIPE['X']}"]
    vcol = "VERSION_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16"
    fcol = "FLDSTRENG_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16"
    d = pd.read_csv(TA.TADPOLE, low_memory=False,
                    usecols=["PTID", "VISCODE", vcol, fcol] + list(dict.fromkeys(need)))
    out = pd.DataFrame({"PTID": d.PTID.values, "VISCODE": d.VISCODE.values,
                        "PIPEVER": d[vcol].astype(str).values, "FIELD": d[fcol].astype(str).values})
    for (c, k, nm) in zip(T.code, T.kind, T.name):
        out[nm] = pd.to_numeric(d[f"ST{c}{k}_{suf}"], errors="coerce").values
    out["ICV_T"] = pd.to_numeric(d[f"ST10CV_{TA.PIPE['X']}"], errors="coerce").values
    a, _ = E.load()
    a = a.dropna(subset=["AGE", "Years_bl", "PTGENDER", "DX"]).copy()
    a["age_visit"] = a.AGE + a.Years_bl
    a["SEX"] = (a.PTGENDER == "Male").astype(int)
    a = a[["PTID", "VISCODE", "age_visit", "SEX", "DX"]].drop_duplicates(["PTID", "VISCODE"])
    m = out.merge(a, on=["PTID", "VISCODE"], how="inner")
    names = list(T.name)
    m = m.dropna(subset=names + ["age_visit"]).reset_index(drop=True)
    # stably healthy: CN at every visit
    stable = m.groupby("PTID").DX.agg(lambda s: (s == "CN").all())
    m = m[m.PTID.map(stable)]
    # one scanner field strength and one pipeline version per person: keep the
    # largest consistent block of visits (ADNI1 -> ADNIGO/2 moved many people 1.5T -> 3T)
    if CONSISTENT:
        m = m.copy()
        m["blk"] = m.FIELD + "|" + m.PIPEVER
        best = m.groupby(["PTID", "blk"]).size().reset_index(name="nv").sort_values("nv").drop_duplicates("PTID", keep="last")
        m = m.merge(best[["PTID", "blk"]], on=["PTID", "blk"], how="inner")
    span = m.groupby("PTID").age_visit.agg(lambda s: s.max() - s.min())
    nvis = m.groupby("PTID").size()
    keep = span[(span >= 2.0) & (nvis >= 3)].index
    m = m[m.PTID.isin(keep)].sort_values(["PTID", "age_visit"]).reset_index(drop=True)
    return m, T, names


def longitudinal_rates(m, names):
    """Per person OLS slope; slope ~ 1 + (baseline age - C0), weighted by span^2;
    between-session noise = pooled residual SD around each person's line."""
    rows = []
    g = m.groupby("PTID")
    base_age = g.age_visit.min()
    span = g.age_visit.agg(lambda s: s.max() - s.min())
    ids = base_age.index.values
    for n in names:
        res2, dof = 0.0, 0
        sub = m[["PTID", "age_visit", n]]
        slopes = {}
        for pid, d in sub.groupby("PTID"):
            x, y = d.age_visit.values, d[n].values.astype(float)
            c = np.polyfit(x, y, 1)
            slopes[pid] = c[0]
            r = y - np.polyval(c, x)
            res2 += (r ** 2).sum()
            dof += len(x) - 2
        s = np.array([slopes[i] for i in ids])
        A = base_age.loc[ids].values + span.loc[ids].values / 2      # mid-follow-up age
        w = span.loc[ids].values ** 2
        X = np.column_stack([np.ones_like(A), A - P.C0])
        W = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(X * W[:, None], s * W, rcond=None)
        eps = np.sqrt(res2 / max(dof, 1))
        rows.append(dict(region=n, k0=coef[0], k1=coef[1], eps_long=eps, n=len(ids)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ (b)
def whole_brain_reg(m, names, cn_mask, lams):
    Y = np.column_stack([m[n].values.astype(float) for n in names])
    age = m.age_visit.values.astype(float)
    sex = m.SEX.values.astype(float)
    icv = m.ICV_T.values.astype(float)
    f = E.folds_of(m.PTID.values)
    n, p = Y.shape
    res = {lam: dict(g=np.full(n, np.nan), T2=np.full(n, np.nan), Rb=np.full(n, np.nan)) for lam in lams}
    for k in range(P.FOLDS):
        tr = np.where((f != k) & cn_mask)[0]
        te = np.where(f == k)[0]
        norms = P.fit_all(Y[tr], age[tr], sex[tr], icv[tr])
        Ztr = np.column_stack([nm.z(Y[tr, j], age[tr], sex[tr], icv[tr]) for j, nm in enumerate(norms)])
        Zte = np.column_stack([nm.z(Y[te, j], age[te], sex[te], icv[te]) for j, nm in enumerate(norms)])
        C = LedoitWolf().fit(Ztr).covariance_
        S_te = np.column_stack([nm.k(age[te]) / nm.sigma(age[te]) for nm in norms])
        for lam in lams:
            Cl = (C + lam * np.eye(p)) / (1 + lam) if np.isfinite(lam) else np.eye(p)
            Si = np.linalg.inv(Cl)
            q = np.einsum("ij,jk,ik->i", S_te, Si, S_te)
            num = np.einsum("ij,jk,ik->i", S_te, Si, Zte)
            gi = num / q
            D2 = np.einsum("ij,jk,ik->i", Zte, Si, Zte)
            res[lam]["g"][te] = gi
            res[lam]["Rb"][te] = 1 / np.sqrt(q)
            res[lam]["T2"][te] = D2 - gi ** 2 * q
    return res


def ecdf_centile(ref, x):
    ref = np.sort(ref)
    return np.searchsorted(ref, x, side="right") / len(ref)


def main():
    out = {}
    # ---------------- (a)
    ml, T, names = load_long("L")
    rates = longitudinal_rates(ml, names)
    out["long_subjects"] = int(ml.PTID.nunique())
    out["long_visits"] = int(len(ml))
    out["long_median_span"] = float(ml.groupby("PTID").age_visit.agg(lambda s: s.max() - s.min()).median())
    cross = pd.read_csv(os.path.join(OUT, "region_table_all98.csv"))
    rows = []
    for _, r in rates.iterrows():
        for A in AGES:
            cr = cross[(cross.region == r.region) & (cross.age == A)].iloc[0]
            kl = r.k0 + r.k1 * (A - P.C0)
            rows.append(dict(region=r.region, cls=cr.cls, age=A, k_cross=cr.k, k_long=kl,
                             sigma=cr.sigma, mu=cr.mu, R_cross=cr.R,
                             R_long=cr.sigma / max(abs(kl), 1e-12),
                             pct_cross=100 * cr.k / cr.mu, pct_long=100 * kl / cr.mu,
                             eps_long=r.eps_long, M_long=r.eps_long / max(abs(kl), 1e-12)))
    lr = pd.DataFrame(rows)
    lr.to_csv(os.path.join(OUT, "longitudinal_vs_cross.csv"), index=False)
    for A in AGES:
        d = lr[lr.age == A]
        out[f"age{int(A)}_median_ratio_klong_over_kcross"] = float(np.median(d.k_long / d.k_cross))
        out[f"age{int(A)}_corr_klong_kcross"] = float(np.corrcoef(d.k_long, d.k_cross)[0, 1])
        out[f"age{int(A)}_median_R_cross"] = float(d.R_cross.median())
        out[f"age{int(A)}_median_R_long"] = float(d.R_long.median())
        out[f"age{int(A)}_min_R_long"] = float(d.R_long.min())
    hip = lr[lr.region.isin(["L_hippocampus", "R_hippocampus"])].groupby("age")[["pct_cross", "pct_long", "R_cross", "R_long", "M_long"]].mean()
    out["hippocampus_by_age"] = hip.round(3).to_dict()
    lv = lr[lr.region.isin(["L_latvent", "R_latvent"])].groupby("age")[["pct_cross", "pct_long", "R_cross", "R_long", "M_long"]].mean()
    out["latvent_by_age"] = lv.round(3).to_dict()

    # ---------------- (b)
    m, T, names, comp = P.load()
    cn = (m.DX == "CN").values
    lams = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf]
    res = whole_brain_reg(m, names, cn, lams)
    wb_old = np.load(os.path.join(OUT, "whole_brain.npz"), allow_pickle=True)
    gap = wb_old["gap"]
    sel = m.DX.isin(["CN", "Dementia"]).values
    y = (m.DX.values[sel] == "Dementia").astype(int)
    rows = []
    for lam in lams:
        g, T2, Rb = res[lam]["g"], res[lam]["T2"], res[lam]["Rb"]
        cent = np.array([ecdf_centile(np.delete(T2[cn], np.where(np.where(cn)[0] == i)[0]) if cn[i] else T2[cn], T2[i])
                         for i in range(len(T2))])
        rows.append(dict(lam=lam,
                         R_theory_CN=float(np.sqrt(np.mean(Rb[cn] ** 2))),
                         SD_g_CN=float(np.std(g[cn], ddof=1)),
                         calib_ratio=float(np.std(g[cn], ddof=1) / np.sqrt(np.mean(Rb[cn] ** 2))),
                         AUC_g=float(roc_auc_score(y, g[sel])),
                         AUC_T2=float(roc_auc_score(y, T2[sel])),
                         corr_g_gap=float(np.corrcoef(g, gap)[0, 1]),
                         g_MCI=float(np.mean(g[(m.DX == "MCI").values])),
                         g_AD=float(np.mean(g[(m.DX == "Dementia").values])),
                         T2cent_CN=float(np.median(cent[cn])),
                         T2cent_MCI=float(np.median(cent[(m.DX == "MCI").values])),
                         T2cent_AD=float(np.median(cent[(m.DX == "Dementia").values]))))
        if lam == 1.0:
            np.savez(os.path.join(OUT, "whole_brain_reg_lam1.npz"), g=g, T2=T2, Rb=Rb, T2cent=cent,
                     DX=m.DX.values, gap=gap, age=m.age_visit.values)
    reg = pd.DataFrame(rows)
    reg.to_csv(os.path.join(OUT, "regularisation_sweep.csv"), index=False)
    out["classic_gap_AUC"] = float(roc_auc_score(y, gap[sel]))
    json.dump(out, open(os.path.join(OUT, "summary_p1b.json"), "w"), indent=2, default=float)
    print(json.dumps(out, indent=1, default=float))
    print(reg.round(3).to_string(index=False))
    show = lr[lr.region.isin(["L_hippocampus", "R_hippocampus", "L_latvent", "L_inflatvent", "brainstem",
                              "L_putamen", "L_entorhinal", "L_precuneus", "L_thalamus", "L_cerebellumcortex"])]
    print(show.pivot_table(index="region", columns="age", values=["pct_cross", "pct_long"]).round(2).to_string())


if __name__ == "__main__":
    main()
