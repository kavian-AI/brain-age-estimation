# Phase 1 quantities across the adult lifespan, per dataset, in FastSurfer space.
#
# For every dataset separately (IXI, DLBS wave 1, OASIS-2 baseline) and region r:
#   mu_r(a)    natural cubic spline in age + sex + head size (+ site for IXI)
#   sigma_r(a) spread: |residual| on a spline in age, x sqrt(pi/2)
#   k_r(a)     d mu / d a (cross-sectional aging rate), %/yr = 100 k / mu
#   S_r = |k| / sigma (SD per year), R_r = sigma / |k| (years)
# with subject-bootstrap CIs, so curves can be compared between datasets.
# DLBS adds within-person rates (2-3 waves over ~9 years, same scanner) across 21-97,
# the lifespan version of the ADNI result that within-person aging accelerates
# while cross-sectional slopes flatten. Same-session second runs give epsilon.
# The whole-brain precision floor R_brain(a) = 1/sqrt(s' Sigma^-1 s) is computed in
# age windows.
import json
import os
import sys

import numpy as np
import pandas as pd
import patsy
from sklearn.covariance import LedoitWolf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2_lifespan_data import BILAT, DKT, SINGLE, load_dlbs, load_ixi, load_oasis2

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p2_out")
os.makedirs(OUT, exist_ok=True)
FEAT = ["hippocampus", "amygdala", "entorhinal", "parahippocampal", "lateral_ventricles",
        "inf_lat_ventricles", "third_ventricle", "thalamus", "putamen", "caudate", "pallidum",
        "accumbens", "ventral_dc", "brainstem", "cerebellum_cortex", "cerebellum_wm", "cerebral_wm",
        "cortex", "precuneus", "superiorfrontal", "rostralmiddlefrontal", "lateraloccipital",
        "precentral", "postcentral", "insula", "superiortemporal", "middletemporal", "fusiform",
        "inferiorparietal", "posteriorcingulate", "total_brain"]
HEMI = [p + n for n in list(BILAT) + list(DKT.values()) if n not in ("choroid_plexus",)
        for p in ("L_", "R_")] + ["third_ventricle", "fourth_ventricle", "brainstem"]
NBOOT = 200
RNG = np.random.default_rng(0)


class LNorm:
    """Lifespan normative model for one region."""

    def __init__(self, d, y, df_mu=4, df_sd=3, site=False):
        self.lo, self.hi = d.age.min(), d.age.max()
        self.sx, self.ic = d.sex.mean(), d.icv.mean()
        self.site_levels = sorted(d.site.unique()) if site else None
        self.f_mu = f"cr(age, df={df_mu}, lower_bound={self.lo}, upper_bound={self.hi})"
        self.f_sd = f"cr(age, df={df_sd}, lower_bound={self.lo}, upper_bound={self.hi})"
        X = self._X(d)
        self.b, *_ = np.linalg.lstsq(X, y, rcond=None)
        e = y - X @ self.b
        Xs = patsy.dmatrix(self.f_sd, d, return_type="dataframe").values
        self.Xs_info = patsy.dmatrix(self.f_sd, d).design_info
        self.d_, *_ = np.linalg.lstsq(Xs, np.abs(e), rcond=None)
        self.floor = 0.25 * np.sqrt(np.pi / 2) * np.abs(e).mean()

    def _X(self, d):
        if not hasattr(self, "B_info"):
            B = patsy.dmatrix(self.f_mu, d, return_type="dataframe")
            self.B_info = B.design_info
        else:
            B = patsy.build_design_matrices([self.B_info], d, return_type="dataframe")[0]
        cols = [B.values, (d.sex.values - self.sx)[:, None], (d.icv.values - self.ic)[:, None]]
        if self.site_levels and len(self.site_levels) > 1:
            cols.append(np.column_stack([(d.site.values == s).astype(float) - np.mean(d.site.values == s)
                                         for s in self.site_levels[1:]]))
        return np.column_stack(cols)

    def _grid(self, ages):
        g = pd.DataFrame({"age": np.asarray(ages, float), "sex": self.sx, "icv": self.ic,
                          "site": self.site_levels[0] if self.site_levels else "x"})
        return g

    def mu(self, ages):
        g = self._grid(ages)
        X = self._X(g)
        if self.site_levels and len(self.site_levels) > 1:
            X[:, -(len(self.site_levels) - 1):] = 0.0      # site-averaged curve
        return X @ self.b

    def mu_at(self, d):
        return self._X(d) @ self.b

    def sigma(self, ages):
        g = self._grid(ages)
        Xs = patsy.build_design_matrices([self.Xs_info], g, return_type="dataframe")[0].values
        return np.maximum(np.sqrt(np.pi / 2) * (Xs @ self.d_), self.floor)

    def k(self, ages, h=0.5):
        a = np.clip(np.asarray(ages, float), self.lo + h, self.hi - h)
        return (self.mu(a + h) - self.mu(a - h)) / (2 * h)

    def z(self, d, y):
        return (y - self.mu_at(d)) / self.sigma(d.age.values)


def grid_for(d):
    lo, hi = d.age.min(), d.age.max()
    return np.array([a for a in range(25, 91, 5) if lo + 3 <= a <= hi - 3], float)


def atlas(d, name, site=False):
    ages = grid_for(d)
    rows = []
    idx = np.arange(len(d))
    subj = d.subject.values
    us = np.unique(subj)
    for r in FEAT:
        y = d[r].values.astype(float)
        if np.nanstd(y) == 0:
            continue
        m = LNorm(d, y, site=site)
        mu, sd, k = m.mu(ages), m.sigma(ages), m.k(ages)
        boot = {"R": [], "pct": [], "sd": []}
        for _ in range(NBOOT):
            pick = RNG.choice(us, len(us))
            ii = np.concatenate([idx[subj == s] for s in pick]) if len(us) < len(d) else \
                RNG.choice(idx, len(idx))
            db = d.iloc[ii].reset_index(drop=True)
            try:
                mb = LNorm(db, y[ii], site=site)
            except Exception:
                continue
            kb, sb, mub = mb.k(ages), mb.sigma(ages), mb.mu(ages)
            boot["R"].append(sb / np.maximum(np.abs(kb), 1e-9))
            boot["pct"].append(100 * kb / mub)
            boot["sd"].append(100 * sb / mub)
        for j, a in enumerate(ages):
            R = sd[j] / max(abs(k[j]), 1e-9)
            bR = np.array([b[j] for b in boot["R"]])
            bp = np.array([b[j] for b in boot["pct"]])
            rows.append({"dataset": name, "region": r, "age": a, "mu": mu[j], "sigma": sd[j],
                         "sd_pct": 100 * sd[j] / mu[j], "k": k[j], "pct_per_yr": 100 * k[j] / mu[j],
                         "pct_lo": np.percentile(bp, 2.5), "pct_hi": np.percentile(bp, 97.5),
                         "S": abs(k[j]) / sd[j], "R": min(R, 999),
                         "R_lo": min(np.percentile(bR, 2.5), 999), "R_hi": min(np.percentile(bR, 97.5), 999),
                         "n": len(d)})
    return pd.DataFrame(rows)


def floor_by_age(d, name, half=8.0, site=False):
    """R_brain(a) and the held-out spread of the aging-like deviation g, in age windows."""
    feats = [f for f in HEMI if f in d.columns and d[f].std() > 0]
    Z = np.zeros((len(d), len(feats)))
    models = []
    for j, f in enumerate(feats):
        m = LNorm(d, d[f].values.astype(float), site=site)
        Z[:, j] = m.z(d, d[f].values.astype(float))
        models.append(m)
    out = []
    for a in grid_for(d):
        w = np.abs(d.age.values - a) <= half
        if w.sum() < 40:
            continue
        s = np.array([m.k([a])[0] / m.sigma([a])[0] for m in models])
        Zw = Z[w]
        Si = np.linalg.pinv(LedoitWolf().fit(Zw).covariance_)
        info = s @ Si @ s
        # held-out spread of g: Sigma estimated without the person (5 folds)
        fold = RNG.permutation(len(Zw)) % 5
        g = np.zeros(len(Zw))
        for f in range(5):
            Sif = np.linalg.pinv(LedoitWolf().fit(Zw[fold != f]).covariance_)
            g[fold == f] = (Zw[fold == f] @ Sif @ s) / (s @ Sif @ s)
        out.append({"dataset": name, "age": a, "n_window": int(w.sum()), "p": len(feats),
                    "R_brain": 1 / np.sqrt(info), "SD_g_heldout": float(np.std(g)),
                    "best_single_R": float(np.min(np.abs(1 / s)))})
    return pd.DataFrame(out)


def dlbs_longitudinal(dl):
    """Within-person rates from DLBS waves (run 1 only).

    The skull does not change in adults, yet SynthStrip ICV drifts by ~-0.15 %/yr
    within DLBS people at every age and independently of their brain change: a
    geometric scale drift of the scanner between waves. It affects every volume by
    the same factor, so each scan is rescaled by (person's mean ICV / scan ICV)."""
    d = dl[(dl.run == 1) & dl.icv.notna()].copy()
    scale = d.groupby("subject").icv.transform("mean") / d.icv
    for r in FEAT:
        d[r] = d[r] * scale
    cnt = d.groupby("subject").size()
    d = d[d.subject.isin(cnt[cnt >= 2].index)]
    rows = []
    for r in FEAT:
        per = []
        resid = []
        for s, g in d.groupby("subject"):
            a, y = g.age.values.astype(float), g[r].values.astype(float)
            if np.ptp(a) < 2:
                continue
            A = np.column_stack([np.ones_like(a), a - a.mean()])
            b, *_ = np.linalg.lstsq(A, y, rcond=None)
            per.append((a.mean(), b[1], np.ptp(a), y.mean()))
            if len(a) >= 3:
                resid += list(y - A @ b)
        P = pd.DataFrame(per, columns=["mid", "slope", "span", "ybar"])
        F = pd.DataFrame({"age": P.mid})
        lo, hi = P.mid.min(), P.mid.max()
        X = patsy.dmatrix(f"cr(age, df=3, lower_bound={lo}, upper_bound={hi})", F, return_type="dataframe")
        W = P.span.values ** 2
        b, *_ = np.linalg.lstsq(X.values * np.sqrt(W)[:, None], P.slope.values * np.sqrt(W), rcond=None)
        ages = np.array([a for a in range(30, 91, 5) if lo + 2 <= a <= hi - 2], float)
        Xg = patsy.build_design_matrices([X.design_info], pd.DataFrame({"age": ages}),
                                         return_type="dataframe")[0].values
        kl = Xg @ b
        # level at each age, from the people themselves (for %/yr)
        Xm = patsy.build_design_matrices([X.design_info], F, return_type="dataframe")[0].values
        bm, *_ = np.linalg.lstsq(Xm, P.ybar.values, rcond=None)
        lev = Xg @ bm
        eps = np.sqrt(np.sum(np.square(resid)) / max(len(resid) - 2 * len(P), 1)) if resid else np.nan
        for j, a in enumerate(ages):
            rows.append({"region": r, "age": a, "k_long": kl[j], "pct_long": 100 * kl[j] / lev[j],
                         "eps_long": eps, "n_people": len(P)})
    return pd.DataFrame(rows)


def repeat_noise(dl):
    """Same-session second MPRAGE runs (DLBS): epsilon = SD(diff)/sqrt(2)."""
    rep = dl[dl.run == 2]
    rows = []
    for _, r2 in rep.iterrows():
        r1 = dl[(dl.subject == r2.subject) & (dl.wave == r2.wave) & (dl.run == 1)]
        if len(r1):
            rows.append((r2, r1.iloc[0]))
    out = {}
    for f in FEAT:
        dif = np.array([a[f] - b[f] for a, b in rows])
        out[f] = float(np.std(dif, ddof=1) / np.sqrt(2)) if len(dif) > 2 else np.nan
    return out, len(rows)


def main():
    sets = {}
    for name, fn in (("IXI", load_ixi), ("DLBS", load_dlbs), ("OASIS2", load_oasis2)):
        try:
            d = fn()
        except Exception as e:
            print(name, "not available:", type(e).__name__, e)
            continue
        if len(d) == 0:
            print(name, "no rows yet")
            continue
        if name == "DLBS":
            sets["DLBS_long"] = d
            d = d[(d.wave == 1) & (d.run == 1)]
        if name == "OASIS2":
            d = d[d.wave == 1]
        d = d[d.icv.notna()].reset_index(drop=True)
        sets[name] = d
        print(f"{name}: {len(d)} people, age {d.age.min():.0f}-{d.age.max():.0f}")
    tabs, floors = [], []
    for name in ("IXI", "DLBS", "OASIS2"):
        if name in sets:
            tabs.append(atlas(sets[name], name, site=(name == "IXI")))
            floors.append(floor_by_age(sets[name], name, site=(name == "IXI")))
    A = pd.concat(tabs)
    A.to_csv(os.path.join(OUT, "lifespan_atlas.csv"), index=False)
    Fl = pd.concat(floors)
    Fl.to_csv(os.path.join(OUT, "lifespan_floor.csv"), index=False)
    print(Fl.round(2).to_string(index=False))
    show = ["hippocampus", "entorhinal", "lateral_ventricles", "thalamus", "putamen", "brainstem",
            "cerebellum_cortex", "cortex", "total_brain"]
    piv = A[A.region.isin(show)].pivot_table(index=["region", "dataset"], columns="age", values="R")
    print("\nResolution R (years) by age\n", piv.round(1).to_string())
    piv = A[A.region.isin(show)].pivot_table(index=["region", "dataset"], columns="age", values="pct_per_yr")
    print("\nCross-sectional %/yr by age\n", piv.round(2).to_string())
    if "DLBS_long" in sets:
        L = dlbs_longitudinal(sets["DLBS_long"])
        L.to_csv(os.path.join(OUT, "dlbs_longitudinal.csv"), index=False)
        C = A[A.dataset == "DLBS"][["region", "age", "k", "sigma", "pct_per_yr"]]
        M = L.merge(C, on=["region", "age"])
        M["ratio"] = M.k_long / M.k
        M["R_cross"] = M.sigma / M.k.abs()
        M["R_long"] = M.sigma / M.k_long.abs()
        M["M_long"] = M.eps_long / M.k_long.abs()
        M.to_csv(os.path.join(OUT, "dlbs_long_vs_cross.csv"), index=False)
        print("\nDLBS within-person %/yr\n",
              M[M.region.isin(show)].pivot_table(index="region", columns="age", values="pct_long").round(2).to_string())
        print("\nmedian k_long/k_cross by age\n", M.groupby("age").ratio.median().round(2).to_string())
        print("\nmedian R_cross / R_long by age\n",
              M.groupby("age")[["R_cross", "R_long"]].median().round(1).to_string())
        eps, n = repeat_noise(sets["DLBS_long"])
        json.dump({"n_pairs": n, "eps": eps}, open(os.path.join(OUT, "dlbs_repeat_noise.json"), "w"), indent=1)
        print("\nDLBS same-session pairs:", n)


if __name__ == "__main__":
    main()
