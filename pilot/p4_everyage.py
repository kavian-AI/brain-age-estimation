# Year-by-year normative tables, as wide in age as the data allow.
#
# Pooled healthy reference = p3 pool (ADNI, DLBS, IXI x3, OASIS-2; ages 20-93) +
# young controls already on disk (ABIDE-I controls, ADHD-200 typically developing;
# FreeSurfer 5.x tables, ages 6.5-56) as extra sites -> the curves reach age 7.
# Model: p3_pooled.LS (Gaussian location-scale, exact ML; site-specific spread).
#
# For every region and every single year of age the table holds:
#   reference volume for a woman and for a man with the reference head size on the
#   average scanner, centiles 2.5/5/25/50/75/95/97.5 (mm3), spread (%),
#   cross-sectional and within-person aging rate (%/yr), sensitivity S, resolution R,
#   measurement resolution M; plus a second file with the per-site shift and spread
#   factor and the head-size exponent, i.e. everything needed to score a new person.
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3_pooled as P  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p4_out")
os.makedirs(OUT, exist_ok=True)
YOUTH_REG = {"hippocampus": ("Left-Hippocampus", "Right-Hippocampus"),
             "amygdala": ("Left-Amygdala", "Right-Amygdala"),
             "lateral ventricles": ("Left-Lateral-Ventricle", "Right-Lateral-Ventricle"),
             "inf. lat. ventricles": ("Left-Inf-Lat-Vent", "Right-Inf-Lat-Vent"),
             "third ventricle": ("3rd-Ventricle",),
             "thalamus": ("Left-Thalamus-Proper", "Right-Thalamus-Proper"),
             "putamen": ("Left-Putamen", "Right-Putamen"),
             "brainstem": ("Brain-Stem",),
             "cerebellum cortex": ("Left-Cerebellum-Cortex", "Right-Cerebellum-Cortex"),
             "cortex": ("Cortex",), "cerebral WM": ("CorticalWhiteMatter",)}
CENT = [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]


def youth():
    a = pd.read_csv(os.path.join(ROOT, "data", "abide_merged.csv"))
    b = pd.read_csv(os.path.join(ROOT, "data", "adhd200_merged.csv"))
    A = pd.DataFrame({"subject": "ABIDE" + a.SUB_ID.astype(str), "dataset": "ABIDE",
                      "site": "ABIDE " + a.SITE_ID.astype(str), "age": a.AGE_AT_SCAN.astype(float),
                      "sex": (a.SEX == 1).astype(float),
                      "icv": a.EstimatedTotalIntraCranialVol.fillna(a.IntraCranialVol)})[(a.DX_GROUP == 2).values]
    for k, cols in YOUTH_REG.items():
        A[k] = sum(a[c].astype(float) for c in cols)[(a.DX_GROUP == 2).values]
    B = pd.DataFrame({"subject": "ADHD" + b.subject.astype(str), "dataset": "ADHD200",
                      "site": "ADHD200 " + b.Site.astype(str), "age": pd.to_numeric(b.Age, errors="coerce"),
                      "sex": (b.Gender == 1).astype(float),
                      "icv": b.EstimatedTotalIntraCranialVol.fillna(b.IntraCranialVol)})[(b.DX.astype(str) == "0").values]
    for k, cols in YOUTH_REG.items():
        B[k] = sum(b[c].astype(float) for c in cols)[(b.DX.astype(str) == "0").values]
    Y = pd.concat([A, B], ignore_index=True)
    Y["icv_method"] = "FreeSurfer5 eTIV (youth)"
    Y = Y[(Y.icv > 0) & Y.age.notna()].copy()
    small = Y.site.map(Y.site.value_counts()) < 20                       # tiny sites pooled per study
    Y.loc[small, "site"] = Y.loc[small, "dataset"] + " other sites"
    return Y


def main():
    cross, D, O = P.assemble()
    Y = youth()
    cross = cross.copy()
    cross["icv_method"] = np.where(cross.dataset == "ADNI", "FreeSurfer eTIV (ADNI)", "SynthStrip")
    allp = pd.concat([cross, Y], ignore_index=True)
    groups = ["SynthStrip", "FreeSurfer eTIV (ADNI)", "FreeSurfer5 eTIV (youth)"]
    allp["licv"] = np.log(allp.icv) - np.log(allp.icv).mean()
    ref_icv = float(np.exp(np.log(allp.icv).mean()))
    print(f"people: {len(allp)} ({len(cross)} adult pool + {len(Y)} young controls); sites {allp.site.nunique()}; "
          f"ages {allp.age.min():.1f}-{allp.age.max():.1f}; reference head {ref_icv/1000:.0f} cm3")
    per, resid = P.within_person(D.rename(columns={v: k for k, v in P.FS_REG.items()}),
                                 O.rename(columns={v: k for k, v in P.FS_REG.items()}), P.REGIONS)
    eps_rep, _ = P.repeat_noise(P.FS_REG)
    rows, params = [], []
    for r in P.REGIONS:
        d, ly = P.region_data(allp, r)
        gr = [g for g in groups if (d.icv_method == g).any()]
        young = d.age.min() < 15          # curve flexibility matched to the region's age span
        m = P.LS(d, ly, agesex=(r in P.AGESEX), df_mu=7 if young else 5, df_sd=4 if young else 3,
                 lo=5.0 if young else 18.0, hi=98.0, icv_groups=gr)
        lo, hi = np.ceil(d.age.quantile(0.002)), np.floor(d.age.quantile(0.998))
        ages = np.arange(max(lo, 6), min(hi, 95) + 1, 1.0)
        sl = m.slope(ages)
        # reference spread at each age = average (log scale) over the scanners that actually
        # scanned people of that age, weighted by how many (within +-5 y); otherwise the
        # many young-only FreeSurfer-5 sites would set the "average scanner" at age 80 too
        nS = len(m.Bs.column_names)
        so = list(m.bs[nS:nS + len(m.sites) - 1])
        so = np.array([-sum(so)] + so)
        W = np.array([[((d.site.values == s) & (np.abs(d.age.values - a) <= 5)).sum() for s in m.sites] for a in ages], float)
        W = W / np.maximum(W.sum(1, keepdims=True), 1)
        sg = (m.sigma(ages) / 100) * np.exp((W @ so) / 2)
        if len(per[r]) > 30:
            pl = pd.DataFrame(per[r], columns=["s", "mid", "slope", "span"])
            lo_l, hi_l = pl.mid.min() + 2, pl.mid.max() - 2
            kl_all, *_ = P.long_curve(per[r], ages, boot=False)
            kl = np.where((ages >= lo_l) & (ages <= hi_l), kl_all, np.nan)
        else:
            kl = np.full(len(ages), np.nan)
        n_near = [int((np.abs(d.age.values - a) <= 2.5).sum()) for a in ages]
        for sex, lab in ((0.0, "women"), (1.0, "men")):
            g = pd.DataFrame({"age": ages, "sex": sex, "licv": 0.0, "site": m.sites[0], "icv_method": gr[0]})
            mu = m._Xm(g, zero_site=True) @ m.bm
            for j, a in enumerate(ages):
                row = {"region": r, "sex": lab, "age": a, "n_within_2.5y": n_near[j],
                       "spread_pct": 100 * sg[j], "slope_cross": sl[j], "slope_long": kl[j],
                       "S_cross": abs(sl[j]) / (100 * sg[j]), "R_cross": min(100 * sg[j] / max(abs(sl[j]), 1e-6), 999),
                       "R_long": min(100 * sg[j] / max(abs(kl[j]), 1e-6), 999) if np.isfinite(kl[j]) else np.nan,
                       "M_repeat": min(eps_rep.get(r, np.nan) / max(abs(kl[j]), 1e-6), 999) if np.isfinite(kl[j]) else np.nan}
                for c in CENT:
                    row[f"c{c*100:g}"] = float(np.exp(mu[j] + sg[j] * stats.norm.ppf(c)))
                rows.append(row)
        nB = m.nB
        offs = list(m.bm[m.site_start:m.site_start + len(m.sites) - 1])
        offs = [-sum(offs)] + offs
        nS = len(m.Bs.column_names)
        soffs = list(m.bs[nS:nS + len(m.sites) - 1])
        soffs = [-sum(soffs)] + soffs
        icvm = dict(zip(d.site, d.icv_method))
        bstr = ", ".join(f"{g.split()[0]} {m.bm[nB + 1 + i]:.2f}" for i, g in enumerate(gr))
        for s, o, so in zip(m.sites, offs, soffs):
            params.append({"region": r, "site": s, "n": int((d.site == s).sum()),
                           "mean_shift_pct": 100 * (np.exp(o) - 1), "spread_factor": float(np.exp(so / 2)),
                           "head_exponent_b": float(m.bm[nB + 1 + gr.index(icvm[s])]) if s in icvm else np.nan,
                           "sex_coef_log": m.b_sex / 100})
        print(f"  {r:22s} n={len(d):5d} sites={len(m.sites):2d} ages {ages[0]:.0f}-{ages[-1]:.0f} b: {bstr}", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "everyage_table.csv"), index=False)
    Pm = pd.DataFrame(params)
    Pm.to_csv(os.path.join(OUT, "everyage_site_params.csv"), index=False)
    json.dump({"reference_head_cm3": ref_icv / 1000, "n_people": len(allp), "sites": int(allp.site.nunique()),
               "age_range": [float(allp.age.min()), float(allp.age.max())]},
              open(os.path.join(OUT, "everyage_meta.json"), "w"), indent=1)
    show = T[(T.sex == "women") & T.age.isin([8, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 60, 70, 80, 89])]
    print(show.pivot(index="region", columns="age", values="R_cross").round(0).to_string())


if __name__ == "__main__":
    main()
