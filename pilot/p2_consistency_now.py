# Consistency checks possible before the Kaggle data arrive.
#
# Same lifespan model (p2_lifespan_atlas.LNorm: spline mean + sex + head size,
# spline spread) on every subset, so differences are data, not method:
#   1. dataset:   ADNI CN (FreeSurfer, multi-site) vs OASIS-2 nondemented
#                 (FastSurfer, one Siemens 1.5T)
#   2. scanner:   ADNI CN scanned at 1.5T vs at 3T
#   3. sex:       ADNI CN men vs women
#   4. head size: with vs without the ICV covariate (how much of sigma is head size)
#   5. scanner covariate: field strength added to the mean model
# Reported per region at 70 / 75 / 80: %/yr, spread as % of mean, resolution R,
# each with a subject-bootstrap 95% interval.
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p1_definitions as P1  # noqa: E402
import tadpole_atlas as TA  # noqa: E402
from p2_lifespan_atlas import LNorm  # noqa: E402
from p2_lifespan_data import load_oasis2  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p2_out")
os.makedirs(OUT, exist_ok=True)
AGES = np.array([70.0, 75.0, 80.0])
NB = 300
RNG = np.random.default_rng(1)
REG = {  # display name: (ADNI composite, OASIS-2 region)
    "hippocampus": ("hippocampus", "hippocampus"),
    "entorhinal": ("entorhinal", "entorhinal"),
    "amygdala": ("amygdala", "amygdala"),
    "lateral ventricles": ("lateral ventricles", "lateral_ventricles"),
    "inf. lat. ventricles": ("inf. lat. ventricles", "inf_lat_ventricles"),
    "thalamus": ("thalamus", "thalamus"),
    "putamen": ("putamen", "putamen"),
    "brainstem": ("brainstem", "brainstem"),
    "cerebellum cortex": ("cerebellum cortex", "cerebellum_cortex"),
    "precuneus": ("precuneus", "precuneus"),
}


def adni():
    m, T, names, comp = P1.load()
    t = pd.read_csv(TA.TADPOLE, low_memory=False,
                    usecols=["PTID", "VISCODE", "FLDSTRENG_bl"])
    t = t[t.VISCODE == "bl"].drop_duplicates("PTID")
    fs = dict(zip(t.PTID, t["FLDSTRENG_bl"].astype(str)))
    d = pd.DataFrame({"subject": m.PTID.values, "age": m.age_visit.values.astype(float),
                      "sex": m.SEX.values.astype(float), "icv": m.ICV_T.values.astype(float),
                      "site": [fs.get(p, "nan") for p in m.PTID], "dx": m.DX.values,
                      "fsver": m.FSVERSION_T.values if "FSVERSION_T" in m else "?"})
    for k, (a, _) in REG.items():
        d[k] = comp[a]
    d = d[(d.dx == "CN")].reset_index(drop=True)
    fl = pd.to_numeric(d.site.str.extract(r"([0-9.]+)")[0], errors="coerce")
    d["site"] = np.where(fl < 2, "1.5T", np.where(fl >= 2, "3T", "unknown"))
    return d


def oasis():
    o = load_oasis2()
    o = o[o.wave == 1].reset_index(drop=True)
    d = o[["subject", "age", "sex", "icv", "site"]].copy()
    for k, (_, b) in REG.items():
        d[k] = o[b].values
    return d


def curves(d, r, site=False, use_icv=True):
    dd = d if use_icv else d.assign(icv=0.0)
    m = LNorm(dd, dd[r].values.astype(float), df_mu=3, df_sd=3, site=site)
    mu, k, s = m.mu(AGES), m.k(AGES), m.sigma(AGES)
    return 100 * k / mu, 100 * s / mu, s / np.maximum(np.abs(k), 1e-9)


def summarise(d, label, site=False, use_icv=True):
    rows = []
    n = len(d)
    for r in REG:
        pct, sdp, R = curves(d, r, site, use_icv)
        B = []
        for _ in range(NB):
            i = RNG.integers(0, n, n)
            try:
                B.append(curves(d.iloc[i].reset_index(drop=True), r, site, use_icv))
            except Exception:
                pass
        B = np.array(B)          # NB x 3 x ages
        for j, a in enumerate(AGES):
            rows.append({"set": label, "n": n, "region": r, "age": a,
                         "pct_yr": pct[j], "pct_lo": np.percentile(B[:, 0, j], 2.5),
                         "pct_hi": np.percentile(B[:, 0, j], 97.5),
                         "sd_pct": sdp[j], "sd_lo": np.percentile(B[:, 1, j], 2.5),
                         "sd_hi": np.percentile(B[:, 1, j], 97.5),
                         "R": min(R[j], 999), "R_lo": min(np.percentile(B[:, 2, j], 2.5), 999),
                         "R_hi": min(np.percentile(B[:, 2, j], 97.5), 999)})
    return rows


def main():
    A = adni()
    O = oasis()
    print("ADNI CN", len(A), "field strength:", A.site.value_counts().to_dict(),
          "| men", int(A.sex.sum()), "women", int((1 - A.sex).sum()))
    print("OASIS-2 nondemented baseline", len(O))
    rows = []
    rows += summarise(A, "ADNI all")
    rows += summarise(O, "OASIS-2")
    rows += summarise(A[A.site == "1.5T"].reset_index(drop=True), "ADNI 1.5T")
    rows += summarise(A[A.site == "3T"].reset_index(drop=True), "ADNI 3T")
    rows += summarise(A[A.sex == 1].reset_index(drop=True).assign(sex=0.0), "ADNI men")
    rows += summarise(A[A.sex == 0].reset_index(drop=True), "ADNI women")
    rows += summarise(A, "ADNI no-ICV", use_icv=False)
    rows += summarise(A[A.site != "unknown"].reset_index(drop=True), "ADNI +field", site=True)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, "consistency_now.csv"), index=False)
    fmt = lambda v, lo, hi, p: f"{v:.{p}f} [{lo:.{p}f}, {hi:.{p}f}]"
    for a in (75.0,):
        t = D[D.age == a]
        for col, lo, hi, p, title in (("pct_yr", "pct_lo", "pct_hi", 2, "aging rate, %/yr"),
                                      ("sd_pct", "sd_lo", "sd_hi", 1, "spread sigma, % of mean"),
                                      ("R", "R_lo", "R_hi", 1, "resolution R, years")):
            tab = t.assign(v=[fmt(x, y, z, p) for x, y, z in zip(t[col], t[lo], t[hi])]).pivot(
                index="region", columns="set", values="v")
            print(f"\n== {title} at age {a:.0f} ==")
            print(tab[[c for c in ["ADNI all", "OASIS-2", "ADNI 1.5T", "ADNI 3T", "ADNI men", "ADNI women",
                                   "ADNI no-ICV", "ADNI +field"] if c in tab]].to_string())


if __name__ == "__main__":
    main()
