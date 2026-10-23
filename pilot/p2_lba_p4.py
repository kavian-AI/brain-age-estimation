# Test of P4 on the Irimia-lab local brain age (PNAS 2026) model, IXI healthy adults.
#
# P4: if each region's "age" is a shrunken prediction (slope beta_r < 1 toward the
# training mean) and one shared correction is applied to all regions, then in
# HEALTHY people the apparent regional gap is
#       gap_r - gap_brain  ~  (beta_r - beta_bar) * (age - a0)
# so regions that track age well (high beta_r) look "older" in people above a0 and
# "younger" below it, and the ranking flips at one common age a0 (the training mean).
#
# Inputs (Kaggle ixi-lba-p0/p1): per-person mean LBA over the brain and per
# FastSurfer label (cap_L<label>), plus the uncapped-intensity variant (raw_*) on a
# third of the scans.
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2_lifespan_data import BILAT, DKT, SINGLE, load_ixi

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p2_out")
os.makedirs(OUT, exist_ok=True)


def label_names():
    n = {}
    for k, (l, r) in BILAT.items():
        n[l], n[r] = "L_" + k, "R_" + k
    for k, l in SINGLE.items():
        n[l] = k
    for k, v in DKT.items():
        n[1000 + k], n[2000 + k] = "L_" + v, "R_" + v
    return n


def main():
    d = load_ixi()
    names = label_names()
    print(f"IXI with LBA: {len(d)} people, age {d.age.min():.0f}-{d.age.max():.0f}")
    res = {"n": len(d)}
    # 1. global LBA
    g = d.cap_mean.values
    a = d.age.values
    gap = g - a
    b = np.polyfit(a, g, 1)
    res["global"] = {"MAE": float(np.mean(np.abs(gap))), "mean_gap": float(gap.mean()),
                     "r": float(np.corrcoef(a, g)[0, 1]), "slope": float(b[0]), "intercept": float(b[1]),
                     "corr_gap_age": float(np.corrcoef(a, gap)[0, 1]),
                     "within_brain_sd_median": float(d.cap_sd.median())}
    by_site = d.assign(gap=gap).groupby("site").agg(n=("gap", "size"), mean_gap=("gap", "mean"),
                                                    MAE=("gap", lambda x: np.mean(np.abs(x))))
    res["by_site"] = by_site.round(2).to_dict()
    if "raw_mean" in d:
        w = d.raw_mean.notna()
        res["cap_vs_raw"] = {"n": int(w.sum()),
                             "mean_diff": float((d.raw_mean[w] - d.cap_mean[w]).mean()),
                             "r": float(np.corrcoef(d.raw_mean[w], d.cap_mean[w])[0, 1]),
                             "MAE_raw": float(np.mean(np.abs(d.raw_mean[w] - d.age[w]))),
                             "MAE_cap": float(np.mean(np.abs(d.cap_mean[w] - d.age[w])))}
    # 2. regional LBA: slope on age, crossing age of the relative gap
    rows = []
    for col in [c for c in d.columns if c.startswith("cap_L")]:
        lab = int(col[5:])
        if lab not in names:
            continue
        y = d[col].values
        ok = np.isfinite(y)
        if ok.sum() < 0.9 * len(d):
            continue
        br = np.polyfit(a[ok], y[ok], 1)
        rel = y[ok] - g[ok]                      # regional minus whole-brain LBA
        c = np.polyfit(a[ok], rel, 1)            # rel = c0 * age + c1
        rows.append({"label": lab, "region": names[lab], "beta_r": br[0], "icpt_r": br[1],
                     "rel_slope": c[0], "rel_icpt": c[1],
                     "cross_age": -c[1] / c[0] if abs(c[0]) > 1e-4 else np.nan,
                     "rel_young": float(rel[a[ok] < 40].mean()), "rel_old": float(rel[a[ok] > 65].mean()),
                     "r_age": float(np.corrcoef(a[ok], y[ok])[0, 1])})
    R = pd.DataFrame(rows).sort_values("beta_r")
    R.to_csv(os.path.join(OUT, "ixi_lba_regional.csv"), index=False)
    beta_bar = np.median(R.beta_r)
    res["regional"] = {
        "n_regions": len(R), "beta_median": float(beta_bar),
        "beta_range": [float(R.beta_r.min()), float(R.beta_r.max())],
        # P4: relative slope should equal beta_r - beta_global
        "corr_relslope_vs_beta": float(np.corrcoef(R.rel_slope, R.beta_r - b[0])[0, 1]),
        "relslope_on_betadiff": [float(v) for v in np.polyfit(R.beta_r - b[0], R.rel_slope, 1)],
        "cross_age_median": float(np.nanmedian(R.cross_age)),
        "cross_age_IQR": [float(np.nanpercentile(R.cross_age, 25)), float(np.nanpercentile(R.cross_age, 75))],
        # sign flip: across regions, corr(beta_r, relative gap) in young vs old people
        "corr_beta_relgap_young": float(np.corrcoef(R.beta_r, R.rel_young)[0, 1]),
        "corr_beta_relgap_old": float(np.corrcoef(R.beta_r, R.rel_old)[0, 1]),
    }
    # 3. per age band: correlation across regions between beta_r and mean relative gap
    bands = []
    for lo in range(20, 86, 10):
        w = (a >= lo) & (a < lo + 10)
        if w.sum() < 15:
            continue
        relg = [np.nanmean(d.loc[w, f"cap_L{l}"].values - g[w]) for l in R.label]
        bands.append({"band": f"{lo}-{lo+9}", "n": int(w.sum()),
                      "corr_beta_relgap": float(np.corrcoef(R.beta_r, relg)[0, 1]),
                      "spread_relgap_sd": float(np.std(relg))})
    res["bands"] = bands
    json.dump(res, open(os.path.join(OUT, "ixi_lba_p4.json"), "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
    show = R[R.region.str.contains("hippocampus|ventricles|putamen|thalamus|brainstem|cerebellum_cortex|precuneus|superiorfrontal|lateraloccipital|entorhinal|insula")]
    print(show[["region", "beta_r", "r_age", "rel_young", "rel_old", "cross_age"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
