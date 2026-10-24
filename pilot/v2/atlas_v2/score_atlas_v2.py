"""Score people against atlas v2.  Needs only numpy + pandas and the tables in this folder.

    python score_atlas_v2.py people.csv out.csv [--site-controls controls.csv]

people.csv columns: subject, age (years), sex (0 = female, 1 = male), icv (mm3, SynthStrip), site
                    (a site name from sites_v2.csv, or any new name), plus one column per atlas feature
                    with the FastSurfer 2.4.2 volume in mm3 (names as in coefs_v2.csv; missing columns are skipped).
--site-controls: healthy people scanned on the NEW scanner(s), same columns. Their scanner offsets are
                 estimated with the shrinkage rule validated in the transfer test (aim for >= 20 controls;
                 with 0 controls a new scanner's z can be off by up to ~1 SD).
Output: one z column and one centile column per feature; z > 0 always means 'larger than expected'
(atrophy shows as negative z for tissue, positive z for ventricles / CSF)."""
import os, sys, json, numpy as np, pandas as pd
from math import erf, sqrt
HERE = os.path.dirname(os.path.abspath(__file__))
_GX, _GW = np.polynomial.hermite_e.hermegauss(80); _GW = _GW / _GW.sum()

def load_atlas(folder=HERE):
    meta = json.load(open(os.path.join(folder, "atlas_v2_meta.json")))
    grid = pd.read_csv(os.path.join(folder, "grid_v2.csv.gz"))
    coefs = pd.read_csv(os.path.join(folder, "coefs_v2.csv")).set_index("feature")
    sites = pd.read_csv(os.path.join(folder, "sites_v2.csv"))
    return {"meta": meta, "grid": grid, "coefs": coefs, "sites": sites}

def _raw(atlas, df, f):
    """z0 relative to the reference curve, head size applied, NO scanner offset."""
    G = atlas["grid"][atlas["grid"].feature == f]; c = atlas["coefs"].loc[f]
    lo, hi = atlas["meta"]["age_range"]
    out_of_range = ((df.age < lo) | (df.age > hi)).values       # never extrapolate: these get no score
    l0 = np.where(df.sex.values == 1, np.interp(df.age, G[G.sex == 1].age, G[G.sex == 1].loc0),
                  np.interp(df.age, G[G.sex == 0].age, G[G.sex == 0].loc0))
    s0 = np.where(df.sex.values == 1, np.interp(df.age, G[G.sex == 1].age, G[G.sex == 1].logsd0),
                  np.interp(df.age, G[G.sex == 0].age, G[G.sex == 0].logsd0))
    x = np.log(df.icv.values) - atlas["meta"]["log_icv_mean"]
    loc = l0 + c.b_head * x + c.c_age_head * (df.age.values - 60) / 10 * x
    lsd = s0 + c.d1_head_logsd * x + c.d2_head2_logsd * x ** 2
    loc = np.where(out_of_range, np.nan, loc)
    return np.log(df[f].values.astype(float)), loc, lsd

def adapt_sites(atlas, controls):
    """per (feature, new site): location shift (log units) and log-sd offset, empirical-Bayes shrunk."""
    rows = []
    for f in atlas["coefs"].index:
        if f not in controls: continue
        c = atlas["coefs"].loc[f]
        m0 = (_GW * np.sinh((np.arcsinh(_GX) + c.nu) / c.tau)).sum()
        s0 = np.sqrt((_GW * np.sinh((np.arcsinh(_GX) + c.nu) / c.tau) ** 2).sum() - m0 ** 2)
        for s, g in controls.groupby("site"):
            g = g[g[f].notna() & (g[f] > 0)]
            if len(g) < 2: continue
            y, loc, lsd = _raw(atlas, g, f); u = (y - loc) / np.exp(lsd); u = u[np.isfinite(u)]; n = len(u)
            if n < 2: continue
            b_raw = np.std(u, ddof=1) / s0; ws = c.tau_site_logsd ** 2 / (c.tau_site_logsd ** 2 + 1 / (2 * (n - 1)))
            b = np.exp(ws * np.log(b_raw)); a_raw = np.mean(u) - b * m0
            wm = c.tau_site_shift_z ** 2 / (c.tau_site_shift_z ** 2 + b ** 2 / n); a = wm * a_raw
            rows.append({"feature": f, "site": s, "n": n, "a_z0": a, "b_scale": b})
    return pd.DataFrame(rows)

def score(atlas, people, adapted=None):
    out = people[["subject"]].copy() if "subject" in people else pd.DataFrame(index=people.index)
    cols = {}
    known = atlas["sites"]
    for f in atlas["coefs"].index:
        if f not in people: continue
        c = atlas["coefs"].loc[f]; ok = people[f].notna() & (people[f] > 0)
        p = people[ok]; y, loc, lsd = _raw(atlas, p, f)
        ks = known[known.feature == f].set_index("site")
        shift = p.site.map(ks.shift_log); lsdo = p.site.map(ks.logsd_offset)
        z0 = (y - loc - shift.fillna(0).values) / np.exp(lsd + lsdo.fillna(0).values)
        new = shift.isna().values
        if new.any():
            if adapted is not None and len(adapted):
                ad = adapted[adapted.feature == f].set_index("site")
                a = p.site.map(ad.a_z0).fillna(0).values; b = p.site.map(ad.b_scale).fillna(1).values
                z0 = np.where(new, ((y - loc) / np.exp(lsd) - a) / b, z0)
            else:
                print(f"WARNING {f}: {int(new.sum())} people on scanners unknown to the atlas and no --site-controls; "
                      f"their z may be off by up to ~1 SD", file=sys.stderr)
        z = np.sinh(c.tau * np.arcsinh(z0) - c.nu)
        if np.isnan(loc).any():
            lo, hi = atlas['meta']['age_range']
            print(f'NOTE {f}: {int(np.isnan(loc).sum())} people outside the atlas age range {lo}-{hi} were not scored', file=sys.stderr)
        zc = pd.Series(np.nan, index=people.index); zc[ok] = z
        cols[f"z_{f}"] = zc; cols[f"centile_{f}"] = zc.map(lambda v: np.nan if np.isnan(v) else 50 * (1 + erf(v / sqrt(2))))
    return pd.concat([out, pd.DataFrame(cols, index=people.index)], axis=1)

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2: sys.exit(__doc__)
    atlas = load_atlas(); people = pd.read_csv(args[0])
    adapted = None
    if "--site-controls" in args:
        adapted = adapt_sites(atlas, pd.read_csv(args[args.index("--site-controls") + 1]))
        print(adapted.groupby("site").n.max().to_string(), file=sys.stderr)
    score(atlas, people, adapted).to_csv(args[1], index=False); print(f"wrote {args[1]}")
