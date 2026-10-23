# The full healthy reference: everything processed so far, one pooled normative model.
# Runs on Kaggle (CPU kernel) or locally; all paths are relative to the project root.
#
# People (one scan each, healthy only), ALL through one pipeline (FastSurfer 2.4.2 + SynthStrip ICV):
#   earlier pool   IXI (3 sites), DLBS wave 1, OASIS-2 baseline nondemented
#   new queue      OpenNeuro healthy sets (Precision Aging Network x4 sites, NIMH volunteers, ...),
#                  OpenBHB (ABIDE, CoRR, GSP, NAR, NPC, RBP, MPI-Leipzig, Localizer; many sites),
#                  SALD, OASIS-1 nondemented
# ADNI (FreeSurfer 4.3/5.1 tables) and the ADHD-200 FreeSurfer 5 tables were dropped on 2026-10-20
# (user decision): a single program defines every region for everyone. ADNI stays for the disease phase;
# ADHD-200 raw scans may be added later through the same pipeline.
# Every scanner / dataset combination is a site (small OpenBHB sites pooled per study).
#
# Parts (command line):  A = model choice + every-age tables + atlas + calibration + floor + figures
#                        B = model contest (adults >= 18)
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import on_fs_load as OL  # noqa: E402
import shash  # noqa: E402
import p3_pooled as P  # noqa: E402
from p2_lifespan_atlas import HEMI  # noqa: E402
from p2_lifespan_data import regions_from_labels  # noqa: E402
from p2_strata import FS_REG  # noqa: E402

OUT = os.path.join(HERE, "p5_out")
os.makedirs(OUT, exist_ok=True)
CENT = [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]
MODEL = os.environ.get("NORM_MODEL", "SHASH")      # SHASH (main, won the model contest) or GAUSS
NB = 60 if MODEL == "SHASH" else 100
RNG = np.random.default_rng(5)
ICV_GROUPS = ["SynthStrip"]                        # one head-size method for everyone since ADNI / FS5 were dropped


def _sex(v):
    s = str(v).strip().lower()
    return 1.0 if s[:1] == "m" else (0.0 if s[:1] == "f" else np.nan)


# ------------------------------------------------------------------ data
def new_queue():
    """Rows from the Kaggle queue (OpenNeuro, OpenBHB, SALD, OASIS-1) with age, sex, site, health."""
    R = OL.load_rows()
    R = R[~R.qc_fail]
    J = pd.read_csv(os.path.join(ROOT, "data", "all_jobs.csv")).set_index("scan_id")
    R = R[R.scan.isin(J.index)].copy()
    R["ds"] = R.scan.map(J.ds)
    R["sub"] = R.scan.map(J["sub"])
    R["age"] = R.scan.map(J.age)
    R["primary"] = R.scan.map(J.primary).astype(bool)
    R["sex"] = np.nan
    R["site"] = R.ds
    R["healthy"] = True
    on_dir = os.path.join(ROOT, "data", "openneuro_participants")
    for ds in R.ds.unique():
        if not ds.startswith("ds"):
            continue
        p = pd.read_csv(os.path.join(on_dir, f"{ds}.tsv"), sep="\t", na_values=["n/a"])
        sc = "sex" if "sex" in p else ("gender" if "gender" in p else None)
        w = R.ds == ds
        if sc:
            R.loc[w, "sex"] = R.loc[w, "sub"].map(dict(zip(p.participant_id, p[sc].map(_sex))))
        if "site" in p and ds in ("ds007522", "ds003592"):
            name = "PAN" if ds == "ds007522" else ds
            R.loc[w, "site"] = name + " " + R.loc[w, "sub"].map(dict(zip(p.participant_id, p.site.astype(str))))
    # OpenBHB: study + acquisition site; tiny sites pooled per study
    b = pd.read_csv(os.path.join(ROOT, "data", "openbhb", "participants.tsv"), sep="\t")
    b["pid"] = b.participant_id.astype(str)
    w = R.ds.str.startswith("openbhb")
    pid = R.loc[w, "scan"].str.split("__").str[1]
    R.loc[w, "sex"] = pid.map(dict(zip(b.pid, b.sex.map(_sex)))).values
    st = pid.map(dict(zip(b.pid, "OpenBHB s" + b.study.astype(str) + " site" + b.site.astype(int).astype(str))))
    R.loc[w, "site"] = st.values
    # SALD
    x = pd.read_excel(os.path.join(ROOT, "data", "indi", "sald_sub_information.xlsx"))
    w = R.ds == "sald"
    R.loc[w, "sex"] = R.loc[w, "sub"].map(dict(zip("sub-0" + x.Sub_ID.astype(int).astype(str), x.Sex.map(_sex))))
    R.loc[w, "site"] = "SALD"
    # OASIS-1 (CDR > 0 kept aside for the disease phase)
    o = pd.read_csv(os.path.join(ROOT, "data", "oasis_cross-sectional.csv"))
    w = R.ds == "oasis1"
    sess = R.loc[w, "scan"].str.split("__").str[1]
    R.loc[w, "sex"] = sess.map(dict(zip(o.ID, o["M/F"].map(_sex)))).values
    cdr = sess.map(dict(zip(o.ID, o.CDR)))
    R.loc[w, "healthy"] = (cdr.isna() | (cdr == 0)).values
    R.loc[w, "site"] = "OASIS-1 Siemens Vision 1.5T"
    # ADHD-200 typically developing children (raw T1 from INDI, same pipeline) - prepared but not processed yet
    # (postponed 2026-10-20); used automatically once their results exist
    w = R.ds == "adhd200"
    if w.any():
        a = pd.read_csv(os.path.join(ROOT, "data", "adhd200_raw", "adhd200_tdc_jobs.csv"))
        R.loc[w, "sex"] = R.loc[w, "scan"].map(dict(zip(a.scan_id, a.sex))).values
        R.loc[w, "site"] = ("ADHD200 " + R.loc[w, "scan"].str.split("__").str[1]).values
    # sites with >= 15 people keep their own offsets (pooled "other sites" mix acquisitions with different
    # head-size-mask and contrast behaviour)
    n = R.site.map(R[R.primary].site.value_counts()).fillna(0)
    small = R.site.str.startswith("OpenBHB") & (n < 15)
    R.loc[small, "site"] = R.loc[small, "site"].str.replace(r" site\d+$", " other sites", regex=True)
    return R


def to_regions(R):
    rows = []
    for r in R.to_dict("records"):
        d = {"subject": r["ds"] + ":" + str(r["sub"]), "scan": r["scan"], "dataset": r["ds"], "site": r["site"],
             "age": r["age"], "sex": r["sex"], "icv": r.get("icv_synthstrip"), "healthy": r["healthy"],
             "primary": r["primary"], "icv_method": "SynthStrip"}
        reg = regions_from_labels(r, "V")
        d.update(reg)
        for k, v in FS_REG.items():
            d[k] = reg[v]
        rows.append(d)
    return pd.DataFrame(rows)


INFRA = ["cerebellum cortex", "cerebellum_cortex", "L_cerebellum_cortex", "R_cerebellum_cortex", "cerebellum_wm",
         "L_cerebellum_wm", "R_cerebellum_wm", "brainstem", "fourth_ventricle", "total brain", "total_brain"]
FOV_PRONE = ("openbhb", "ds005405")      # the only sources with an excess low-cerebellum tail (5 % / 1.75 % below z = -3)


def fov_qc(N):
    """Field-of-view check: a scan whose slab misses the bottom of the head loses cerebellum and brainstem.
    Robust z of log(cerebellum / ICV) and log(brainstem / ICV), after age (spline), sex and site medians,
    with the reference taken from healthy primary scans.
      z < -5 (or a lower-brain label with zero voxels)  -> whole scan excluded (head-size mask is cut too)
      -5 <= z < -3 in a cut-off-prone source              -> only cerebellum / brainstem / total brain dropped
    ds005405 (up to 4 scans per person, cut-off varies by session): the least cut scan becomes primary."""
    from patsy import dmatrix
    N = N.copy()
    B = np.asarray(dmatrix("bs(age, df=8, include_intercept=True) - 1", {"age": N.age.fillna(40).clip(5, 95)}))
    X = np.column_stack([B, N.sex.fillna(0.5)])
    ref = (N.healthy & N.primary & N.age.notna() & N.sex.notna() & (N.icv > 0)).values
    for col, reg in (("z_cb", "cerebellum cortex"), ("z_bs", "brainstem")):
        y = np.log(N[reg] / N.icv).values
        ok = ref & np.isfinite(y)
        e = pd.Series(y - X @ np.linalg.lstsq(X[ok], y[ok], rcond=None)[0], index=N.index)
        e = e - N.site.map(e[ok].groupby(N.site[ok]).median()).fillna(0)
        mad = 1.4826 * np.median(np.abs(e[ok] - np.median(e[ok])))
        N[col] = (e - np.median(e[ok])) / mad
    zmin = N[["z_cb", "z_bs"]].min(axis=1)
    missing = N[["cerebellum_wm", "fourth_ventricle", "brainstem", "cerebellum_cortex"]].isna().any(axis=1)
    N["fov_cut"] = (zmin < -5) | missing
    prone = N.dataset.str.startswith(FOV_PRONE)
    N["fov_mild"] = prone & ~N.fov_cut & (zmin < -3)
    N.loc[N.fov_mild, INFRA] = np.nan
    d5 = (N.dataset == "ds005405") & N.healthy & ~N.fov_cut
    if d5.any():
        best = zmin[d5].groupby(N.subject[d5]).idxmax()
        N.loc[N.dataset == "ds005405", "primary"] = False
        N.loc[best.values, "primary"] = True
    N[["dataset", "site", "subject", "scan", "age", "primary", "z_cb", "z_bs", "fov_cut", "fov_mild"]][
        N.fov_cut | N.fov_mild].to_csv(os.path.join(OUT, "qc_fov_flags.csv"), index=False)
    return N[~N.fov_cut]


BAD_SITES = {"OpenBHB s1 site53"}   # ABIDE, 0.86 x 1.5 x 0.86 mm children: cortex visibly under-segmented,
                                     # Euler median -103 (worst 5 % of OpenBHB), half the site extreme on brain/ICV


def image_qc(N):
    """OpenBHB's published image-quality metrics: CAT12 IQR > 3 (grade worse than C, 'satisfactory') -> excluded
    (low-cortex outliers rise from < 1 % to 12-20 % above that grade). Plus BAD_SITES."""
    q = pd.read_csv(os.path.join(ROOT, "data", "openbhb", "qc.tsv"), sep="\t")
    iqr = dict(zip(q.participant_id.astype(str), q["cat12vbm-iqr"]))
    ob = N.dataset.str.startswith("openbhb")
    N = N.copy()
    N["cat12_iqr"] = np.where(ob, N.scan.str.split("__").str[1].map(iqr), np.nan)
    N["iqr_fail"] = ob & (N.cat12_iqr > 3.0)
    N["bad_site"] = N.site.isin(BAD_SITES)
    N[["dataset", "site", "subject", "scan", "age", "primary", "cat12_iqr", "iqr_fail", "bad_site"]][
        N.iqr_fail | N.bad_site].to_csv(os.path.join(OUT, "qc_image_flags.csv"), index=False)
    return N[~(N.iqr_fail | N.bad_site)]


def full_pool():
    cross, D, O = P.assemble()
    cross = cross[cross.dataset != "ADNI"].copy()            # ADNI: other program -> disease phase only
    cross["icv_method"] = "SynthStrip"
    N = image_qc(fov_qc(to_regions(new_queue())))
    newx = N[N.primary & N.healthy & N.age.notna() & N.sex.notna() & N.icv.notna()]
    allp = pd.concat([cross, newx], ignore_index=True)
    allp = allp[allp.age.notna() & allp.sex.notna() & (allp.icv > 0)].reset_index(drop=True)
    allp["licv"] = np.log(allp.icv) - np.log(allp.icv).mean()
    return allp, D, O, N


def repeat_pairs(N):
    """Same-session repeats: OASIS-1 mpr-1 vs mpr-2, ds005405 run-1 vs run-2 (same session),
    plus OASIS-2 (P.repeat_noise). Returns eps (%, log units) per region and the number of pairs."""
    pairs = []
    o1 = N[N.dataset == "oasis1"].copy()
    o1["sess"] = o1.scan.str.split("__").str[1]
    o1["mpr"] = o1.scan.str.split("__").str[2]
    for s, g in o1.groupby("sess"):
        if {"mpr-1", "mpr-2"} <= set(g.mpr):
            pairs.append((g[g.mpr == "mpr-1"].iloc[0], g[g.mpr == "mpr-2"].iloc[0], "OASIS-1"))
    J = pd.read_csv(os.path.join(ROOT, "data", "all_jobs.csv")).set_index("scan_id")
    d5 = N[N.dataset == "ds005405"].copy()
    d5["key"] = d5.scan.map(J.key)
    d5["ses"] = d5.key.str.extract(r"(ses-[^/_]+)")[0]
    d5["run"] = d5.key.str.extract(r"run-(\d)")[0]
    for (s, ses), g in d5.groupby(["subject", "ses"]):
        if {"1", "2"} <= set(g.run):
            pairs.append((g[g.run == "1"].iloc[0], g[g.run == "2"].iloc[0], "ds005405"))
    eps, npairs = {}, {"OASIS-1": 0, "ds005405": 0}
    for _, _, src in pairs:
        npairs[src] += 1
    per_src = []
    for r in P.REGIONS:
        dif = np.array([np.log(b[r]) - np.log(a[r]) for a, b, _ in pairs if a[r] > 0 and b[r] > 0])
        # robust SD (1.4826 x MAD): a few ds005405 runs cut off the lower cerebellum
        eps[r] = 100 * 1.4826 * np.median(np.abs(dif - np.median(dif))) / np.sqrt(2) if len(dif) > 10 else np.nan
        for src in ("OASIS-1", "ds005405"):
            ds_ = np.array([np.log(b[r]) - np.log(a[r]) for a, b, s_ in pairs if s_ == src and a[r] > 0 and b[r] > 0])
            if len(ds_) > 10:
                per_src.append({"region": r, "source": src, "pairs": len(ds_),
                                "eps_sd_pct": 100 * ds_.std(ddof=1) / np.sqrt(2),
                                "eps_robust_pct": 100 * 1.4826 * np.median(np.abs(ds_ - np.median(ds_))) / np.sqrt(2),
                                "share_diff_gt_10pct": float(np.mean(np.abs(ds_) > 0.10))})
    pd.DataFrame(per_src).to_csv(os.path.join(OUT, "repeat_noise_by_source.csv"), index=False)
    e2, n2 = P.repeat_noise(FS_REG)
    npairs["OASIS-2"] = n2
    return eps, e2, npairs


# ------------------------------------------------------------------ part A
def fit_region(d, ly, r):
    gr = [g for g in ICV_GROUPS if (d.icv_method == g).any()]
    young = d.age.min() < 15
    cls = shash.SHASH if MODEL == "SHASH" else P.LS
    return cls(d, ly, agesex=(r in P.AGESEX), df_mu=7 if young else 5, df_sd=4 if young else 3,
               lo=5.0 if young else 18.0, hi=98.0, icv_groups=gr), gr


def ref_sigma(m, d, ages):
    nS = len(m.Bs.column_names)
    so = list(m.bs[nS:nS + len(m.sites) - 1])
    so = np.array([-sum(so)] + so)
    W = np.array([[((d.site.values == s) & (np.abs(d.age.values - a) <= 5)).sum() for s in m.sites] for a in ages], float)
    W = W / np.maximum(W.sum(1, keepdims=True), 1)
    return (m.sigma(ages) / 100) * np.exp((W @ so) / 2)


TRAJ = ("hippocampus", "lateral ventricles", "cortex")     # regions whose individual trajectories are drawn
BANDS = [(20, 40), (40, 60), (60, 70), (70, 80), (80, 100)]


def long_scans(D, O, licv_mean):
    """Every DLBS wave / OASIS-2 visit of people scanned >= 2 times, as model input rows. Head size =
    the person's mean ICV (the skull does not change); volumes rescaled by mean ICV / scan ICV, which
    removes scanner scale drift between visits (as in P.within_person)."""
    L = pd.concat([D[(D.run == 1) & D.icv.notna()].assign(site=P.SITE_NAME["DLBS"]),
                   O[O.icv.notna()].assign(site=P.SITE_NAME["OASIS2"])], ignore_index=True)
    L = L.rename(columns={v: k for k, v in FS_REG.items()})
    mi = L.groupby("subject").icv.transform("mean")
    for r in P.REGIONS:
        if r in L:
            L[r] = L[r] * mi / L.icv
    L["licv"] = np.log(mi) - licv_mean
    L["icv_method"] = "SynthStrip"
    n = L.groupby("subject").age.transform("size")
    return L[n >= 2].reset_index(drop=True)


def trajectory(m, gr, L, r, ages, sg):
    """Each repeat scan -> its SHASH deviate z (own age, sex, head size, scanner) -> the volume with the same
    centile for the reference person (reference head, average scanner): every person on one common chart."""
    Lr = L[L[r] > 0]
    z = m.z(Lr, np.log(Lr[r].values))
    g = pd.DataFrame({"age": Lr.age.values, "sex": Lr.sex.values, "licv": 0.0, "site": m.sites[0], "icv_method": gr[0]})
    loc = m._Xm(g, zero_site=True) @ m.bm
    nu, tau, csd = getattr(m, "nu", 0.0), getattr(m, "tau", 1.0), getattr(m, "c_sd", 1.0)
    scale = np.interp(Lr.age.values, ages, sg) / csd
    adj = np.exp(loc + scale * np.sinh((np.arcsinh(z) + nu) / tau))
    return pd.DataFrame({"region": r, "subject": Lr.subject.values, "source": Lr.site.values, "age": Lr.age.values,
                         "sex": Lr.sex.values, "z": z, "volume_reference_scale": adj, "volume_raw": Lr[r].values})


def individual_rates(L, eps):
    """How fast single people change. Per person: OLS slope of 100*log(volume) on age (%/yr). Longitudinal
    noise eps_long: scatter of people with >= 3 scans around their own straight line (pooled SSE / dof),
    which includes visit-to-visit effects that a same-session repeat cannot see. The noise of a person's slope
    is eps_long^2 / sum((t - mean t)^2); subtracting its mean from the observed variance of slopes leaves the
    TRUE person-to-person spread of aging speed."""
    rows, noise = [], []
    for r in P.REGIONS:
        if r not in L:
            continue
        per = []
        sse, dof = {}, {}
        for s, g in L[L[r] > 0].groupby("subject"):
            t, y = g.age.values.astype(float), 100 * np.log(g[r].values)
            if len(t) < 2 or np.ptp(t) < 2:
                continue
            sxx = ((t - t.mean()) ** 2).sum()
            b = ((t - t.mean()) * (y - y.mean())).sum() / sxx
            if len(t) >= 3:
                res = y - y.mean() - b * (t - t.mean())
                src = g.site.iloc[0]
                sse[src] = sse.get(src, 0.0) + (res ** 2).sum()
                dof[src] = dof.get(src, 0) + len(t) - 2
            per.append((t.mean(), b, sxx, np.ptp(t), len(t)))
        if len(per) < 30 or sum(dof.values()) < 20:
            continue
        p = pd.DataFrame(per, columns=["mid", "slope", "sxx", "span", "n"])
        e_long = np.sqrt(sum(sse.values()) / sum(dof.values()))
        noise.append({"region": r, "eps_same_session_pct": eps.get(r, np.nan), "eps_longitudinal_pct": e_long,
                      **{f"eps_longitudinal_{k.split()[0]}": np.sqrt(sse[k] / dof[k]) for k in sse if dof[k] >= 10},
                      "people": len(p), "people_3plus_scans": int((p.n >= 3).sum())})
        p["noise_var"] = e_long ** 2 / p.sxx
        for lo, hi in BANDS:
            q = p[(p.mid >= lo) & (p.mid < hi)]
            if len(q) < 15:
                continue
            med = q.slope.median()
            mad = 1.4826 * np.median(np.abs(q.slope - med))
            q = q[np.abs(q.slope - med) <= 5 * max(mad, 1e-6)]          # drop gross outliers (bad scans)
            ov, nv = q.slope.var(ddof=1), q.noise_var.mean()
            rows.append({"region": r, "age_band": f"{lo}-{hi}", "people": len(q), "followup_yrs_median": q.span.median(),
                         "median_rate_pct_yr": med, "p10": q.slope.quantile(.1), "p90": q.slope.quantile(.9),
                         "observed_sd": np.sqrt(ov), "noise_sd": np.sqrt(nv), "true_sd": np.sqrt(max(ov - nv, 0.0)),
                         "share_true": max(ov - nv, 0.0) / ov, "share_growing_observed": float((q.slope > 0).mean())})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "individual_rates.csv"), index=False)
    pd.DataFrame(noise).to_csv(os.path.join(OUT, "noise_budget.csv"), index=False)
    print(pd.DataFrame(rows).round(2).to_string(index=False), flush=True)
    print(pd.DataFrame(noise).round(2).to_string(index=False), flush=True)
    return {n["region"]: n["eps_longitudinal_pct"] for n in noise}


def r_ci(sp_lo, sp_hi, s_lo, s_hi):
    """CI of R = spread / |rate| from the CIs of both (conservative: extremes combined)."""
    if not (np.isfinite(s_lo) and np.isfinite(s_hi)):
        return np.nan, np.nan
    a, b = sorted((abs(s_lo), abs(s_hi)))
    lo = 100 * sp_lo / max(b, 1e-6)
    hi = 999.0 if s_lo * s_hi <= 0 else 100 * sp_hi / max(a, 1e-6)
    return min(lo, 999.0), min(hi, 999.0)


def part_a():
    t0 = time.time()
    allp, D, O, N = full_pool()
    ref_icv = float(np.exp(np.log(allp.icv).mean()))
    summ = allp.groupby("dataset").agg(n=("age", "size"), age_min=("age", "min"), age_max=("age", "max"),
                                       men=("sex", "mean"), sites=("site", "nunique"))
    summ.to_csv(os.path.join(OUT, "pool_summary.csv"))
    print("normative model:", MODEL, flush=True)
    print(f"FULL POOL: {len(allp)} healthy people, {allp.site.nunique()} sites, ages {allp.age.min():.1f}-"
          f"{allp.age.max():.1f}; aged 60+: {(allp.age >= 60).sum()}", flush=True)
    print(summ.round(2).to_string(), flush=True)
    per, resid = P.within_person(D.rename(columns={v: k for k, v in FS_REG.items()}),
                                 O.rename(columns={v: k for k, v in FS_REG.items()}), P.REGIONS)
    eps, eps_o2, npairs = repeat_pairs(N)
    print("repeat pairs:", npairs, flush=True)
    Lsc = long_scans(D, O, float(np.log(allp.icv).mean()))
    eps_long = individual_rates(Lsc, eps)          # visit-to-visit noise (years apart), per region
    rows, params, choice, traj = [], [], [], []
    for r in P.REGIONS:
        d, ly = P.region_data(allp, r)
        m, gr = fit_region(d, ly, r)
        lo, hi = np.ceil(d.age.quantile(0.002)), np.floor(d.age.quantile(0.998))
        ages = np.arange(max(lo, 6), min(hi, 95) + 1, 1.0)
        sl = m.slope(ages)
        sg = ref_sigma(m, d, ages)
        # bootstrap (resample people within site) for slope / spread / R
        groups = [np.where(d.site.values == s)[0] for s in m.sites]
        B = []
        for _ in range(NB):
            ii = np.concatenate([g[RNG.integers(0, len(g), len(g))] for g in groups])
            try:
                db = d.iloc[ii].reset_index(drop=True)
                mb, _ = fit_region(db, ly[ii], r)
                B.append((mb.slope(ages), ref_sigma(mb, db, ages)))
            except Exception:
                pass
        B = np.array(B)                                     # NB x 2 x ages
        Rb = 100 * B[:, 1] / np.maximum(np.abs(B[:, 0]), 1e-6)
        if len(per[r]) > 30:
            pl = pd.DataFrame(per[r], columns=["s", "mid", "slope", "span"])
            kl_all, kl_lo, kl_hi, _ = P.long_curve(per[r], ages, boot=True)
            inr = (ages >= pl.mid.min() + 2) & (ages <= pl.mid.max() - 2)
            kl = np.where(inr, kl_all, np.nan)
            kll, klh = np.where(inr, kl_lo, np.nan), np.where(inr, kl_hi, np.nan)
        else:
            kl = kll = klh = np.full(len(ages), np.nan)
        if r in TRAJ:
            traj.append(trajectory(m, gr, Lsc, r, ages, sg))
        e = eps.get(r, np.nan)
        if not np.isfinite(e):
            e = eps_o2.get(r, np.nan)
        n_near = [int((np.abs(d.age.values - a) <= 2.5).sum()) for a in ages]
        for sex, lab in ((0.0, "women"), (1.0, "men")):
            g = pd.DataFrame({"age": ages, "sex": sex, "licv": 0.0, "site": m.sites[0], "icv_method": gr[0]})
            loc = m._Xm(g, zero_site=True) @ m.bm
            nu, tau, csd = getattr(m, "nu", 0.0), getattr(m, "tau", 1.0), getattr(m, "c_sd", 1.0)
            scale_ref = sg / csd
            for j, a in enumerate(ages):
                sp_lo, sp_hi = np.percentile(B[:, 1, j], 2.5), np.percentile(B[:, 1, j], 97.5)
                rl_lo, rl_hi = r_ci(sp_lo, sp_hi, kll[j], klh[j])
                row = {"region": r, "sex": lab, "age": a, "n_within_2.5y": n_near[j],
                       "spread_pct": 100 * sg[j], "spread_lo": 100 * np.percentile(B[:, 1, j], 2.5),
                       "spread_hi": 100 * np.percentile(B[:, 1, j], 97.5),
                       "slope_cross": sl[j], "slope_cross_lo": np.percentile(B[:, 0, j], 2.5),
                       "slope_cross_hi": np.percentile(B[:, 0, j], 97.5), "slope_long": kl[j],
                       "slope_long_lo": kll[j], "slope_long_hi": klh[j],
                       "S_cross": abs(sl[j]) / (100 * sg[j]),
                       "R_cross": min(100 * sg[j] / max(abs(sl[j]), 1e-6), 999),
                       "R_cross_lo": min(np.percentile(Rb[:, j], 2.5), 999),
                       "R_cross_hi": min(np.percentile(Rb[:, j], 97.5), 999),
                       "R_long": min(100 * sg[j] / max(abs(kl[j]), 1e-6), 999) if np.isfinite(kl[j]) else np.nan,
                       "R_long_lo": rl_lo, "R_long_hi": rl_hi,
                       "eps_repeat_pct": e,
                       "M_repeat": min(e / max(abs(kl[j]), 1e-6), 999) if np.isfinite(kl[j]) else np.nan,
                       # monitoring: one follow-up scan's noise (visit to visit) in years of normal aging
                       "eps_long_pct": eps_long.get(r, np.nan),
                       "M_long": (min(eps_long[r] / max(abs(kl[j]), 1e-6), 999)
                                  if r in eps_long and np.isfinite(kl[j]) else np.nan),
                       "ICC": 1 - (e / (100 * sg[j])) ** 2 if np.isfinite(e) else np.nan}
                for c in CENT:
                    q = loc[j] + scale_ref[j] * np.sinh((np.arcsinh(stats.norm.ppf(c)) + nu) / tau)
                    row[f"c{c*100:g}"] = float(np.exp(q))
                rows.append(row)
        nB = m.nB
        offs = list(m.bm[m.site_start:m.site_start + len(m.sites) - 1])
        offs = [-sum(offs)] + offs
        nS = len(m.Bs.column_names)
        soffs = list(m.bs[nS:nS + len(m.sites) - 1])
        soffs = [-sum(soffs)] + soffs
        icvm = dict(zip(d.site, d.icv_method))
        for s, o, so in zip(m.sites, offs, soffs):
            params.append({"region": r, "site": s, "n": int((d.site == s).sum()),
                           "age_min": float(d.age[d.site == s].min()), "age_max": float(d.age[d.site == s].max()),
                           "mean_shift_pct": 100 * (np.exp(o) - 1), "spread_factor": float(np.exp(so / 2)),
                           "head_exponent_b": float(m.bm[nB + 1 + gr.index(icvm[s])]),
                           "sex_coef_log": m.b_sex / 100})
        # LR test: scanner-specific spread vs common spread
        kw = dict(agesex=(r in P.AGESEX), df_mu=7 if d.age.min() < 15 else 5,
                  df_sd=4 if d.age.min() < 15 else 3, lo=m.lo, hi=m.hi, icv_groups=gr)
        mg = P.LS(d, ly, site_sd=True, **kw)
        mc = P.LS(d, ly, site_sd=False, **kw)
        lr = max(2 * (mg.loglik - mc.loglik), 0)
        dfree = mg.k_params - mc.k_params
        choice.append({"region": r, "n": len(d), "sites": len(m.sites), "LR_site_spread": lr, "df": dfree,
                       "p": stats.chi2.sf(lr, dfree), "dAIC": -lr + 2 * dfree,
                       "shash_nu": getattr(m, "nu", np.nan), "shash_tau": getattr(m, "tau", np.nan),
                       "loglik_gain_shash_vs_gauss": m.loglik - mg.loglik})
        print(f"  {r:22s} n={len(d):5d} sites={len(m.sites):3d} ages {ages[0]:.0f}-{ages[-1]:.0f} "
              f"LR p={stats.chi2.sf(lr, dfree):.1e}  ({time.time()-t0:.0f}s)", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "everyage_table.csv"), index=False)
    pd.DataFrame(params).to_csv(os.path.join(OUT, "site_params.csv"), index=False)
    pd.DataFrame(choice).to_csv(os.path.join(OUT, "model_choice.csv"), index=False)
    json.dump({"reference_head_cm3": ref_icv / 1000, "n_people": int(len(allp)), "sites": int(allp.site.nunique()),
               "age_range": [float(allp.age.min()), float(allp.age.max())], "aged_60_plus": int((allp.age >= 60).sum()),
               "repeat_pairs": npairs}, open(os.path.join(OUT, "everyage_meta.json"), "w"), indent=1)
    pd.concat(traj, ignore_index=True).to_csv(os.path.join(OUT, "trajectories.csv"), index=False)
    calibration(allp)
    floor(allp, D, O)
    print(f"part A done in {(time.time()-t0)/60:.1f} min", flush=True)


def calibration(allp):
    subj = allp.subject.unique()
    fold_of = dict(zip(subj, np.random.default_rng(0).permutation(len(subj)) % 5))
    out = []
    for r in P.REGIONS:
        d, ly = P.region_data(allp, r)
        f = d.subject.map(fold_of).values
        z = np.full(len(d), np.nan)
        for k in range(5):
            tr, te = f != k, f == k
            m, _ = fit_region(d[tr].reset_index(drop=True), ly[tr], r)
            dte = d[te].reset_index(drop=True)
            dte = dte[dte.site.isin(m.sites)]
            z[np.where(te)[0][d[te].site.isin(m.sites).values]] = m.z(dte, np.log(dte[r].values.astype(float)))
        out.append(pd.DataFrame({"region": r, "site": d.site, "dataset": d.dataset, "age": d.age, "z": z}))
    Z = pd.concat(out).dropna(subset=["z"])
    Z["band"] = pd.cut(Z.age, [0, 18, 40, 60, 75, 100], labels=["<18", "18-40", "40-60", "60-75", "75+"])
    summ = lambda g: pd.Series({"n": len(g), "sd_z": g.z.std(), "within_1.96": np.mean(np.abs(g.z) < 1.96),
                                "within_1": np.mean(np.abs(g.z) < 1), "skew": stats.skew(g.z),
                                "excess_kurt": stats.kurtosis(g.z)})
    for k in ("dataset", "band", "region"):
        Z.groupby(k).apply(summ).to_csv(os.path.join(OUT, f"calibration_by_{k}.csv"))
    print("held-out calibration:", summ(Z).round(3).to_dict(), flush=True)
    print(Z.groupby("band").apply(summ).round(3).to_string(), flush=True)


def floor(allp, D, O):
    """Whole-brain precision floor from pooled z-scores, FastSurfer sites, hemisphere features."""
    fs = allp[allp.icv_method == "SynthStrip"].reset_index(drop=True)
    feats = [h for h in HEMI if h in fs and fs[h].notna().all() and (fs[h] > 0).all()]
    per, _ = P.within_person(D, O, feats)
    ages = np.arange(10, 91, 5).astype(float)
    Z = np.zeros((len(fs), len(feats)))
    Sx, Sl = [], []
    for j, h in enumerate(feats):
        ly = np.log(fs[h].values.astype(float))
        m, _ = fit_region(fs, ly, h)
        Z[:, j] = m.z(fs, ly)
        sg = ref_sigma(m, fs, ages)
        Sx.append(m.slope(ages) / 100 / sg)
        kl, *_ = P.long_curve(per[h], ages, boot=False) if len(per[h]) > 30 else (np.full(len(ages), np.nan),)
        Sl.append(kl / 100 / sg)
    Sx, Sl = np.array(Sx), np.array(Sl)
    from sklearn.covariance import LedoitWolf
    rows = []
    for j, a in enumerate(ages):
        w = np.abs(fs.age.values - a) <= 5
        if w.sum() < 100:
            continue
        Zw = Z[w]
        row = {"age": a, "n_window": int(w.sum()), "p": len(feats)}
        for nm, s in (("cross", Sx[:, j]), ("long", Sl[:, j])):
            if not np.isfinite(s).all():
                continue
            Si = np.linalg.pinv(LedoitWolf().fit(Zw).covariance_)
            fold = RNG.permutation(len(Zw)) % 5
            g = np.zeros(len(Zw))
            for f in range(5):
                Sif = np.linalg.pinv(LedoitWolf().fit(Zw[fold != f]).covariance_)
                g[fold == f] = (Zw[fold == f] @ Sif @ s) / (s @ Sif @ s)
            row[f"R_brain_{nm}"] = 1 / np.sqrt(s @ Si @ s)
            row[f"SD_g_heldout_{nm}"] = float(np.std(g))
            row[f"best_single_R_{nm}"] = float(np.min(1 / np.abs(s)))
        rows.append(row)
    F = pd.DataFrame(rows)
    F.to_csv(os.path.join(OUT, "floor_by_age.csv"), index=False)
    print(F.round(2).to_string(index=False), flush=True)


# ------------------------------------------------------------------ part B
def part_b():
    import p4_models as M4
    allp, D, O, N = full_pool()
    adults = allp[allp.age >= 18].reset_index(drop=True)
    # the adult filter leaves some mostly-children scanner groups with 1-11 adults: pool them per
    # study, drop any group still under 10 (a 1-person group breaks the per-site spread in CV folds)
    n = adults.site.map(adults.site.value_counts())
    adults.loc[n < 20, "site"] = adults.loc[n < 20, "dataset"] + " other sites"
    n = adults.site.map(adults.site.value_counts())
    adults = adults[n >= 10].reset_index(drop=True)
    adults["licv"] = np.log(adults.icv) - np.log(adults.icv).mean()
    print(f"model contest on {len(adults)} adults, {adults.site.nunique()} sites", flush=True)
    json.dump({"people": len(adults), "sites": int(adults.site.nunique())}, open(os.path.join(OUT, "contest_meta.json"), "w"))

    _gfit = M4.GPR.fit

    def gpr_fit(self, d, y):                                 # GP is O(n^3): fit on <= 2500 people
        if len(d) > 2500:
            i = np.random.default_rng(0).choice(len(d), 2500, replace=False)
            d, y = d.iloc[i].reset_index(drop=True), y[i]
        return _gfit(self, d, y)
    M4.GPR.fit = gpr_fit
    M4.OUT = OUT
    M4.P.assemble = lambda: (adults, D, O)                  # evaluate() pulls the pool from here
    M4.evaluate()


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "A"
    {"A": part_a, "B": part_b}[part]()
