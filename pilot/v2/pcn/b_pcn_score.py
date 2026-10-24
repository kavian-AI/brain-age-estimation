"""Step B (pcntoolkit 0.35 env): score people with PCN braincharts lifespan_57K_82sites.

Mirrors the braincharts transfer notebook exactly (age+sex covariates, cubic B-spline on age
with xmin=-5/xmax=110, site dummies over the 82 training sites, BLR predict with adaptation).

  python b_pcn_score.py selftest   -> PCN's own OpenNeuro transfer example (usage check)
  python b_pcn_score.py oasis      -> OASIS-1 / OASIS-2, 5 folds, adaptation = pool minus fold-k controls
"""
import os, sys, zipfile, tempfile, contextlib, io
import numpy as np, pandas as pd
from pcntoolkit.normative import predict
from pcntoolkit.util.utils import create_design_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
# clone of github.com/predictive-clinical-neuroscience/braincharts (see pilot/v2/pcn/README.md)
PCN = os.environ.get("PCN_BRAINCHARTS", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "external", "pcn_braincharts"))
ZIP = os.path.join(PCN, "models", "lifespan_57K_82sites.zip")
SITES_TR = open(os.path.join(PCN, "docs", "site_ids_ct_82sites.txt")).read().splitlines()
WORK = os.path.join(HERE, "pcn_models")
COV = ["age", "sex"]

def model_dir(ph):
    d = os.path.join(WORK, "lifespan_57K_82sites", ph, "Models")
    if not os.path.isfile(os.path.join(d, "NM_0_0_estimate.pkl")):
        with zipfile.ZipFile(ZIP) as z:
            for n in z.namelist():
                if n.startswith(f"lifespan_57K_82sites/{ph}/"):
                    z.extract(n, WORK)
    assert os.path.isfile(os.path.join(d, "NM_0_0_estimate.pkl")), f"no model for {ph}"
    return d

def X(df):
    return create_design_matrix(df[COV], site_ids=df["site"], all_sites=SITES_TR,
                                basis="bspline", xmin=-5, xmax=110)

def score(ph, te, ad):
    """Z for te, PCN model for phenotype ph, site effect adapted on ad (same var-group id)."""
    md = model_dir(ph)
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd(); os.chdir(tmp)
        try:
            f = lambda n, a: (np.savetxt(n, a), os.path.join(tmp, n))[1]
            cte, rte = f("cov_te.txt", X(te)), f("resp_te.txt", te[ph].to_numpy(float))
            cad, rad = f("cov_ad.txt", X(ad)), f("resp_ad.txt", ad[ph].to_numpy(float))
            gte = f("vg_te.txt", te["sitenum"].to_numpy(int))
            gad = f("vg_ad.txt", ad["sitenum"].to_numpy(int))
            with contextlib.redirect_stdout(io.StringIO()):
                res = predict(cte, alg="blr", respfile=rte, model_path=md, inputsuffix="estimate",
                              adaptrespfile=rad, adaptcovfile=cad,
                              adaptvargroupfile=gad, testvargroupfile=gte)
            Z = res[2] if isinstance(res, tuple) and len(res) >= 3 else np.loadtxt("Z_predict.txt")
        finally:
            os.chdir(cwd)
    return np.asarray(Z, float).ravel()

def selftest():
    te = pd.read_csv(os.path.join(PCN, "docs", "OpenNeuroTransfer_ct_te.csv"))
    ad = pd.read_csv(os.path.join(PCN, "docs", "OpenNeuroTransfer_ct_ad.csv"))
    print(f"PCN self-test on its own OpenNeuro transfer example: test n={len(te)}, adapt n={len(ad)}, "
          f"sites={te.site.nunique()}")
    rows = []
    for ph in ["Left-Hippocampus", "Right-Hippocampus", "Left-Lateral-Ventricle", "Left-Thalamus-Proper",
               "Brain-Stem", "Left-Amygdala"]:
        z = score(ph, te, ad)
        rows.append({"phenotype": ph, "n": int(np.isfinite(z).sum()), "mean_z": np.nanmean(z),
                     "sd_z": np.nanstd(z), "within_1.96": float(np.nanmean(np.abs(z) < 1.96))})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

def oasis():
    ppl = pd.read_csv(os.path.join(HERE, "people.csv"))
    pool = pd.read_csv(os.path.join(HERE, "adapt_pool.csv"))
    meta = {"subject", "age", "sex", "icv", "cohort", "group", "is_patient", "fold"}
    PHENO = [c for c in ppl.columns if c not in meta]
    for d in (ppl, pool):
        d["site"] = d.cohort.str.replace("-", "")       # 'OASIS1' / 'OASIS2' : unseen by PCN
        d["sitenum"] = 1
    out = []
    for coh in ("OASIS-1", "OASIS-2"):
        P_, A_ = ppl[ppl.cohort == coh], pool[pool.cohort == coh]
        for k in sorted(P_.fold.unique()):
            te = P_[P_.fold == k].reset_index(drop=True)
            held = set(te.loc[~te.is_patient, "subject"])
            ad = A_[~A_.subject.isin(held)].reset_index(drop=True)
            z = pd.DataFrame({"subject": te.subject, "group": te.group, "cohort": coh, "fold": k})
            for ph in PHENO:
                z[ph] = score(ph, te, ad)
            out.append(z)
            print(f"  {coh} fold {k}: scored {len(te)} (held-out controls {len(held)}), adapted on {len(ad)}",
                  flush=True)
    Z = pd.concat(out, ignore_index=True)
    Z.to_csv(os.path.join(HERE, "pcn_z.csv"), index=False)
    print(f"wrote pcn_z.csv: {len(Z)} people x {len(PHENO)} phenotypes")

if __name__ == "__main__":
    {"selftest": selftest, "oasis": oasis}[sys.argv[1]]()
