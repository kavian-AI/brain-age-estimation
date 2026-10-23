# Disease test of the normative atlas (phase 6): can single-scan deviations and within-person change
# separate dementia from healthy ageing, and how do they compare with a brain-age gap?
#
# Data: everything through the same FastSurfer 2.4.2 + SynthStrip pipeline as the healthy atlas.
#   OASIS-1  93 patients (CDR 0.5 = 68, CDR >= 1 = 25) vs healthy OASIS-1 controls aged >= 55
#   OASIS-2  baseline: 64 demented + 14 converted (CDR 0 at entry, demented later) vs 72 nondemented
#            follow-up: per-person rates of change (2-5 visits)
# Fairness: controls are scored by atlas models that never saw them. Controls and patients of both
# cohorts are split into 5 folds; for fold k the 17 regional SHASH models (and the ridge brain-age model)
# are refitted without fold k's controls, then fold k is scored.
#
# Scores (higher = more abnormal for every score):
#   bag_ridge   conventional brain-age gap: ridge on 85 hemisphere log volumes + log ICV + sex, healthy
#               training set, bias-corrected (gap regressed on age in out-of-fold training predictions)
#   bag_atlas   atlas brain-age gap in years: GLS projection of the 17-region z vector on the healthy
#               cross-sectional ageing direction at the person's age (the floor estimator)
#   z_<region>  single-region normative deviation, sign flipped so that atrophy / enlargement is positive
#   ad_mean     mean of hippocampus, entorhinal, amygdala deviations and temporal-horn enlargement
#   learned     cross-validated logistic regression on the 17 deviations (patients vs controls of other folds)
#   n_outliers  number of regions beyond 1.96 in the disease direction (Verdi et al. 2023 style)
# Outputs: pilot/p6_out/scores.csv, summary tables (printed + csv), rates.csv.
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import cross_val_predict

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p2_lifespan_data as L2  # noqa: E402
import p3_pooled as P  # noqa: E402
import p5_full_pool as F  # noqa: E402
from p2_lifespan_atlas import HEMI  # noqa: E402

OUT = os.path.join(HERE, "p6_out")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(11)
K = 5
REG = P.REGIONS
FLUID = {"lateral ventricles", "inf. lat. ventricles", "third ventricle"}
SIGN = {r: (1.0 if r in FLUID else -1.0) for r in REG}          # multiply z by this: + = abnormal
AD_SET = ["hippocampus", "entorhinal", "amygdala", "inf. lat. ventricles"]


def build():
    allp, D, O, N = F.full_pool()
    licv_mean = float(np.log(allp.icv).mean())
    cs = pd.read_csv(os.path.join(F.ROOT, "data", "oasis_cross-sectional.csv")).set_index("ID")
    # OASIS-1 patients (CDR > 0), primary scan, same QC as the atlas
    p1 = N[(N.dataset == "oasis1") & ~N.healthy & N.primary].copy()
    sess = p1.scan.str.split("__").str[1]
    p1["cdr"] = sess.map(cs.CDR).values
    p1["group"] = np.where(p1.cdr >= 1, "OASIS-1 CDR>=1", "OASIS-1 CDR 0.5")
    p1["cohort"] = "OASIS-1"
    # OASIS-2 baseline of demented and converted people
    o2 = L2.load_oasis2(nondemented_only=False)
    demo = pd.read_csv(os.path.join(F.ROOT, "data", "oasis_longitudinal.csv")).drop_duplicates("Subject ID").set_index("Subject ID")
    o2["grp"] = o2.subject.map(demo.Group)
    o2 = o2.rename(columns={v: k for k, v in F.FS_REG.items()})
    o2["site"] = P.SITE_NAME["OASIS2"]
    b2 = o2[(o2.wave == 1) & o2.grp.isin(["Demented", "Converted"])].copy()
    b2["group"] = "OASIS-2 " + b2.grp.str.lower()
    b2["cohort"] = "OASIS-2"
    pat = pd.concat([p1, b2], ignore_index=True)
    pat = pat[(pat.icv > 0) & pat.sex.notna()].copy()
    pat["licv"] = np.log(pat.icv) - licv_mean
    pat["icv_method"] = "SynthStrip"
    pat["is_patient"] = True
    # held-out controls: OASIS-1 healthy aged >= 55, OASIS-2 nondemented baseline
    ctl_mask = ((allp.dataset == "oasis1") & (allp.age >= 55)) | (allp.dataset == "OASIS2")
    ctl = allp[ctl_mask].copy()
    ctl["group"] = np.where(ctl.dataset == "oasis1", "OASIS-1 controls", "OASIS-2 nondemented")
    ctl["cohort"] = np.where(ctl.dataset == "oasis1", "OASIS-1", "OASIS-2")
    ctl["is_patient"] = False
    ctl["pool_index"] = ctl.index
    pat["pool_index"] = -1
    people = pd.concat([ctl, pat], ignore_index=True)
    people["fold"] = RNG.permutation(len(people)) % K
    return allp, o2, people, licv_mean


def ridge_bag(train, test):
    feats = [h for h in HEMI if h in train and train[h].notna().all() and (train[h] > 0).all() and test[h].notna().all()]
    X = lambda d: np.column_stack([np.log(d[feats].values.astype(float)), np.log(d.icv.values), d.sex.values])
    tr = train[train.age >= 18]
    mu, sd = X(tr).mean(0), X(tr).std(0) + 1e-9
    m = RidgeCV(alphas=np.logspace(-2, 4, 25))
    pred_tr = cross_val_predict(m, (X(tr) - mu) / sd, tr.age.values, cv=5)
    a, b = np.polyfit(tr.age.values, pred_tr - tr.age.values, 1)[::-1]          # bias of the gap vs age
    m.fit((X(tr) - mu) / sd, tr.age.values)
    gap = m.predict((X(test) - mu) / sd) - test.age.values
    sd_h = np.std(pred_tr - tr.age.values - (a + b * tr.age.values))
    return gap - (a + b * test.age.values), sd_h


def main():
    t0 = time.time()
    allp, o2, people, licv_mean = build()
    print("people scored:", people.groupby("group").size().to_dict(), flush=True)
    Z = np.full((len(people), len(REG)), np.nan)
    bag_ridge = np.full(len(people), np.nan)
    bag_atlas = np.full(len(people), np.nan)
    sd_ridge = []
    for k in range(K):
        held = people[people.fold == k]
        drop = held.pool_index[held.pool_index >= 0].values
        train = allp.drop(index=drop)
        test = held.copy()
        zs_train, slopes, sig = {}, {}, {}
        ages = np.arange(18, 98, 1.0)
        for j, r in enumerate(REG):
            d, ly = P.region_data(train, r)
            m, gr = F.fit_region(d, ly, r)
            ok = test[r].notna() & (test[r] > 0)
            Z[test.index[ok], j] = m.z(test[ok], np.log(test.loc[ok, r].values))
            zs_train[r] = pd.Series(m.z(d, ly), index=d.index)
            slopes[r] = m.slope(ages) / 100.0
            sig[r] = F.ref_sigma(m, d, ages)
        # atlas brain-age gap: GLS projection on the ageing direction (z units per year)
        Zt = pd.DataFrame(zs_train).reindex(train.index)
        for i in test.index:
            a = int(np.clip(round(people.age[i]), 18, 97))
            s = np.array([slopes[r][a - 18] / sig[r][a - 18] for r in REG])
            w = (np.abs(train.age.values - a) <= 5)
            Zw = Zt[w].dropna().values
            Si = np.linalg.pinv(LedoitWolf().fit(Zw).covariance_)
            z = Z[i]
            if np.isfinite(z).all():
                bag_atlas[i] = (s @ Si @ z) / (s @ Si @ s)        # years along the ageing direction: + = older
        g, sdh = ridge_bag(train, test)
        bag_ridge[test.index] = g
        sd_ridge.append(sdh)
        print(f"fold {k}: {len(test)} people scored ({time.time()-t0:.0f}s)", flush=True)
    S = people[["group", "cohort", "is_patient", "fold", "age", "sex", "subject"]].copy()
    for j, r in enumerate(REG):
        S["z_" + r] = SIGN[r] * Z[:, j]
    S["bag_ridge"] = bag_ridge
    S["bag_atlas"] = bag_atlas
    S["ad_mean"] = S[["z_" + r for r in AD_SET]].mean(axis=1)
    S["n_outliers"] = (S[["z_" + r for r in REG]] > 1.96).sum(axis=1)
    # learned multivariate score: logistic regression trained on the other folds
    S["learned"] = np.nan
    cols = ["z_" + r for r in REG]
    for k in range(K):
        tr = S[(S.fold != k)].dropna(subset=cols)
        te = S[(S.fold == k)].dropna(subset=cols)
        clf = LogisticRegression(C=0.3, max_iter=2000).fit(tr[cols], tr.is_patient)
        S.loc[te.index, "learned"] = clf.decision_function(te[cols])
    S.to_csv(os.path.join(OUT, "scores.csv"), index=False)
    print("healthy SD of the ridge brain-age gap (training, bias-corrected): %.1f y" % np.mean(sd_ridge))
    summarize(S)
    rates(o2)
    print(f"done in {(time.time()-t0)/60:.1f} min")


def summarize(S):
    scores = ["bag_ridge", "bag_atlas", "z_hippocampus", "z_entorhinal", "z_amygdala", "z_inf. lat. ventricles",
              "z_lateral ventricles", "z_total brain", "z_cortex", "ad_mean", "learned", "n_outliers"]
    rows = []
    for coh, ctl_name in (("OASIS-1", "OASIS-1 controls"), ("OASIS-2", "OASIS-2 nondemented")):
        c = S[(S.cohort == coh) & ~S.is_patient]
        for grp in sorted(S[(S.cohort == coh) & S.is_patient].group.unique()):
            p = S[S.group == grp]
            for sc in scores:
                cv, pv = c[sc].dropna().values, p[sc].dropna().values
                thr = np.percentile(cv, 95)
                auc = stats.mannwhitneyu(pv, cv).statistic / (len(pv) * len(cv))
                boot = []
                for _ in range(500):
                    a, b = RNG.choice(pv, len(pv)), RNG.choice(cv, len(cv))
                    boot.append(stats.mannwhitneyu(a, b).statistic / (len(a) * len(b)))
                rows.append({"comparison": f"{grp} (n={len(pv)}) vs {ctl_name} (n={len(cv)})", "score": sc,
                             "patients_median": np.median(pv), "controls_median": np.median(cv), "controls_sd": np.std(cv),
                             "AUC": auc, "AUC_lo": np.percentile(boot, 2.5), "AUC_hi": np.percentile(boot, 97.5),
                             "sens_at_95spec": float((pv > thr).mean()),
                             "share_beyond_1.96": float((pv > 1.96).mean()) if sc.startswith("z_") or sc == "ad_mean" else np.nan})
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(T.round(2).to_string(index=False), flush=True)


def rates(o2):
    """Within-person change in OASIS-2 (ICV-rescaled, like the atlas): demented / converted / nondemented."""
    o2 = o2[o2.icv > 0].copy()
    mi = o2.groupby("subject").icv.transform("mean")
    rows = []
    for r in ("hippocampus", "entorhinal", "amygdala", "inf. lat. ventricles", "lateral ventricles", "total brain", "cortex"):
        for s, g in o2.groupby("subject"):
            t = g.age.values.astype(float)
            if len(t) < 2 or np.ptp(t) < 1:
                continue
            y = 100 * np.log(g[r].values * mi[g.index].values / g.icv.values)
            rows.append({"region": r, "subject": s, "group": g.grp.iloc[0], "mid_age": t.mean(), "span": np.ptp(t),
                         "visits": len(t), "rate": np.polyfit(t, y, 1)[0]})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "rates.csv"), index=False)
    out = []
    for r, g in R.groupby("region"):
        sign = 1.0 if r in FLUID else -1.0
        c = sign * g[g.group == "Nondemented"].rate.values
        for grp in ("Converted", "Demented"):
            p = sign * g[g.group == grp].rate.values
            for lab, sel in (("all", np.ones(len(p), bool)), (">=2y", g[g.group == grp].span.values >= 2)):
                cc = c if lab == "all" else sign * g[(g.group == "Nondemented") & (g.span >= 2)].rate.values
                pp = p[sel]
                if len(pp) < 5:
                    continue
                out.append({"region": r, "group": grp, "followup": lab, "n_pat": len(pp), "n_ctl": len(cc),
                            "pat_median_rate": sign * np.median(pp), "ctl_median_rate": sign * np.median(cc),
                            "AUC": stats.mannwhitneyu(pp, cc).statistic / (len(pp) * len(cc)),
                            "sens_at_95spec": float((pp > np.percentile(cc, 95)).mean())})
    T = pd.DataFrame(out)
    T.to_csv(os.path.join(OUT, "rates_summary.csv"), index=False)
    print(T.round(2).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
