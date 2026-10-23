#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PREDICTING BRAIN AGE ACROSS DISORDERS FROM PUBLISHED EFFECT MAPS ALONE.

E67 showed that the alignment eta = cos^2_S(w, S^-1 mu) predicts the disease AUC of
any linear brain summary (r = 0.96 on 500 unseen summaries), and E67b showed it
survives with a healthy-only covariance and a published-style Cohen's d map
(r = 0.95, error 0.02 AUC). So the law can be run on data nobody here has.

The ENIGMA Toolbox publishes regional case-control Cohen's d maps (68 cortical
thickness + 16 subcortical volumes) for a dozen disorders. Here:
  - a brain-age model is trained on ADNI controls in EXACTLY that feature space
  - its weights are expressed in normative-deviation coordinates
  - the healthy covariance comes from the same controls
  - for every disorder, from its d map alone, the law predicts:
        the brain-age gap's effect size (Cohen's d of patients vs controls)
        the best effect size any linear summary could reach
        eta: the share of that best separation brain age captures
        and the same numbers for hippocampal volume alone
Sign conventions were checked per file against the original ENIGMA papers: every
file is patients-minus-controls except schizophrenia, which is reversed (its
hippocampus, thalamus and cortex come out positive, against van Erp 2016/2018).

CAVEAT carried into every number: the controls are ADNI (55-90 y), the brain-age
model and covariance are therefore late-life, while most ENIGMA psychiatric cohorts
are young adults. Predictions are about the SHAPE of each disorder relative to
late-life aging, and are compared with published values on rank order first.
"""

import glob
import io
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp53_units_and_dimension as E  # noqa: E402
import tadpole_atlas as TA  # noqa: E402
from sklearn.linear_model import LinearRegression, RidgeCV  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

BASE = E.BASE
ENIG = os.path.join(BASE, "data", "enigma")
XP = TA.PIPE["X"]
SUBMAP = {"LLatVent": (37, "SV"), "RLatVent": (96, "SV"), "Lthal": (61, "SV"), "Rthal": (120, "SV"),
          "Lcaud": (16, "SV"), "Rcaud": (75, "SV"), "Lput": (53, "SV"), "Rput": (112, "SV"),
          "Lpal": (42, "SV"), "Rpal": (101, "SV"), "Lhippo": (29, "SV"), "Rhippo": (88, "SV"),
          "Lamyg": (12, "SV"), "Ramyg": (71, "SV"), "Laccumb": (11, "SV"), "Raccumb": (70, "SV")}
DISORDERS = {
    "schizophrenia": ("Schizophrenia_case-controls_CortThick", "Schizophrenia_case-control_SubVol", -1),
    "bipolar (adult)": ("bd_case-controls_CortThick_adult", "bd_case-controls_SubVol_typeI", 1),
    "depression (adult)": ("mddadult_case-controls_CortThick", "mdd_case-controls_SubVol", 1),
    "OCD (adult)": ("ocdadults_case-controls_CortThick", "ocdadult_case-controls_SubVol", 1),
    "ADHD (adult)": ("adhdadult_case-controls_CortThick", "adhdadult_case-controls_SubVol", 1),
    "autism": ("asd_meta-analysis_case-controls_CortThick", "asd_meta-analysis_case-controls_SubVol", 1),
    "epilepsy (all)": ("allepi_case-controls_CortThick", "allepi_case-controls_SubVol", 1),
    "22q11.2 deletion": ("22q_case-controls_CortThick", "22q_case-controls_SubVol", 1),
    "Parkinson HY1": ("parkinsons_case-controls_CortThick_HY1PDvsCN", "parkinsons_case-controls_SubVol_HY1PDvsCN", 1),
    "Parkinson HY2": ("parkinsons_case-controls_CortThick_HY2PDvsCN", "parkinsons_case-controls_SubVol_HY2PDvsCN", 1),
    "Parkinson HY3": ("parkinsons_case-controls_CortThick_HY3PDvsCN", "parkinsons_case-controls_SubVol_HY3PDvsCN", 1),
    "Parkinson HY4-5": ("parkinsons_case-controls_CortThick_HY4PDandHY5PDvsCN",
                        "parkinsons_case-controls_SubVol_HY4PDandHY5PDvsCN", 1),
}


def read_map(name):
    t = open(os.path.join(ENIG, name + ".csv"), encoding="utf-8", errors="ignore").read()
    d = pd.read_csv(io.StringIO(t), sep=None, engine="python", on_bad_lines="skip")
    st = d.iloc[:, 1].astype(str).str.replace("_thickavg", "").str.replace("_asy_thick.csv", "")
    return dict(zip(st, pd.to_numeric(d.iloc[:, 2], errors="coerce")))


def main():
    # ---------------------------------------------------------- feature space
    ct_names = [f"{s}_{nm}" for s in ("L", "R") for nm in TA.CORTICAL.values()]
    ct_codes = [(n + off, "TA") for off in (0, 59) for n in TA.CORTICAL.keys()]
    sv_names = list(SUBMAP.keys())
    cols = [f"ST{c}{k}_{XP}" for c, k in ct_codes] + [f"ST{c}{k}_{XP}" for c, k in SUBMAP.values()]
    feat_names = ct_names + sv_names
    t = pd.read_csv(TA.TADPOLE, low_memory=False, usecols=["PTID", "VISCODE"] + cols
                    + [f"ST10CV_{XP}"])
    t = t[t.VISCODE == "bl"].drop_duplicates("PTID")
    F = pd.DataFrame({"PTID": t.PTID.values, "ICV_T": pd.to_numeric(t[f"ST10CV_{XP}"], errors="coerce")})
    for nm, c in zip(feat_names, cols):
        F[nm] = pd.to_numeric(t[c], errors="coerce").values
    a, b = E.load()
    m = b.merge(F, on="PTID").dropna(subset=feat_names + ["ICV_T"]).reset_index(drop=True)
    cn = m.DX.values == "CN"
    C = m[cn].reset_index(drop=True)
    print(f"ADNI controls in ENIGMA feature space: n={len(C)} "
          f"(66 cortical thickness + 16 subcortical volumes), age {C.age_visit.min():.0f}-{C.age_visit.max():.0f}")

    ag = C.age_visit.values.astype(float)
    Xc = np.column_stack([ag, ag ** 2, C.SEX.values, C.ICV_T.values])
    Y = C[feat_names].values.astype(float)
    sc = StandardScaler().fit(Xc)
    lr = LinearRegression().fit(sc.transform(Xc), Y)
    R_ = Y - lr.predict(sc.transform(Xc))
    sd = R_.std(0, ddof=1)
    Z = R_ / sd                                           # normative deviations, controls
    S = np.cov(Z.T)
    S = 0.9 * S + 0.1 * np.diag(np.diag(S))

    # brain-age model in the same space, out of fold, then its weights in z coords
    Xa = np.column_stack([Y, C.ICV_T.values, C.SEX.values])
    f = E.folds_of(C.PTID.values)
    g = np.zeros(len(C))
    mae = []
    for k in range(5):
        tr, te = f != k, f == k
        s2 = StandardScaler().fit(Xa[tr])
        md = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(s2.transform(Xa[tr]), ag[tr])
        p = md.predict(s2.transform(Xa[te]))
        mae.append(np.abs(p - ag[te]))
        bb = LinearRegression().fit(ag[tr].reshape(-1, 1), md.predict(s2.transform(Xa[tr])) - ag[tr])
        g[te] = (p - ag[te]) - bb.predict(ag[te].reshape(-1, 1))
    w_age = LinearRegression().fit(Z, g).coef_
    print(f"brain-age model: MAE {np.concatenate(mae).mean():.2f} y in controls; "
          f"variance of gap explained by the z-space weights "
          f"{LinearRegression().fit(Z, g).score(Z, g):.2f}")
    hippo = np.array([n in ("Lhippo", "Rhippo") for n in feat_names], float)
    # sign: in z coordinates a smaller hippocampus is negative; flip so larger = worse
    w_hip = -hippo

    def predict(mu):
        dstar = np.sqrt(mu @ np.linalg.solve(S, mu))
        out = {}
        for nm, w in [("brain age", w_age), ("hippocampus", w_hip)]:
            d = (w @ mu) / np.sqrt(w @ S @ w)
            out[nm] = (d, d ** 2 / dstar ** 2)
        return dstar, out

    # ---------------------------------------------------------- AD as a known anchor
    Aall = m[m.DX.isin(["CN", "Dementia"])]
    Za = ((Aall[feat_names].values - lr.predict(sc.transform(np.column_stack(
        [Aall.age_visit, Aall.age_visit ** 2, Aall.SEX, Aall.ICV_T])))) / sd)
    ya = (Aall.DX == "Dementia").values
    pool_sd = np.sqrt(0.5 * (Za[ya].var(0, ddof=1) + Za[~ya].var(0, ddof=1)))
    d_ad = (Za[ya].mean(0) - Za[~ya].mean(0)) / pool_sd

    rows = [("Alzheimer (ADNI, anchor)", d_ad)]
    for dis, (ctf, svf, sgn) in DISORDERS.items():
        ct, sv = read_map(ctf), read_map(svf)
        v = []
        for nm in feat_names:
            if nm in sv_names:
                x = sv.get(nm, np.nan)
                if nm.endswith("LatVent") and dis == "schizophrenia":
                    x = np.nan                               # missing in that file (0, CI 0-0)
            else:
                x = ct.get(nm, np.nan)
            v.append(sgn * x if np.isfinite(x) else np.nan)
        rows.append((dis, np.array(v)))

    print("\n" + "=" * 100)
    print("PREDICTIONS FROM EFFECT MAPS ALONE (Cohen's d, patients vs controls)")
    print("=" * 100)
    print(f"  {'disorder':26s} {'regions':>7s} {'best d*':>8s} {'brain-age d':>12s} {'eta_BA':>7s}"
          f" {'hippocampus d':>14s} {'eta_hip':>8s}  {'brain age uses':>15s}")
    out = []
    for dis, v in rows:
        ok = np.isfinite(v)
        mu = np.where(ok, v, 0.0)
        dstar, pr = predict(mu)
        dba, eba = pr["brain age"]
        dh, eh = pr["hippocampus"]
        out.append((dis, ok.sum(), dstar, dba, eba, dh, eh))
        print(f"  {dis:26s} {ok.sum():7d} {dstar:8.2f} {dba:+12.2f} {eba:7.3f} {dh:+14.2f} {eh:8.3f}"
              f"  {100*eba:13.0f}%")
    O = pd.DataFrame(out, columns=["disorder", "n_regions", "dstar", "d_ba", "eta_ba", "d_hip", "eta_hip"])
    O.to_csv(os.path.join(BASE, "pilot", "exp68_enigma_predictions.csv"), index=False)
    # the Parkinson severity gradient
    P = O[O.disorder.str.startswith("Parkinson")]
    print("\n  Parkinson by Hoehn-Yahr stage: predicted brain-age d "
          + " -> ".join(f"{x:+.2f}" for x in P.d_ba) + "   eta "
          + " -> ".join(f"{x:.2f}" for x in P.eta_ba))


if __name__ == "__main__":
    main()
