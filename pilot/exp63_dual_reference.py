#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
THE DUAL-REFERENCE MODEL. (The idea is the thesis author's; the controls are ours.)

Every brain-age paper fits the age model on healthy people and calls a person sick
when their gap is positive. The proposal here turns that around and then doubles it:

  model H   age model fitted on CONTROLS      gap_H > t_H  ->  "sick"
  model D   age model fitted on AD PATIENTS   gap_D < t_D  ->  "healthy"
            (an AD-trained model reads a preserved brain as a YOUNG patient,
             so a healthy person comes out with a negative gap)

  both agree            -> the call is made
  the two contradict    -> FLAGGED for a physician

A contradiction can happen in two geometrically different ways, and they should
not be lumped together:

  H says sick,  D says healthy   the brain is too atrophied for the healthy norm
                                 AND too preserved for the Alzheimer norm:
                                 it FITS NEITHER reference
  H says fine,  D says not-fine  the brain is typical of healthy people AND
                                 typical of patients: it FITS BOTH (the overlap)

WHAT HAS TO BE SHOWN, AND WHAT COULD SINK IT
  1. Does it work at all?  Accuracy on the non-flagged, and how many get flagged.
  2. THE FAIR COMPARISON.  Any single score can abstain too: flag the people
     whose score is closest to its threshold. If model H alone, abstaining on the
     same NUMBER of people, is just as accurate on the rest, the second model has
     added nothing and the flag is only "these are the borderline cases".
  3. WHO GETS FLAGGED - the discovery half. Applied to MCI (neither reference
     group), the four categories are compared on conversion, amyloid, tau and
     cognition. Pre-registered prediction: people who FIT NEITHER are atypical for
     Alzheimer's as well as for health, so they should be ENRICHED for
     amyloid-negative neurodegeneration (SNAP, "suspected non-Alzheimer
     pathophysiology"). The multivariate version, which can detect an atypical
     PATTERN and not only an atypical AMOUNT, is where that should show most.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp53_units_and_dimension as E  # noqa: E402
from lifelines import CoxPHFitter  # noqa: E402
from scipy.stats import chi2, fisher_exact  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

V = E.V
FOLDS = 5
Q_SPEC = 0.90          # calibrated thresholds: 90% of the reference group on the right side


def num(s):
    return pd.to_numeric(s.astype(str).str.replace("<", "").str.replace(">", ""),
                         errors="coerce")


# --------------------------------------------------------------- the models
def dual_gaps(b):
    """Out-of-fold gap under a CONTROL-trained and an AD-trained age model.
    Each model is bias-corrected against its own training group."""
    f = E.folds_of(b.PTID.values)
    ag = b.age_visit.values.astype(float)
    X = np.column_stack([b[v].values for v in V] + [b.ICV.values, b.SEX.values])
    out = {}
    for ref in ["CN", "Dementia"]:
        grp = (b.DX.values == ref)
        g = np.zeros(len(b))
        for k in range(FOLDS):
            tr, te = np.where((f != k) & grp)[0], np.where(f == k)[0]
            sc = StandardScaler().fit(X[tr])
            m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(X[tr]), ag[tr])
            bb = LinearRegression().fit(ag[tr].reshape(-1, 1),
                                        m.predict(sc.transform(X[tr])) - ag[tr])
            g[te] = (m.predict(sc.transform(X[te])) - ag[te]) \
                - bb.predict(ag[te].reshape(-1, 1))
        out[ref] = g
    return out["CN"], out["Dementia"]


def dual_normative(b):
    """Six-region deviation under each reference, out of fold, plus that
    reference's covariance - the ingredients of a Mahalanobis typicality test."""
    f = E.folds_of(b.PTID.values)
    ag = b.age_visit.values.astype(float)
    Xc = np.column_stack([ag, ag ** 2, b.SEX.values, b.ICV.values])
    Y = np.column_stack([b[v].values for v in V]).astype(float)
    res = {}
    for ref in ["CN", "Dementia"]:
        grp = (b.DX.values == ref)
        Z = np.zeros_like(Y)
        for k in range(FOLDS):
            tr, te = np.where((f != k) & grp)[0], np.where(f == k)[0]
            sc = StandardScaler().fit(Xc[tr])
            lr = LinearRegression().fit(sc.transform(Xc[tr]), Y[tr])
            r = Y[tr] - lr.predict(sc.transform(Xc[tr]))
            sd = r.std(0, ddof=1)
            Z[te] = (Y[te] - lr.predict(sc.transform(Xc[te]))) / sd
        C = np.cov(Z[grp].T)
        Ci = np.linalg.inv(C + 1e-6 * np.eye(len(V)))
        res[ref] = np.einsum("ij,jk,ik->i", Z, Ci, Z)      # squared Mahalanobis
    return res["CN"], res["Dementia"]


# --------------------------------------------------------------- scoring
def categories(sick_H, healthy_D):
    c = np.full(len(sick_H), "", dtype=object)
    c[sick_H & ~healthy_D] = "SICK (agree)"
    c[~sick_H & healthy_D] = "HEALTHY (agree)"
    c[sick_H & healthy_D] = "FLAG: fits neither"
    c[~sick_H & ~healthy_D] = "FLAG: fits both"
    return c


def selective_single(score, y, n_flag):
    """Model H alone, abstaining on the n_flag people nearest its own threshold.
    Threshold is the one that maximises balanced accuracy, so the single model
    gets its best shot."""
    ts = np.quantile(score, np.linspace(0.05, 0.95, 181))
    best = None
    for t in ts:
        pred = (score > t).astype(int)
        ba = 0.5 * ((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean())
        if best is None or ba > best[0]:
            best = (ba, t)
    t = best[1]
    keep = np.argsort(-np.abs(score - t))[:len(score) - n_flag]
    pred = (score[keep] > t).astype(int)
    return (pred == y[keep]).mean(), 0.5 * ((pred[y[keep] == 1] == 1).mean()
                                             + (pred[y[keep] == 0] == 0).mean())


def main():
    a, b = E.load()
    gH, gD = dual_gaps(b)
    b = b.assign(gap_H=gH, gap_D=gD)
    print("=" * 90)
    print("WHAT THE TWO MODELS SAY ABOUT EACH GROUP (mean gap, years)")
    print("=" * 90)
    for g in ["CN", "MCI", "Dementia"]:
        q = b.DX.values == g
        print(f"  {g:9s} n={q.sum():4d}   healthy-trained gap {gH[q].mean():+6.2f}"
              f"   AD-trained gap {gD[q].mean():+6.2f}")
    print(f"  correlation between the two gaps: {np.corrcoef(gH, gD)[0,1]:+.3f}")

    # ---------------------------------------------------- PART 1: AD vs CN
    sel = b.DX.isin(["CN", "Dementia"]).values
    S = b[sel].reset_index(drop=True)
    y = (S.DX.values == "Dementia").astype(int)
    rng = np.random.RandomState(0)
    half = rng.rand(len(S)) < 0.5
    print("\n" + "=" * 90)
    print("PART 1  AD vs CN - thresholds set on one half, scored on the other, then swapped")
    print("=" * 90)
    for rule in ["literal (thresholds at zero)", "calibrated (90% of each reference)"]:
        acc, bal, cov, nflag, sing, singb = [], [], [], [], [], []
        types = {"FLAG: fits neither": 0, "FLAG: fits both": 0}
        for cal, ev in [(half, ~half), (~half, half)]:
            if rule.startswith("literal"):
                tH, tD = 0.0, 0.0
            else:
                tH = np.quantile(S.gap_H.values[cal & (y == 0)], Q_SPEC)
                tD = np.quantile(S.gap_D.values[cal & (y == 1)], 1 - Q_SPEC)
            c = categories(S.gap_H.values[ev] > tH, S.gap_D.values[ev] < tD)
            yy = y[ev]
            agree = np.isin(c, ["SICK (agree)", "HEALTHY (agree)"])
            pred = (c[agree] == "SICK (agree)").astype(int)
            acc.append((pred == yy[agree]).mean())
            bal.append(0.5 * ((pred[yy[agree] == 1] == 1).mean()
                              + (pred[yy[agree] == 0] == 0).mean()))
            cov.append(agree.mean())
            nflag.append((~agree).sum())
            for t in types:
                types[t] += (c == t).sum()
            a1, b1 = selective_single(S.gap_H.values[ev], yy, (~agree).sum())
            sing.append(a1)
            singb.append(b1)
        print(f"\n  {rule}")
        print(f"    decided {100*np.mean(cov):.1f}% of people, flagged {100*(1-np.mean(cov)):.1f}%"
              f"   ({types['FLAG: fits neither']} fit neither, {types['FLAG: fits both']} fit both)")
        print(f"    DUAL model on the decided:      accuracy {100*np.mean(acc):.1f}%"
              f"   balanced {100*np.mean(bal):.1f}%")
        print(f"    SINGLE model H, same # flagged: accuracy {100*np.mean(sing):.1f}%"
              f"   balanced {100*np.mean(singb):.1f}%")
        print(f"    -> dual minus single: {100*(np.mean(acc)-np.mean(sing)):+.1f} points accuracy,"
              f" {100*(np.mean(bal)-np.mean(singb)):+.1f} balanced")

    # bootstrap the calibrated comparison, since it is the one that matters
    dd = []
    rng = np.random.RandomState(1)
    for _ in range(300):
        i = rng.randint(0, len(S), len(S))
        Si, yi = S.iloc[i].reset_index(drop=True), y[i]
        hf = rng.rand(len(Si)) < 0.5
        r_ = []
        for cal, ev in [(hf, ~hf), (~hf, hf)]:
            if (yi[cal] == 0).sum() < 20 or (yi[cal] == 1).sum() < 20:
                continue
            tH = np.quantile(Si.gap_H.values[cal & (yi == 0)], Q_SPEC)
            tD = np.quantile(Si.gap_D.values[cal & (yi == 1)], 1 - Q_SPEC)
            c = categories(Si.gap_H.values[ev] > tH, Si.gap_D.values[ev] < tD)
            ag_ = np.isin(c, ["SICK (agree)", "HEALTHY (agree)"])
            if ag_.sum() < 30:
                continue
            pr = (c[ag_] == "SICK (agree)").astype(int)
            s1, _ = selective_single(Si.gap_H.values[ev], yi[ev], (~ag_).sum())
            r_.append((pr == yi[ev][ag_]).mean() - s1)
        if r_:
            dd.append(np.mean(r_))
    dd = np.array(dd)
    lo, hi = np.percentile(dd, [2.5, 97.5])
    p = 2 * min((dd <= 0).mean(), (dd >= 0).mean())
    print(f"\n  calibrated dual minus single at matched flag rate, bootstrap:"
          f" {100*dd.mean():+.1f} points [{100*lo:+.1f}, {100*hi:+.1f}] p={p:.3f}"
          f"  {'REAL' if p < 0.05 else 'not distinguishable'}")

    # ---------------------------------------------------- PART 2 + 3: MCI
    tH = np.quantile(gH[b.DX.values == "CN"], Q_SPEC)
    tD = np.quantile(gD[b.DX.values == "Dementia"], 1 - Q_SPEC)
    dH, dD = dual_normative(b)
    crit = chi2.ppf(Q_SPEC, df=len(V))
    M = b[b.DX.values == "MCI"].copy()
    qM = b.DX.values == "MCI"
    M["cat_gap"] = categories(gH[qM] > tH, gD[qM] < tD)
    M["cat_mv"] = categories(dH[qM] > crit, dD[qM] < crit)   # atypical for H, typical for AD ...
    # for the multivariate version the natural labels are about TYPICALITY:
    typH, typD = dH[qM] <= crit, dD[qM] <= crit
    mv = np.full(qM.sum(), "", dtype=object)
    mv[typH & ~typD] = "HEALTHY-like only"
    mv[~typH & typD] = "AD-like only"
    mv[typH & typD] = "FLAG: typical of both"
    mv[~typH & ~typD] = "FLAG: typical of NEITHER"
    M["cat_mv"] = mv

    # biomarkers at baseline
    bl = a[a.VISCODE == "bl"].drop_duplicates("PTID").set_index("PTID")
    for c in ["ABETA", "PTAU", "TAU", "AV45", "FDG", "ADAS13", "MMSE", "CDRSB", "APOE4"]:
        M[c] = num(bl.reindex(M.PTID)[c]).values
    Apos = np.where(M.AV45.notna(), M.AV45 > 1.11, np.where(M.ABETA.notna(), M.ABETA < 980, np.nan))
    M["A_pos"] = Apos.astype(float)
    M["T_pos"] = np.where(M.PTAU.notna(), (M.PTAU > 23).astype(float), np.nan)

    keep, dur, ev = E.survival(a, M.PTID.values, ["Dementia"])
    o = {s: i for i, s in enumerate(keep)}
    M["T_dur"] = [dur[o[s]] if s in o else np.nan for s in M.PTID]
    M["E_ev"] = [ev[o[s]] if s in o else np.nan for s in M.PTID]

    for col, title, ref in [("cat_gap", "PART 2  MCI SORTED BY THE DUAL GAP MODEL", "HEALTHY (agree)"),
                            ("cat_mv", "PART 3  MCI SORTED BY DUAL MULTIVARIATE TYPICALITY",
                             "HEALTHY-like only")]:
        print("\n" + "=" * 90)
        print(title)
        print("=" * 90)
        print(f"  {'category':26s} {'n':>4s} {'conv%':>6s} {'HR vs ref':>10s} {'A+%':>6s} "
              f"{'T+%':>6s} {'ADAS13':>7s} {'MMSE':>6s} {'APOE4+%':>8s} {'age':>5s}")
        cats = [c for c in M[col].unique() if c]
        D = M.dropna(subset=["T_dur", "E_ev"]).copy()
        hrs = {}
        for c in cats:
            if c == ref:
                continue
            q = D[col].isin([c, ref])
            if (D[col] == c).sum() < 10:
                continue
            X = pd.DataFrame({"T": D.T_dur[q], "E": D.E_ev[q],
                              "x": (D[col][q] == c).astype(int), "age": D.age_visit[q]})
            try:
                cf = CoxPHFitter().fit(X, "T", "E")
                hrs[c] = (np.exp(cf.params_["x"]), *np.exp(cf.confidence_intervals_.loc["x"]),
                          cf.summary.loc["x", "p"])
            except Exception:
                pass
        for c in sorted(cats, key=lambda s: (s != ref, s)):
            q = M[col] == c
            qd = D[col] == c
            hr = hrs.get(c)
            hrs_ = "  ref" if c == ref else (f"{hr[0]:5.2f}" if hr else "   - ")
            print(f"  {c:26s} {q.sum():4d} {100*D.E_ev[qd].mean():6.1f} {hrs_:>10s} "
                  f"{100*np.nanmean(M.A_pos[q]):6.1f} {100*np.nanmean(M.T_pos[q]):6.1f} "
                  f"{np.nanmean(M.ADAS13[q]):7.1f} {np.nanmean(M.MMSE[q]):6.1f} "
                  f"{100*np.nanmean(M.APOE4[q] > 0):8.1f} {M.age_visit[q].mean():5.1f}")
        for c, hr in hrs.items():
            print(f"    HR {c:26s} {hr[0]:.2f} [{hr[1]:.2f}, {hr[2]:.2f}] p={hr[3]:.4f}")

        # the pre-registered test: is 'neither' enriched for amyloid-NEGATIVE?
        neither = [c for c in cats if "NEITHER" in c.upper()]
        if neither:
            nq = (M[col] == neither[0]) & M.A_pos.notna()
            oq = (M[col] != neither[0]) & M.A_pos.notna() & (M[col] != "")
            tab = [[int((M.A_pos[nq] == 0).sum()), int((M.A_pos[nq] == 1).sum())],
                   [int((M.A_pos[oq] == 0).sum()), int((M.A_pos[oq] == 1).sum())]]
            if min(sum(tab[0]), sum(tab[1])) >= 5:
                orr, pp = fisher_exact(tab)
                print(f"\n  PRE-REGISTERED: amyloid-NEGATIVE among '{neither[0]}' "
                      f"{100*tab[0][0]/max(1,sum(tab[0])):.1f}% (n={sum(tab[0])})  vs everyone else "
                      f"{100*tab[1][0]/max(1,sum(tab[1])):.1f}% (n={sum(tab[1])})"
                      f"   odds ratio {orr:.2f}  Fisher p={pp:.4f}")
                snap = nq & (M.T_pos == 0)
                print(f"  and among those amyloid-negative in '{neither[0]}': "
                      f"converted {100*np.nanmean(M.E_ev[(M[col]==neither[0]) & (M.A_pos==0)]):.1f}%")

    # ---------------------------------------------------- PART 4
    print("\n" + "=" * 90)
    print("PART 4  DOES THE FLAG CARRY PROGNOSTIC INFORMATION BEYOND THE GAP ITSELF?")
    print("=" * 90)
    D = M.dropna(subset=["T_dur", "E_ev"]).copy()
    X = pd.DataFrame({"T": D.T_dur, "E": D.E_ev, "gap_H": D.gap_H, "age": D.age_visit,
                      "flag_neither": (D.cat_gap == "FLAG: fits neither").astype(int),
                      "flag_both": (D.cat_gap == "FLAG: fits both").astype(int),
                      "gap_D": D.gap_D})
    for cols in [["gap_H", "age"], ["gap_H", "gap_D", "age"],
                 ["gap_H", "age", "flag_neither", "flag_both"]]:
        cf = CoxPHFitter(penalizer=0.01).fit(X[cols + ["T", "E"]], "T", "E")
        print(f"    {' + '.join(cols):40s} C = {cf.concordance_index_:.3f}   "
              + "  ".join(f"{c} p={cf.summary.loc[c,'p']:.3f}" for c in cols if c != "age"))


if __name__ == "__main__":
    main()
