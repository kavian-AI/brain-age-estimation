#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Does the RATE of brain ageing, measured from images, beat the rate measured from
volumes?

This targets the report's strongest result rather than its baseline comparison.
E8/E9 showed that how fast the brain-age gap GROWS predicts MCI -> dementia
conversion better than the gap's value (+0.048 C-index on ADNI, replicated on
OASIS at +0.047). Everything there came from 6 FreeSurfer volumes. 813 of the
cached ADNI subjects have repeat scans, so the same question can now be asked of
the image itself.

LANDMARK DESIGN (the part that is easy to get wrong)
----------------------------------------------------
Rate is estimated ONLY from scans in [0, L]; conversion is counted ONLY after L;
follow-up is censored at the subject's last known diagnosis. Without the
landmark, a subject who converts early contributes both the steep slope and the
event it caused, and the model scores its own input.

AGE AT VISIT
------------
ADNIMERGE's AGE column is age at BASELINE and is constant within a subject, so a
follow-up scan must be scored against AGE + Years_bl. Using the baseline age at
every visit would make the gap grow by one year per year of follow-up in every
subject, healthy or not, and manufacture a "rate" that is pure bookkeeping.

TRAINING SET
------------
The age model is fitted on CN subjects only, who are disjoint from the MCI
subjects being scored, so no MCI subject is ever in its own training set. The
Cox models are cross-validated by subject.

ARMS
----
  value arms   the gap at the first scan            (what the classic method gives)
  rate arms    slope of the gap over the window     (E8's contribution)
  fair baseline    the volumes themselves, and their rate of change, with age -
                   because a gap that only tracks shrinking hippocampus adds
                   nothing over measuring the hippocampus twice.
"""

import os
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "pilot")
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
N_PCA = 100
LANDMARKS = (1.0, 2.0)
N_REP = 20


def brain_mask(A):
    s = A.std(0)
    return s > 0.01 * s.max()


def load():
    X = np.load(os.path.join(OUT, "adni_vox32.npz"))["X"]
    m = pd.read_csv(os.path.join(OUT, "adni_vox32_meta.csv"))
    a = pd.read_csv(os.path.join(DATA, "ADNIMERGE_05Mar2026.csv"), low_memory=False)
    t = a[["PTID", "VISCODE", "Years_bl"]].drop_duplicates(["PTID", "VISCODE"])
    m = m.merge(t, on=["PTID", "VISCODE"], how="left")
    m["age_visit"] = m.AGE + m.Years_bl          # AGE is baseline age
    hist = a[["PTID", "Years_bl", "DX"]].dropna(subset=["DX"])
    return X, m, hist


def fit_age_model(Xtr, tabtr, agetr, use_img, use_tab):
    """Return a scorer mapping (img, tab, age) -> bias-corrected gap."""
    parts, pack = [], {}
    if use_tab:
        sc = StandardScaler().fit(tabtr)
        parts.append(sc.transform(tabtr))
        pack["tab"] = sc
    if use_img:
        msk = brain_mask(Xtr)
        p = PCA(n_components=min(N_PCA, len(Xtr) - 2), svd_solver="randomized",
                random_state=0).fit(Xtr[:, msk])
        parts.append(p.transform(Xtr[:, msk]) / np.sqrt(p.explained_variance_))
        pack["img"] = (msk, p)
    D = np.hstack(parts)
    mdl = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(D, agetr)
    b = LinearRegression().fit(agetr.reshape(-1, 1), mdl.predict(D) - agetr)

    def score(Xq, tabq, ageq):
        ps = []
        if use_tab:
            ps.append(pack["tab"].transform(tabq))
        if use_img:
            msk, p = pack["img"]
            ps.append(p.transform(Xq[:, msk]) / np.sqrt(p.explained_variance_))
        Dq = np.hstack(ps)
        return (mdl.predict(Dq) - ageq) - b.predict(ageq.reshape(-1, 1))

    return score


def slopes(df, col, tcol="Years_bl"):
    """OLS slope per subject; NaN when a subject has <2 distinct time points."""
    out = {}
    for s, g in df.groupby("PTID"):
        t, v = g[tcol].values, g[col].values
        ok = np.isfinite(t) & np.isfinite(v)
        if ok.sum() < 2 or np.ptp(t[ok]) < 0.2:
            continue
        out[s] = np.polyfit(t[ok], v[ok], 1)[0]
    return pd.Series(out)


def cindex_cv(D, dur, ev, n_rep=N_REP):
    """Subject-level CV C-index for a Cox model on the given columns."""
    n = len(dur)
    sc = []
    for rep in range(n_rep):
        r = np.random.RandomState(rep)
        f = r.permutation(n) % 5
        risk = np.zeros(n)
        for k in range(5):
            tr, te = f != k, f == k
            d = D[tr].copy()
            d["T"], d["E"] = dur[tr], ev[tr]
            try:
                cph = CoxPHFitter(penalizer=0.1).fit(d, "T", "E")
                risk[te] = cph.predict_partial_hazard(D[te]).values
            except Exception:
                risk[te] = 0
        sc.append(concordance_index(dur, -risk, ev))
    return np.array(sc)


def main():
    X, m, hist = load()
    print(f"cached volumes {X.shape}, subjects {m.PTID.nunique()}")

    complete = m.DX.notna() & m.ICV.notna() & m[V].notna().all(1) & m.age_visit.notna()
    m = m[complete].reset_index(drop=True)
    X = X[np.where(complete.values)[0]]

    # --- age model trained on CN subjects only (disjoint from the MCI cohort)
    cn = m[(m.DX == "CN")].drop_duplicates("PTID")
    icn = cn.index.values
    tab_all = np.column_stack([m[V].values, m.ICV.values, m.SEX.values])
    print(f"age model trained on {len(cn)} CN subjects (one scan each)")

    scorers = {}
    for nm, ui, ut in [("TAB", False, True), ("IMG", True, False), ("BOTH", True, True)]:
        scorers[nm] = fit_age_model(X[icn], tab_all[icn], cn.age_visit.values,
                                    ui, ut)
        g = scorers[nm](X[icn], tab_all[icn], cn.age_visit.values)
        print(f"  {nm:5s} in-sample |gap| on CN: {np.abs(g).mean():.2f} y")

    base_dx = m[m.VISCODE == "bl"][["PTID", "DX"]].rename(columns={"DX": "DX_bl"})
    mci = set(base_dx[base_dx.DX_bl == "MCI"].PTID)
    d = m[m.PTID.isin(mci)].copy()
    di = np.where(m.PTID.isin(mci).values)[0]
    for nm, f in scorers.items():
        d[f"gap_{nm}"] = f(X[di], tab_all[di], d.age_visit.values)

    for L in LANDMARKS:
        print("\n" + "=" * 74)
        print(f"LANDMARK L = {L} y   (rate from scans in [0,{L+0.25}], "
              f"conversion counted after {L})")
        print("=" * 74)

        w = d[d.Years_bl <= L + 0.25]
        feats = {}
        for nm in scorers:
            feats[f"rate_{nm}"] = slopes(w, f"gap_{nm}")
            first = w.sort_values("Years_bl").groupby("PTID")[f"gap_{nm}"].first()
            feats[f"value_{nm}"] = first
        # fair baseline: the volumes themselves and how fast they change
        w2 = w.copy()
        w2["hipp"] = w2.Hippocampus / w2.ICV
        feats["rate_hipp"] = slopes(w2, "hipp") * 1e3
        feats["value_hipp"] = w2.sort_values("Years_bl").groupby("PTID").hipp.first() * 1e3

        F = pd.DataFrame(feats).dropna()
        fut = hist[hist.PTID.isin(F.index) & (hist.Years_bl > L)]
        conv = fut[fut.DX == "Dementia"].groupby("PTID").Years_bl.min()
        last = fut.groupby("PTID").Years_bl.max()
        sub = sorted(set(last.index) & set(F.index))
        F = F.loc[sub]
        dur = np.array([conv[s] - L if s in conv.index else last[s] - L for s in sub])
        ev = np.array([1 if s in conv.index else 0 for s in sub])
        ok = dur > 0
        F, dur, ev = F[ok], dur[ok], ev[ok]
        age0 = d.drop_duplicates("PTID").set_index("PTID").AGE.loc[F.index].values
        print(f"  n = {len(F)}   converters = {int(ev.sum())}   "
              f"median follow-up {np.median(dur):.1f} y")
        if ev.sum() < 25:
            print("  too few events; skipping")
            continue

        arms = [
            ("age only", ["__age"]),
            ("hippocampus at baseline", ["value_hipp", "__age"]),
            ("hippocampus RATE", ["rate_hipp", "__age"]),
            ("gap value  (tabular)", ["value_TAB", "__age"]),
            ("gap RATE   (tabular)", ["rate_TAB", "__age"]),
            ("gap value  (image)", ["value_IMG", "__age"]),
            ("gap RATE   (image)", ["rate_IMG", "__age"]),
            ("gap RATE   (both)", ["rate_BOTH", "__age"]),
            ("gap RATE tabular + image", ["rate_TAB", "rate_IMG", "__age"]),
        ]
        F = F.copy()
        F["__age"] = age0
        res = {}
        for nm, cols in arms:
            c = cindex_cv(F[cols], dur, ev)
            res[nm] = c
            print(f"  {nm:28s} C = {c.mean():.3f} "
                  f"[{np.percentile(c,2.5):.3f}, {np.percentile(c,97.5):.3f}]")

        print()
        for lbl, a, b in [
                ("rate vs value (tabular)", "gap RATE   (tabular)", "gap value  (tabular)"),
                ("rate vs value (image)", "gap RATE   (image)", "gap value  (image)"),
                ("image rate vs tabular rate", "gap RATE   (image)", "gap RATE   (tabular)"),
                ("image rate vs hippocampus rate", "gap RATE   (image)", "hippocampus RATE"),
                ("tabular rate vs hippocampus rate", "gap RATE   (tabular)", "hippocampus RATE"),
                ("does image ADD to tabular rate?", "gap RATE tabular + image", "gap RATE   (tabular)")]:
            dd = res[a] - res[b]
            p = 2 * min((dd <= 0).mean(), (dd >= 0).mean())
            print(f"  {lbl:34s} {dd.mean():+.3f}   p={p:.3f}")

        np.savez(os.path.join(OUT, f"exp24_rate_L{L}.npz"),
                 **{k.replace(" ", "_"): v for k, v in res.items()})

    print(f"\nsaved -> {OUT}\\exp24_rate_L*.npz")


if __name__ == "__main__":
    main()
