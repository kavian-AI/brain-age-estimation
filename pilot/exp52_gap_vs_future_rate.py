#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DOES THE BRAIN-AGE GAP PREDICT HOW FAST THE BRAIN WILL ACTUALLY CHANGE?

The entire interpretive foundation of brain age is that a positive gap means
"this brain is aging faster". Vidal-Pineiro et al. (2021, PMC8580481) reported no
association between the cross-sectional gap and the longitudinal rate of brain
change, and attributed the gap to early-life factors instead. If that is right,
the standard interpretation is simply wrong: the gap is a level, not a speed.

Before believing either answer, two things have to be handled.

THE SHARED-SCAN ARTEFACT. The baseline gap and the rate are normally computed
from overlapping scans. Noise e in the baseline scan inflates the gap AND
deflates the rate, because the rate is (later - baseline)/dt. That manufactures a
NEGATIVE correlation out of nothing. A true positive association and this
artefact can cancel to exactly zero - which is what "no association" looks like.
    FIX: the gap comes from scan 1. The rate is fitted through scans 2..n ONLY.
    No shared scan, no shared noise. Both versions are reported so the size of
    the artefact is visible.

MEASURABILITY. If an individual's rate of change is mostly noise, then a null
correlation says nothing about biology - it says the measurement failed. So the
reliability of both quantities is estimated first, and the resulting attenuation
ceiling is printed BEFORE any correlation is interpreted.

And the question E48 raises: if the gap does not predict the rate, does the
better single number do any better? Same test, same subjects.
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler

BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
DATA = os.path.join(BASE, "data")
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
FA = {"Hippocampus": "هیپوکامپ", "Entorhinal": "آنتورینال", "Fusiform": "دوکی",
      "MidTemp": "گیجگاهی میانی", "Ventricles": "بطن‌ها", "WholeBrain": "کل مغز"}
GFA = {"CN": "سالم", "MCI": "MCI", "Dementia": "آلزایمر"}
FOLDS, N_BOOT = 5, 2000


def load():
    a = pd.read_csv(os.path.join(DATA, "ADNIMERGE_05Mar2026.csv"), low_memory=False)
    d = a.dropna(subset=["AGE", "Years_bl", "ICV", "PTGENDER", "DX"] + V).copy()
    d["SEX"] = (d.PTGENDER == "Male").astype(int)
    d["age_visit"] = d.AGE + d.Years_bl
    d = d.sort_values(["PTID", "Years_bl"]).reset_index(drop=True)
    d["DX0"] = d.PTID.map(d.groupby("PTID").DX.first())
    return d


def folds_of(d, seed=0):
    uq = np.unique(d.PTID.values)
    fm = dict(zip(uq, np.random.RandomState(seed).permutation(len(uq)) % FOLDS))
    return np.array([fm[s] for s in d.PTID.values])


def gap_every_visit(d):
    """CN-trained, bias-corrected brain-age gap at EVERY visit, out of fold."""
    f = folds_of(d)
    cn = (d.DX == "CN").values
    age = d.age_visit.values.astype(float)
    X = np.column_stack([d[v].values for v in V] + [d.ICV.values, d.SEX.values])
    g = np.zeros(len(d))
    for k in range(FOLDS):
        tr, te = np.where((f != k) & cn)[0], np.where(f == k)[0]
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(X[tr]), age[tr])
        b = LinearRegression().fit(age[tr].reshape(-1, 1),
                                   m.predict(sc.transform(X[tr])) - age[tr])
        g[te] = (m.predict(sc.transform(X[te])) - age[te]) \
            - b.predict(age[te].reshape(-1, 1))
    return g


def z_every_visit(d):
    """CN-trained normative deviation per region at every visit, out of fold."""
    f = folds_of(d)
    cn = (d.DX == "CN").values
    age = d.age_visit.values.astype(float)
    X = np.column_stack([age, age ** 2, d.SEX.values, d.ICV.values])
    Z = np.zeros((len(d), len(V)))
    for j, v in enumerate(V):
        y = d[v].values.astype(float)
        for k in range(FOLDS):
            tr, te = np.where((f != k) & cn)[0], np.where(f == k)[0]
            sc = StandardScaler().fit(X[tr])
            m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(X[tr]), y[tr])
            s = (y[tr] - m.predict(sc.transform(X[tr]))).std(ddof=1)
            Z[te, j] = (y[te] - m.predict(sc.transform(X[te]))) / max(s, 1e-9)
    return Z


def cn_sd(d):
    b = d[(d.VISCODE == "bl") & (d.DX == "CN")]
    ag = b.age_visit.values
    X = np.column_stack([ag, ag ** 2, b.SEX.values, b.ICV.values])
    out = {}
    for v in V:
        y = b[v].values.astype(float)
        out[v] = (y - LinearRegression().fit(X, y).predict(X)).std(ddof=1)
    return out


def slope(t, y):
    return np.polyfit(t, y, 1)[0]


def partial_r(x, y, C):
    """Correlation of x and y after both are residualised on C."""
    C = np.column_stack([np.ones(len(x))] + list(C.T))
    rx = x - C @ np.linalg.lstsq(C, x, rcond=None)[0]
    ry = y - C @ np.linalg.lstsq(C, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def boot_r(x, y, C, n=N_BOOT):
    rng = np.random.RandomState(0)
    out = []
    for _ in range(n):
        i = rng.randint(0, len(x), len(x))
        try:
            out.append(partial_r(x[i], y[i], C[i]))
        except Exception:
            pass
    o = np.array(out)
    return np.percentile(o, 2.5), np.percentile(o, 97.5)


def line(label, r, ci, ceil):
    lo, hi = ci
    sig = "REAL" if (lo > 0 or hi < 0) else "indistinguishable from zero"
    frac = f"{100*abs(r)/ceil:.0f}% of ceiling" if ceil > 0.05 else "-"
    print(f"    {label:34s} r = {r:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
          f"   {sig:26s} {frac}")


def main():
    d = load()
    sd = cn_sd(d)
    G = gap_every_visit(d)
    Z = z_every_visit(d)
    d = d.assign(gap=G)
    for j, v in enumerate(V):
        d["z_" + v] = Z[:, j]

    # ---- the healthy aging direction, from CN subjects (a direction, not a
    #      prediction: used only to turn a 6-vector rate into one number)
    cn_rates = []
    for s, g in d[d.DX0 == "CN"].groupby("PTID"):
        t = g.Years_bl.values.astype(float)
        if len(g) >= 3 and t.max() - t.min() >= 2:
            cn_rates.append([slope(t, g[v].values.astype(float)) / sd[v] for v in V])
    u = np.mean(cn_rates, axis=0)
    u = u / np.linalg.norm(u)
    print("healthy aging direction (unit vector), from "
          f"{len(cn_rates)} controls:")
    print("  " + "  ".join(f"{FA[v]} {w:+.3f}" for v, w in zip(V, u)))

    # ---- per subject: gap at scan 1, rate from scans 2..n  (no shared scan)
    rows = []
    for s, g in d.groupby("PTID"):
        g = g.sort_values("Years_bl")
        if len(g) < 4:
            continue
        t = g.Years_bl.values.astype(float)
        if t.max() - t.min() < 2.0:
            continue
        fut = g.iloc[1:]
        tf = fut.Years_bl.values.astype(float)
        if len(fut) < 3 or tf.max() - tf.min() < 1.5:
            continue
        rv_f = np.array([slope(tf, fut[v].values.astype(float)) / sd[v] for v in V])
        rv_c = np.array([slope(t, g[v].values.astype(float)) / sd[v] for v in V])
        # within-subject noise of the gap: residual of a line through its visits
        gg = g.gap.values
        res = gg - np.polyval(np.polyfit(t, gg, 1), t)
        r = dict(PTID=s, DX0=g.DX0.values[0], age=g.age_visit.values[0],
                 SEX=g.SEX.values[0], n_vis=len(g), span=t.max() - t.min(),
                 gap0=gg[0], rate_future=float(rv_f @ u), rate_shared=float(rv_c @ u),
                 gaprate_future=slope(tf, fut.gap.values),
                 gap_noise_sq=float((res ** 2).sum() / max(len(t) - 2, 1)),
                 gap_icc_within=float(np.var(gg, ddof=1)) if len(gg) > 1 else np.nan)
        for j, v in enumerate(V):
            r["z0_" + v] = g["z_" + v].values[0]
        # split-half reliability of the FUTURE rate: odd vs even future visits
        if len(fut) >= 6:
            a_, b_ = fut.iloc[0::2], fut.iloc[1::2]
            ta, tb = a_.Years_bl.values.astype(float), b_.Years_bl.values.astype(float)
            if len(a_) >= 3 and len(b_) >= 3 and ta.max() - ta.min() > 1 \
                    and tb.max() - tb.min() > 1:
                ra = np.array([slope(ta, a_[v].values.astype(float)) / sd[v] for v in V])
                rb = np.array([slope(tb, b_[v].values.astype(float)) / sd[v] for v in V])
                r["rate_half_a"], r["rate_half_b"] = float(ra @ u), float(rb @ u)
        rows.append(r)
    S = pd.DataFrame(rows)
    print(f"\nsubjects with >=4 scans, span >=2 y, and >=3 FUTURE scans: {len(S)}")
    print(S.DX0.value_counts().to_string())

    # ---- E48's component, fitted on baseline deviations of these subjects
    Z0 = S[[f"z0_{v}" for v in V]].values
    pc = PCA(n_components=1, random_state=0).fit(StandardScaler().fit_transform(Z0))
    S["pc0"] = pc.transform(StandardScaler().fit_transform(Z0))[:, 0]
    if np.corrcoef(S.pc0, S.z0_Hippocampus)[0, 1] > 0:   # sign convention: worse = higher
        S["pc0"] = -S.pc0

    # ================================================== MEASURABILITY FIRST
    print("\n" + "=" * 78)
    print("MEASURABILITY — established BEFORE any correlation is interpreted")
    print("=" * 78)
    H = S.dropna(subset=["rate_half_a", "rate_half_b"])
    r_half = np.corrcoef(H.rate_half_a, H.rate_half_b)[0, 1]
    rel_rate = 2 * r_half / (1 + r_half)          # Spearman-Brown
    noise_var = S.gap_noise_sq.mean()
    rel_gap = 1 - noise_var / S.gap0.var(ddof=1)
    ceil = np.sqrt(max(rel_rate, 0) * max(rel_gap, 0))
    print(f"  rate, split-half over future visits (n={len(H)}): r = {r_half:.3f}")
    print(f"     -> reliability of a full-length rate  {rel_rate:.3f}")
    print(f"  gap, within-subject residual SD = {np.sqrt(noise_var):.2f} y, "
          f"between-subject SD = {S.gap0.std():.2f} y")
    print(f"     -> reliability of a single-visit gap  {rel_gap:.3f}")
    print(f"\n  ATTENUATION CEILING for any correlation between them: {ceil:.3f}")
    if ceil < 0.3:
        print("  WARNING: ceiling is low. A null result here would be a")
        print("           measurement failure, not a biological statement.")

    # ================================================== THE ARTEFACT
    print("\n" + "=" * 78)
    print("THE SHARED-SCAN ARTEFACT — how much does it move the answer?")
    print("=" * 78)
    C = S[["age", "SEX"]].values.astype(float)
    for lab, col in [("rate SHARES the baseline scan", "rate_shared"),
                     ("rate from scans 2..n only", "rate_future")]:
        r = partial_r(S.gap0.values, S[col].values, C)
        print(f"    {lab:34s} r = {r:+.3f}")
    print("    the gap between these two rows is pure measurement artefact.")

    # ================================================== THE QUESTION
    for grp in ["ALL", "CN", "MCI", "Dementia"]:
        T = S if grp == "ALL" else S[S.DX0 == grp]
        if len(T) < 40:
            continue
        print("\n" + "=" * 78)
        print(f"DOES A BASELINE MEASURE PREDICT THE FUTURE RATE?   "
              f"{'همه' if grp=='ALL' else GFA[grp]}  (n={len(T)})")
        print("=" * 78)
        Ct = T[["age", "SEX"]].values.astype(float)
        y = T.rate_future.values
        for lab, x in [("brain-age gap at baseline", T.gap0.values),
                       ("E48 component at baseline", T.pc0.values),
                       ("hippocampal z at baseline", T.z0_Hippocampus.values),
                       ("ventricular z at baseline", T.z0_Ventricles.values)]:
            r = partial_r(x, y, Ct)
            line(lab, r, boot_r(x, y, Ct), ceil)
        # all six at once, as a multiple correlation
        X6 = T[[f"z0_{v}" for v in V]].values
        Cc = np.column_stack([np.ones(len(T)), Ct])
        ry = y - Cc @ np.linalg.lstsq(Cc, y, rcond=None)[0]
        rX = X6 - Cc @ np.linalg.lstsq(Cc, X6, rcond=None)[0]
        pred = rX @ np.linalg.lstsq(rX, ry, rcond=None)[0]
        print(f"    {'all six deviations (multiple R)':34s} R = "
              f"{np.corrcoef(pred, ry)[0,1]:+.3f}   (in-sample, an upper bound)")

    # ================================================== SHARPENING
    print("\n" + "=" * 78)
    print("SHARPENING THE TWO ANSWERS")
    print("=" * 78)
    print(f"  sanity: corr(E48 component, brain-age gap) = "
          f"{np.corrcoef(S.pc0, S.gap0)[0,1]:+.3f}")

    # how strong a correlation can the CONTROL null actually exclude?
    T = S[S.DX0 == "CN"]
    Ct = T[["age", "SEX"]].values.astype(float)
    lo, hi = boot_r(T.gap0.values, T.rate_future.values, Ct)
    print(f"\n  the control null is informative, not just underpowered:")
    print(f"    ceiling {ceil:.2f}, n={len(T)}, 95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"    -> anything stronger than |r| = {max(abs(lo), abs(hi)):.2f} is "
          f"excluded. The true value is near zero.")

    # is the MCI result just "these people are converting"?
    a = pd.read_csv(os.path.join(DATA, "ADNIMERGE_05Mar2026.csv"), low_memory=False)
    conv = set(a[a.DX == "Dementia"].PTID.unique())
    M = S[S.DX0 == "MCI"]
    NC = M[~M.PTID.isin(conv)]
    print(f"\n  is the MCI result just conversion? MCI who NEVER converted, "
          f"n={len(NC)}:")
    if len(NC) >= 40:
        Cn = NC[["age", "SEX"]].values.astype(float)
        for lab, x in [("brain-age gap", NC.gap0.values),
                       ("E48 component", NC.pc0.values)]:
            r = partial_r(x, NC.rate_future.values, Cn)
            line(lab + " (non-converters)", r, boot_r(x, NC.rate_future.values, Cn),
                 ceil)

    # does E48's component beat the gap at this task too?
    print("\n  E48 component minus brain-age gap, as predictors of the rate:")
    rng = np.random.RandomState(0)
    for grp, T2 in [("CN", S[S.DX0 == "CN"]), ("MCI", M), ("MCI non-conv", NC)]:
        if len(T2) < 40:
            continue
        C2 = T2[["age", "SEX"]].values.astype(float)
        g2, p2, y2 = T2.gap0.values, T2.pc0.values, T2.rate_future.values
        dd = []
        for _ in range(N_BOOT):
            i = rng.randint(0, len(T2), len(T2))
            try:
                dd.append(abs(partial_r(p2[i], y2[i], C2[i]))
                          - abs(partial_r(g2[i], y2[i], C2[i])))
            except Exception:
                pass
        dd = np.array(dd)
        l2, h2 = np.percentile(dd, [2.5, 97.5])
        pv = 2 * min((dd <= 0).mean(), (dd >= 0).mean())
        print(f"    {grp:14s} {dd.mean():+.3f}  95% CI [{l2:+.3f}, {h2:+.3f}]"
              f"  p={pv:.3f}  -> {'REAL' if pv < 0.05 else 'not distinguishable'}")

    # ================================================== TRAIT OR STATE
    print("\n" + "=" * 78)
    print("IS THE GAP A TRAIT OR A SPEED?")
    print("=" * 78)
    print("  If the gap were a speed it would grow steadily. If it is a level")
    print("  set early in life it should be nearly constant within a person.")
    bw = S.gap0.var(ddof=1)
    wi = S.gap_noise_sq.mean()
    print(f"    between-subject variance of the gap   {bw:.2f} y^2")
    print(f"    within-subject residual variance      {wi:.2f} y^2")
    print(f"    -> {100*(1-wi/bw):.0f}% of the variance in a person's gap is the "
          f"person, not the moment")
    for grp in ["CN", "MCI", "Dementia"]:
        T = S[S.DX0 == grp]
        if len(T) < 40:
            continue
        r = partial_r(T.gap0.values, T.gaprate_future.values,
                      T[["age", "SEX"]].values.astype(float))
        print(f"    {GFA[grp]:10s} baseline gap vs FUTURE rate of the gap itself: "
              f"r = {r:+.3f}")


if __name__ == "__main__":
    main()
