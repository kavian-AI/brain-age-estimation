#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WHAT BIOLOGICAL PROCESS DOES THE BRAIN-AGE GAP ACTUALLY INDEX?

We have established what the gap is NOT: not a speed (E52), not a clinical
predictor in healthy people (E53a), not weighted where Alzheimer's lives (E61,
E62: its regional weights are uncorrelated with a disease model's, rank r=-0.03).
This asks the positive question, with the ADNI biomarkers that were never used.

  A  THE A/T/N MAP.  Alzheimer's is defined biologically by Amyloid, Tau and
     Neurodegeneration. For each structural summary - brain-age gap, hippocampal
     deviation, uniform average - which of CSF amyloid, CSF p-tau, amyloid PET
     and FDG-PET does it track, in controls and in MCI separately?
  B  PRECLINICAL ALZHEIMER'S.  Among cognitively normal people, those with
     amyloid already present are on the way. Is their brain age older? Is their
     hippocampus smaller? A marker blind to this stage is blind to the stage that
     matters most for prevention.
  C  COGNITIVE DECLINE IN HEALTHY PEOPLE.  E53a showed the gap does not predict
     conversion in controls - but conversion is rare. Cognitive slope is
     continuous and far better powered. Does baseline gap predict how fast a
     healthy person's cognition declines?
  D  SEX.  The age model leans hard on ICV (the single largest weight in E61), and
     ICV differs by sex. Is the gap biased by sex in healthy people, and does it
     work equally well in women and men?
  E  HORIZON.  If the gap is a trait (E52: 94% person, 6% moment), it may be
     relatively better at long horizons and worse at short ones. Time-dependent
     discrimination at 1, 2, 3 and 5 years.
  F  WHO DOES THE GAP MISS?  MCI converters that the gap ranks as low-risk while
     the hippocampus ranks them high-risk - what are they?
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
from scipy.stats import mannwhitneyu  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

SIGN = np.array([-1., -1., -1., -1., +1., -1.])
N_BOOT = 1000


def num(s):
    return pd.to_numeric(s.astype(str).str.replace("<", "").str.replace(">", ""),
                         errors="coerce")


def partial_r(x, y, C):
    ok = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(C), 1)
    x, y, C = x[ok], y[ok], C[ok]
    C1 = np.column_stack([np.ones(len(x)), C])
    rx = x - C1 @ np.linalg.lstsq(C1, x, rcond=None)[0]
    ry = y - C1 @ np.linalg.lstsq(C1, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1]), int(ok.sum())


def boot_pr(x, y, C, n=N_BOOT):
    ok = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(C), 1)
    x, y, C = x[ok], y[ok], C[ok]
    rng = np.random.RandomState(0)
    out = []
    for _ in range(n):
        i = rng.randint(0, len(x), len(x))
        out.append(partial_r(x[i], y[i], C[i])[0])
    return np.percentile(out, [2.5, 97.5])


def main():
    a, b = E.load()
    Z = E.normative_z(b)
    gap = E.brain_age_gap(b)
    b = b.assign(gap=gap, hipp=-Z[:, 0], uni=(Z * SIGN).mean(1))
    bl = a[a.VISCODE == "bl"].drop_duplicates("PTID").set_index("PTID")
    for c in ["ABETA", "PTAU", "TAU", "AV45", "FDG", "PTEDUCAT", "APOE4"]:
        b[c] = num(bl.reindex(b.PTID)[c]).values
    b["A_pos"] = np.where(b.AV45.notna(), (b.AV45 > 1.11).astype(float),
                          np.where(b.ABETA.notna(), (b.ABETA < 980).astype(float), np.nan))
    ARMS = [("brain-age gap", "gap"), ("hippocampal deviation", "hipp"),
            ("uniform average", "uni")]

    # ============================================================== A
    print("=" * 92)
    print("A  THE A/T/N MAP  (partial r, controlling age and sex; sign: + means 'worse with worse')")
    print("=" * 92)
    # orient every biomarker so that higher = more pathological
    bms = [("CSF amyloid (low=bad)", -b.ABETA.values), ("amyloid PET AV45", b.AV45.values),
           ("CSF p-tau", b.PTAU.values), ("CSF total tau", b.TAU.values),
           ("FDG-PET (low=bad)", -b.FDG.values)]
    for grp in ["CN", "MCI"]:
        q = (b.DX.values == grp)
        print(f"\n  {grp}")
        print(f"    {'':24s}" + "".join(f"{nm[:22]:>24s}" for nm, _ in ARMS))
        for bn, bv in bms:
            cells = []
            for _, col in ARMS:
                C = np.column_stack([b.age_visit.values[q], b.SEX.values[q]])
                r, n = partial_r(b[col].values[q], bv[q], C)
                lo, hi = boot_pr(b[col].values[q], bv[q], C, n=400)
                star = "*" if (lo > 0 or hi < 0) else " "
                cells.append(f"{r:+.3f}{star} (n={n:4d})")
            print(f"    {bn:24s}" + "".join(f"{c:>24s}" for c in cells))
    print("    * = 95% bootstrap CI excludes zero")

    # ============================================================== B
    print("\n" + "=" * 92)
    print("B  PRECLINICAL ALZHEIMER'S: amyloid-POSITIVE vs amyloid-NEGATIVE cognitively normal")
    print("=" * 92)
    q = (b.DX.values == "CN") & b.A_pos.notna()
    Ap, An = q & (b.A_pos == 1), q & (b.A_pos == 0)
    print(f"  A+ n={Ap.sum()}  A- n={An.sum()}  (age {b.age_visit[Ap].mean():.1f} vs {b.age_visit[An].mean():.1f})")
    for nm, col in ARMS:
        x = b[col].values
        # adjust for age and sex before comparing, since A+ are older
        C1 = np.column_stack([np.ones(q.sum()), b.age_visit.values[q], b.SEX.values[q]])
        r = x[q] - C1 @ np.linalg.lstsq(C1, x[q], rcond=None)[0]
        rp, rn = r[b.A_pos.values[q] == 1], r[b.A_pos.values[q] == 0]
        sdp = np.std(r, ddof=1)
        rng = np.random.RandomState(0)
        dd = [rng.choice(rp, len(rp)).mean() - rng.choice(rn, len(rn)).mean() for _ in range(N_BOOT)]
        lo, hi = np.percentile(dd, [2.5, 97.5])
        auc = roc_auc_score(np.r_[np.ones(len(rp)), np.zeros(len(rn))], np.r_[rp, rn])
        print(f"    {nm:24s} A+ minus A- = {(rp.mean()-rn.mean())/sdp:+.3f} SD  "
              f"[{lo/sdp:+.3f}, {hi/sdp:+.3f}]   AUC {auc:.3f}   "
              f"{'DETECTS' if lo > 0 else 'blind'}")

    # ============================================================== C
    print("\n" + "=" * 92)
    print("C  DOES BASELINE BRAIN AGE PREDICT COGNITIVE DECLINE IN HEALTHY PEOPLE?")
    print("=" * 92)
    rows = []
    for s, g in a[a.PTID.isin(b.PTID[b.DX == "CN"])].groupby("PTID"):
        for cog in ["mPACCdigit", "ADAS13"]:
            gg = g[["Years_bl", cog]].dropna()
            gg = gg.assign(v=num(gg[cog]))
            gg = gg.dropna()
            if len(gg) >= 3 and gg.Years_bl.max() - gg.Years_bl.min() >= 2:
                rows.append((s, cog, np.polyfit(gg.Years_bl.values, gg.v.values, 1)[0]))
    S = pd.DataFrame(rows, columns=["PTID", "cog", "slope"])
    for cog, worse in [("mPACCdigit", -1), ("ADAS13", +1)]:
        T = S[S.cog == cog].merge(b[["PTID", "gap", "hipp", "uni", "age_visit", "SEX",
                                     "PTEDUCAT"]], on="PTID")
        y = worse * T.slope.values          # higher = faster decline
        C = np.column_stack([T.age_visit.values, T.SEX.values, T.PTEDUCAT.values])
        print(f"\n  {cog} slope (n={len(T)} controls, >=3 visits over >=2 y), "
              f"sign: + means worse baseline -> faster decline")
        for nm, col in ARMS:
            r, n = partial_r(T[col].values, y, C)
            lo, hi = boot_pr(T[col].values, y, C, n=600)
            print(f"    {nm:24s} partial r = {r:+.3f}  [{lo:+.3f}, {hi:+.3f}]"
                  f"  {'REAL' if lo > 0 or hi < 0 else '-'}")

    # ============================================================== D
    print("\n" + "=" * 92)
    print("D  SEX")
    print("=" * 92)
    q = b.DX.values == "CN"
    for sx, lab in [(1, "men"), (0, "women")]:
        qq = q & (b.SEX.values == sx)
        print(f"  healthy {lab:6s} n={qq.sum():4d}  mean gap {b.gap[qq].mean():+.2f} y   "
              f"mean ICV {b.ICV[qq].mean()/1e6:.3f} L")
    gm, gw = b.gap[q & (b.SEX == 1)].values, b.gap[q & (b.SEX == 0)].values
    rng = np.random.RandomState(0)
    dd = [rng.choice(gm, len(gm)).mean() - rng.choice(gw, len(gw)).mean() for _ in range(N_BOOT)]
    print(f"  men minus women, healthy: {np.mean(dd):+.2f} y  [{np.percentile(dd,2.5):+.2f},"
          f" {np.percentile(dd,97.5):+.2f}]")
    sel = b.DX.isin(["CN", "Dementia"]).values
    for sx, lab in [(1, "men"), (0, "women")]:
        qq = sel & (b.SEX.values == sx)
        y = (b.DX.values[qq] == "Dementia").astype(int)
        ag = b.age_visit.values[qq].reshape(-1, 1)
        s = b.PTID.values[qq]
        print(f"  AD vs CN in {lab:6s} (n={qq.sum()}):  " + "   ".join(
            f"{nm.split()[0]} {E.auc_cv(np.column_stack([b[c].values[qq], ag]), y, s):.3f}"
            for nm, c in ARMS))

    # ============================================================== E
    print("\n" + "=" * 92)
    print("E  HORIZON: discrimination of conversion BY a given year (cumulative/dynamic AUC)")
    print("=" * 92)
    pool = b[b.DX == "MCI"]
    keep, dur, ev = E.survival(a, pool.PTID.values, ["Dementia"])
    M = pool[pool.PTID.isin(set(keep))].copy()
    o = {s: i for i, s in enumerate(keep)}
    M["tt"] = [dur[o[s]] for s in M.PTID]
    M["ee"] = [ev[o[s]] for s in M.PTID]
    print(f"  MCI n={len(M)}, conversions {int(M.ee.sum())}")
    print(f"    {'horizon':10s}" + "".join(f"{nm[:22]:>24s}" for nm, _ in ARMS) + "   cases/controls")
    for h in [1, 2, 3, 5]:
        case = (M.tt <= h) & (M.ee == 1)
        ctrl = M.tt > h
        use = case | ctrl
        yy = case[use].astype(int).values
        cells = [f"{roc_auc_score(yy, M[c].values[use]):.3f}" for _, c in ARMS]
        print(f"    by {h} y    " + "".join(f"{c:>24s}" for c in cells)
              + f"   {int(yy.sum())}/{int((1-yy).sum())}")

    # ============================================================== F
    print("\n" + "=" * 92)
    print("F  WHO DOES THE GAP MISS?  converters the gap calls low-risk but the hippocampus flags")
    print("=" * 92)
    C_ = M[M.ee == 1].copy()
    gr = M.gap.rank(pct=True).reindex(C_.index)
    hr_ = M.hipp.rank(pct=True).reindex(C_.index)
    missed = (gr < 0.5) & (hr_ >= 0.5)
    caught = (gr >= 0.5) & (hr_ >= 0.5)
    for c in ["ABETA", "PTAU", "AV45", "APOE4", "PTEDUCAT"]:
        C_[c] = b.set_index("PTID").reindex(C_.PTID)[c].values
    print(f"  converters: {len(C_)}   missed by the gap but flagged by the hippocampus: {missed.sum()}"
          f"   caught by both: {caught.sum()}")
    print(f"    {'':22s} {'missed':>10s} {'caught by both':>15s}   Mann-Whitney p")
    for c, lab in [("age_visit", "age"), ("ICV", "ICV (L)"), ("SEX", "male fraction"),
                   ("PTEDUCAT", "education (y)"), ("APOE4", "APOE4 alleles"),
                   ("AV45", "amyloid PET"), ("PTAU", "CSF p-tau"), ("tt", "years to conversion")]:
        x1, x2 = C_[c][missed].astype(float), C_[c][caught].astype(float)
        sc = 1e6 if c == "ICV" else 1
        try:
            p = mannwhitneyu(x1.dropna(), x2.dropna()).pvalue
        except Exception:
            p = np.nan
        print(f"    {lab:22s} {x1.mean()/sc:10.2f} {x2.mean()/sc:15.2f}   {p:.4f}")


if __name__ == "__main__":
    main()
