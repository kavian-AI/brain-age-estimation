#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IS "BRAIN AGING" THE SAME PROCESS IN CN, MCI AND AD?

The whole normative-modelling field, and brain age with it, rests on one
assumption that is almost never tested: there is ONE aging trajectory. A
reference curve is fitted in healthy people, and everyone else is scored as a
deviation FROM it. That only means something if patients are travelling along
the same road, just further down it. If disease moves the brain in a different
DIRECTION, then "brain age" is measuring the projection of a disease vector onto
an aging axis, and the deviation is a shadow, not a distance.

E48 already showed the weighting brain age inherits from age is the wrong one for
disease. This asks the question underneath that result: is the age axis itself
even the same object in the three groups?

The trap: coefficients fitted in two different samples ALWAYS differ. Any
difference we measure is meaningless on its own. So every comparison here is
scored against a null built from TWO DISJOINT HALVES OF THE CONTROLS - same
statistic, same sample sizes, but a process that is identical by construction.
If CN-vs-AD sits inside that null, the processes are indistinguishable.

    PART 1  cross-sectional age slope per region, per group
    PART 2  longitudinal rate direction per group   <- the strong version
    PART 3  the brain-age model itself: does a model fitted in patients
            read a control's age as well as a control-fitted one?
    PART 4  everything again on an age-matched subsample

Units: every regional measure is divided by the CN residual SD for that region,
so the six regions are on one comparable scale. The age trend is NOT removed -
that is the thing being compared.
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler

BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
DATA = os.path.join(BASE, "data")
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
FA = {"Hippocampus": "هیپوکامپ", "Entorhinal": "آنتورینال", "Fusiform": "دوکی",
      "MidTemp": "گیجگاهی میانی", "Ventricles": "بطن‌ها", "WholeBrain": "کل مغز"}
GRP = ["CN", "MCI", "Dementia"]
GFA = {"CN": "سالم", "MCI": "MCI", "Dementia": "آلزایمر"}
N_NULL, N_BOOT, MIN_SPAN = 300, 400, 1.5


# ----------------------------------------------------------------- loading
def load():
    a = pd.read_csv(os.path.join(DATA, "ADNIMERGE_05Mar2026.csv"), low_memory=False)
    d = a.dropna(subset=["AGE", "Years_bl", "ICV", "PTGENDER", "DX"] + V).copy()
    d["SEX"] = (d.PTGENDER == "Male").astype(int)
    d["age_visit"] = d.AGE + d.Years_bl
    d = d.sort_values(["PTID", "Years_bl"]).reset_index(drop=True)
    dx0 = d.groupby("PTID").DX.first()
    d["DX0"] = d.PTID.map(dx0)
    return d


def cn_scale(d):
    """Residual SD of each region among CONTROLS, after age/sex/ICV.

    One constant per region. Dividing by it puts the six regions on a common
    scale WITHOUT removing the age trend, which is what we are comparing.
    """
    b = d[d.VISCODE == "bl"]
    cn = b[b.DX == "CN"]
    ag = cn.age_visit.values
    X = np.column_stack([ag, ag ** 2, cn.SEX.values, cn.ICV.values])
    sd = {}
    for v in V:
        y = cn[v].values.astype(float)
        m = LinearRegression().fit(X, y)
        sd[v] = (y - m.predict(X)).std(ddof=1)
    return sd


# ------------------------------------------------------ the two statistics
def xsec_slope(df, sd):
    """Age slope per region, in CN-SD units per year, adjusted for sex and ICV."""
    ag = df.age_visit.values.astype(float)
    X = np.column_stack([ag, df.SEX.values, df.ICV.values])
    return np.array([LinearRegression().fit(X, df[v].values.astype(float)).coef_[0] / sd[v]
                     for v in V])


def long_rate(sub, sd):
    """Mean annualised rate of change per region, in CN-SD units per year.

    `sub` is one row per subject carrying a fitted slope per region.
    """
    return np.array([sub["r_" + v].values.mean() for v in V])


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def null_sizes(n_cn, n_grp):
    """Sizes for the two disjoint control subsets used as the null.

    The real comparison is n_cn against n_grp, so the null should be too. The
    two null arms must be disjoint, so the first arm can only reach n_cn-n_grp.
    That leaves the null slightly wider than the truth, i.e. conservative - it
    makes 'DIFFERENT' harder to declare, never easier.
    """
    n2 = min(n_grp, n_cn // 2)
    return n_cn - n2, n2


def null_cosine(pool, n1, n2, stat, rng, n_null=N_NULL):
    """Cosine between the statistic computed on two DISJOINT control subsets.

    This is what the statistic looks like when the process really is identical
    and only the sample differs. Sizes are matched to the real comparison.
    """
    ids = np.unique(pool.PTID.values)
    out = []
    for _ in range(n_null):
        p = rng.permutation(len(ids))
        s1 = set(ids[p[:n1]])
        s2 = set(ids[p[n1:n1 + n2]])
        try:
            c = cosine(stat(pool[pool.PTID.isin(s1)]),
                       stat(pool[pool.PTID.isin(s2)]))
            if np.isfinite(c):
                out.append(c)
        except Exception:
            pass
    return np.array(out)


def verdict(obs, null):
    """Where does the observed cosine sit in the identical-process null?"""
    p = (null <= obs).mean()
    lo = np.percentile(null, 2.5)
    tag = "SAME (indistinguishable)" if obs >= lo else "DIFFERENT"
    return lo, p, tag


def report(label, obs, null):
    lo, p, tag = verdict(obs, null)
    print(f"    {label:28s} cos = {obs:+.4f}   null[2.5%] = {lo:+.4f}   "
          f"p = {p:.3f}   -> {tag}")


# ------------------------------------------------------------------- parts
def part1(d, sd, tag=""):
    print("\n" + "=" * 78)
    print(f"PART 1  CROSS-SECTIONAL AGE SLOPE PER REGION {tag}")
    print("=" * 78)
    b = d[d.VISCODE == "bl"].reset_index(drop=True)
    S = {}
    for g in GRP:
        sub = b[b.DX == g]
        S[g] = xsec_slope(sub, sd)
        print(f"\n  {GFA[g]:8s} (n={len(sub)})  SD per year of age:")
        print("    " + "  ".join(f"{FA[v]} {s:+.4f}" for v, s in zip(V, S[g])))

    print("\n  --- magnitude: how fast does the whole vector move? ---")
    for g in GRP:
        print(f"    {GFA[g]:8s} |slope| = {np.linalg.norm(S[g]):.4f} SD/year")

    print("\n  --- direction: is the PATTERN across regions the same? ---")
    rng = np.random.RandomState(0)
    pool = b[b.DX == "CN"]
    nc = int((b.DX == "CN").sum())
    for g in ["MCI", "Dementia"]:
        ng = int((b.DX == g).sum())
        n1, n2 = null_sizes(nc, ng)
        null = null_cosine(pool, n1, n2, lambda x: xsec_slope(x, sd), rng)
        report(f"سالم در برابر {GFA[g]}", cosine(S["CN"], S[g]), null)
    return S


def build_rates(d, sd, window=None):
    """One row per subject: fitted slope per region over their visits.

    `window` caps the follow-up in years. This matters more than it looks:
    AD subjects drop out early, so their slopes rest on ~2 years while a
    control's rest on ~4.5. A short window makes a slope noisier, and a noisier
    slope looks more OFF-AXIS for free. Capping everyone at the same window is
    the only way the comparison means anything.
    """
    src = d if window is None else d[d.Years_bl <= window + 0.35]
    rows = []
    for s, g in src.groupby("PTID"):
        t = g.Years_bl.values.astype(float)
        if len(g) < 3 or (t.max() - t.min()) < MIN_SPAN:
            continue
        r = dict(PTID=s, DX0=g.DX0.values[0], age=g.age_visit.values[0],
                 SEX=g.SEX.values[0], ICV=g.ICV.values[0], n_vis=len(g),
                 span=t.max() - t.min(),
                 stable=int(g.DX.nunique() == 1))
        for v in V:
            r["r_" + v] = np.polyfit(t, g[v].values.astype(float), 1)[0] / sd[v]
        rows.append(r)
    return pd.DataFrame(rows)


def part2(R, sd, tag="", stable_only=False):
    print("\n" + "=" * 78)
    print(f"PART 2  LONGITUDINAL RATE - THE ACTUAL PROCESS {tag}"
          + ("  [non-converters only]" if stable_only else ""))
    print("=" * 78)
    Q = R[R.stable == 1] if stable_only else R
    M = {}
    for g in GRP:
        sub = Q[Q.DX0 == g]
        if len(sub) < 40:
            print(f"  {GFA[g]}: n={len(sub)}, too few; skipped")
            continue
        M[g] = long_rate(sub, sd)
        print(f"\n  {GFA[g]:8s} (n={len(sub)}, median {sub.n_vis.median():.0f} scans"
              f" over {sub.span.median():.1f} y)  SD per year:")
        print("    " + "  ".join(f"{FA[v]} {s:+.4f}" for v, s in zip(V, M[g])))
    if len(M) < 3:
        return M

    print("\n  --- magnitude: speed of the process ---")
    base = np.linalg.norm(M["CN"])
    for g in GRP:
        n = np.linalg.norm(M[g])
        print(f"    {GFA[g]:8s} |rate| = {n:.4f} SD/year   "
              f"= {n/base:.2f}x the healthy speed")

    print("\n  --- direction: same road, or a different road? ---")
    rng = np.random.RandomState(1)
    pool = Q[Q.DX0 == "CN"]
    nc = len(pool)
    for g in ["MCI", "Dementia"]:
        ng = int((Q.DX0 == g).sum())
        n1, n2 = null_sizes(nc, ng)
        null = null_cosine(pool, n1, n2, lambda x: long_rate(x, sd), rng)
        report(f"سالم در برابر {GFA[g]}", cosine(M["CN"], M[g]), null)
    # MCI vs AD, with a null of the same sizes drawn from controls
    nm, na = int((Q.DX0 == "MCI").sum()), int((Q.DX0 == "Dementia").sum())
    n1, n2 = min(nm, nc - min(na, nc // 2)), min(na, nc // 2)
    null = null_cosine(pool, n1, n2, lambda x: long_rate(x, sd), rng)
    report("MCI در برابر آلزایمر", cosine(M["MCI"], M["Dementia"]), null)

    print("\n  --- how much of each group is NOT healthy aging? ---")
    print("    CAREFUL: an off-axis component appears from sampling noise alone.")
    print("    The null column is what two identical processes produce at this n.")
    u = M["CN"] / np.linalg.norm(M["CN"])
    for g in ["MCI", "Dementia"]:
        ng = int((Q.DX0 == g).sum())
        n1, n2 = null_sizes(nc, ng)
        nl = null_cosine(pool, n1, n2, lambda x: long_rate(x, sd), rng, n_null=200)
        proj = M[g] @ u
        resid = np.linalg.norm(M[g] - proj * u)
        off = resid / np.linalg.norm(M[g])
        off_null = np.sqrt(np.clip(1 - np.median(nl) ** 2, 0, 1))
        print(f"    {GFA[g]:8s} along healthy axis {proj:+.4f}  off it {resid:.4f}"
              f"   off-axis {100*off:.0f}%   null off-axis {100*off_null:.0f}%"
              f"   excess {100*(off-off_null):+.0f} points")
    return M


def part3(d, rng):
    print("\n" + "=" * 78)
    print("PART 3  THE BRAIN-AGE MODEL ITSELF - TRAINED WHERE, TESTED WHERE?")
    print("=" * 78)
    print("  A one-way test would be unfair: a control-trained model tested on")
    print("  controls enjoys a matched distribution. So the grid is SYMMETRIC.")
    print("  Every training set: same size, same age distribution. Every test")
    print("  set: held out by subject.\n")
    b = d[d.VISCODE == "bl"].reset_index(drop=True)
    N_TEST, N_TR, REPS = 100, 200, 40
    pool, held = {}, {}
    for g in GRP:
        gg = b[b.DX == g]
        ids = np.unique(gg.PTID.values)
        p = rng.permutation(len(ids))
        held[g] = gg[gg.PTID.isin(set(ids[p[:N_TEST]]))]
        pool[g] = gg[gg.PTID.isin(set(ids[p[N_TEST:]]))]
        print(f"  {GFA[g]:8s} train pool {len(pool[g]):4d}   held out {len(held[g])}")

    # one fixed age grid for every training draw, so they differ only in DIAGNOSIS
    edges = np.percentile(b.age_visit.values, [5, 24, 43, 62, 81, 95])

    def matched_draw(src, r):
        take = []
        for k in range(5):
            c = src[(src.age_visit >= edges[k]) & (src.age_visit < edges[k + 1])]
            if len(c) == 0:
                continue
            take.append(c.sample(n=N_TR // 5, replace=len(c) < N_TR // 5,
                                 random_state=r))
        return pd.concat(take)

    def feats(f):
        return np.column_stack([f[v].values for v in V] + [f.ICV.values, f.SEX.values])

    MAE = {(a, c): [] for a in GRP for c in GRP}
    coefs = {g: [] for g in GRP}
    for rep in range(REPS):
        for g in GRP:
            tr = matched_draw(pool[g], rep)
            sc = StandardScaler().fit(feats(tr))
            m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(
                sc.transform(feats(tr)), tr.age_visit.values)
            coefs[g].append(m.coef_[:len(V)])
            for t in GRP:
                MAE[(g, t)].append(np.abs(
                    m.predict(sc.transform(feats(held[t])))
                    - held[t].age_visit.values).mean())

    print("\n  MAE in years.  rows = trained on, columns = tested on")
    print("            " + "".join(f"{GFA[t]:>12s}" for t in GRP))
    for g in GRP:
        print(f"    {GFA[g]:8s}" + "".join(
            f"{np.mean(MAE[(g,t)]):>12.3f}" for t in GRP))

    print("\n  is the diagonal the best column? (own-group advantage)")
    for t in GRP:
        best = min(GRP, key=lambda g: np.mean(MAE[(g, t)]))
        own = np.mean(MAE[(t, t)])
        print(f"    tested on {GFA[t]:8s} best trainer = {GFA[best]:8s}"
              f"   own-group MAE {own:.3f}"
              f"   {'YES' if best == t else 'NO'}")

    print("\n  cost of training on the wrong group, tested on CONTROLS:")
    for g in ["MCI", "Dementia"]:
        dd = np.array(MAE[(g, "CN")]) - np.array(MAE[("CN", "CN")])
        lo, hi = np.percentile(dd, [2.5, 97.5])
        print(f"    {GFA[g]:8s} {dd.mean():+.3f} y  95% CI [{lo:+.3f}, {hi:+.3f}]"
              f"   ({100*dd.mean()/np.mean(MAE[('CN','CN')]):+.0f}%)")
    print("\n  cost of training on the wrong group, tested on ALZHEIMER:")
    for g in ["CN", "MCI"]:
        dd = np.array(MAE[(g, "Dementia")]) - np.array(MAE[("Dementia", "Dementia")])
        lo, hi = np.percentile(dd, [2.5, 97.5])
        print(f"    {GFA[g]:8s} {dd.mean():+.3f} y  95% CI [{lo:+.3f}, {hi:+.3f}]"
              f"   ({100*dd.mean()/np.mean(MAE[('Dementia','Dementia')]):+.0f}%)")

    print("\n  and the coefficient vectors themselves:")
    C = {g: np.mean(coefs[g], axis=0) for g in GRP}
    for g in GRP:
        print(f"    {GFA[g]:8s} " + "  ".join(f"{FA[v]} {c:+.2f}"
                                              for v, c in zip(V, C[g])))
    h = len(coefs["CN"]) // 2
    n_cos = np.array([cosine(coefs["CN"][i], coefs["CN"][j])
                      for i in range(h) for j in range(h, 2 * h)])
    for g in ["MCI", "Dementia"]:
        report(f"سالم در برابر {GFA[g]}", cosine(C["CN"], C[g]), n_cos)


def main():
    d = load()
    sd = cn_scale(d)
    print("CN residual SD per region (the common unit):")
    print("  " + "  ".join(f"{FA[v]} {sd[v]:.0f}" for v in V))

    S = part1(d, sd)
    R = build_rates(d, sd)
    print(f"\nsubjects with >=3 scans spanning >={MIN_SPAN} y: {len(R)}")
    print(R.DX0.value_counts().to_string())
    part2(R, sd)
    part2(R, sd, stable_only=True)
    part3(d, np.random.RandomState(0))

    # ------------------------------------------------ PART 4: age matched
    print("\n" + "=" * 78)
    print("PART 4  THE AGE CONFOUND - AD subjects are ~2 years older.")
    print("        Everything again on a sample matched to 65-85 only.")
    print("=" * 78)
    dm = d[(d.age_visit >= 65) & (d.age_visit <= 85)].reset_index(drop=True)
    part1(dm, sd, tag="[65-85 only]")
    Rm = R[(R.age >= 65) & (R.age <= 85)]
    part2(Rm, sd, tag="[65-85 only]")

    # ------------------------------------------- PART 5: matched follow-up
    print("\n" + "=" * 78)
    print("PART 5  THE FOLLOW-UP CONFOUND, WHICH IS THE DANGEROUS ONE.")
    print("        Controls are followed 4.5 y, AD only 2.0 y. A slope fitted")
    print("        over 2 y is noisier, and noise alone pushes a vector")
    print("        off-axis. Everyone capped at the same 2-year window.")
    print("=" * 78)
    R2 = build_rates(d, sd, window=2.0)
    print(f"\n  subjects surviving the 2-year cap with >=3 scans: {len(R2)}")
    print(R2.DX0.value_counts().to_string())
    print("  median span by group: " + "  ".join(
        f"{GFA[g]} {R2[R2.DX0==g].span.median():.1f}y" for g in GRP
        if (R2.DX0 == g).any()))
    part2(R2, sd, tag="[2-year window for EVERYONE]")


if __name__ == "__main__":
    main()
