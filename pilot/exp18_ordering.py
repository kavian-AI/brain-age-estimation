#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DOES REGIONAL AGEING FOLLOW AN ORDER, AND DO REGIONS TALK TO EACH OTHER?

Three questions, none of which need to improve prediction to be a finding:

  Q1 SEQUENCE  - in converters, which region starts ageing abnormally FIRST,
                 and how many years later do the others follow?
                 Method: event-based ordering. For each region, find the time
                 (relative to conversion) at which its gap crosses the normal
                 threshold (95th pct of stable subjects).

  Q2 COUPLING  - are regional ageing RATES correlated with each other?
                 Does the correlation structure differ between people who
                 convert and people who stay stable?

  Q3 DIRECTION - does ageing in region A at time t predict ageing in region B
                 at t+1?  (lagged cross-correlation = a crude causal probe)

Data: ADNI, subjects with >=3 scans.
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
DATA = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\data"
OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
SHORT = ["Hipp", "Ento", "Fusi", "MidT", "Vent", "Whole"]


def build_trajectories():
    """Per subject: the full time-series of regional brain-age gaps."""
    d = pd.read_csv(f"{DATA}/ADNIMERGE_05Mar2026.csv", low_memory=False)
    d = d.sort_values(["RID", "Years_bl"])
    cnb = d.dropna(subset=V + ["ICV", "AGE", "PTGENDER", "DX"]).groupby("RID").first()
    cnb = cnb[cnb.DX == "CN"]
    models = []
    for v in V:
        X = np.column_stack([cnb[v].values, cnb.ICV.values,
                             (cnb.PTGENDER == "Male").astype(int).values])
        sc = StandardScaler().fit(X)
        m = LinearRegression().fit(sc.transform(X), cnb.AGE.values)
        b = LinearRegression().fit(cnb.AGE.values.reshape(-1, 1),
                                   m.predict(sc.transform(X)) - cnb.AGE.values)
        models.append((sc, m, b))

    def gap(r, vol, age, icv, sex):
        sc, m, b = models[r]
        X = sc.transform(np.column_stack([vol, icv, sex]))
        return (m.predict(X) - age) - b.predict(np.asarray(age).reshape(-1, 1))

    trajs = []
    for rid, g in d.groupby("RID"):
        g = g.sort_values("Years_bl")
        if g.iloc[0]["DX"] == "Dementia":
            continue
        gv = g.dropna(subset=V + ["ICV", "AGE", "PTGENDER"])
        if gv.Years_bl.nunique() < 3:
            continue
        sex = (gv.PTGENDER == "Male").astype(int).values
        G = np.column_stack([gap(r, gv[v].values, gv.AGE.values, gv.ICV.values, sex)
                             for r, v in enumerate(V)])
        conv = g[g["DX"] == "Dementia"]
        t_conv = conv.iloc[0]["Years_bl"] if len(conv) else np.nan
        trajs.append(dict(rid=rid, bdx=g.iloc[0]["DX"], t=gv.Years_bl.values,
                          G=G, converter=int(len(conv) > 0), t_conv=t_conv))
    return trajs


def main():
    trajs = build_trajectories()
    conv = [t for t in trajs if t["converter"] == 1 and np.isfinite(t["t_conv"])]
    stab = [t for t in trajs if t["converter"] == 0]
    print(f"subjects with >=3 scans: {len(trajs)}  "
          f"(converters {len(conv)}, stable {len(stab)})\n")

    # ---------- normal threshold per region, from stable subjects ----------
    allstab = np.vstack([t["G"] for t in stab])
    thr = np.percentile(allstab, 95, axis=0)
    print("abnormality threshold per region (95th pct of stable), years:")
    print("  " + "  ".join(f"{s}={x:+.2f}" for s, x in zip(SHORT, thr)))

    # ================= Q1: SEQUENCE =================
    print("\n" + "=" * 72)
    print("Q1. IN WHAT ORDER DO REGIONS BECOME ABNORMAL?")
    print("     (time of first threshold crossing, relative to conversion)")
    print("=" * 72)
    cross = {v: [] for v in V}
    for t in conv:
        rel = t["t"] - t["t_conv"]          # negative = before conversion
        for r, v in enumerate(V):
            over = np.where(t["G"][:, r] > thr[r])[0]
            if len(over):
                cross[v].append(rel[over[0]])
    rows = []
    for r, v in enumerate(V):
        c = np.array(cross[v])
        if len(c) >= 20:
            rows.append((v, np.median(c), np.mean(c), len(c)))
    rows.sort(key=lambda x: x[1])
    print(f"  {'region':12s} {'median yrs vs conversion':>26s} {'mean':>8s} {'n':>6s}")
    for v, med, mn, n in rows:
        print(f"  {v:12s} {med:26.2f} {mn:8.2f} {n:6d}")
    if len(rows) >= 2:
        first, last = rows[0], rows[-1]
        print(f"\n  -> {first[0]} becomes abnormal FIRST, "
              f"{last[0]} LAST, separated by {last[1]-first[1]:.2f} years")
        # significance of the first-vs-last difference
        a, b = np.array(cross[first[0]]), np.array(cross[last[0]])
        print(f"     Mann-Whitney p = {stats.mannwhitneyu(a, b).pvalue:.2e}")

    # pairwise: how often does region i cross before region j (within subject)?
    print("\n  within-subject ordering (P(row crosses before column)):")
    P = np.full((len(V), len(V)), np.nan)
    for i in range(len(V)):
        for j in range(len(V)):
            if i == j:
                continue
            wins, tot = 0, 0
            for t in conv:
                oi = np.where(t["G"][:, i] > thr[i])[0]
                oj = np.where(t["G"][:, j] > thr[j])[0]
                if len(oi) and len(oj):
                    tot += 1
                    if t["t"][oi[0]] < t["t"][oj[0]]:
                        wins += 1
            if tot >= 20:
                P[i, j] = wins / tot
    print("        " + "".join(f"{s:>7s}" for s in SHORT))
    for i, s in enumerate(SHORT):
        print(f"  {s:6s}" + "".join(
            "      -" if i == j else
            ("      ." if np.isnan(P[i, j]) else f"{P[i,j]:7.2f}")
            for j in range(len(V))))
    order = np.nanmean(P, axis=1)
    seq = [SHORT[i] for i in np.argsort(-order)]
    print(f"\n  -> inferred sequence (earliest first): {' -> '.join(seq)}")

    # ================= Q2: COUPLING =================
    print("\n" + "=" * 72)
    print("Q2. ARE REGIONAL AGEING RATES COUPLED? (and does it differ by group?)")
    print("=" * 72)

    def rates_of(group):
        R = []
        for t in group:
            tt = t["t"] - t["t"][0]
            R.append([np.polyfit(tt, t["G"][:, r], 1)[0] for r in range(len(V))])
        return np.array(R)

    Rc, Rs = rates_of(conv), rates_of(stab)
    Cc, Cs = np.corrcoef(Rc.T), np.corrcoef(Rs.T)
    print("  correlation of ageing rates — CONVERTERS (upper) / STABLE (lower)")
    print("        " + "".join(f"{s:>7s}" for s in SHORT))
    for i, s in enumerate(SHORT):
        line = f"  {s:6s}"
        for j in range(len(V)):
            if i == j:
                line += "      -"
            elif j > i:
                line += f"{Cc[i,j]:7.2f}"
            else:
                line += f"{Cs[i,j]:7.2f}"
        print(line)
    iu = np.triu_indices(len(V), 1)
    print(f"\n  mean |r| converters = {np.abs(Cc[iu]).mean():.3f}, "
          f"stable = {np.abs(Cs[iu]).mean():.3f}")
    # Fisher z test on the difference of mean coupling
    zc = np.arctanh(np.clip(Cc[iu], -.999, .999))
    zs = np.arctanh(np.clip(Cs[iu], -.999, .999))
    print(f"  paired t-test on Fisher-z coupling: p = "
          f"{stats.ttest_rel(zc, zs).pvalue:.4f}")
    big = sorted(zip(np.abs(Cc[iu] - Cs[iu]), [(SHORT[i], SHORT[j])
                 for i, j in zip(*iu)]), reverse=True)[:3]
    print("  largest converter-vs-stable coupling differences:")
    for d, (x, y) in big:
        i, j = SHORT.index(x), SHORT.index(y)
        print(f"    {x}-{y}: converters r={Cc[i,j]:+.2f}, stable r={Cs[i,j]:+.2f} "
              f"(diff {d:.2f})")

    # ================= Q3: LAGGED DIRECTION =================
    print("\n" + "=" * 72)
    print("Q3. DOES AGEING IN ONE REGION PRECEDE ANOTHER? (lagged correlation)")
    print("=" * 72)
    lag_pairs = []
    for i in range(len(V)):
        for j in range(len(V)):
            if i == j:
                continue
            vals = []
            for t in conv:
                if len(t["t"]) < 3:
                    continue
                di = np.diff(t["G"][:, i])
                dj = np.diff(t["G"][:, j])
                if len(di) >= 2:
                    vals.append(np.corrcoef(di[:-1], dj[1:])[0, 1])
            vals = [v for v in vals if np.isfinite(v)]
            if len(vals) >= 30:
                m = np.mean(vals)
                p = stats.ttest_1samp(vals, 0).pvalue
                lag_pairs.append((m, p, SHORT[i], SHORT[j], len(vals)))
    lag_pairs.sort(reverse=True)
    print("  top lagged couplings  (change in A at t -> change in B at t+1):")
    for m, p, x, y, n in lag_pairs[:6]:
        star = "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else "ns"))
        print(f"    {x:5s} -> {y:5s}  r = {m:+.3f}  p = {p:.1e} {star}  (n={n})")

    np.savez(f"{OUT}/exp18_ordering.npz", thr=thr, P=P, Cc=Cc, Cs=Cs,
             Rc=Rc, Rs=Rs)
    print(f"\nsaved -> {OUT}/exp18_ordering.npz")


if __name__ == "__main__":
    main()
