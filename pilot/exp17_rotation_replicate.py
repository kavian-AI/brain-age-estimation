#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Is the "rotate early, then only scale" finding real, or a one-dataset fluke?

The claim from E14: going CN -> MCI the mean regional brain-age profile
rotates 31.7 degrees; going MCI -> AD it rotates only 3.9 degrees, while its
magnitude doubles. i.e. the SPATIAL PATTERN is set early, then just amplifies.

Replication tests:
  1. bootstrap CIs on the angles (is 31.7 vs 3.9 stable, or noise?)
  2. a permutation null: what angle do we get if labels are shuffled?
  3. independent split-half of ADNI (two disjoint halves of subjects)
  4. a completely different staging variable: CDR global (0 / 0.5 / 1+)
  5. OASIS-1, a different cohort entirely (CDR 0 / 0.5 / 1+)
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
DATA = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\data"
OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
rng = np.random.default_rng(0)


def angle(u, w):
    c = u @ w / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def regional_gaps(ref, tgt, cols, icv="ICV", age="AGE", sexcol="PTGENDER"):
    G = np.zeros((len(tgt), len(cols)))
    rs = (ref[sexcol] == "Male").astype(int).values if ref[sexcol].dtype == object \
        else ref[sexcol].values
    ts = (tgt[sexcol] == "Male").astype(int).values if tgt[sexcol].dtype == object \
        else tgt[sexcol].values
    for r, c in enumerate(cols):
        Xh = np.column_stack([ref[c].values, ref[icv].values, rs])
        sc = StandardScaler().fit(Xh)
        m = LinearRegression().fit(sc.transform(Xh), ref[age].values)
        b = LinearRegression().fit(ref[age].values.reshape(-1, 1),
                                   m.predict(sc.transform(Xh)) - ref[age].values)
        X = sc.transform(np.column_stack([tgt[c].values, tgt[icv].values, ts]))
        G[:, r] = (m.predict(X) - tgt[age].values) \
            - b.predict(tgt[age].values.reshape(-1, 1))
    return G


def angles_from(G, y):
    m0, m1, m2 = G[y == 0].mean(0), G[y == 1].mean(0), G[y == 2].mean(0)
    return (angle(m0, m1), angle(m1, m2), angle(m0, m2),
            np.linalg.norm(m0), np.linalg.norm(m1), np.linalg.norm(m2))


def boot_angles(G, y, n=2000):
    a01, a12 = [], []
    idx = {k: np.where(y == k)[0] for k in (0, 1, 2)}
    for _ in range(n):
        bs = np.concatenate([rng.choice(v, len(v), replace=True) for v in idx.values()])
        yy = np.concatenate([np.full(len(v), k) for k, v in idx.items()])
        r = angles_from(G[bs], yy)
        a01.append(r[0]); a12.append(r[1])
    return np.array(a01), np.array(a12)


def perm_null(G, y, n=1000):
    out = []
    for _ in range(n):
        yy = rng.permutation(y)
        r = angles_from(G, yy)
        out.append(r[0])
    return np.array(out)


def report(G, y, title):
    a01, a12, a02, n0, n1, n2 = angles_from(G, y)
    print(f"\n--- {title}")
    print(f"    magnitudes: stage0 {n0:.2f} -> stage1 {n1:.2f} -> stage2 {n2:.2f}")
    print(f"    angle 0->1 = {a01:5.1f} deg | angle 1->2 = {a12:5.1f} deg"
          f" | ratio = {a01/max(a12,1e-9):.1f}x")
    return a01, a12


def main():
    d = pd.read_csv(f"{DATA}/ADNIMERGE_05Mar2026.csv", low_memory=False)
    a = d.dropna(subset=V + ["ICV", "AGE", "DX", "PTGENDER"]).sort_values("Years_bl")
    base = a.groupby("RID").first().reset_index()
    cn_all = base[base.DX == "CN"]

    print("=" * 72)
    print("TEST 1 — ADNI full cohort, with bootstrap CIs and permutation null")
    print("=" * 72)
    ref = cn_all.iloc[::2]          # half the CN as the age reference
    rest = base[~base.RID.isin(ref.RID)]
    sub = rest[rest.DX.isin(["CN", "MCI", "Dementia"])]
    y = sub.DX.map({"CN": 0, "MCI": 1, "Dementia": 2}).values
    G = regional_gaps(ref, sub, V)
    a01, a12 = report(G, y, f"ADNI (n={len(y)}, ref CN n={len(ref)})")

    b01, b12 = boot_angles(G, y)
    print(f"    bootstrap 0->1: {b01.mean():5.1f} [{np.percentile(b01,2.5):.1f}, "
          f"{np.percentile(b01,97.5):.1f}]")
    print(f"    bootstrap 1->2: {b12.mean():5.1f} [{np.percentile(b12,2.5):.1f}, "
          f"{np.percentile(b12,97.5):.1f}]")
    diff = b01 - b12
    p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
    print(f"    is 0->1 > 1->2 ?  diff {diff.mean():.1f} deg, p={p:.4f}")
    nul = perm_null(G, y)
    print(f"    permutation null for 0->1: {nul.mean():.1f} deg "
          f"[{np.percentile(nul,2.5):.1f}, {np.percentile(nul,97.5):.1f}]"
          f"  -> observed {a01:.1f} is {'OUTSIDE' if a01 > np.percentile(nul,97.5) else 'inside'} the null")

    print("\n" + "=" * 72)
    print("TEST 2 — split-half of ADNI (disjoint subjects)")
    print("=" * 72)
    ids = sub.RID.values.copy()
    rng.shuffle(ids)
    h1, h2 = set(ids[:len(ids)//2]), set(ids[len(ids)//2:])
    for nm, hh in [("half A", h1), ("half B", h2)]:
        s = sub[sub.RID.isin(hh)]
        yy = s.DX.map({"CN": 0, "MCI": 1, "Dementia": 2}).values
        report(regional_gaps(ref, s, V), yy, f"ADNI {nm} (n={len(yy)})")

    print("\n" + "=" * 72)
    print("TEST 3 — different staging variable: CDR global (0 / 0.5 / >=1)")
    print("=" * 72)
    c = a.groupby("RID").first().reset_index()
    c = c.dropna(subset=["CDRSB"])
    # CDR-SB bands as an independent severity axis
    yy = pd.cut(c.CDRSB, [-0.1, 0.0, 2.5, 100], labels=[0, 1, 2]).astype(float)
    ok = yy.notna()
    c2, yv = c[ok], yy[ok].astype(int).values
    refc = c2[(yv == 0)].iloc[::2]
    keep = ~c2.RID.isin(refc.RID)
    report(regional_gaps(refc, c2[keep], V), yv[keep.values],
           f"ADNI by CDR-SB (n={keep.sum()})")

    print("\n" + "=" * 72)
    print("TEST 4 — OASIS-1, a completely different cohort")
    print("=" * 72)
    o = pd.read_csv(f"{DATA}/oasis_cross-sectional.csv")
    o = o.dropna(subset=["Age", "nWBV", "eTIV", "CDR"]).copy()
    o["sex"] = (o["M/F"] == "M").astype(int)
    o["stage"] = pd.cut(o.CDR, [-0.1, 0.0, 0.5, 10], labels=[0, 1, 2]).astype(int)
    # OASIS has only global measures -> use the 3 available as a mini-profile
    cols = ["nWBV", "eTIV"]
    refo = o[o.stage == 0].iloc[::2]
    resto = o[~o.index.isin(refo.index)]
    Go = regional_gaps(refo, resto, cols, icv="eTIV", age="Age", sexcol="sex")
    report(Go, resto.stage.values, f"OASIS-1 (n={len(resto)}, only {len(cols)} measures)")
    print("    NOTE: OASIS has no regional parcellation, so this is a weak test.")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print("  Look for: angle(0->1) consistently LARGER than angle(1->2)")


if __name__ == "__main__":
    main()
