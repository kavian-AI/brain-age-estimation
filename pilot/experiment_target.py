#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment: does the TRAINING TARGET matter?

Symmetric comparison. All models see the SAME inputs (nWBV, eTIV, sex),
the SAME subjects, and the SAME cross-validation folds. Only the training
target changes:

  M0  benchmark   : nWBV alone (no model)
  M1  age target  : predict chronological age -> brain age gap  (standard brain age)
  M2  disease tgt : predict time-to-conversion (Cox, censoring-aware)
  M3  direct      : predict conversion status  (logistic)

Chronological age is deliberately NOT an input feature for any model,
otherwise M2/M3 would gain an unfair advantage over the brain-age framing.

Outcome: Harrell's C-index for time-to-conversion, on pooled out-of-fold
predictions, repeated over many CV seeds.

Cohort: OASIS-2 subjects non-demented at baseline.
"""

import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\data"
OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"
FEATS = ["nWBV", "eTIV", "sex"]
N_REPEATS = 50
rng = np.random.default_rng(0)


def build_cohort(label_rule="cdr"):
    """Subjects non-demented at baseline; event = first CDR>0 visit."""
    d = pd.read_csv(f"{DATA}/oasis_longitudinal.csv").sort_values(
        ["Subject ID", "MR Delay"])
    rows = []
    for sid, g in d.groupby("Subject ID"):
        g = g.dropna(subset=["CDR", "nWBV", "eTIV"])
        if len(g) < 2:
            continue
        base = g.iloc[0]
        if base["CDR"] > 0:
            continue                          # prevalent case, not at risk
        conv = g[g["CDR"] > 0]                 # event time is always first CDR>0 visit
        if label_rule == "cdr":
            ev = len(conv) > 0
        else:
            # sensitivity check: trust the dataset's curated subject-level label
            ev = (base["Group"] == "Converted") and len(conv) > 0
        t = (conv.iloc[0] if ev else g.iloc[-1])["MR Delay"] / 365.25
        if t <= 0:
            continue
        rows.append(dict(sid=sid, t=t, event=int(ev), age=base["Age"],
                         nWBV=base["nWBV"], eTIV=base["eTIV"],
                         sex=1 if base["M/F"] == "M" else 0))
    return pd.DataFrame(rows)


def run_cv(df, seed):
    """One repeat of stratified 5-fold; returns pooled out-of-fold risk scores."""
    X = df[FEATS].values
    y_ev, y_t, y_age = df.event.values, df.t.values, df.age.values
    n = len(df)
    scores = {k: np.zeros(n) for k in ("M0", "M1", "M2", "M3")}

    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y_ev):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])

        # M0 - lower brain volume = higher risk
        scores["M0"][te] = -df.nWBV.values[te]

        # M1 - age target. Fit on non-converters in the training fold only
        # (they are the "healthy reference"), then gap = predicted - actual.
        healthy = tr[y_ev[tr] == 0]
        sc_h = StandardScaler().fit(X[healthy])
        m_age = LinearRegression().fit(sc_h.transform(X[healthy]), y_age[healthy])
        gap_tr = m_age.predict(sc_h.transform(X[healthy])) - y_age[healthy]
        bias = LinearRegression().fit(y_age[healthy].reshape(-1, 1), gap_tr)
        gap_te = (m_age.predict(sc_h.transform(X[te])) - y_age[te]
                  - bias.predict(y_age[te].reshape(-1, 1)))
        scores["M1"][te] = gap_te

        # M2 - disease-informed target: time to conversion, censoring-aware
        tr_df = pd.DataFrame(Xtr, columns=FEATS)
        tr_df["t"], tr_df["event"] = y_t[tr], y_ev[tr]
        try:
            cph = CoxPHFitter(penalizer=0.1).fit(tr_df, "t", "event")
            scores["M2"][te] = cph.predict_partial_hazard(
                pd.DataFrame(Xte, columns=FEATS)).values
        except Exception:
            scores["M2"][te] = np.nan

        # M3 - direct supervision on the outcome label
        if len(np.unique(y_ev[tr])) == 2:
            lr = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, y_ev[tr])
            scores["M3"][te] = lr.predict_proba(Xte)[:, 1]
        else:
            scores["M3"][te] = np.nan
    return scores


def cindex(df, s):
    """Higher score = higher risk -> negate for concordance_index convention."""
    ok = ~np.isnan(s)
    return concordance_index(df.t.values[ok], -s[ok], df.event.values[ok])


def main():
    print("=" * 74)
    print("EXPERIMENT: does the training target matter?")
    print("=" * 74)

    for rule in ("cdr", "group"):
        df = build_cohort(rule)
        ev = int(df.event.sum())
        print(f"\n--- cohort definition = '{rule}':  n={len(df)}, events={ev}, "
              f"median follow-up={df.t.median():.2f} y")

        res = {k: [] for k in ("M0", "M1", "M2", "M3")}
        for r in range(N_REPEATS):
            sc = run_cv(df, seed=r)
            for k, v in sc.items():
                res[k].append(cindex(df, v))

        names = {"M0": "M0  nWBV alone (benchmark)",
                 "M1": "M1  age target -> brain age gap",
                 "M2": "M2  disease target -> Cox risk",
                 "M3": "M3  direct supervision"}
        print(f"    {'model':34s} {'C-index':>9s}  {'95% range over CV repeats':>26s}")
        for k in ("M0", "M1", "M2", "M3"):
            a = np.array(res[k])
            print(f"    {names[k]:34s} {a.mean():9.3f}  "
                  f"[{np.percentile(a,2.5):.3f}, {np.percentile(a,97.5):.3f}]")

        if rule == "cdr":
            keep = df, res

    # ---------------- paired comparison on the primary (CDR) definition
    df, res = keep
    d21 = np.array(res["M2"]) - np.array(res["M1"])
    print("\n" + "=" * 74)
    print("PRIMARY COMPARISON  (disease target vs age target)")
    print("=" * 74)
    print(f"  delta C-index (M2 - M1) = {d21.mean():+.3f}   "
          f"SD across CV repeats = {d21.std():.3f}")

    # subject-level bootstrap for a proper CI
    boot = []
    for _ in range(400):
        idx = rng.choice(len(df), len(df), replace=True)
        sub = df.iloc[idx].reset_index(drop=True)
        if sub.event.sum() < 3:
            continue
        s = run_cv(sub, seed=int(rng.integers(1e6)))
        boot.append(cindex(sub, s["M2"]) - cindex(sub, s["M1"]))
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    print(f"  subject-level bootstrap: {boot.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  p={p:.3f}")
    print("  ->", "disease target BETTER" if p < .05 and boot.mean() > 0
          else "no significant difference at this sample size")

    # ---------------- how many events would we need?
    print("\n" + "=" * 74)
    print("POWER: how many conversion events are needed?")
    print("=" * 74)
    n_ev = int(df.event.sum())
    se = boot.std()
    obs = abs(boot.mean())
    print(f"  observed |delta| = {obs:.3f}, bootstrap SE = {se:.3f} at {n_ev} events")
    if obs > 1e-6:
        se_needed = obs / (1.96 + 0.84)          # 80% power, alpha=0.05
        need = int(np.ceil(n_ev * (se / se_needed) ** 2))
        print(f"  events needed for 80% power to detect this effect: ~{need}")
        print(f"  ADNI typically yields 250-350 MCI converters -> "
              f"{'sufficient' if need <= 350 else 'still marginal; widen window or pool cohorts'}")
    else:
        print("  effect indistinguishable from zero; cannot extrapolate a sample size")

    pd.DataFrame(res).to_csv(f"{OUT}/experiment_target_cindex.csv", index=False)
    np.savetxt(f"{OUT}/experiment_target_bootstrap.csv", boot, delimiter=",")
    print(f"\nsaved -> {OUT}/experiment_target_cindex.csv")


if __name__ == "__main__":
    main()
