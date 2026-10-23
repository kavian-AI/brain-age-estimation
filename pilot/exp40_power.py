#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
POWER ANALYSIS: how large a cohort is needed to say anything about a regional
brain-age cascade?

E38/E39 found that measurement precision explains between half and all of the
apparent onset spread, with p-values swinging non-monotonically between 0.000 and
1.000 across thresholds. That instability is the symptom of an underpowered test,
not of a subtle effect. This answers the question that should have been asked
first: given the noise we measured, how many converters does it take to detect a
TRUE difference in onset of a given size?

THE SIMULATION
--------------
Ground truth: each region starts deviating at its own true onset T_r and grows
linearly to the amplitude A_r that we actually observe at conversion. Regions
differ ONLY in T_r (the effect) and in their measured noise (the nuisance).

Observation: bin means carry sampling noise noise_r / sqrt(n in that bin), with
bin occupancy scaled from the real ADNI visit distribution.

Test: read each region's onset with the reliability criterion (deviation >= k x
its own per-person noise), take the spread, and compare it against a null in
which every region shares one onset. Power is the fraction of simulated datasets
in which that comparison reaches p < 0.05.

Both real per-region noise values and real amplitudes are taken from the data, so
the only thing being varied is cohort size and true effect size.
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
FA = {"Hippocampus": "هیپوکامپ", "Entorhinal": "آنتورینال", "Fusiform": "دوکی",
      "MidTemp": "گیجگاهی میانی", "Ventricles": "بطن‌ها", "WholeBrain": "کل مغز"}

# measured in E38/E39 on 379 converters + 650 stable MCI
NOISE = {"Hippocampus": 0.42, "Entorhinal": 0.40, "Fusiform": 0.42,
         "MidTemp": 0.39, "Ventricles": 0.10, "WholeBrain": 0.36}
AMPL = {"Hippocampus": 4.45, "Entorhinal": 1.68, "Fusiform": 2.55,
        "MidTemp": 3.16, "Ventricles": 1.23, "WholeBrain": 3.21}
BIN_N_REF = np.array([414, 308, 214, 113, 91, 76])     # visits per bin, n=379
N_REF = 379
MID = np.array([0.125, 1.25, 2.25, 3.25, 4.375, 6.0])
K = 2.0
N_SIM = 400
N_NULL = 400


def ramp(t_before, onset, ampl):
    """Deviation at a given time before conversion: zero before onset, then
    linear growth to `ampl` at conversion."""
    x = np.clip((onset - t_before) / max(onset, 1e-6), 0, 1)
    return ampl * x


def onset_of(curve, thresh):
    if not np.isfinite(curve[0]) or curve[0] < thresh:
        return np.nan
    for i in range(1, len(curve)):
        if curve[i] < thresh:
            f = (curve[i - 1] - thresh) / (curve[i - 1] - curve[i])
            return MID[i - 1] + f * (MID[i] - MID[i - 1])
    return MID[-1]


def simulate(true_onsets, n_conv, rng, sigma_between=0.0):
    """One simulated study: observed onset spread.

    sigma_between is the SUBJECT-TO-SUBJECT spread in when disease starts. The
    first version of this analysis left it at zero - every patient deviating on
    exactly the same schedule - and returned 100% power everywhere, which
    contradicted the unstable p-values the real data produced. Patients do not
    share a schedule, and averaging over a population whose onsets are spread out
    smears the group curve into a gentle slope, which is what actually destroys
    the ability to locate an onset.
    """
    bn = np.maximum(8, (BIN_N_REF * n_conv / N_REF)).astype(int)
    on = []
    for v in V:
        if sigma_between > 0:
            # group mean over subjects whose individual onsets are spread out.
            # Vectorised over subjects: the per-subject Python loop made a single
            # sweep take longer than the entire rest of the analysis.
            subj = np.clip(true_onsets[v]
                           + rng.normal(0, sigma_between, min(n_conv, 1500)),
                           0.3, 12.0)
            x = np.clip((subj[None, :] - MID[:, None]) / subj[None, :], 0, 1)
            clean = AMPL[v] * x.mean(axis=1)
        else:
            clean = np.array([ramp(m, true_onsets[v], AMPL[v]) for m in MID])
        obs = clean + rng.normal(0, NOISE[v] / np.sqrt(bn))
        o = onset_of(obs, K * NOISE[v])
        if np.isfinite(o):
            on.append(o)
    return (max(on) - min(on)) if len(on) >= 2 else np.nan


def one_study(true_onsets, n_conv, rng, sb=0.0):
    """Observed spread, and p against a null where all onsets are equal."""
    obs = simulate(true_onsets, n_conv, rng, sb)
    if not np.isfinite(obs):
        return np.nan
    flat = {v: np.mean(list(true_onsets.values())) for v in V}
    null = np.array([simulate(flat, n_conv, rng, sb) for _ in range(N_NULL)])
    null = null[np.isfinite(null)]
    if len(null) < 20:
        return np.nan
    return (null >= obs).mean()


def main():
    print("measured inputs (from 379 converters):")
    print(f"  {'region':16s} {'noise (y)':>10} {'amplitude':>11}")
    for v in V:
        print(f"  {FA[v]:16s} {NOISE[v]:10.2f} {AMPL[v]:11.2f}")
    print(f"\n  ventricles measured {np.mean([NOISE[x] for x in V if x!='Ventricles'])/NOISE['Ventricles']:.1f}x "
          f"more precisely than the average region")

    rng = np.random.RandomState(0)
    sizes = [200, 379, 600, 1000, 2000, 4000]
    seps = [0.0, 1.0, 2.0, 3.0]
    import sys
    SB = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    print(f"\n  subject-to-subject spread in onset: {SB:.1f} y")

    print("\n" + "=" * 78)
    print("POWER to detect a TRUE onset separation, at p<0.05")
    print("=" * 78)
    print("  rows: converters in the cohort.  columns: true separation between")
    print("  the earliest and latest region, in years.\n")
    print(f"  {'converters':>11}" + "".join(f"{s:>10.0f}y" for s in seps))
    grid = {}
    for n in sizes:
        row = []
        for sep in seps:
            # spread the true onsets evenly over `sep`, hippocampus earliest
            order = ["Hippocampus", "MidTemp", "WholeBrain", "Fusiform",
                     "Entorhinal", "Ventricles"]
            t = {v: 4.0 - sep * i / max(1, len(order) - 1)
                 for i, v in enumerate(order)}
            # The null depends only on cohort size and heterogeneity, not on
            # the observed study, so it is built ONCE per cell instead of once
            # per simulated study - the earlier nested version was N_SIM x
            # N_NULL calls per cell and did not finish.
            flat = {v: np.mean(list(t.values())) for v in V}
            null = np.array([simulate(flat, n, rng, SB) for _ in range(N_NULL)])
            null = null[np.isfinite(null)]
            obs = np.array([simulate(t, n, rng, SB) for _ in range(N_SIM)])
            obs = obs[np.isfinite(obs)]
            if len(null) < 20 or len(obs) < 20:
                pw = np.nan
            else:
                crit = np.percentile(null, 95)
                pw = (obs > crit).mean()
            row.append(pw)
            grid[(n, sep)] = pw
        print(f"  {n:>11}" + "".join(
            f"{100*x:>9.0f}%" if np.isfinite(x) else "      n/a" for x in row))

    print("\n  (the 0-year column is the FALSE POSITIVE rate - it should sit")
    print("   near 5%. Anything much above that means the test is not valid.)")

    print("\n" + "=" * 78)
    print("WHAT THIS MEANS FOR THE PROPOSAL")
    print("=" * 78)
    fp = grid[(379, 0.0)]
    print(f"  false-positive rate at the current cohort size: {100*fp:.0f}%")
    for sep in [1.0, 2.0, 3.0]:
        need = [n for n in sizes if np.isfinite(grid[(n, sep)])
                and grid[(n, sep)] >= 0.80]
        got = grid[(379, sep)]
        s = f"{need[0]} converters" if need else f"more than {sizes[-1]}"
        print(f"  true separation {sep:.0f} y: power at n=379 is {100*got:3.0f}%; "
              f"80% power needs {s}")

    pd.DataFrame({f"sep_{s}": [grid[(n, s)] for n in sizes] for s in seps},
                 index=sizes).to_csv(os.path.join(OUT, "exp40_power.csv"))
    print(f"\nsaved -> {OUT}\\exp40_power.csv")


if __name__ == "__main__":
    main()
