# Atlas v2 — healthy-brain normative atlas (FastSurfer volumes)

Built 2026-10-24 (code: `pilot/v2/`, run order in `pilot/v2/README.md`) from the same 7,603 healthy people (77 scanners, 17 sources, ages 6–94) as the published
`p5_out` atlas, with the same pipeline (FastSurfer 2.4.2 aseg + DKT, SynthStrip ICV) and the same QC.

## What it does

**Input**, per person: age, sex (0 = female, 1 = male), ICV (mm³), scanner, and FastSurfer volumes (mm³).
**Output**, per structure: a z-score and centile = where this person sits among healthy people of the same
age, sex, head size and scanner. `z > 0` always means *larger than expected* (atrophy = negative z for tissue,
positive z for ventricles / CSF).

**59 structures**: 23 bilateral (the 17 published regions + caudate, pallidum, accumbens, ventral DC,
cerebellar WM, 4th ventricle) and 36 left/right (every paired structure modelled per hemisphere).

## The model

Per structure, on `y = log volume` (SHASH location–scale model, extending `pilot/shash.py`):

```
y      = mu + sigma * sinh((asinh(Z) + nu) / tau),   Z ~ N(0,1)

mu     = f(age)                         natural spline, 7 df
       + beta * sex                     average male–female difference
       + g(age) * sex                   ... changing smoothly with age (spline, +2 df)            [v2]
       + b * log ICV                    head-size power law, exponent per structure
       + c * (age-60)/10 * log ICV      exponent changing with age: temporal horn, 3rd ventricle   [v2]
       + shift[scanner]                 empirical-Bayes shrunk toward the typical scanner          [v2]

log sd = h(age)                         natural spline, 4 df
       + spread[scanner]                empirical-Bayes shrunk                                      [v2]
       + d1 * log ICV + d2 * log ICV^2  CSF spaces only (lateral, temporal horn, 3rd ventricle)     [v2]

nu, tau  one skew and one tail-weight parameter per structure
```

Because the model is on log volume, every `+` is a `×` on volume (power laws, multiplicative scanner and
sex factors). The age curves are free-form splines.

### Why each v2 term is there (evidence on this pool)

| term | evidence |
|---|---|
| age × sex (curve) | published atlas: healthy older men's tissue z 0.2–0.37 SD low, CSF z 0.2–0.37 SD high (total brain 60–75: −0.29, 5.2 SE). LR-significant in 15/17 regions; curve fixes 75+ (hippocampus gap −0.47 → −0.12) |
| age × head size (temporal horn, 3rd ventricle) | head-size exponent 0.8 (young) → 1.4–1.6 (after 40); LR p = 4e-4 and 1e-6 on top of age × sex |
| head size in spread (CSF only) | held-out gain only in CSF; SD at largest heads 1.105 → 1.053 (3rd), 1.070 → 1.028 (lateral) |
| scanner shrinkage (BrainChart's random effects) | held-out log-likelihood better in 17/17 regions; small sites (n < 40): SD z 1.13 → 1.06, inside ±1.96 92.1% → 94.0% |
| left/right models (PCN braincharts) | +0.027 AUC in mild dementia (p = .015) |
| extra subcortical structures (PCN) | full set now matches PCN's 27 subcortical phenotypes |
| **not** added: spread by sex (CentileBrain) | no held-out gain (6/17 regions better) |

## Validation

**Held-out calibration** (5-fold, 59 models): mean z ≈ 0; SD 1.009–1.024 (median 1.019; published ≈ 1.03);
inside ±1.96 94.4–95.5% (median 94.8%; published 94.3%); age bands with a significant male–female bias:
0 in 57/59 models (published: up to 4 of 5); small sites inside ±1.96 median 94.2% (published ≈ 92%).

**New scanner** (10 scanners left out in turn, 23 structures, 20 random draws):

| local healthy controls | 0 | 5 | 10 | 20 | 40 | 80 |
|---|---|---|---|---|---|---|
| typical bias, median \|mean z\| | 0.40 | 0.27 | 0.22 | **0.18** | 0.13 | 0.10 |
| 90th percentile bias | 0.97 | 0.39 | 0.30 | 0.23 | 0.17 | 0.13 |
| inside ±1.96 | 92.8% | 94.2% | 94.6% | **94.9%** | 95.1% | 95.2% |

→ never use uncalibrated on a new scanner; **≥ 20 healthy local controls**, 40 if possible.

**Dementia (OASIS-1/2, published p6 folds, AUC within sex)**:

| | CDR 0.5 | CDR ≥ 1 | OASIS-2 demented |
|---|---|---|---|
| v2 learned (27 subcortical L/R) | 0.886 | 0.930 | 0.891 |
| published learned (17 bilateral) | 0.843 | 0.909 | 0.878 |
| PCN braincharts learned (27) | 0.897 | 0.947 | 0.902 |

v2 vs published: +0.043 (p = .026) in mild dementia; v2 vs PCN: all differences ≤ 0.017, none significant.
The pooled AD-signature AUC is ~0.017 lower than published: the published atlas was flattered by sex
(patients include more men; sex alone gives AUC 0.58–0.63). Within sex the two are identical.

**Calibration on held-out OASIS controls**: v2 SD 1.02–1.06, 94.2–94.6% inside ±1.96, |r(z, head size)| ≈ 0.06;
PCN SD 0.67–0.80, 97.7–98.3%, |r| 0.34–0.43 (no head-size adjustment).

**Deep-learning comparison**: a neural network with SHASH outputs (3-network ensemble, same folds) was worse
in held-out log-likelihood in 8/8 regions and flattened aging rates (entorhinal 0.25×, hippocampus 0.64×,
temporal horn 0.64× of v2's rate at 70), as the MLP in the published model contest did (median 0.83×).
Black-box residual auditors (boosted trees, MLP) could not detect a planted 0.2 SD bias in men ≥ 60 and only
borderline 0.3 SD, so the targeted sex × age / head-size / site checks are the evidence of completeness.

## Using it

```
python pilot/v2/atlas_v2/score_atlas_v2.py people.csv scores.csv                                # scanners already in the atlas
python pilot/v2/atlas_v2/score_atlas_v2.py people.csv scores.csv --site-controls controls.csv   # new scanner: >= 20 healthy controls
```

`people.csv`: `subject, age, sex, icv, site` + one column per structure (names as in `coefs_v2.csv`, mm³).
Ages outside 5–95 are not scored. Volumes must come from FastSurfer 2.4.2 with SynthStrip ICV.

## Files

| file | content |
|---|---|
| `grid_v2.csv.gz` | per structure × sex × age (0.25 y steps): reference location and log-SD |
| `coefs_v2.csv` | per structure: head-size exponent b, age × head-size c, spread head-size terms, ν, τ, scanner-variance priors |
| `sites_v2.csv` | per structure × scanner: location shift and log-SD offset |
| `everyage_table_v2.csv` | per structure × sex × age 7–89: median, spread, aging rate, resolution R, centiles 2.5–97.5 (reference head 1,538 cm³, average scanner) |
| `site_params_v2.csv` | per structure × scanner: shift %, spread factor (same layout as `p5_out/site_params.csv`) |
| `score_atlas_v2.py` | standalone scorer (numpy + pandas only) |
| `portable_check.csv` | portable tables vs fitted model: max \|Δz\| = 2.5e-4 |

## Known limitations / open items

1. Healthy older men have a slightly heavier low tail in brainstem, cerebellum, ventral DC and total brain
   (e.g. brainstem 3.8% vs 2.1% below z = −1.96). Not explained by field-of-view cut-off (not concentrated in
   cut-off datasets or large heads). Candidates: subclinical disease more common in older men, or sex-specific
   skew/tail shape (untested).
2. Temporal horn: 3.5% of healthy over-60s (both sexes) beyond +1.96 — plausibly preclinical disease in "healthy" samples.
3. All atlases detect dementia less well in men than women here (male control groups are small: 22–32).
4. No single-scan score separates future converters (AUC 0.54–0.63, n = 14).
5. New-scanner spread adaptation is effectively off (spread prior estimated from shrunk fits); uncalibrated
   spread was already fine (SD 0.98–0.99) but the prior should be re-estimated from unshrunk fits.
6. Within-person aging rates (`R_long`) and the DLBS drift issue are untouched by v2.

## Ready-to-paste FINDINGS_LOG entry

```
# ATLAS V2 (2026-10-24) — pilot/v2/
Same 7,603-person pool. Adds: age x sex curve (all), age x ICV (temporal horn, 3rd vent), ICV in spread
(3 CSF), empirical-Bayes scanner shrinkage, L/R models, PCN's extra subcortical structures (59 models).
Held-out: SD 1.009-1.024, in +-1.96 94.4-95.5%, sex-biased age bands 0/5 in 57/59 (published up to 4/5).
New scanner: bias 0.40 -> 0.18 SD with 20 controls (0.13 with 40). OASIS within-sex AUC: v2 learned27
0.886/0.930/0.891 vs published 0.843/0.909/0.878 (+0.043, p .026 CDR0.5) vs PCN 0.897/0.947/0.902 (n.s.).
Published pooled AUC partly sex-confounded (patients more male; sex alone AUC 0.58-0.63).
Neural-SHASH: worse held-out logp 8/8, aging rate @70 flattened (entorhinal 0.25x, hippocampus 0.64x).
Open: older men's heavier low tail (brainstem/cerebellum), not FOV.
```
