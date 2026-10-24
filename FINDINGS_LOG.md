# Brain Age Project — Findings Log

Running record of every experiment, result, figure, and script. For later reporting.
All experiments use open data (OASIS, ADNI-via-ADNIMERGE, ABIDE), run locally on CPU.
Everything reproducible from `pilot/*.py`; data in `data/`.

---

## Datasets acquired

| Dataset | n | Content | File |
|---|---|---|---|
| OASIS-1 | 436 | cross-sectional, CN + impaired, nWBV/eTIV | `data/oasis_cross-sectional.csv` |
| OASIS-2 | 150 subj / 373 visits | longitudinal, 14 converters, nWBV | `data/oasis_longitudinal.csv` |
| ADNIMERGE | 2430 subj / 16421 visits | 6 FreeSurfer regions + ICV, CSF, cognition, 379 MCI→dem converters | `data/ADNIMERGE_05Mar2026.csv` |
| ABIDE | 1103 | 32 bilateral FreeSurfer ROIs, autism vs control, ages 6.5–64 | `data/abide_merged.csv` (harvested from S3 via `fetch_abide_freesurfer.py`) |

---

## Architecture (what the model does)

Two stages. Stage 1 = **regression**: inputs = regional volumes + ICV + sex; target = chronological age; trained on **healthy only**; output = predicted age → gap = predicted − real. Stage 2 = **classification**: input = gap / profile / volumes; target = disease label; output = P(disease). Figure: `pilot/fig6_architecture.png`.

---

## Experiment results

### E1 — Reproduction + "does the gap beat volume?" (OASIS)
`pilot_brainage.py` → `fig_pilot_results.png`, `fig2_comparison.png`, `fig3_longitudinal.png`
- Age model on healthy: r=0.78, MAE=6.9y. Impaired vs healthy gap = **+5.83y** (d=0.76, p=4e-7). Reproduces Franke 2010.
- **Gap AUC 0.709 vs nWBV-alone 0.696, p=0.65 → gap does NOT beat simple volume.**
- Matches Tan 2025 (AUC 0.70 from a 53,542-subject deep model) with 3 numbers + linear regression.
- Longitudinal: converters' gap slope 0.73 vs stable 0.24 (p=0.09); offset does NOT separate (p=0.47).

### E2 — Does the training TARGET matter? (OASIS-2, Cox)
`experiment_target.py` → `fig4_target.png`
- 4 targets (age / time-to-conversion / direct / none), same inputs. With 15 events, no target beats the others.
- **No trained model beat nWBV alone.** Power calc: ~550 events needed → ADNI alone may be insufficient, pool cohorts.

### E3+E4 — The fair-baseline CONTROL experiment (ADNI + ABIDE)  ★ novel angle
`exp2_regional.py`, `exp3_prognosis.py` → `fig5_control.png`
- The honest baseline is **volumes + age**, not volumes alone (the gap smuggles age in).
- Across 3 tasks, 2 diseases, all p<0.001:
  - **Compression is real**: scalar gap → regional profile gains +0.02–0.04 AUC every time.
  - **But the brain-age transform NEVER beats its own inputs (volumes+age).**
  | task | global gap | regional profile | shape only | volumes+age |
  |---|---|---|---|---|
  | ADNI AD vs CN | 0.893 | 0.933 | 0.902 | **0.943** |
  | ADNI MCI→dem | 0.722 | 0.746 | 0.713 | **0.752** |
  | ABIDE autism | 0.537 | 0.567 | 0.566 | **0.583** |
- Bonus geometry finding: in AD, shape-only stays diagnostic (0.902) → spatial pattern matters; in autism, shape ≈ full → no severity axis. **Two diseases, different brain-age geometry.**

### E5 — Fusion ideas: multi-task / transfer / ensemble (ADNI, PyTorch)
`exp4_fusion.py` → `fig7_fusion.png`
- Real shared-trunk multi-task, real fine-tuning. None beat volumes+age logistic.
- **pretrain→fine-tune was WORST** (0.913 diag) — independently replicates Tan 2025's transfer-learning null.
- Reason: 6 tabular features → MLP overfits vs logistic. Not compute-limited.

### E6 — Direct model vs gap: do they agree? + do false alarms convert? (ADNI)
`exp5_agreement.py`, `exp6_prognostic.py` → `fig8_agreement.png`, `fig9_prognostic.png`
- corr(direct score, gap) = 0.77 — mostly agree.
- **21% of direct-flagged-diseased have a NORMAL gap** → direct model detects via shape, not "looking old".
- **False alarms DO convert**: CN flagged diseased convert at **16% vs 4%** (4×). They were early detections, not errors.
- CN split by gap: high-gap 10% vs low-gap 1% convert (**10×**).
- Cox: gap has independent hazard HR=1.35 (p<0.001) after adjusting for direct, but no whole-cohort C-index gain (r=0.77). Value concentrated in CN (C 0.691→0.714).

### E7 — Diagnosis vs progression: which marker wins? (ADNI)
`exp7_progression.py`
- Diagnosis AUC: direct 0.948 > gap 0.903. Progression (ADAS-slope) Spearman: direct 0.49 > gap 0.43.
- Direct wins both cross-sectional tasks. Age alone ≈ 0.

### E8 — RATE of brain aging (the thing the snapshot CANNOT do) (ADNI)  ★ headline
`exp8_rate.py` → `fig10_rate.png`
- Landmark design (≥3 scans estimate rate, predict conversion strictly after). n=1100, 160 conversions.
- **Rate alone C=0.774 ≈ direct snapshot 0.772; direct + rate = 0.820 (+0.048).**
- In full Cox, **rate is the STRONGEST predictor: HR=1.84, p=5e-22** > direct HR=1.51. Independent of baseline gap + direct.
- Converters age ~2× faster: 0.97 vs 0.51 gap-yr/yr.
- NOTE: this general idea is already published (DunedinPACNI, Nature Aging 2025; Jin 2025) — our pipeline replicates it, so it validates our method but is not novel by itself.

### E9 — Regional RATE profile + combined score (ADNI)  ★ novel angle
`exp9_regional_rate.py` → part of `fig11_final.png`
- direct 0.770 | scalar rate 0.782 | **regional rate profile 0.799** | direct+scalar 0.822 | **direct+regional profile 0.831**
- Regional profile beats scalar rate (+0.009, p<0.001) — compression hurts here too.
- Fastest-aging regions in converters: **hippocampus (+0.36), mid-temporal (+0.34)** — AD-signature, interpretable.

### E10 — External validation on OASIS-2  ★ the replication
`exp10_oasis_validate.py` → part of `fig11_final.png`
- Independent cohort, different scanner, different features (nWBV not FreeSurfer). Reference models from OASIS-1.
- **direct+rate gain REPLICATES: +0.047** (0.633→0.680), vs ADNI +0.048. Converters age faster (0.69 vs 0.26).
- Underpowered for single-term significance (15 events, rate HR=1.48 p=0.11) but effect direction + C-index gain match ADNI.

---

## What is NOVEL vs what REPLICATES (be honest in any report)

- **Replicates existing published work** (validates our pipeline): the rate-of-brain-aging → conversion finding (E8); brain age elevated in patients (E1); gap predicts conversion in CN/SCD.
- **Genuinely under-explored / our contribution**:
  1. The **fair-baseline audit** (E3/E4): systematically showing the brain-age transform doesn't beat volumes+age across tasks/diseases, incl. the "gap smuggles age" critique.
  2. The **regional RATE profile** (E9): spatial × temporal, beats scalar rate, with interpretable region ranking.
  3. The **AD vs autism geometry** difference (E4): severity axis exists in AD, absent in autism.
  4. Two-cohort replication of the combined-score gain (E10).

---

### E11 — Differential diagnosis (disease A vs disease B) ★ new direction
`exp11_differential.py` → `exp11_differential.npz`
Added dataset: **ADHD-200** (659 subj, 232 ADHD / 399 controls, 32 ROIs) harvested from
open fcp-indi S3 via `data/fetch_adhd200_freesurfer.py`. Gotcha: FreeSurfer 5.1 (ABIDE)
writes `IntraCranialVol`, 5.3 (ADHD-200) writes `EstimatedTotalIntraCranialVol` — must
coalesce or the cohort silently collapses to n≈2.

**Age-matched** (greedy 1:1, tol 1y) so the answer can't just be chronological age.

| comparison | profile | shape | volumes+age | age only |
|---|---|---|---|---|
| autism vs ADHD (children, n=392, 28 ROIs) | 0.925 | 0.909 | **0.938** | 0.530 |
| Alzheimer's vs MCI (elderly, n=714, 6 ROIs) | 0.705 | 0.668 | **0.733** | 0.466 |

- Age-only ≈ chance in both → age matching worked, no age confound.
- Profile shape alone still separates → spatial pattern carries disease-specific info.
- **AD vs MCI at 0.705 is a genuine result**: same cohort, same scanner protocol, so no
  site confound. Differential diagnosis along a severity continuum works.

⚠️ **CRITICAL CONTROL — the autism-vs-ADHD result is CONFOUNDED, do not report as-is.**
Ran the obvious check:
- within ABIDE, autism vs its *own* controls: AUC **0.579**
- within ADHD-200, ADHD vs its *own* controls: AUC **0.678**
- **healthy ABIDE vs healthy ADHD-200 controls: AUC 0.942**

The cross-cohort 0.925 is essentially reading *scanner/site*, not disease. Real disease
effects in these cohorts are 0.58–0.68. Any cross-cohort differential-diagnosis claim
needs site harmonisation (ComBat) or within-cohort designs. **Only AD-vs-MCI (within-ADNI)
is trustworthy.**

### E12 — ComBat site harmonisation: does autism-vs-ADHD survive? ★ the honest correction
`exp12_combat.py` → `exp12_combat.npz`, `exp12_harmonised.csv`
Harmonised across **27 acquisition sites** (20 ABIDE + 7 ADHD-200), covariates age+sex.
Design note: harmonise by **SITE, never by COHORT** — cohort is perfectly confounded with
diagnosis here, so ComBat on cohort would delete the disease signal itself.

| | RAW | ComBat |
|---|---|---|
| **healthy-vs-healthy (pure site confound)** | **0.942** | **0.702** |
| autism vs ADHD — volumes+age | 0.937 | 0.637 |
| autism vs ADHD — regional profile | 0.936 | 0.624 |
| autism vs ADHD — profile shape | 0.933 | 0.632 |

**Verdict: the spectacular 0.93 was ~85% scanner.** After harmonisation everything drops to
0.62–0.64, and residual site signal is still 0.702 — so even the harmonised cross-cohort
number is NOT clean. Cross-cohort differential diagnosis on these two datasets is not
salvageable without same-scanner data. Only **within-cohort** and **within-ADNI (AD vs MCI,
0.705)** comparisons are trustworthy.

### E13 — Does giving the profile MORE regions close the gap? (harmonised, within-cohort)
`exp13_nregions.py` → `exp13_autism.csv`, `exp13_adhd.csv`
Swept R = 2,4,8,16,28 regions (ranked by |corr with age| in controls).

| R | autism: profile − baseline | ADHD: profile − baseline |
|---|---|---|
| 2 | −0.092 | −0.038 |
| 4 | −0.007 | −0.031 |
| 8 | +0.001 | −0.026 |
| 16 | +0.000 | −0.019 |
| 28 | **+0.009** | **−0.020** |

- The 6-region objection is answered: more regions **does** close the gap (autism −0.092 → +0.009).
- But it converges **to parity, not superiority**. At best +0.009; ADHD stays negative throughout.
- Scalar gap stays near chance for autism (0.49–0.51) while the profile reaches 0.577 —
  **compression remains the biggest single loss**, consistent with E3/E4/E9.
- Real within-cohort disease effects are modest: autism ~0.58, ADHD ~0.63–0.67.

**Consolidated conclusion across E3, E4, E9, E11, E13:** the brain-age transform converges to
its own inputs but does not exceed them, at any region count, in any disease, on any task.
Its value is elsewhere — longitudinal RATE (E8/E9), risk stratification in the healthy (E6).

### E14 — Three-way CN → MCI → AD within ADNI ★ the clean differential test
`exp14_threeway.py`, `fig12_threeway.py` → `fig12_threeway.png`, `exp14_threeway.npz`
One cohort, one protocol → **no site confound**. Groups matched on age AND sex
(234 triplets; age ANOVA p=1.000). Age model trained on 484 held-out CN.

3-way balanced accuracy (chance 0.333): scalar 0.530 | profile 0.576 | shape 0.552 | volumes+age 0.591

| adjacent pair | scalar gap | regional profile | volumes+age |
|---|---|---|---|
| CN vs MCI (early detection) | 0.694 | **0.734** | 0.735 |
| MCI vs AD (staging) | 0.707 | 0.744 | **0.760** |
| CN vs AD (easy reference) | 0.869 | 0.921 | **0.936** |

- **CN vs MCI is the one task where the profile essentially TIES the fair baseline
  (0.734 vs 0.735).** Early detection is where brain age is least redundant.
- Profile > scalar everywhere (+0.03 to +0.05), consistent with all prior experiments.

**★ NEW FINDING — the profile shape rotates early, then only amplifies:**
mean regional gap magnitude: CN 0.98 → MCI 4.07 → AD 7.84 (8× growth)
angle between mean profiles: **CN→MCI 31.7°, MCI→AD 3.9°, CN→AD 34.5°**

So the *spatial pattern* of accelerated ageing is established almost entirely in the
CN→MCI transition; from MCI to AD the same pattern just scales up ~2×. Hippocampus leads
at every stage (+0.2 → +2.6 → +5.2 years) and its dominance grows.
Interpretation: the profile carries **stage-onset information early**, but becomes
redundant with a single severity number later — which explains why scalar gap and profile
converge on MCI-vs-AD but separate on CN-vs-MCI.

### E15 — Does the profile ADD to volumes? (correcting an earlier overstatement)
`exp15_addvalue.py`
Earlier I said "brain age never beats volumes+age". True but INCOMPLETE — it asked whether
brain age *replaces* volumetry. The user asked whether it *adds*. It does:

| task | volumes only | volumes+age | volumes+age+profile | gain from profile |
|---|---|---|---|---|
| AD vs CN | 0.9377 | 0.9426 | **0.9508** | **+0.0082** (p<0.001) |
| CN vs MCI | 0.7332 | 0.7467 | **0.7544** | **+0.0076** (p<0.001) |
| MCI→dementia | 0.7507 | 0.7520 | **0.7560** | **+0.0040** (p<0.001) |

Also: on CN vs MCI the profile ALONE (0.7510) beats volumes+age (0.7467) — the only task
where brain age wins unaided. **Correct framing: brain age is a COMPLEMENT, not a replacement.**

### E16 — WHICH regions carry the signal? (user's idea) ★ strong new result
`exp16_which_regions.py` → `exp16_rates.csv`
Landmark design, n=1100, 160 conversions. Per-region ageing RATE.

Solo C-index: **MidTemp 0.765 > Ventricles 0.746 > Hippocampus 0.741** > WholeBrain 0.710
> Fusiform 0.690 > Entorhinal 0.675

All-regions Cox (independent contribution):
Hippocampus HR=1.47 p=8e-08 *** | Ventricles HR=1.40 p=2e-07 *** | MidTemp HR=1.26 p=0.003 **
| Entorhinal HR=1.15 p=0.04 * | **Fusiform HR=0.97 p=0.66 ns** | **WholeBrain HR=1.02 p=0.78 ns**

Greedy forward selection: MidTemp (0.765) → +Hippocampus (0.790) → +Ventricles (0.801) →
then everything else adds ≤0.002 or HURTS.
**2 regions = 0.790 = 98.8% of the full 6-region model.**
→ Confirms the user's hypothesis: ageing in *specific* regions carries the Alzheimer's signal;
ageing of fusiform and whole-brain is essentially IRRELEVANT to conversion.

### E17 — Does the "rotation" finding replicate? ⚠️ ARTIFACT FOUND AND CORRECTED
`exp17_rotation_replicate.py`
Ran bootstrap CIs, permutation null, split-half, alternative staging (CDR-SB), and OASIS-1.
The direction (angle CN→MCI ≫ angle MCI→AD) replicated everywhere: ratios 18.7×, 21.3×,
33.9×, 41.3×, 59.3×; permutation null 3.3° vs observed 117°.

**BUT the CN→MCI angle is an ARTIFACT and must NOT be reported as "31.7°".**
Reason: the age model is trained on CN, so the CN mean profile is ≈0 by construction
(|m|=0.26 vs MCI 4.35, AD 9.68). The *direction* of a near-zero vector is pure noise, so the
CN→MCI angle is unstable — bootstrap CI [45.9°, 150.8°], and it swung 70.6° vs 156.1° between
split halves. The earlier 31.7° figure was one draw from that noise.

**Corrected, stable version of the same idea** — per-subject alignment with the AD direction
(cosine similarity to the mean AD profile):
CN **−0.071** → MCI **+0.436** → AD **+0.794**  (CN vs MCI p=2e-46; MCI vs AD p=3e-28)
Magnitude: 0.26 → 4.35 → 9.68.
→ The real, defensible statement: **the AD-specific spatial pattern emerges progressively and
is already ~55% established at MCI, while magnitude keeps growing.** The MCI→AD angle of
3.5–3.9° IS stable (bootstrap [2.0°, 6.9°]) — so late-stage change really is near-pure scaling.

### E18/E19 — Temporal ORDER and network COUPLING between regions ★ user's idea
`exp18_ordering.py`, `exp19_network.py` → `exp18_ordering.npz`, `exp19_autism_regions.csv`
Question: not "which region matters" but "do regions age in a SEQUENCE, and do they
influence each other?" Method: per-visit regional gap trajectories; abnormality = crossing
the 95th pct of stable subjects; converters n≈170, stable n≈930.

**Q1 ORDERING — partly real, mostly not.**
Median crossing time relative to conversion was nearly identical across regions
(Entorhinal −1.03y, Hippocampus −1.00, MidTemp −1.00, WholeBrain −0.99, Fusiform −0.98,
Ventricles −0.64) — Mann-Whitney first-vs-last p=0.07, i.e. NOT significant.
Within-subject pairwise binomial tests found only ONE robust fact:
**everything precedes the Ventricles** — Hipp 0.74 (p=0.002), MidT 0.72 (p=0.005),
Fusi 0.74 (p=0.011), Ento 0.69 (p=0.024). All other 11 pairs ns (P≈0.49–0.57).
Bootstrap ranking (500 resamples) is stable at the ends, ambiguous in the middle:
Ento [1,2] ≈ Hipp [1,2] → MidT [3,4] ≈ Fusi [3,5] → Whole [4,5] → **Vent [6,6] always last**.
→ **Defensible claim: ventricular enlargement is a LATE event; medial-temporal ageing is
early. The fine-grained order among temporal regions is NOT resolvable at 6 ROIs.**

**Q2 COUPLING — converters are more synchronised, but weakly.**
mean |r| between regional ageing rates: converters 0.353 vs stable 0.311, paired
Fisher-z p=0.15 (ns overall). Largest specific differences: MidT–Vent (0.51 vs 0.23),
Fusi–MidT (0.58 vs 0.38), Fusi–Vent (0.41 vs 0.24).

**Q3 LAGGED DIRECTION — negative result.** No lagged coupling survived: best was
Hipp→Ento r=+0.075, p=0.09 ns. ADNI's ~1-2y scan spacing is too coarse for this.

**C. PARTIAL correlations** (global ageing factor removed) show a temporal-lobe hub:
MidT–Whole +0.34, Fusi–MidT +0.32, MidT–Vent +0.27, Hipp–Whole +0.25.
→ MidTemp is the most connected node in the ageing network.

**B. AUTISM regional signature (28 ROIs, site-harmonised) — striking contrast with AD.**
Best single region for autism = Right-Lateral-Ventricle **AUC 0.564**; nothing exceeds 0.57.
Compare AD, where MidTemp alone reaches **0.765**.
→ **AD ageing is FOCAL (2 regions = 98.8% of the signal); autism ageing is DIFFUSE
(no region carries meaningful signal alone).** This is a clean, defensible qualitative
difference between a neurodegenerative and a neurodevelopmental condition, and it echoes
the E4 geometry finding (severity axis present in AD, absent in autism).

### E20 — Is "FOCAL vs DIFFUSE" a general rule? ⚠️ HYPOTHESIS NOT SUPPORTED
`exp20_focal_diffuse.py` → `exp20_focal_diffuse.npz`
Tested the E19 idea (neurodegenerative = focal, neurodevelopmental = diffuse) by adding
ADHD as a second neurodevelopmental condition. Used a scale-free metric (Gini of per-region
effect sizes) plus "how many regions to reach 90% of achievable signal", specifically so
that "diffuse" could not be confused with "weak".

| condition | best solo | all regions | Gini | k for 90% | % of regions |
|---|---|---|---|---|---|
| Alzheimer's (6 ROIs) | 0.923 | 0.933 | 0.114 | 1 | 17% |
| Autism (28 ROIs) | 0.540 | 0.561 | 0.099 | 6 | 21% |
| ADHD (28 ROIs) | 0.566 | 0.510 | 0.184 | 1 | 4% |

**The rule does NOT hold.** Gini is essentially the same for AD (0.114) and autism (0.099),
and ADHD — the second neurodevelopmental condition — is the MOST concentrated (0.184), the
opposite of the prediction. AD needing only 1 region for 90% is real, but by the scale-free
measures AD is not more focal than the developmental conditions.

**What the earlier E19 contrast actually was:** AD has a much STRONGER signal (best solo
0.923 vs 0.54–0.57), not a more CONCENTRATED one. The "focal vs diffuse" story confused
effect size with spatial concentration. Retracted.

⚠️ Also caught a broken statistic: ADHD showed "693% of total signal" for k=1, which is
impossible. Cause: the denominator (all-28-region AUC = 0.510) is *lower* than the best
single region (0.566) because with 28 noisy features and only 233 patients the full model
overfits. Verified by regularisation sweep: C=0.01 → 0.539, C=0.1 → 0.526, C=1.0 → 0.513.
So for ADHD the "total achievable signal" denominator is meaningless and any percentage
built on it is invalid. Autism does not have this problem (28-region 0.561 ≥ best solo 0.540).

**Lesson worth keeping:** with many regions and few patients, "all regions" is not an upper
bound on achievable signal, so normalising by it is unsafe.

## Open next directions (see conversation for the chosen one)
- Scale regional rate to full ~40–100 FreeSurfer regions (does the gain grow?).
- Uncertainty / calibrated single-subject prediction (conformal).
- Disease-specificity: can the rate PROFILE discriminate disease A vs disease B (not just vs control)?
- Personalized "brain-age profile card" per patient (which regions old + aging fast).

---

# PART II — REAL IMAGE DATA (the report's stated main limitation)

Every experiment E1–E20 used pre-computed volumetric features, never the image itself.
Two image datasets were downloaded to close that gap:

| Dataset | what | n | notes |
|---|---|---|---|
| OASIS-1 JPEG slices (`ninadaithal/imagesoasis`) | 86,437 axial slices, 61 levels × 3–4 repeat acquisitions | 347 subj | **not** registered, **not** skull-stripped |
| ADNI tensors (`fabriziopacheco/adni-mri-tensors`) | 9,072 volumes, 128³ float32 | 1,246 subj | registered, skull-stripped |

Join to demographics is exact: 86,437/86,437 OASIS slices matched, and the folder
labels reproduce CDR perfectly (147 missing-CDR subjects are all young, all "Non Demented").

### E21/E22 — OASIS 2D slices, linear recipe  ✗ the Franke recipe FAILS here
`exp21_prepare_images.py`, `exp22a_cache_oasis.py`, `exp22_image_brainage.py`, `exp22b_diagnose.py`

| input | age MAE (60+) | disease AUC (gap) | direct (+age) |
|---|---|---|---|
| TAB nWBV+eTIV+sex (3 numbers) | **11.0 y** | **0.729** | 0.712 |
| IMG 41,600 px → 80 PCA | 15.8 y | 0.432 | 0.526 |
| BOTH | 15.4 y | 0.433 | 0.649 |

**Diagnosed, not just reported.** Two separate bugs were found and fixed before believing
the result:

1. ⚠️ **Intensity normalisation cancelled the signal.** Dividing each acquisition by its
   foreground *mean* is the obvious choice and is wrong: atrophy replaces bright tissue with
   dark CSF, so that mean is itself a strong predictor (**AUC 0.398**). It drove a hand-made
   CSF feature from **0.607 → 0.523**. Measured every candidate: p50 0.410, p75 0.397,
   p90 0.484, **p95 0.512**, p98 0.515, mean 0.398 → use **p95** (tracks white matter,
   diagnosis-neutral). Fixing it moved IMG+age 0.499 → 0.526 — real, but not the main problem.
2. **The main problem is missing registration.** PC2–PC5 correlate |r| = 0.42–0.58
   with head centroid / head area, against a maximum of 0.36 with age across the first
   ten components. PCA spends its capacity
   describing where the head sits in the frame. Crude centroid+scale alignment only moved
   AUC 0.407 → 0.536.

**The headline:** one hand-made feature (dark-CSF fraction) = **0.607**, while 80 components
over 41,600 pixels = **0.526**. The information IS in the image; the representation throws it away.

### E22-GPU — OASIS convnet (Kaggle T4)  → better, still loses to 3 numbers
`kaggle/brainage_oasis_cnn/` — 3D-ish CNN over 16 slices as channels, 5-fold × 3 seeds, subject-level.

| arm | AUC |
|---|---|
| CNN brain-age gap | 0.637 |
| CNN features + age (fair baseline) | 0.570 |
| CNN trained directly on disease | 0.513 |
| **nWBV + eTIV + sex + age** | **0.711** |

Convolution fixes the translation problem (0.53 → 0.64) but three tabular numbers still win.
Age MAE (60+) 11.7 y vs 11.0 y tabular — a dead heat.

★ **Unplanned finding — age pretraining as a rescue for small samples.** The age-derived gap
(0.637) beats a CNN trained directly on the disease label (0.513). With 163 subjects there is
not enough label to train a classifier, but age is free and abundant. This is direct support
for the pretrain-on-age → fine-tune-on-disease idea.

### E23 — ADNI 3D volumes, linear recipe (n=549 AD/CN, n=774 MCI/CN)
`exp23a_stream_adni.py` (streams from the 5.9 GB zip; it expands to 33.6 GB, which we do not
have — block-mean 128³→32³, output 558 MB), `exp23_adni_image.py`

Registration verified: bounding boxes agree to 0.6–2.3 voxels on the 32 grid (vs OASIS, where
head position dominated). So here the recipe gets a fair hearing.

**AD vs CN** (age MAE / r on healthy test subjects, before bias correction)

| input | age MAE | r | gap AUC | direct +age |
|---|---|---|---|---|
| TAB volumes+ICV+sex | **4.3 y** | 0.57 | **0.923** | **0.952** |
| IMG voxel PCA | 4.9 y | 0.38 | 0.627 | 0.648 |
| BOTH | 4.5 y | 0.53 | 0.873 | 0.931 |

**MCI vs CN**: TAB gap 0.764 / direct 0.785; IMG gap 0.587 / direct 0.613; BOTH 0.711 / 0.739.

- **Does the image ADD to the volumes? No.** −0.021 (AD), −0.046 (MCI), both p<0.001.
- **Gap loses to the direct model again**, now at n=549 and n=774: −0.029 (AD), −0.021 (MCI).
  This replicates E1/E3 and matches Tan 2025 on a much larger sample.

⚠️ **Two bugs caught in my own code here — both would have produced publishable-looking nonsense:**

1. **"age MAE 568 years."** 63.7% of the 32³ grid is identically zero (background) and among
   the rest the smallest SD is 3e-9, so `StandardScaler` multiplied dead voxels by up to
   **3.4×10⁸** and PCA fitted amplified noise. Images are not tabular features — they already
   share one intensity scale, so PCA should *centre* and nothing more. Fix: variance mask
   (keeps 8,179 of 32,768 voxels = 25%) + no per-voxel scaling.
2. **A metric that flatters useless models.** I first reported "age MAE" as the *bias-corrected*
   |gap|, which made IMG look BETTER than TAB (2.0 y vs 2.7 y). It is an artifact: a model with
   no signal predicts everyone at the mean age, the bias correction regresses that away, and its
   |gap| collapses toward zero. Reported uncorrected, the truth reverses: IMG 4.9 y vs TAB 4.3 y.
   **The corrected gap must never be quoted as age accuracy.**

### Longitudinal capacity unlocked
`adni_vox32.npz` holds 4,261 volumes from 1,246 subjects; **813 have ≥2 visits** (up to 9),
which enables an image-based *rate* of brain ageing — the image analogue of E8/E9, the
strongest result in the report (+0.048 C-index, replicated on OASIS at +0.047).

### E22-Q3 / test-retest — the gap's noise floor  ★ novel, and it replicates
Each OASIS subject has 3–4 acquisitions **from the same session** — same brain, minutes apart,
so any variation in their gap is measurement noise rather than biology.

| model | within-subject SD | between-subject SD | reliability |
|---|---|---|---|
| linear (PCA) | 1.89 y | 10.95 y | 0.993 |
| convnet (T4) | 1.98 y | 12.54 y | 0.994 |

Two very different models agree: **rescanning the same person minutes later moves their brain-age
gap by ~1.9 years.** That is a concrete floor on how small an individual gap change can be
believed — directly relevant to E8's "rate of ageing", where the effect being measured is of
comparable size.

## What Part II changes about the report's conclusions
- The main stated limitation ("we never used the images") is now **closed on two cohorts**.
- The answer is negative and consistent: **at this resolution and sample size the raw image does
  not beat, and does not add to, the volumetric features** — which strengthens rather than
  weakens the report's existing argument that simple volumes are a hard baseline.
- The "gap loses to the direct model" finding now replicates at n=549 / n=774 with images.
- **Caveat, stated plainly:** this is 32³/64³ resolution, ≤1,000 subjects, and shallow models.
  Published image models that beat volumetrics use ~10⁴–10⁵ subjects. The honest claim is
  "not at this scale", not "images are useless".

### E24 — Image-based RATE of brain ageing, and a re-validation of E8  ★ the important one
`exp24_image_rate.py`, `exp24b_verify.py`, `exp8b_rate_validated.py`

813 cached ADNI subjects have repeat scans, so E8's question could finally be asked of the
image. Landmark design: rate estimated only from scans in [0, L], conversion counted only
after L, censored at last known DX. **Age at a follow-up visit must be `AGE + Years_bl`** —
ADNIMERGE's `AGE` is baseline age and is constant within subject, so using it at every visit
would make every gap grow 1 year per year of follow-up and manufacture a fake "rate".

**Image-matched cohort (n=207, 127 converters), L=2.0:**

| arm | C-index |
|---|---|
| gap value (tabular) | 0.629 |
| **gap RATE (tabular)** | **0.683** |
| hippocampus RATE | 0.638 |
| gap value (image) | 0.518 |
| gap RATE (image) | 0.509 |

Landmark sweep (rate − value): −0.003 @L=1.0, +0.019 @1.5, +0.055 @2.0, +0.038 @2.5,
+0.077 @3.0 → the rate needs a long enough window to be estimable; at L=1 it is pure noise.
**The image rate carries no signal at all** (0.509), consistent with E23.

#### ⚠️ Method audit of E8 — and the verdict
Testing the image version exposed that `exp8_rate.py` **fits the Cox model and computes the
C-index on the same cohort** — in-sample, no CV, no CI. Re-ran properly on the full tabular
cohort (**n=562, 238 conversions**), out-of-fold + 500-resample subject bootstrap:

| test | estimate | 95% CI | verdict |
|---|---|---|---|
| rate **beats** value | +0.001 | [−0.038, +0.044] | crosses zero |
| **rate ADDS to value** (= E8's actual claim) | **+0.052** | [+0.029, +0.077] | **holds** |
| rate vs hippocampus RATE (fair baseline) | +0.046 | [+0.020, +0.071] | **holds** |
| rate vs hippocampus VALUE (toughest baseline) | +0.021 | [−0.021, +0.059] | crosses zero |
| permutation (rate alone) | 0.730 vs null 0.499 | — | p < 0.0001 |

**E8 SURVIVES.** Its +0.048 replicates as +0.052 with a CI excluding zero. Two honest
qualifications, though:

1. **The missing CV mattered less than feared.** Measured optimism is only +0.002 to +0.004
   across every arm — a Cox model with three covariates barely overfits. The flaw was real,
   the consequence was not. (Worth remembering before assuming in-sample ⇒ inflated.)
2. **My own first reading was wrong.** exp24 tested "rate beats value" and found it crossed
   zero, which I initially framed as E8 failing. E8 never claimed that — it claimed the rate
   *adds to* the snapshot. Comparing the wrong pair of arms nearly produced a false retraction.
3. ★ **New and genuinely limiting:** the brain-age rate beats the hippocampal-atrophy *rate*
   (+0.046, holds) but does **not** clearly beat simply measuring the hippocampus *once*
   (+0.021, CI crosses zero). So the brain-age machinery earns its keep against the
   rate-based baseline, not against the simplest snapshot baseline. This is the sharpest
   fair-baseline caveat in the project so far and belongs in the report.

Converters age +0.76 gap-years/year vs +0.14 for non-converters (n=562).

**Lesson worth keeping:** confidence intervals from repeated cross-validation measure split
noise, not sampling noise. At n=207 the CV-repeat interval on rate−value was ±0.006 while
the subject bootstrap was ±0.07 — an order of magnitude apart. Always bootstrap over subjects.

### E26 — ADNI 3D convnet  ✗ INCONCLUSIVE (sample-size failure, not an image result)
`kaggle/brainage_adni_cnn/` — 3D CNN, 64³, 5-fold × 3 seeds, T4, ~2.5 h

| arm | AD vs CN | MCI vs CN |
|---|---|---|
| tabular gap | 0.926 | 0.762 |
| CNN gap | 0.491 | 0.481 |
| CNN features + age | 0.519 | 0.510 |
| CNN direct | 0.507 | 0.529 |
| CNN age-pretrained → fine-tuned | 0.507 | 0.518 |
| volumes+ICV+sex+age | **0.953** | **0.786** |

**Every CNN arm is at chance — including below the linear voxel PCA (0.627/0.587) on the same
data.** A convnet losing to PCA on identical input means it did not train. Cause: the age model
gets ~313 CN baseline volumes per fold, and a 3D CNN from scratch cannot learn from that. The
OASIS 2D net did learn because it had ~850 training samples (266 subjects × ~4 acquisitions)
and far fewer parameters.

⚠️ **Do NOT file this as evidence that images carry no signal.** It is a statement about n and
architecture. The pretraining arm is likewise uninterpretable here (+0.000) — you cannot measure
a transfer benefit from a source model that never learned. (The OASIS evidence for pretraining,
gap 0.637 vs direct 0.513, stands and is unaffected.)

### E27 — THE ANATOMICAL BOTTLENECK: image → volumes → age → disease  ★★ the best image result
`kaggle/brainage_bottleneck/`, `exp25_rerun_predicted.py`

Supervisor's-eye summary: rather than making a network discover a good representation from
~1,000 subjects, fine-tune a pretrained ResNet-18 to output the 7 FreeSurfer numbers, then run
those predicted numbers through the identical pipeline every earlier experiment used. Trained
out-of-fold by subject on all visits of training subjects (3,286 scans, 1,132 subjects).

**Stage A — can the image reproduce the FreeSurfer numbers?**

| target | r | | target | r |
|---|---|---|---|---|
| Ventricles | **0.920** | | WholeBrain | 0.617 |
| ICV | 0.644 | | Hippocampus | 0.584 |
| MidTemp | 0.564 | | Fusiform | 0.524 |
| | | | **Entorhinal** | **0.443** |

Large high-contrast structures are learned well; the small medial-temporal ones that carry the
AD signature are learned worst — which is exactly where the downstream loss comes from.

**Stage B/C — the full suite rerun, identical folds, true vs predicted**

| task | true volumes | predicted | voxel PCA | 3D CNN |
|---|---|---|---|---|
| AD vs CN (n=621) | 0.953 | **0.824** | 0.648 | 0.507 |
| MCI vs CN (n=880) | 0.791 | **0.686** | 0.613 | 0.529 |
| AD vs MCI, age-matched (n=356) | 0.751 | **0.649** | — | — |

**Recovery of ABOVE-CHANCE signal** (see the metric warning below):

| task | voxel PCA | bottleneck |
|---|---|---|
| AD vs CN | 32.7% | **71.5%** |
| MCI vs CN | 38.8% | **63.9%** |
| AD vs MCI | — | 59.4% |

**The bottleneck more than doubles what the image recovers**, and recovery degrades as the task
gets harder (71.5 → 63.9 → 59.4%).

**T4, the longitudinal arm — the biggest change.** Voxel PCA's image-derived rate was pure
chance (C = 0.509). Through the bottleneck it works:

| arm | true | predicted | recovery |
|---|---|---|---|
| gap value | 0.639 | 0.576 | 54.7% |
| gap RATE | 0.681 | **0.604** | 57.5% |
| value + RATE | 0.712 | 0.636 | 64.2% |
| hippocampus RATE | 0.644 | 0.620 | 83.3% |

**The rate-beats-value pattern survives the trip through the image** (+0.028 predicted vs +0.042
true) — a qualitative replication of the report's headline finding, now from pixels.

⚠️ **Metric trap caught here.** "Recovered" was first computed as `AUC_pred / AUC_true`, which
gave 86.4 / 86.7 / 86.4% — suspiciously identical across three tasks of very different
difficulty. AUC has a floor of 0.5, so that ratio is inflated and compresses real differences.
On the above-chance scale the same numbers are 71.5 / 63.9 / 59.4%, revealing a trend the naive
ratio hid entirely. **Never quote a ratio of AUCs as "percent recovered".**

## What E27 changes about the report's conclusions
Part II's negative result needs restating. It is NOT "images do not help at this scale". It is:

> **Raw images do not help at this scale; images supervised through an anatomical bottleneck
> recover most of the signal.** The limitation was the representation, not the information —
> which was always the logically necessary answer, since the volumes are computed FROM the image.

The bottleneck still does not beat true FreeSurfer volumes, and should not be expected to: it is
imitating them, so it is capped by Stage A accuracy. Its practical value is different — it needs
no FreeSurfer run (hours per scan) and degrades gracefully, which matters for deployment, not
for accuracy.

### E28/E29 — FROZEN pretrained backbones (ViT-S, ResNet-50, ConvNeXt-T), ADNI n=1064
`exp28_frozen_features.py`, `exp29_frozen_analysis.py` — no GPU; forward passes only, on CPU.

Run because the Kaggle account lost GPU **and** internet (both gated behind phone
verification), so fine-tuning was unavailable. Frozen features need neither. Real ImageNet
weights, downloaded locally and verified loaded.

**Stage A — features → 7 FreeSurfer volumes (mean r over targets):**

| representation | mean r | Ventricles | Hippocampus | Entorhinal |
|---|---|---|---|---|
| voxel PCA (no pretraining) | 0.378 | 0.776 | 0.300 | 0.246 |
| frozen ResNet-50 | 0.379 | 0.576 | 0.275 | 0.223 |
| frozen ViT-S/16 | 0.452 | 0.784 | 0.402 | 0.286 |
| frozen ConvNeXt-T | **0.520** | 0.782 | 0.442 | 0.363 |
| **FINE-TUNED ResNet-18 (E27)** | **0.614** | **0.920** | **0.584** | **0.443** |

**Age on CN:** ViT-S r 0.461 / MAE 4.51y vs voxel PCA r 0.284 / 4.90y.

**Disease (AD vs CN, n=621):** best frozen route 0.732 (ConvNeXt path A), fine-tuned
bottleneck 0.824, ceiling 0.953. MCI vs CN: best frozen 0.665, ceiling 0.791.

★ **Two findings that point opposite ways, and both matter:**

1. **Frozen pretrained features beat voxel PCA at reading ANATOMY** — mean r 0.520 vs 0.378
   (+0.142) for volumes, r 0.461 vs 0.284 for age. So ImageNet features do transfer to brain
   MRI structure.
2. **But they do NOT beat voxel PCA at DISEASE** — −0.017 to −0.052 across every backbone and
   both tasks. Better anatomy reading did not become better diagnosis.

★ **Fine-tuning is where the value is, not the pretrained features per se:**

| | Stage A mean r | AD vs CN | % of ceiling (above chance) |
|---|---|---|---|
| frozen best | 0.520 | 0.732 | 51% |
| fine-tuned | 0.614 (+0.094) | 0.824 (+0.092) | 72% |

This supports the supervisor's position that pretrained weights must be FINE-TUNED, and
quantifies it: frozen features capture roughly half of what fine-tuning reaches.

⚠️ **Caveats, both real:**
- The fine-tuned comparator trained on 3,286 scans (all visits); the frozen arms use 1,064
  baselines. Some of the +0.094 is sample size, not fine-tuning.
- The "pretraining vs none" comparison is NOT perfectly controlled: voxel PCA sees 32³
  whole-volume, the backbones see 6 slices at 224². Different views of the same scan, so
  resolution and coverage are confounded with pretraining. The Stage A advantage is large
  enough to survive this; the disease deficit is not clearly interpretable.

### ⚠ E-VOID — Kaggle screening kernel (`mohammadparsakavian/brainage-backbones`)
Cancelled after 20 h. `device: cpu` (GPU requested, not granted) and DNS failure to
huggingface.co (internet requested, not granted) — the new account is not phone-verified.
Every arm ran with RANDOM weights. **Numbers discarded, not reported.** The explicit
weight-load check added to this kernel is what caught it; the earlier E27 bottleneck run
produced an empty log and its pretrained status was never confirmed.

### E28/E29 — pretrained backbones: frozen vs fine-tuned, at matched sample size
`pilot/exp28_frozen_features.py`, `pilot/exp29_frozen_analysis.py`,
`kaggle/brainage_backbones/` (Tesla T4, verified GPU + verified weight download)

Supervisor framing: the point of working from pixels is NOT to beat the volumes.
The volumes are computed FROM the image, so that contest is meaningless. In deployment
there is no FreeSurfer table, only a scan — so everything must be reachable from pixels.
True volumes are therefore demoted from competitor to CEILING.

**Stage A (image → 7 FreeSurfer numbers), all trained on the SAME 3,286 scans, out-of-fold:**

| representation | mean r |
|---|---|
| voxel PCA (no pretraining) | 0.317 |
| frozen ResNet-50 | 0.428 |
| frozen ViT-S | 0.476 |
| frozen ConvNeXt-T | 0.507 |
| **FINE-TUNED ResNet-18 (E27)** | **0.614** |

- pretraining alone (frozen): **+0.190**
- fine-tuning on top: **+0.107**

**Fine-tuned screening on GPU (1 fold, ~2,400 scans, 7 arms):**

| arm | weights | mean r | age MAE |
|---|---|---|---|
| ResNet-50 | pretrained | 0.504 | 4.37 y |
| ViT-S/16 | pretrained | 0.462 | 4.24 y |
| ConvNeXt-T | pretrained | 0.454 | 4.32 y |
| ViT-B/16 | pretrained | 0.426 | 4.43 y |
| ViT-S frozen | pretrained | 0.379 | 4.30 y |
| ResNet-50 scratch | random | 0.377 | 4.49 y |
| DINOv2-S | pretrained | 0.339 | 4.78 y |

- **pretraining control: +0.127 mean r** (ResNet-50 vs identical arch from scratch)
- **fine-tuning control: +0.083 mean r** (ViT-S fine-tuned vs frozen)
- scratch (0.377) ≈ frozen (0.379): pretrained weights you cannot adapt are worth
  about as much as no pretrained weights
- ResNet-50 beats every ViT; ViT-B is WORSE than ViT-S (capacity hurts at this n)
- DINOv2 worst, but `img_size=224` was forced onto a patch-14 model, resampling its
  position embeddings — treat as NOT PROPERLY TESTED, not as a verdict

**AD vs CN, bottleneck path, above-chance recovery:** voxel PCA 40.0% →
frozen ViT-S 47.8% → fine-tuned 71.5%.

⚠️ **CORRECTION to the n=1,064 report.** At baseline-only sample size, frozen pretrained
features beat voxel PCA by only +0.008 (51.2% vs 49.4%), and I reported that "pretraining
alone buys almost nothing". At n=3,286 the gap is +0.035 on AD and +0.190 on Stage A.
**That earlier conclusion was sample-size limited and is withdrawn.** Both pretraining AND
fine-tuning matter; pretraining's value grows with data rather than shrinking.

⚠️ Comparability caveat: E27's downstream was baseline-only (n=621, ceiling 0.953) while
E28/E29's is scan-level (n=1,883, ceiling 0.950). Stage A mean r IS matched (both 3,286
scans, same targets, out-of-fold) and is the clean comparison. The 71.5% vs 47.8% figure
mixes the two designs and should be quoted as indicative, not exact.

⚠️ The +0.083 fine-tuning control slightly overstates: the frozen arm's head was 6 epochs
of AdamW, whereas closed-form ridge on the same backbone reached 0.476 rather than 0.379.
The +0.127 pretraining control has no such issue — both arms are identical but for the
initial weights.

### E30 — PHASE 2: full pixel pipeline, ResNet-50, 5-fold  ⚠️ WORSE THAN E27
`kaggle/brainage_pixels/`, `pilot/_exp30_rate.log` (Tesla T4, verified pretrained)

Four routes from pixels + the ceiling. AD vs CN, n=621 baselines (identical task,
identical n to E27, so directly comparable).

| path | route | AUC | above chance |
|---|---|---|---|
| A direct | image → volumes → +age → disease | **0.782** | 62.4% |
| A gap | image → volumes → age → gap | 0.769 | 59.5% |
| B | image → age → gap | 0.714 | 47.3% |
| D | image → disease, end-to-end | 0.707 | 45.8% |
| C | image features (2048) + age | 0.652 | 33.6% |
| CEILING | true FreeSurfer volumes | 0.952 | 100% |

**THE ORDERING IS THE RESULT**, and it validates the two-stage design:

    bottleneck 0.782  >  direct age 0.714  >  end-to-end 0.707  >  raw features 0.652

- **bottleneck beats end-to-end by +0.075.** PATH D sees ~500 binary labels; PATH A sees
  3,286 × 7 real numbers. Supervision density, not architecture.
- **bottleneck beats direct brain age by +0.068** — routing through anatomy beats
  predicting age straight from pixels.
- **PATH C is WORST.** 2,048 features at n=621 overfit badly; the 7-number bottleneck
  acts as a strong regulariser. NOTE: this is a dimensionality effect, NOT evidence that
  the gap "distills" information — the docstring's framing was wrong and is corrected here.

⚠️ **PHASE 2 UNDERPERFORMED E27 ON EVERY ARM**, despite using the screening WINNER:

| | E27 (ResNet-18) | E30 (ResNet-50) |
|---|---|---|
| Stage A mean r | **0.614** | 0.599 |
| AD vs CN (A direct) | **0.824** | 0.782 |
| MCI→dementia gap RATE | **0.604** | 0.521 (≈chance) |

Differences: E27 used 14 epochs / 12 slices / 128²; E30 used 8 epochs / 6 slices / 224².
**The 1-fold, 6-epoch screening ranked backbones while all of them were undertrained, and
that ranking did not transfer to full training.** Lesson: screen at the training budget you
intend to use, or the ranking measures who learns fastest rather than who ends up best.
Epochs and slice count mattered MORE than backbone choice here.

The rate arm is the sharpest regression: 0.521 vs E27's 0.604. A rate is a difference
between two noisy estimates, so it amplifies Stage A noise — the arm that degrades first
when the volume predictor gets weaker. Best image-derived rate remains **E27's 0.604**.

### E31 — PHASE 3: same pipeline, bigger training budget  ★ NEW BEST, hypothesis confirmed
`kaggle/brainage_phase3/` (ResNet-18, 20 epochs, 12 slices, 224², 5-fold, T4)

E30 lost to E27 while using the screening WINNER. The stated hypothesis was that the
1-fold/6-epoch screening ranked backbones while all were undertrained, so the limit was
BUDGET not architecture. This run holds everything fixed and raises only the budget.

**Confirmed.**

| | E27 (R18, 14ep, 12sl, 128²) | E30 (R50, 8ep, 6sl, 224²) | **E31 (R18, 20ep, 12sl, 224²)** |
|---|---|---|---|
| Stage A mean r | 0.614 | 0.599 | **0.682** |
| AD vs CN | 0.824 | 0.782 | **0.836** |
| MCI vs CN | 0.686 | 0.690 | **0.700** |
| MCI→dementia RATE | 0.604 | 0.521 | 0.551 |

**Stage A, per target — the weak link improved most:**

| target | E30 | E31 |
|---|---|---|
| Entorhinal | 0.407 | **0.504** |
| Hippocampus | 0.538 | **0.637** |
| MidTemp | 0.545 | **0.646** |
| WholeBrain | 0.628 | **0.726** |
| Ventricles | 0.917 | **0.951** |

AD vs CN above-chance recovery: E30 62.4% → E27 71.7% → **E31 74.3%**.

**Full path ordering at the higher budget (AD vs CN, n=621):**

    bottleneck direct 0.836 > bottleneck gap 0.791 > end-to-end 0.724
      > direct brain age 0.703 > raw features 0.693     [ceiling 0.952]

The E30 ordering is reproduced with more training, so it is a property of the design and
not of one run: routing pixels through predicted anatomy beats every alternative route.

⚠️ **The RATE arm did NOT recover.** 0.551 here vs E27's 0.604, despite E31 beating E27 on
every cross-sectional measure. So the rate is not simply a function of Stage A accuracy —
E31 has the better volume predictor and the worse rate. Most likely cause: E27 capped
training at 3 visits/subject and E31 at 4, changing which visits the slope is fitted to;
a rate is a difference of noisy estimates and is far more fragile than a cross-sectional
value. **Best image-derived rate remains E27's 0.604, and it is NOT yet explained.**
This is the open question for the next run, not a solved arm.

Methodological lesson worth keeping: **screen at the training budget you intend to use.**
An undertrained screening measures which model learns fastest, not which ends up best —
it picked ResNet-50, which lost to ResNet-18 by 0.054 AUC at full budget.

### E32 — LONGITUDINAL ATTENUATION  ★ strongest novelty candidate so far
`pilot/exp32_longitudinal_attenuation.py` (local, no GPU — uses saved per-visit predictions)

Chasing the rate puzzle. Two hypotheses tested, BOTH REJECTED, and a third finding
survived that is more interesting than either.

**Hypothesis 1 (mine): longer training makes the net lock onto subject identity,
flattening slopes.** REJECTED — E31 tracks change BETTER than E30 (sensitivity 0.245 vs
0.181, attenuation 0.424 vs 0.306), not worse.

**Hypothesis 2: rate C-index follows longitudinal sensitivity.** REJECTED —

| run | cross-sec r | sensitivity | attenuation | rate C |
|---|---|---|---|---|
| E27 R18 14ep | 0.614 | 0.184 | 0.285 | **0.604** |
| E30 R50 8ep | 0.599 | 0.181 | 0.306 | 0.521 |
| E31 R18 20ep | **0.682** | **0.245** | **0.424** | 0.551 |

corr(cross-sec r, rate C) = +0.004; corr(sensitivity, rate C) = −0.114. The run with the
BEST rate tracks change WORST of the three. (n=3 runs, so these correlations are
descriptive only — but the ordering is plainly non-monotonic.) **The rate C-index is
still unexplained**; the MAX_VISITS control is the next test and is blocked on quota.

**THE FINDING THAT STANDS — measured, robust, and directly relevant to the thesis:**

Image-derived volumes are **severely attenuated as longitudinal measures**, and the
attenuation is NOT predictable from cross-sectional accuracy:

| target | cross-sec r | % of real change reported |
|---|---|---|
| Ventricles | 0.951 | **94.0%** |
| ICV | 0.744 | 39.3% |
| WholeBrain | 0.726 | 42.5% |
| MidTemp | 0.646 | 29.2% |
| Hippocampus | 0.637 | 52.9% |
| Fusiform | 0.565 | 26.0% |
| **Entorhinal** | 0.504 | **12.8%** |

Note ICV (r 0.744) reports 39% while Hippocampus (r 0.637) reports 53% — rank order
differs from cross-sectional r, so the two are genuinely different properties.

**Why this matters.** A model that looks respectable on a single scan can be nearly blind
to within-subject change. Entorhinal cortex — the AD signature region, where Braak staging
begins — is reproduced at r=0.50 cross-sectionally but only 13% of its real change is
reported. Any rate computed from such a predictor is shrunk toward zero BEFORE any
survival model sees it, which is a sufficient explanation for why image-derived rates
have failed throughout this project (voxel PCA 0.509, E30 0.521, E31 0.551) while
image-derived cross-sectional measures reach 0.836.

**Constructive next step:** train the network on CHANGE (Δvolume between visit pairs)
rather than on absolute volume, and measure whether attenuation improves. Nothing in the
pipeline currently asks the network to be sensitive to change — it is optimised entirely
for absolute accuracy, and gets exactly what it optimises for.

⚠️ NOT yet checked against the literature. Attenuation/regression-dilution in imaging
biomarkers is a known phenomenon in general; whether this specific measurement for
CNN-predicted volumetrics is published needs a search before any novelty is claimed.

### LITERATURE CHECK on the two novelty candidates (Sept 2026)

**Candidate A — test-retest noise floor of the gap (~1.9 y): NOT NOVEL. Withdraw the claim.**
- Dular et al., *NeuroImage* (PMC10619370) compares accuracy AND test-retest reliability of
  publicly available brain-age packages — exactly this measurement, done systematically.
- Independent work reports BAG test-retest ICC 0.93 over 6 months; ours was 0.993 over minutes.
- *Nature Communications* 2026, "Precision estimates of longitudinal brain aging", uses cluster
  scanning (densely repeated rapid structural scans) to estimate measurement error directly —
  a stronger version of our design.
→ Our E22-Q3 number **replicates** known work. Report it as validation of the pipeline, NOT as
  a contribution. The FINDINGS_LOG entry labelling it "★ novel" is WRONG and is corrected here.

**Candidate B — longitudinal attenuation of regression-predicted volumes: PARTIALLY OPEN.**
- The general problem (deep models are longitudinally inconsistent) is known and actively worked:
  DeepBVC (*Front. Neurosci.* 2023) and GeoLongSeg (*NeuroImage: Clinical* 2024).
- BUT both address **segmentation** methods, and both measure **consistency** (test-retest PBVC,
  multi-step agreement, Bland-Altman, variance ratios). Verified by reading DeepBVC directly:
  it does NOT quantify an attenuation slope, and does NOT test whether cross-sectional accuracy
  predicts longitudinal sensitivity.
- Our setting differs: a **regression** model predicting scalar volumes from slices, scored by
  the slope of Δpredicted on Δtrue, plus the demonstrated **dissociation** (ICV r=0.744 → 39%
  while Hippocampus r=0.637 → 53%; rank order differs from cross-sectional r).
→ Narrow but real. Attenuation/regression-dilution is a classical statistical idea, so the
  contribution is the MEASUREMENT and the DISSOCIATION in this pipeline, not the concept.
  **Not a paradigm shift — a methodological caution with a constructive fix.**

### E34/E35 — MAX_VISITS control + SIAMESE delta model, and ⚠️ A SELF-CORRECTION THAT VOIDS THE PREMISE

**E34, the MAX_VISITS control** (`mohammadparsakavian/brainage-phase3b`, MAX_VISITS=3, sole change):

| | Stage A mean r | AD vs CN | gap RATE |
|---|---|---|---|
| E31 (MV=4) | 0.682 | 0.836 | 0.551 |
| E34 (MV=3) | 0.675 | 0.829 | 0.561 |

MAX_VISITS accounts for **+0.010** of the 0.053 gap to E27. Hypothesis REJECTED.

**E35, the siamese delta model** (`mohammadparsakavian/brainage-siamese`): shared ResNet-18
over both scans of a pair, head on [f1, f2, f2−f1], same augmentation applied to both scans.
1,068 landmark-window pairs from 237 MCI subjects.

| | sensitivity | attenuation | rate C |
|---|---|---|---|
| E31 absolute | 0.245 | **0.424** | 0.551 |
| E35 siamese | **0.301** | 0.189 | 0.579 |

Correlates BETTER with real change, scaled DOWN more. **Attenuation was the wrong yardstick
for the downstream task** — Cox ranks subjects, so the slope cancels and only the ordering
matters. Correcting that, siamese ranks change better (corr with true rate 0.688).

⚠️⚠️ **THE SELF-CORRECTION. The "rate puzzle" was never established as real.**

Bootstrap over subjects, n=203, 123 events:

    E27 rate C 0.611   E31 rate C 0.561   difference +0.049
    E27 95% CI [0.556, 0.666]  — width 0.110
    difference 95% CI [−0.019, +0.115]   p = 0.141   NOT SIGNIFICANT

**At this sample size a C-index cannot be pinned to better than about ±0.055.** Every
between-run rate difference I interpreted — 0.521 / 0.551 / 0.561 / 0.579 / 0.604 — sits
inside one noise band. I spent two GPU experiments explaining a difference that was never
shown to exist, and wrote three FINDINGS_LOG entries treating it as a phenomenon
("the rate arm did NOT recover", "the sharpest regression", "still unexplained").
Those framings are WITHDRAWN.

**The error:** I applied subject-level bootstrap rigorously to E8's claims (correctly
catching that "rate beats value" crossed zero) and then failed to apply it to my own
between-run comparisons for five consecutive experiments. Precision was reported as a
3-decimal point estimate with no interval, which invites exactly this mistake.

**What actually survives:**
1. Image-derived rates cluster at 0.52–0.60 against a true-volume ceiling of 0.67–0.68.
   That gap is large and consistent across five independent runs — it is the real finding.
2. Attenuation is measured and reproducible (E32): 94% ventricles, 53% hippocampus,
   13% entorhinal. It is a property of the representation; the siamese target changed the
   scale but not the fundamental limit.
3. Cross-sectional performance IS separable from rate performance in the sense that
   E31 clearly beats E27 cross-sectionally (0.836 vs 0.824 with a much larger Stage A
   margin) while their rates are indistinguishable. But "E27's rate is better" is not
   supported.

**RULE going forward: no between-run comparison gets reported without a subject-level
bootstrap interval. Point estimates of C-index at n≈200 are worth ±0.055 and no more.**

### E36 — ACCELERATION of the gap: measurable, but prognostically empty
`pilot/exp36_acceleration.py`  (679 subjects with >=4 visits inside a 2-year window)

Before any survival model, two checks on whether a second derivative is even
estimable given the gap's ~1.9 y noise floor:

    observed spread of acceleration   1.353 y/y^2
    null spread from noise alone      0.845 [0.799, 0.892]
    ratio                             1.60  -> 61% of its variance is real signal

So acceleration IS measurable. But it predicts nothing:

| model | C |
|---|---|
| age only | 0.569 |
| value | 0.719 |
| **rate** | **0.754** |
| **acceleration** | **0.563** |
| value + rate | **0.792** |
| value + rate + acceleration | 0.790 |

Adding acceleration: **-0.001, 95% CI [-0.006, +0.004], p=0.570.**

→ The gap trajectory in MCI is *informationally linear*: the curvature is real
and measurable, but carries no disease information. Worth reporting precisely
because the check for measurability came out POSITIVE — this is not "we could
not measure it", it is "we measured it and it is empty".

### E37 — A TEMPORAL CASCADE IN REGIONAL BRAIN-AGE GAPS  ★★ best candidate yet
`pilot/exp37_regional_cascade.py`  (379 converters, 722 stable MCI)

Visits re-expressed as years BEFORE each converter's own conversion, so subjects
at different stages align to a common clock. Per-region age models (region + ICV
+ sex, CN-trained, out of fold). Stable MCI are the reference, so normal ageing
and model bias cancel.

**Deviation of converters from stable MCI (gap-years):**

| region | -0.1 y | -1.2 | -2.2 | -3.2 | -4.4 | -6.0 |
|---|---|---|---|---|---|---|
| Hippocampus | 4.45 | 3.89 | 3.42 | 2.83 | 2.25 | 0.86 |
| MidTemp | 3.16 | 2.52 | 1.71 | 1.65 | 1.15 | 0.27 |
| WholeBrain | 3.21 | 2.56 | 1.72 | 1.70 | 0.87 | 0.72 |
| Fusiform | 2.55 | 1.96 | 1.42 | 1.44 | 0.78 | 0.30 |
| Entorhinal | 1.68 | 1.26 | 1.20 | 1.19 | 0.52 | 0.42 |
| Ventricles | 1.23 | 0.80 | 0.44 | 0.25 | 0.44 | 0.04 |

**Onset, absolute threshold (>= 1 gap-year):** Hippocampus 5.83 → MidTemp 4.63 →
WholeBrain 4.22 → Fusiform 4.00 → Entorhinal 3.57 → Ventricles 0.75.

**Onset, RELATIVE threshold (50% of each region's own deviation at conversion):**
Hippocampus 4.41 → Entorhinal 3.84 → Fusiform 3.50 → MidTemp 3.29 →
WholeBrain 3.28 → Ventricles 1.74.

The relative version is the one that matters: a region with a smaller total
deviation crosses a FIXED threshold late even with identical timing, so only the
magnitude-free version tests ORDER. **The ordering survives it.**

| separation, hippocampus vs ventricles | value | 95% CI | p |
|---|---|---|---|
| absolute threshold | +4.96 y | [+3.99, +5.70] | <0.001 |
| **relative threshold** | **+2.49 y** | **[+1.02, +3.60]** | **0.028** |

**Why this is more than a statistical curiosity:** the magnitude-free ordering
(hippocampus → entorhinal → fusiform/midtemp → ventricles) recovers the known
neuropathological sequence — medial temporal structures first, ventricular
enlargement as a late global consequence — WITHOUT ever being told about
pathology. The method was given only volumes and chronological age.

Note that entorhinal moves from 5th place on the absolute threshold to 2nd on
the relative one, because its total deviation is small (1.68 y) but begins early.
That reversal is the single clearest argument for using the relative criterion.

⚠️ **What is NOT established.** Only the extremes are separated with confidence;
adjacent regions have overlapping intervals, so the full six-way ordering is
suggestive, not demonstrated. The far bins are thin (76 visits at 5-7 y).
Conversion time is the first visit coded Dementia, so it inherits the visit
schedule's coarseness. And this has NOT yet been checked against the literature —
regional brain-age timing may already be published; novelty cannot be claimed
until that search is done.

### LITERATURE CHECK on E37 (regional cascade): NOT NOVEL — published twice over
- *Alzheimer's & Dementia*, "Local brain aging maps proximity to AD-related cognitive
  impairment": regional brain aging binned by time-to-conversion (0.5-2.5 / 2.5-6 / >6 y).
  Our exact design, at finer parcellation.
- medRxiv 2025, "Heterogeneous ageing of brain networks across the AD continuum":
  progressive MCI deviate first in the default-mode network, then attention/control.
- Behind both, the mature Event-Based Model / SuStaIn literature (Fonteijn 2011,
  Young 2018) already orders regional atrophy events.
→ E37 is an independent REPRODUCTION. Good pipeline validation, not a contribution.
This is the FIFTH candidate novelty to die on contact with the literature
(noise floor, cascade, delta-training, attenuation-narrow, ...). The pattern itself is
the lesson: this is a crowded field and every natural idea has been tried.

### E38 — THE CASCADE IS LARGELY A MEASUREMENT-PRECISION ARTEFACT  ★★ survives the check
`pilot/exp38_reliability_cascade.py`  (379 converters, 650 stable MCI, 910 CN)

No published cascade corrects for regions being measured with different precision. A
precisely measured region reveals its deviation early; a noisy one hides the same
deviation for years. Per-region measurement noise, from within-subject scatter about a
straight line in CN + stable MCI (people whose trajectory should be straight):

| region | noise (y) | deviation at conversion |
|---|---|---|
| Ventricles | **0.10** | 1.23 |
| WholeBrain | 0.36 | 3.21 |
| MidTemp | 0.39 | 3.16 |
| Entorhinal | 0.40 | 1.68 |
| Hippocampus | 0.42 | 4.45 |
| Fusiform | 0.42 | 2.55 |

**Ventricles are measured four times more precisely than anything else.**

**The ordering is criterion-dependent.** Three defensible criteria, three different orders:

| region | fixed 1 y | relative 50% | reliability 2x own noise |
|---|---|---|---|
| Hippocampus | 1 | 1 | 1 |
| Entorhinal | 5 | **2** | **6** |
| Fusiform | 4 | 3 | 5 |
| MidTemp | 2 | 4 | 4 |
| **Ventricles** | **6** | **6** | **3** |
| WholeBrain | 3 | 5 | **2** |

Only hippocampus holds its rank. Ventricles move from LAST to THIRD; entorhinal from
SECOND to LAST. The published cascades use a fixed threshold and report no sensitivity
analysis.

**Null simulation** — every region given the SAME true onset and shape, differing only
in its measured noise:

    spread produced by precision differences alone   1.50 y [1.30, 1.76]
    observed spread                                  2.10 y     p < 0.001
    noise alone reproduces the extreme ordering in   65% of simulations

→ **There IS a real timing component (2.10 > 1.50, p<0.001), but roughly 71% of the
apparent spread is explained by measurement precision, and the specific extreme ordering
arises from noise alone two thirds of the time.**

⚠️ Two earlier versions of this test were degenerate and their verdicts were discarded:
the first divided by the standard error of the bin mean (noise/sqrt(n)), which with
76-414 visits per bin made every region detectable everywhere and returned all six onsets
at exactly the boundary; the second fixed the observation but left the null on the old
scale, so the null had zero spread by construction and "the cascade survives" was
tautological. Only the third version, with per-person noise on BOTH sides, is reported.

### E39 — ROBUSTNESS OF E38: the "71%" was one point on a wide curve
`pilot/exp39_cascade_robustness.py`

E38 reported that 71% of the apparent cascade spread is explained by measurement
precision. That used ONE threshold (2x noise) and ONE noise estimator. Sweeping both:

| k | line-residual estimator | consecutive-difference estimator |
|---|---|---|
| | % explained (p) | % explained (p) |
| 1.0 | 60% (0.093) | 49% (0.000) |
| 1.5 | 48% (0.000) | 62% (0.000) |
| 2.0 | **71% (0.000)** | 97% (0.330) |
| 2.5 | 93% (0.140) | 120% (1.000) |
| 3.0 | 90% (0.010) | 90% (0.000) |
| 4.0 | 101% (0.740) | 77% (0.000) |

**The fraction ranges from 48% to 120% and the p-values swing non-monotonically
between 0.000 and 1.000.** The specific figure of 71% is not a stable quantity and
must not be quoted as one.

Ventricles are measured **3.9x** more precisely than the average region, and this
holds under both estimators (ratio between estimators 1.15-1.17 across regions), so
the precision imbalance itself IS robust.

**What survives, stated at the strength the data supports:**
Measurement precision accounts for a large share - roughly half to all - of the
apparent onset spread, at every threshold and under both estimators. The exact share
is not identifiable at this sample size and bin resolution. The defensible claim is
therefore: *precision differences between regions are large enough that a regional
cascade cannot be interpreted without reporting them, and no published cascade does.*

⚠️ The non-monotonic p-values are themselves a warning that the null test is unstable
with six bins and 379 converters. Any proposal built on this must include a power
analysis first, not after.

### LITERATURE REVIEW — running list (~26 papers, 6 categories)
Systematic reviews / narrative: J Neurol Sci 2025 (BAG as early AD biomarker);
Brain Sciences 2026 (narrative perspectives); Ageing Research Reviews 2026 (brain
aging clocks); SSRN 2026 fusion-centric roadmap.
Fundamental critiques: Tan 2025 (HBM) scalar gap 0.70 vs direct 0.92; Schulz 2025
(PLOS Biology) inverse accuracy/sensitivity; Vidal-Pineiro 2021 (eLife) gap reflects
brain reserve not rate; "Clinically altered brain activity may not look like aged
brain activity" 2026; Imaging Neuroscience 2026 bias/generalizability.
Reliability: Dular (brain-age package test-retest); Nature Communications 2026
(cluster scanning precision); DeepBVC 2023; GeoLongSeg 2024.
Biomarkers: npj Dementia 2026 (brain age as mediator of p-tau217 -> cognition, N=71);
Frontiers Aging Neuroscience 2026 p-tau network meta-analysis.
Cascade / staging: Alzheimer's & Dementia (local brain aging vs time-to-conversion);
medRxiv 2025 (network-level ordering); Fonteijn 2011 EBM; Young 2018 SuStaIn.
Harmonisation / methods: PMC11215839 (site effects, age range, sample size);
ComBat-Predict 2025; Batch effects in brain foundation model embeddings 2026;
Wen 2020 (Medical Image Analysis) leakage review.

### E40 — POWER ANALYSIS for the regional-cascade question  ★ decides the proposal
`pilot/exp40_power.py` — simulation using the MEASURED per-region noise and amplitudes

False-positive rate 3-8% in every condition, so the test itself is valid. Power depends
overwhelmingly on one quantity nobody has measured: **how much patients differ from each
other in when their disease starts** (sigma).

Power to detect a true onset separation, alpha = 0.05:

| converters | sep 1 y | 2 y | 3 y |
|---|---|---|---|
| **sigma = 0** (every patient on the same schedule) | | | |
| 200 | 100% | 100% | 100% |
| 379 (what we have) | 100% | 100% | 100% |
| **sigma = 1.5 y** | | | |
| 200 | 71% | 100% | 100% |
| 379 | 98% | 100% | 100% |
| **sigma = 3.0 y** | | | |
| 200 | 15% | 84% | 100% |
| 379 | **30%** | **97%** | 100% |
| 1000 | 55% | 100% | 100% |
| 4000 | 80% | 100% | 100% |

**Reading.** For a 2-year true separation we are adequately powered at n=379 even under
strong heterogeneity (97%). For a 1-year separation under strong heterogeneity we would
need roughly 4,000 converters - far beyond ADNI. Our observed spread was 2.10 y, which
sits in the powered regime; but the unstable p-values of E39 are what the underpowered
regime looks like, which suggests either sigma is large or the true separation after
precision correction is nearer 1 year than 2.

**Therefore the first task of any proposal is to MEASURE sigma**, because it decides
whether ADNI alone can answer the question at all. That is a one-experiment question,
not a three-year commitment.

⚠️ Two versions of this analysis were wrong before this one. The first omitted
between-patient heterogeneity entirely (every patient deviating on an identical
schedule) and returned 100% power in every cell - a result that plainly contradicted
the instability seen in real data, which is how the omission was noticed. The second
recomputed the null distribution inside every simulated study, making it N_SIM x N_NULL
calls per cell; it never finished. The null depends only on cohort size and sigma, so it
is now built once per cell.

### E41 — MEASURING sigma: patients differ by ~3 years in when deviation starts
`pilot/exp41_sigma.py` — three independent estimators

| method | estimate |
|---|---|
| A. shape of the group curve, hippocampus | 0.00 y, **95% CI [0.00, 4.51]** — unidentified |
| A. same method, other five regions | 2.00, 2.25, 2.25, 4.00, 4.00 y |
| B. between-subject spread minus noise | 1.5-3.8 gap-years, flat across time |
| **C. per-patient onsets (n=140), noise-corrected** | **3.57 y** (raw 3.80, estimate noise 1.29) |

Method A is poorly identified for hippocampus - its interval covers the whole plausible
range, so its point estimate of zero carries no information and must not be quoted.
Taking the informative estimates together, **sigma is roughly 2-4 years, most plausibly
near 3.**

**CONSEQUENCE, read off the E40 power table at sigma = 3:**

| true separation | power at n=379 |
|---|---|
| 1 year | **30%** |
| 2 years | 97% |

ADNI is adequate ONLY for separations of about two years or more. Our observed spread
was 2.10 y, but E38/E39 showed that measurement precision alone accounts for roughly
half to all of it, so the RESIDUAL true separation is likely well under two years —
exactly the regime where we have ~30% power.

**This explains the E39 instability completely.** p-values swinging non-monotonically
between 0.000 and 1.000 is what a 30%-power test looks like.

### DECISION: the cascade question is NOT answerable with ADNI alone
Not at the effect size that remains after the precision confound is removed. Pursuing it
would be a three-year commitment to an underpowered question. The power analysis cost one
day and prevented that.

**What IS supported by the data in hand** — four direct measurements, each with tight
intervals and none requiring the detection of a small timing difference:
1. noise floor of the gap, ~1.9 y (two independent models agree)
2. longitudinal attenuation, 13%-94% by region, not predictable from cross-sectional r
3. per-region precision imbalance, 3.9x, robust across two estimators
4. acceleration: measurable (61% of its variance is signal) but prognostically empty

These are measurements, not effect-detections, so they are powered. Together they form a
coherent subject: **what the brain-age gap actually measures, and how well.** That is the
defensible thesis direction; the cascade is not.

### E42 — RUNNING THE MODEL BACKWARDS (age -> expected brain) vs forwards
`pilot/exp42_inverse.py`  — user's idea; this is normative modelling
(Marquand et al.; BrainChart, Nature 2022) and is NOT new, but the head-to-head
against the brain-age gap on identical folds and identical features is.

**1. Both directions end up unbiased — but the reverse needs no correction step.**

    forward gap, AFTER its regression-dilution correction :  r = -0.018 with age
    reverse z-scores, NO correction applied               :  r = +0.002

The correction that every brain-age paper applies, and that has to be fitted on
controls and carried to patients, is simply unnecessary when the brain is the
dependent variable. One fewer fitted step between data and claim.

**2. Reliability is IDENTICAL, not better.**

| region | forward | reverse |
|---|---|---|
| Hippocampus | 0.976 | 0.974 |
| Entorhinal | 0.907 | 0.880 |
| Fusiform | 0.963 | 0.960 |
| MidTemp | 0.971 | 0.970 |
| Ventricles | 0.998 | 0.997 |
| WholeBrain | 0.979 | 0.980 |

⚠️ The first printout showed forward noise 0.415 vs reverse 0.152 for hippocampus
and looked like a 3x improvement. It was a UNITS artefact - years against
standard deviations. On the unit-free scale 1 - noise^2/between^2 the two agree to
within 0.003. Nearly reported a dramatic advantage that does not exist.

**3. Disease separation: both differences real, both tiny, opposite signs.**

| task | reverse minus forward | 95% CI | p |
|---|---|---|---|
| AD vs CN | **+0.007** | [+0.003, +0.013] | <0.001 |
| MCI vs CN | **-0.015** | [-0.024, -0.006] | <0.001 |

| arm | AD vs CN | MCI vs CN |
|---|---|---|
| gap scalar | 0.906 | 0.696 |
| gap regional (6) | 0.942 | **0.747** |
| normative z (6) | **0.950** | 0.732 |
| normative distance (1) | 0.881 | 0.637 |
| **volumes + age** | **0.951** | 0.746 |

**Neither direction dominates, and volumes+age still ties or beats both** - the same
fair-baseline result as Reports 3 and 4, now in a third framework.

**4. The compression penalty replicates in the new framework.** Collapsing six
z-scores into one Mahalanobis distance costs 0.069 on AD (0.950 -> 0.881), mirroring
the scalar gap losing to the regional profile (0.906 vs 0.942). Compression hurts
regardless of which direction the model runs.

**5. Interpretability is the genuine gain.** Mean |z| rises monotonically in every
region: hippocampus 0.75 (CN) -> 1.29 (MCI) -> 2.28 (AD); entorhinal 0.74 -> 1.05 ->
1.75; ventricles 0.72 -> 0.82 -> 1.06. The reverse direction states directly which
region deviates and by how much, in that region's own units, without inverting
anything.

→ The forward/reverse choice is a matter of PROCEDURE and INTERPRETABILITY, not of
information. Given age, the two are near-invertible transforms of one another, which
is why they perform the same.

### E43 — ANCHORING TO THE PATIENT'S OWN BASELINE  (user's third idea)
`pilot/exp43_anchored.py` — 1,369 subjects with a baseline and a ~2-year follow-up
(493 CN, 656 MCI, 178 AD); survival arm n=400 MCI, 104 conversions AFTER the
follow-up visit.

**The mechanism works exactly as intended.** Residual SD when predicting a control's
follow-up volume:

| region | population model | anchored model | reduction |
|---|---|---|---|
| Hippocampus | 751 | 288 | 62% |
| Entorhinal | 623 | 358 | 43% |
| Fusiform | 2,103 | 835 | 60% |
| MidTemp | 2,162 | 840 | 61% |
| Ventricles | 15,544 | 2,670 | **83%** |
| WholeBrain | 59,538 | 28,216 | 53% |

**Anchoring removes 60% of the unexplained variance on average.** Everything stable
about a person - head size, shape, reserve - cancels, exactly as the idea predicted.

**But that does NOT translate into better prediction of conversion:**

| arm | C-index |
|---|---|
| age only | 0.559 |
| brain-age gap (scalar) | 0.724 |
| gap rate | 0.712 |
| raw change, 6 regions | 0.718 |
| **population z, 6 regions** | **0.791** |
| anchored z, 6 regions | 0.758 |

| comparison | difference | 95% CI | p |
|---|---|---|---|
| anchored vs population | **-0.033** | [-0.077, +0.013] | 0.185 — not distinguishable |
| anchored vs raw change | **+0.037** | [+0.008, +0.065] | **0.015 — real** |

**Reading.** Anchoring buys a far better *reconstruction* and no better *prediction*.
The variance it removes is between-subject variance, which was never the disease
signal; the disease signal lives in the part that anchoring also partly removes. It
does beat naive subtraction of the baseline, so the model is not decoration - but it
does not beat the population normative model.

★ **The incidental result is the most interesting one: the population normative
6-vector (0.791) beats the classical scalar gap (0.724) and the gap RATE (0.712) at
the same landmark.** A cross-sectional regional deviation vector outperforming a
longitudinal rate is worth following up, and was not what this experiment set out
to test.

⚠️ The first version of the reconstruction table computed corr(y - z*sd_y, y), using
the between-subject SD where the residual SD belonged. It returned NEGATIVE
correlations, which is what exposed it. Recomputed as plain residual SD above.

### E44 — AT MATCHED DIMENSIONALITY, THE RATE ADDS NOTHING  ⚠️ qualifies Report 3
`pilot/exp44_cross_vs_rate.py` — 400 MCI, 104 conversions after the follow-up visit

E43's 0.791 vs 0.712 was six numbers against one. Repeated at matched dimensionality:

| arm | dim | C |
|---|---|---|
| age only | 1 | 0.559 |
| scalar gap | 1 | 0.724 |
| scalar gap RATE | 1 | 0.712 |
| z at baseline | 6 | 0.767 |
| **z at follow-up** | 6 | **0.791** |
| **regional RATE** | 6 | **0.681** |
| z follow + rate | 12 | 0.793 |

| comparison | difference | 95% CI | p |
|---|---|---|---|
| cross-section beats rate | **+0.105** | [+0.042, +0.172] | **<0.001 REAL** |
| **does the rate ADD?** | **+0.007** | [-0.010, +0.029] | **0.488 — NO** |
| follow-up beats baseline | +0.023 | [+0.010, +0.037] | <0.001 REAL |
| regional beats scalar gap | +0.071 | [+0.039, +0.107] | <0.001 REAL |

**This qualifies the headline of Report 3.** That report claimed the rate ADDS to the
value (+0.052, CI [+0.029, +0.077]) — and it does, when the comparator is a SCALAR gap.
Against a regional cross-sectional vector the contribution disappears entirely
(+0.007, CI crosses zero).

**The coherent reading across everything we have measured:** what the rate was really
supplying was information the scalar gap had compressed away. Stop compressing — use
six numbers instead of one — and there is nothing left for the rate to add. This is the
same compression penalty seen in E3/E4 (scalar 0.893 vs profile 0.933), E42 (normative
distance 0.881 vs six z 0.950) and E43.

⚠️ **CAVEAT that limits this conclusion.** The rate here is built from TWO time points,
(z_follow − z_base)/dt, whereas Report 3 fitted a slope through THREE OR MORE scans. A
two-point rate is markedly noisier, so this design handicaps the rate arm. The honest
statement is therefore conditional: *with a two-point rate over ~2 years, the rate adds
nothing to a regional cross-sectional vector.* Whether a multi-point slope would survive
the same comparison is untested and is the obvious next experiment.

The Report 3 claim should not be withdrawn on this evidence — it should be stated with
its comparator named: the rate adds to a SCALAR gap, which is a weaker claim than it
first appeared.

### E45 — THE FAIR VERSION: a properly fitted slope. E44 WAS OVERSTATED.
`pilot/exp45_multipoint_slope.py` — 305 MCI with >=3 scans inside the 2-year window
(median 5 scans each), 79 conversions afterwards

E44 judged the rate using a TWO-point difference and found it added nothing
(+0.007). I flagged that as a handicap at the time. Repeating with a slope fitted
through all scans in the window, both arms seeing exactly the same data:

| arm | C |
|---|---|
| age only | 0.555 |
| cross-section z (6) | 0.798 |
| fitted SLOPE (6) | 0.751 |
| **both (12)** | **0.819** |

| comparison | difference | 95% CI | p |
|---|---|---|---|
| **slope adds to cross-section** | **+0.024** | [-0.002, +0.055] | **0.060 — borderline** |
| cross-section beats slope | +0.044 | [-0.014, +0.105] | 0.120 — no |
| *(E44, two-point rate)* | *+0.007* | *[-0.010, +0.029]* | *0.488* |

**The caveat was correct and E44's conclusion is withdrawn as stated.** Giving the
rate a properly fitted slope more than TRIPLES its contribution, from +0.007 to
+0.024. The two-point design really was handicapping the arm it judged.

**But the fair test does not reach significance either** (p=0.060, lower bound -0.002),
and with only 79 events it is underpowered - the interval is about +-0.029 wide, so an
effect of +0.024 sits right at the edge of what this sample can resolve. The result is
INCONCLUSIVE, leaning toward the slope adding a small amount.

**Where that leaves Report 3.** Its claim (+0.052, CI [+0.029, +0.077], against a
SCALAR gap) is not refuted. Against a REGIONAL comparator the same contribution is
roughly halved (+0.024) and loses significance. So the honest statement is:

> The rate adds substantially to a scalar gap. Against a regional cross-sectional
> vector its contribution is about half the size and not statistically resolvable at
> n=79 events. Whether it survives at larger n is untested.

Note also that the best arm in this whole series is cross-section + slope together
(0.819) - neither summary alone.

### E46/E47 — FIVE QUESTIONS. TWO REAL FINDINGS, THREE CLEAN NULLS.
`pilot/exp46_overnight.py`, `pilot/exp47_followup.py` (ADNI, 1,942 baselines)

★★ **1. THE BRAIN-AGE GAP IS THE WRONG ONE NUMBER — and this corrects four of our
own earlier conclusions.**

| k dimensions | AD vs CN | % of six-dim |
|---|---|---|
| 1 (PCA) | 0.928 | **95%** |
| 3 | 0.938 | 98% |
| 6 | 0.950 | 100% |
| **brain-age gap (also ONE number)** | **0.906** | 90% |

We concluded "compression hurts" in E3/E4, E42, E43 and E44. **That reading was
wrong.** One number retains 95% of six-dimensional performance. What costs accuracy
is not compression but WHICH single number: a well-chosen linear combination of
regional deviations beats the combination brain age computes.

    AD vs CN    one component minus gap  +0.024  95% CI [+0.016, +0.032]  p<0.001  REAL
    MCI vs CN                            +0.005  95% CI [-0.002, +0.014]  p=0.180  no

Verified with PCA fitted INSIDE training folds (no transduction) — identical result.

**The mechanism is visible in the loadings:** WholeBrain +0.451, Hippocampus +0.450,
MidTemp +0.431, Fusiform +0.425, Entorhinal +0.396, Ventricles −0.268 (58% of
variance). The disease-optimal single number is close to a UNIFORM average of the
atrophy regions with ventricles reversed. Brain age instead weights each region by
how well it predicts CHRONOLOGICAL AGE. The mismatch between the age-optimal and the
disease-optimal weighting is what costs 0.024.

⚠️ Holds for AD vs CN only; on MCI vs CN the interval crosses zero. And the PCA
rotation is unsupervised but still fitted on this cohort, so "a better single number
exists" is established, "this particular number generalises" is not.

★★ **2. DISEASE CHANGES THE COUPLING BETWEEN REGIONS, not only the levels.**

| pair | CN | MCI | AD | change |
|---|---|---|---|---|
| Fusiform / MidTemp | 0.34 | 0.56 | 0.58 | **+0.24** |
| Hippocampus / Entorhinal | 0.30 | 0.57 | 0.49 | +0.19 |
| Hippocampus / Ventricles | −0.34 | −0.33 | −0.15 | +0.19 |

    mean |correlation difference| AD vs CN   0.119
    null from label permutation              0.046 [0.026, 0.073]   p < 0.0001

**Not a variance artefact** — Spearman gives 0.124 vs null 0.043, p<0.0001, despite
the group SDs differing (CN 0.98, MCI 1.17, AD 1.25). This is invisible to every
method that examines regions one at a time, which is all of them: brain age,
normative modelling, EBM.

⚠️ But a coupling feature (Mahalanobis under the CN covariance) adds **nothing** to
prediction: +0.001 on AD, −0.001 on MCI. Descriptive finding, not a better predictor.

**3. NO ASYMMETRY.** Splitting the gap into positive and negative tails changes
nothing (AD 0.906 symmetric vs 0.906 split; MCI 0.698 vs 0.696). The field's implicit
symmetry assumption is correct.

**4. EDUCATION DOES NOT PREDICT THE GAP IN HEALTHY PEOPLE.** r = +0.030 with the
scalar gap, |r| <= 0.079 for every region, n=2,785 controls. This is evidence against
reading the gap as brain reserve (Vidal-Pineiro, eLife 2021) — at least with education
as the proxy. ⚠️ ADNI is heavily education-restricted, so range restriction could
mask a real association; this null is weaker than it looks.

**5. APOE4 DOES NOT SHIFT THE GAP IN CONTROLS.** Carriers minus non-carriers
+0.11 y, 95% CI [−0.14, +0.38], p=0.428 (n=1,924 vs 788).

⚠️ E46 crashed on Q4/Q5: ADNIMERGE already contains PTEDUCAT, so merging it again
produced PTEDUCAT_x and the attribute lookup failed. Fixed in E47.

**NOT YET CHECKED AGAINST THE LITERATURE.** Finding 1 is adjacent to Tan 2025 (a
supervised disease model beats the scalar gap, 0.92 vs 0.70) but is a different and
stronger claim: an UNSUPERVISED recombination of the same information beats it, so
the loss is attributable to the age-target weighting rather than to supervision.
Finding 2 needs a search on regional covariance/coupling in AD before any novelty
claim.

### LITERATURE CHECK on E46/E47 — finding 2 dies, finding 1 SURVIVES (narrowly)

**Finding 2 (coupling) — NOT NOVEL. Sixth candidate to die on contact.**
There is an established field called STRUCTURAL COVARIANCE NETWORKS in AD:
- Altered structural covariance of hippocampal subregions in AD (ScienceDirect 2021)
- Alterations via inter-regional connective relationships in AD (PMC10672243)
- Hippocampal atrophy and distinct structural covariance patterns in AD and MCI (PMC9505506)
"Structural covariance connectivity becomes disorganised across brain regions in AD,
while controls maintain more clustered coordination" — that IS our result, already
published, and extended with graph-theoretic measures (path length, small-worldness)
we did not compute. Our version is a re-derivation on six regions.

**Finding 1 — the specific claim SURVIVES.** Checked the most directly competing
paper, "Mind the Gap: Does Brain Age Improve Alzheimer's Disease Prediction?"
(PMC12358809 — the Tan work we already cite; its numbers match, 0.92 direct vs 0.70
gap). Read directly. It does **NOT**:
  - compare the gap against an alternative SINGLE number built from the same features
  - quantify how much is lost to the chronological-age target specifically

| already established | our addition |
|---|---|
| gap (0.70) loses to a direct disease model (0.92) — Tan 2025 | the gap also loses to an **unsupervised** recombination of the same features |
| "chronological age may be a poor pretraining target" — stated | measured: **+0.024, 95% CI [+0.016, +0.032]** |
| lower age accuracy ↔ higher disease sensitivity — Schulz 2025 | the loss DECOMPOSED: compression costs ~5%, weighting costs the rest |
| — | the disease-optimal single number is near-UNIFORM across atrophy regions |

**Why the distinction matters.** The published result invites the reply "of course a
supervised model beats it — supervision helps". Our comparison removes that defence:
PCA sees no labels. So the deficit cannot be attributed to supervision. It localises
to the age target and to the weighting that target induces.

⚠️ Scope: AD vs CN only (on MCI the interval crosses zero), one cohort, six regions,
and the rotation is fitted on this cohort — "a better single number exists" is shown,
"this particular number generalises" is not.

### E48 — DOES IT GENERALISE? Across scanner and study phase, YES.  ★★ verified
`pilot/exp48_generalise.py` — 1,942 baseline scans. Everything (normative model,
rotation, brain-age model) fitted on the TRAIN cohort only; nothing from test touches
a fitted object.

| domain shift | test set | rotation on all | rotation on **CN only** |
|---|---|---|---|
| random half/half | 163 AD | +0.016 [+0.000, +0.031] p=0.050 | **+0.016 [+0.001, +0.030] p=0.045** |
| **ADNI1 → ADNI2/GO/3** | 163 AD | **+0.012 [+0.005, +0.020] p<0.001** | **+0.010 [+0.003, +0.018] p<0.001** |
| **1.5 T → 3 T** | 87 AD | **+0.015 [+0.005, +0.026] p<0.001** | **+0.012 [+0.001, +0.022] p=0.025** |
| FreeSurfer 4.3/5.1 → 6.0 | 60 AD | +0.006 [-0.005, +0.016] p=0.275 | +0.007 [-0.002, +0.017] p=0.180 |

**The finding transfers across a change of scanner field strength and across study
phase — two genuine domain shifts, not a random resplit.** It does not reach
significance across a FreeSurfer version change, but that test has the fewest events
(60 AD against 163 and 87) and its interval [-0.002, +0.017] comfortably contains the
+0.010-0.012 seen elsewhere. That is absence of evidence, not evidence of absence.

★ **The strong version holds: a rotation fitted on TRAINING CONTROLS ONLY — never
having seen a single patient — still beats the brain-age gap** on held-out patients
scanned at a different field strength. The deficit therefore cannot be attributed to
supervision, to patient-specific fitting, or to cohort overfitting.

**Effect size roughly halves on transfer**, from +0.024 within-cohort to +0.010 to
+0.016 across cohorts. The honest headline number is the transferred one.

**The loadings are stable across every split**, which is why it transfers:

    WholeBrain +0.53..0.54  Hippocampus +0.44..0.48  MidTemp +0.39..0.42
    Fusiform +0.33..0.44    Entorhinal +0.23..0.31   Ventricles -0.24..-0.40

Near-uniform positive weights on the atrophy regions with ventricles reversed. Brain
age instead weights regions by their usefulness for predicting chronological age, and
that mismatch is what costs the AUC.

## STATUS OF THE NOVELTY CLAIM after six dead candidates
Established in the literature: the gap loses to supervised disease models (Tan 2025,
0.70 vs 0.92); chronological age is a weak pretraining target (stated, not measured);
lower age accuracy means higher disease sensitivity (Schulz 2025).
Our addition, now verified under domain shift: **the gap also loses to an UNSUPERVISED
recombination of the same features, fitted without seeing any patient — measured at
+0.010 to +0.016 AUC across scanner and cohort shifts — and the loss decomposes into
~5% from compression and the remainder from the age-induced weighting.**


---

# E49 — IS "BRAIN AGING" THE SAME PROCESS IN CN, MCI AND AD?

The assumption under all of normative modelling: patients travel the SAME road as
healthy people, only further along. Never routinely tested. Design: compare the
DIRECTION of the six-region change vector between groups, scored against a null
built from two DISJOINT control subsets matched in size — a process identical by
construction. Units: CN residual SD per region, age trend retained.

## What survived every version
**SPEED.** MCI 1.4-1.95x, AD 2.9-3.3x the healthy rate. Stable across all four
analyses. This is the dominant difference and it is not subtle.

**MCI AND AD ARE ONE PROCESS.** cos(MCI, AD) = 0.975-0.993, never distinguishable
from the null (p = 0.057, 0.443, 0.560, 0.783). Different speeds, one road.

**DIRECTION IS >=94% SHARED WITH HEALTHY AGING IN EVERY SINGLE VERSION.** Even
where "DIFFERENT" is declared the cosine never drops below 0.937. Whatever the
disease-specific component is, it is small.

## What did NOT survive — and the control that killed it
Longitudinal direction CN vs AD looked different in three analyses (p = 0.003,
0.000, 0.003; excess off-axis +20 to +21 points). **Then the follow-up confound:
controls are followed 4.5 y, AD patients 2.0 y, because AD patients drop out. A
slope fitted over 2 years is noisier, and noise alone pushes a vector off-axis.**
Capping EVERYONE at a 2-year window:

    cos(CN, AD)  0.9463 -> 0.9680   p 0.003 -> 0.117   excess off-axis +20 -> +7
    cos(CN, MCI) 0.9772 -> 0.9878   p 0.013 -> 0.563   excess off-axis +11 -> -1

The observed cosine went UP while the null widened. Both point the same way: most
of the apparent direction difference was unequal measurement noise.
**VERDICT: not established.** A +7 point hint at p=0.117, nothing more.

## Cross-sectional age slope (PART 1) — do not use
AD |slope| = 0.089 vs CN 0.188, half the size. That is a floor/selection artefact
in a prevalent sample, not biology. Cross-sectional age slopes inside a patient
group are not a usable measure of an aging process. Reported for completeness only.

## PART 3 — the model, trained where, tested where (symmetric grid, n=200 each,
age-matched training draws, 100 held out per group)

    MAE (y)        tested CN   tested MCI   tested AD
    trained CN       4.295       5.785        6.773
    trained MCI      4.531       5.152        5.774
    trained AD       5.282       5.203        6.075

Own-group is best for CN and MCI but NOT for AD — the MCI-trained model reads an
AD brain's age better (5.774) than the AD-trained one (6.075). The AD-trained
model is worse EVERYWHERE, including on AD itself. The parsimonious reading is
not "a different axis" but **age signal is buried under severity in AD brains**.
Training on AD costs +0.99 y on controls, 95% CI [+0.51, +1.63]. The reverse,
training on CN and testing on AD, costs +0.70 y, CI [-0.06, +1.60], crossing zero.
Ridge coefficient vectors are too unstable at n=200 to distinguish (p = 0.21-0.43);
an earlier run at n=303 said "DIFFERENT" — the test flips with training size and
should not be relied on.

## LITERATURE — the question is not ours
The accelerated-aging-vs-distinct-process debate is established. [European
Radiology 2010] reports MCI and AD share medial temporal atrophy with AD adding
extramedial temporal. [Neurobiol Aging 2021] finds atrophy "did not detectably
differ ... when age ranges were matched, though disorder-specific topographies of
accelerated aging were identified". Current consensus: both acceleration and a
spatially distinct component.
Also relevant and uncomfortable: [Vidal-Pineiro 2021, PMC8580481] found NO
association between cross-sectional brain age and the longitudinal rate of brain
change — a direct challenge to the interpretation of the gap.
Closest neighbour to E48 found so far: [PMC13320867] "a single absolute-deviation
score retained much of the predictive information contained in higher-dimensional
dynamic features" — functional connectivity, not structure, and no comparison
against the brain-age gap. E48 stands.

**WHAT E49 CONTRIBUTES: not the answer, the control.** The follow-up-length
confound reverses the answer, and none of the cited work mentions it. Differential
dropout is universal in dementia cohorts, so any study comparing longitudinal
change direction between groups is exposed to it. That is a methodological note
worth a paragraph in a paper, not a discovery.


---

# E50 — WHAT IF THE AGE MODEL IS TRAINED ON PATIENTS?

Convention: the age model is always fitted in healthy controls. Never tested
against the alternative. E49 part 3 showed a patient-trained model reads AGE
badly, but age accuracy is not the point — E48 showed optimising for age is what
breaks brain age. The question is PROGNOSIS.

Design: four age models, training sets all n=200 drawn to the SAME age strata so
arms differ only in DIAGNOSIS; each gap bias-corrected against its own training
group; MCI arm out-of-fold by subject; each gap averaged over 20 draws; identical
test set. 783 MCI at baseline, 305 converted (39%), median follow-up 3.0 y.

    age only                          C = 0.531
    gap | trained CN   (standard)     C = 0.738
    gap | trained MCI                 C = 0.675    -0.062  CI [-0.081,-0.042] p=0.000
    gap | trained AD                  C = 0.677    -0.061  CI [-0.079,-0.041] p=0.000
    gap | trained all three pooled    C = 0.730    -0.008  CI [-0.014,-0.001] p=0.010
    all four gaps together            C = 0.735    -0.002  CI [-0.007,+0.002] p=0.240
    one PCA component (E48)           C = 0.753    +0.015  CI [+0.004,+0.025] p=0.010
    six deviations (ceiling)          C = 0.757

**ANSWER: NO. Training on patients makes prognosis strictly worse**, by -0.06
C-index for either patient group — a large loss on this scale. Pooling all three
also loses, slightly. **The four gaps together do no better than the standard one
alone** (p=0.240), so the patient-trained gaps carry no complementary information;
they are noisier versions of the same quantity (r = 0.81-0.84 between them).

Mean gap by training group: CN +2.72, MCI +0.04, AD -1.31. An AD-trained model
expects atrophy, so an MCI brain looks YOUNGER than its age against AD norms.

★ **E48 REPLICATES ON A SECOND, DIFFERENT ENDPOINT.** Everything before was AD vs
CN AUC at baseline. Here it is MCI->AD conversion scored by Cox C-index, and the
single PCA component still beats the gap: +0.015, CI [+0.004, +0.025], p=0.010.
Different outcome, different metric, same conclusion.

**The synthesis:** the healthy-controls convention is not arbitrary, it is
correct. Combined with E48 this sharpens the whole story — **the damage is in the
TARGET (chronological age), not in the POPULATION the target is learned from.**
Fix the target, keep the population.

## LITERATURE
The question has been asked. [PMC13536007, "Clinically altered brain activity may
not look like aged brain activity"] and reports that training on mixed populations
changes over-aging behaviour in AD. Also noted: "most studies do not involve
separate training on healthy reference populations and testing on clinical
populations". Our quantification on a conversion endpoint (-0.062 C-index) was
not found, but the question is established, so this is a confirmation with a
number, not a discovery.

---

# E51 — IS THE DEVIATION AXIS DISEASE-SPECIFIC OR ONE GENERIC ILLNESS AXIS?

If the E48 rotation also separates autism and ADHD cases from their controls, it
is a generic "something is wrong" direction and calling it an Alzheimer axis is
wrong. Cohorts: ADNI 1093 (320 AD), ABIDE 1099 (532 autism), ADHD-200 631 (232
ADHD). Axis = mean case deviation in that cohort's own control-SD units, after
age, age^2, sex, ICV and SITE. Null = two disjoint halves of one cohort.

TIER 1, the three regions all three share:

    ADNI    Hippocampus -2.547  Ventricles +0.951  WholeBrain -1.482  |axis| 3.097
    ABIDE   Hippocampus -0.012  Ventricles +0.412  WholeBrain -0.127  |axis| 0.431
    ADHD    Hippocampus -0.234  Ventricles -0.069  WholeBrain -0.272  |axis| 0.365

    ADNI vs ABIDE   cos +0.457  null[2.5%] +0.583  p=0.010  DIFFERENT
    ADNI vs ADHD    cos +0.825  null[2.5%] +0.984  p=0.000  DIFFERENT
    ABIDE vs ADHD   cos +0.058  null[2.5%] +0.534  p=0.000  DIFFERENT

TIER 2, ABIDE vs ADHD-200 across 32 shared segmented structures — the clean
comparison, since both cohorts are young, same feature space, same processing:

    cos = -0.138   null[2.5%] +0.409   p=0.000   DIFFERENT
    null median +0.693 — even a cohort against ITSELF only reaches 0.69

    autism  3rd-Ventricle +0.48, lateral ventricles +0.35..+0.39  (ventricular)
    ADHD    Hippocampus -0.21..-0.23, Right-Caudate -0.21, Accumbens -0.18

**ANSWER: the axes are disease-specific, decisively.** Autism and ADHD are
essentially orthogonal (-0.138). The E48 rotation is about Alzheimer's, not about
illness in general.

**Magnitude matters too: AD's deviation is 7-8x larger than autism's or ADHD's**
(3.10 vs 0.43 vs 0.37). Brain-age-style deviation scoring works in AD partly
because the effect is enormous by neuroimaging standards.

## CAVEATS, which are heavy here
1. ABIDE 6-64 (median 15) and ADHD-200 7-22 (median 11) against ADNI 50-91. The
   normative model is DEVELOPMENT in the young cohorts and AGING in ADNI. Every
   tier-1 comparison involving ADNI is confounded with life stage and cannot be
   attributed to the disease. Tier 2 is free of this.
2. Only three regions are shared with ADNI — ADNIMERGE ships six and no full
   segmentation. Three dimensions is thin.
3. WholeBrain is SupraTentorial in the young cohorts and BrainSegNotVent in ADNI;
   not the same quantity.
4. ADHD cases pool three subtypes. Different FreeSurfer versions across cohorts.

## LITERATURE — NOT NOVEL
[ENIGMA ADHD/ASD/OCD, Am J Psychiatry 2020, n=12,198] already compares effect-size
profiles across ADHD and ASD explicitly to separate shared from distinct effects,
at ten times our sample size. Their ASD subcortical effects (pallidum -0.08,
putamen -0.10, amygdala -0.08, accumbens -0.13) are the same kind of measurement.
Tier 2 is a small-sample reproduction of established work.
**What is not covered: the same framework extended to Alzheimer's** — but our
tier 1 cannot deliver it cleanly because of caveat 1. To make that claim would
need an adult psychiatric cohort age-matched to ADNI, which we do not have.


---

# E52 — DOES THE GAP PREDICT HOW FAST THE BRAIN WILL ACTUALLY CHANGE?

The interpretive foundation of the whole field: a positive gap means "this brain
is aging faster". Vidal-Pineiro 2021 (PMC8580481) reported no association between
the cross-sectional gap and the longitudinal rate of brain change in healthy
adults. Tested here, with the two things that answer needs.

Design: 713 subjects with >=4 scans and >=3 FUTURE scans. **Gap from scan 1, rate
fitted through scans 2..n only — no shared scan.** Rate = the 6-region rate vector
projected onto the healthy-aging direction, in CN-SD per year. CN-trained,
bias-corrected, out of fold by subject.

## MEASURABILITY, established before interpreting anything
    rate, split-half over future visits (n=212)  r = 0.865 -> reliability 0.927
    gap, within-subject noise 0.89 y vs between-subject 3.59 y -> reliability 0.939
    ATTENUATION CEILING 0.933
Both quantities are measured well. **A null here is a biological statement, not a
measurement failure.** This is the check the published null needed.

## THE PREDICTED ARTEFACT DID NOT MATERIALISE
Noise in a shared baseline scan inflates the gap and deflates the rate, which
manufactures a negative correlation. Measured:
    rate shares the baseline scan   r = +0.406
    rate from scans 2..n only       r = +0.396
A 0.010 artefact. Worth checking, not worth worrying about. Reported because the
prediction was made in advance and was wrong.

## THE RESULT — IT DEPENDS ENTIRELY ON WHO YOU ASK
    CONTROLS  n=288   gap -> future rate   r = +0.070  CI [-0.055, +0.186]  NULL
    MCI       n=401   gap -> future rate   r = +0.415  CI [+0.328, +0.492]  REAL
    MCI never-converted n=226              r = +0.186  CI [+0.020, +0.344]  REAL

★ **The control null REPLICATES Vidal-Pineiro, independently, in ADNI, with a
measurability ceiling of 0.93.** With n=288 the CI excludes anything stronger than
|r| = 0.19. The true value is near zero, and that is now a fact about brains and
not about noise.

★ **But in MCI the same gap predicts the future rate at r = +0.42.** Among MCI who
never converted it falls to +0.19 — so most of the MCI association is conversion,
and a small real part survives conversion.

**The reading: the gap is a LEVEL, not a SPEED.** It predicts future change only
where a disease process is present to be predicted, and it predicts it because it
indexes disease burden, not because it measures a rate of aging. Remove the
disease and the predictive relation disappears entirely.

## TRAIT, NOT SPEED — the confirming evidence
    between-subject variance of the gap  12.90 y^2
    within-subject residual variance      0.79 y^2
    -> 94% of the variance in a person's gap is the PERSON, not the moment
    controls: baseline gap vs the future rate OF THE GAP ITSELF   r = +0.027
    MCI:      same                                                r = +0.393
A healthy person's gap sits where it sits and stays there.

## WHERE E48 DOES NOT HELP
corr(E48 component, brain-age gap) = +0.932 — they are nearly the same number, which
makes E48's classification advantage more impressive, not less. But at THIS task the
component is no better:
    CN  -0.008 p=0.727   MCI +0.014 p=0.459   MCI non-conv -0.021 p=0.472
**E48's advantage is specific to disease classification and prognosis. It does not
extend to predicting structural change.** A real limit on the claim, recorded here.

## LITERATURE
Both halves exist separately. Healthy null: Vidal-Pineiro 2021. MCI association:
Franke & Gaser 2013 (PLOS One 0067346) report baseline BrainAGE associated with
accelerated atrophy in MCI, 10% greater AD risk per year of gap. A targeted search
did not find the two put together, nor the level-versus-rate framing, nor a
measurability ceiling attached to the healthy null.
**Contribution: the dissociation in one cohort with one pipeline, plus the ceiling
that makes the null mean something.** Modest, verifiable, and it bears directly on
how every brain-age paper words its interpretation.


---

# E53 — THE UNIT AND THE DIMENSION OF THE NUMBER ITSELF

## PART A — IS ONE YEAR OF GAP WORTH ONE YEAR OF AGE?
Both in one Cox model, so the comparison is of like with like.

    MCI -> dementia   n=783, 305 events, age 55-89
      chronological age             HR 1.020 per year [1.005, 1.036]
      brain-age gap                 HR 1.119 per year [1.068, 1.173]
      hippocampal deviation (1 SD)  HR 1.306          [1.141, 1.496]
      -> ONE YEAR OF GAP = 5.6 YEARS OF AGE   95% CI [2.6, 23.0]
      -> one SD of hippocampal loss = 2.4 years of gap

    CN -> MCI or dementia   n=632, 125 events, age 55-90
      chronological age             HR 1.059 per year [1.025, 1.094]
      brain-age gap                 HR 1.023 per year [0.946, 1.107]   <- CROSSES 1
      hippocampal deviation (1 SD)  HR 1.359          [1.043, 1.769]

★ **The unit is not time.** In MCI a year of gap carries 5.6 years of age worth of
hazard. "This brain is 5 years older" understates by roughly sixfold.

★★ **And in CONTROLS the gap predicts incident MCI/dementia not at all** — the CI
crosses 1 — while a single raw hippocampal deviation predicts it fine (HR 1.36 per
SD). In healthy people the whole brain-age machine is beaten by just looking at
the hippocampus. This is the clinical twin of E52: same population, same verdict,
different outcome. Two independent endpoints now agree that the gap is empty in
controls.

**The multiplier is also inconsistent ACROSS populations** — 5.6 in MCI, undefined
in CN because the numerator is zero. A quantity whose conversion factor to years
changes with the population is not measured in years at all.

CAVEAT stated, not hidden: ADNI's age range is narrow and its subjects are a
memory-clinic sample, so chronological age is a weak predictor and that inflates
the ratio. The direction is safe; the exact multiplier is cohort-dependent.

## PART B — HOW MANY DIMENSIONS DOES THE DEVIATION NEED?
PCA on CONTROLS ONLY, k = 1..6, rotation refitted inside every bootstrap draw.

    AD vs CN at baseline (n=1063: 303 AD, 760 CN)
      gap 0.914 | k1 0.926 | k2 0.925 | k3 0.931 | k4 0.941 | k5 0.940 | k6 0.949
      k=1 minus gap   +0.0132  [+0.0081, +0.0191]  p=0.000  REAL
      k=2 minus k=1   -0.0001  [-0.0020, +0.0032]  p=0.720  -
      k=3 minus k=2   +0.0050  [+0.0005, +0.0112]  p=0.040  REAL
      k=4 minus k=3   +0.0096  [+0.0046, +0.0161]  p=0.000  REAL
      k=5 minus k=4   -0.0005  [-0.0019, +0.0012]  p=0.373  -
      k=6 minus k=5   +0.0078  [+0.0024, +0.0135]  p=0.000  REAL

    MCI -> AD conversion (n=783, 305 events)
      gap 0.732 | k1 0.752 | k2 0.752 | k3 0.751 | k4 0.754 | k5 0.753 | k6 0.756

★ **DIAGNOSIS IS MULTI-DIMENSIONAL; PROGNOSIS IS ESSENTIALLY ONE-DIMENSIONAL.**
Classification keeps gaining out to k=6 (+0.023 over k=1). Conversion prediction
gains +0.020 going from the gap to k=1 and then +0.004 over the next five
components combined.

★ **And the useful directions are NOT the high-variance ones.** Component 2 — the
second largest source of variance among controls — adds exactly nothing (p=0.720),
while components 3, 4 and 6 each add real signal. PCA orders by variance in
healthy people, and variance in healthy people is not relevance to disease. This
is E48's thesis appearing a second time one level down: **the ordering criterion
is wrong for the same reason the target was.**

Interpretation for the proposal: for PROGNOSIS, one number is enough and E48 names
which one. For DIAGNOSIS, no scalar suffices and the output should be a vector.

---

# E54 — DOES E48 SURVIVE WHEN THE VOLUMES COME FROM THE IMAGE?

Rerun of the whole E48 pipeline with the bottleneck CNN's predicted volumes in
place of FreeSurfer's, same 1114 subjects, same normative model, same
controls-only rotation. AD vs CN n=612 (190 AD, 422 CN).

    image-derived   gap 0.834 | one component 0.834 | six deviations 0.834
    FreeSurfer      gap 0.927 | one component 0.943 | six deviations 0.954
    component minus gap:  image -0.0002 p=0.924    FreeSurfer +0.0170 p=0.000

**THE TEST DID NOT ACTUALLY RUN.** All three image arms are identical to three
decimals and the controls-only loadings came out +0.42 +0.42 +0.42 +0.42 -0.33
+0.42 — near-uniform. The diagnosis confirms it:

    mean |correlation| between regional deviations   image 0.908   FreeSurfer 0.495
    variance in the first direction                  image 92.8%   FreeSurfer 58.7%
    EFFECTIVE NUMBER OF DIMENSIONS                   image 1.16    FreeSurfer 2.60

**The network did not produce six regional measurements. It produced one global
atrophy score written six times.** A rotation has nothing to rotate. So E54 is not
evidence against E48 — it is a measurement of the engineering obstacle between
E48 and anything deployable.

Per-region agreement with FreeSurfer: Ventricles r=0.911, WholeBrain 0.623,
Hippocampus 0.618, MidTemp 0.569, Fusiform 0.540, Entorhinal 0.475. The big
easily-seen structure is captured; the small ones are not.

**WHY IT COLLAPSED, and the fix to test:** the six volumes are dominated by a
shared head-size-and-global-atrophy factor. The old kernel standardised each
target, which equalises SCALE but not the shared FACTOR, so MSE is minimised by
learning the dominant direction and stopping. The fix is to train on WHITENED
normative deviations, where every direction carries equal variance in the loss and
the network cannot buy a low loss with the global factor alone. That is the next
GPU run.


---

# E53b — THE COX INCREMENT THAT WOULD NOT CONVERGE, NOW RESOLVED

exp53's Cox bootstrap returned nothing: six correlated components plus age make
the partial likelihood ill-conditioned once a bootstrap draw duplicates rows.
Fixed with a heavier ridge (penalizer 1.0) and a 3-fold split. **250/250 draws
converged**, so the result is not a survivor-of-convergence artefact.

    MCI -> AD conversion, n=783, 305 events
      gap 0.731 | k=1 0.751 | k=2 0.751 | k=6 0.748
      k=1 minus gap   +0.0194  95% CI [+0.0089, +0.0283]  p=0.000  REAL
      k=2 minus k=1   +0.0001  95% CI [-0.0043, +0.0044]  p=0.976  -
      k=6 minus k=1   -0.0003  95% CI [-0.0113, +0.0119]  p=0.896  -

★★ **PROGNOSIS IS EXACTLY ONE-DIMENSIONAL.** Going from the gap to the right
single number is worth +0.019. Going from one number to six is worth nothing at
all, twice over (k=2 and k=6 both flat). Set against E53's classification result -
where k=6 beats k=1 by +0.023 with real increments at k=3, k=4 and k=6 - this is a
clean dissociation:

    DIAGNOSIS   needs four to six dimensions; no scalar suffices
    PROGNOSIS   needs exactly one, and E48 names which one

**This is the sharpest form the proposal has taken.** For the clinically useful
task - who will convert - the answer is not "use more features" and not "use deep
learning". It is: keep one number, and choose it correctly. Brain age keeps one
number and chooses it by predicting chronological age, which costs +0.019 C-index
for nothing.

**Third independent bootstrap of the E48 effect on this endpoint**: E50 gave
+0.015 [+0.004, +0.025], E53b gives +0.019 [+0.009, +0.028]. Same direction, same
order of magnitude, different penalisation and fold structure.


---

# E55 / brainage-whiten — THE TARGET HYPOTHESIS FAILED, AND E54 WAS WRONG

GPU kernel `mohammadparsakavian/brainage-whiten`. One ResNet-18 trunk, three
heads differing ONLY in what they are asked to predict, 5-fold by subject,
1132 subjects (CN 421, MCI 449, AD 190).

## THE HYPOTHESIS IS DEAD
E54 blamed the collapse on the target: six volumes share a huge head-size and
global-atrophy factor, so standardising each target equalises scale but not the
shared factor. Predicted fix: train on normative deviations, or on WHITENED
deviations where every direction carries equal variance in the loss.

    EFFECTIVE DIMENSIONS   head V (raw volumes) 1.28
                           head Z (deviations)  1.26
                           head W (whitened)    1.26
                           FreeSurfer           2.62

**Whitening moved nothing. Normative targeting moved nothing.** Three targets
spanning the full range of the hypothesis produce the same 1.27. The collapse is
not a target-design problem, and at n~900 no reweighting of the loss rescues it.
Prediction made in advance, tested, refuted. Recorded as such.

Per-region agreement with FreeSurfer z is likewise flat across heads:
Ventricles 0.87-0.89, WholeBrain 0.59-0.67, Hippocampus 0.51-0.53,
MidTemp 0.50-0.57, Fusiform 0.48-0.52, Entorhinal 0.44.

## BUT E48 SURVIVES THE IMAGE PIPELINE ANYWAY — E54's VERDICT IS WITHDRAWN
The kernel scored in-sample and printed no interval, so E55 re-scored it locally
with cross-validated AUC by subject and a 500-draw subject-level bootstrap, the
rotation refitted inside every draw. AD vs CN, one visit per subject, n=611
(190 AD, 421 CN):

    arm                      gap     1 comp   six     component minus gap
    head V raw volumes       0.788   0.812   0.816   +0.0231 [+0.0112,+0.0337] p=0.000
    head Z deviations        0.788   0.810   0.807   +0.0208 [+0.0113,+0.0311] p=0.000
    head W whitened          0.788   0.809   0.806   +0.0204 [+0.0107,+0.0304] p=0.000
    FreeSurfer reference     0.914   0.941   0.951   +0.0272 [+0.0197,+0.0352] p=0.000

★ **The effect holds in every image arm, with confidence intervals overlapping
FreeSurfer's.** E54 concluded the opposite (-0.0002, p=0.924) scoring the OLD
bottleneck network, whose effective dimension was 1.16 and whose image gap
reached 0.834. **That conclusion was network-specific and is withdrawn.** The
correct statement: the effect survives a deployable image pipeline, but the image
pipeline itself costs about 0.13 AUC in absolute terms (0.788 vs 0.914 on the
gap, 0.812 vs 0.941 on the component). That absolute cost is the open engineering
problem; the E48 claim is not conditional on FreeSurfer.

## THE LOADINGS ARE THE NEXT LEAD
Every controls-only rotation, on FreeSurfer and on all three image heads, came
back near-uniform: about +0.43 on each atrophy region, about -0.27 on ventricles.
If a plain sign-corrected AVERAGE does what the fitted component does, E48 reduces
from "fit a PCA on your controls" to "weight the regions equally instead of by how
well they predict chronological age" - no rotation to fit, nothing to transfer.
Testing that in E56.


---

# E56 — YOU DO NOT NEED THE ROTATION. THE AVERAGE IS THE ANSWER.

Every controls-only rotation fitted anywhere in this project - FreeSurfer and all
three image heads - came back near-uniform (about +0.43 per atrophy region, about
-0.27 on ventricles). So: does a plain sign-corrected AVERAGE of the six
deviations do what the fitted component does?

    correlation of the uniform average with the fitted component   r = 0.9959
    correlation of the uniform average with the brain-age gap      r = 0.9522

    ENDPOINT 1  AD vs CN, n=1063
      brain-age gap 0.914 | uniform average 0.928 | fitted component 0.926
        uniform minus gap        +0.0159 [+0.0099, +0.0221] p=0.000  REAL
        uniform minus component  +0.0026 [+0.0004, +0.0049] p=0.020  REAL

    ENDPOINT 2  MCI -> AD conversion, n=783, 305 events, 250/250 draws converged
      brain-age gap 0.731 | uniform average 0.750 | fitted component 0.751
        uniform minus gap        +0.0184 [+0.0082, +0.0291] p=0.000  REAL
        uniform minus component  -0.0011 [-0.0038, +0.0015] p=0.488  tied

★★★ **THE UNIFORM AVERAGE MATCHES THE FITTED ROTATION ON BOTH ENDPOINTS AND
SLIGHTLY BEATS IT ON CLASSIFICATION.** It wins on classification because it has no
parameters to estimate and therefore no estimation variance. The PCA was never
doing anything except discovering that the weights should be equal.

**This collapses the whole proposal into one sentence:**

    The brain-age gap's only defect is that it weights regions by how well they
    predict chronological age. Weight them equally instead.
      +0.016 AUC on diagnosis, +0.018 C-index on prognosis.

Why this is strictly better than E48 as a claim:
  - nothing to fit, so no rotation to transfer and no cohort to fit it on
  - a one-line change to any existing normative-modelling pipeline
  - no estimation variance, which is why it wins on the smaller endpoint
  - survives the image pipeline (E55) since the image loadings were uniform too
  - consistent with E53b: prognosis is exactly one-dimensional, and this is it

Everything measured since E48 now points at the same place. E49: patients are
~94% along the healthy direction, so a shared axis exists. E50: the healthy
training population is correct. E51: the axis is disease-specific. E53b: one
dimension is all prognosis needs. E55: it survives image-derived volumes. E56: the
dimension is simply the average.


---

# E57 — THE AVERAGE AGAINST THE SUMMARY THE LITERATURE ALREADY USES

Five summaries of the SAME six normative deviations, so nothing differs except
how six numbers become one.

    ENDPOINT 1  AD vs CN n=1063        ENDPOINT 2  MCI->AD n=783, 305 events
      uniform average    0.928           uniform average    0.750
      fitted component   0.926           fitted component   0.751
      brain-age gap      0.914           brain-age gap      0.731
      outlier count      0.893           outlier count      0.719
      mean |z|           0.890           mean |z|           0.718

    uniform minus outlier count  +0.0357 [+0.0212,+0.0499] p=0.000 | +0.0303 [+0.0095,+0.0491] p=0.000
    uniform minus mean |z|       +0.0399 [+0.0270,+0.0516] p=0.000 | +0.0314 [+0.0172,+0.0488] p=0.000
    outlier count minus gap      -0.0198 [-0.0332,-0.0052] p=0.016 | -0.0119 [-0.0337,+0.0103] p=0.296

★ **The sign-corrected mean dominates every alternative summary**, including both
established ones. And the published normative summary - outlier count - LOSES to
the brain-age gap on classification. So this is not brain age versus normative
modelling; it is that both of their usual summaries are beaten by the average.

**The mechanism is visible in what the two losers discard.** mean |z| keeps
magnitude and throws away direction. Outlier count keeps direction and throws away
magnitude. The uniform average keeps both, and that is the whole difference.

## CAVEAT THAT MUST TRAVEL WITH THIS RESULT
Outlier count is degenerate at six regions:

    CN        mean 0.19 of 6 flagged, 86% of subjects have ZERO
    MCI       mean 0.85 of 6 flagged, 59% have ZERO
    Dementia  mean 2.22 of 6 flagged, 19% have ZERO

Most subjects score exactly zero, so the statistic has almost no range. It was
designed for parcellations with hundreds of regions, where a count has room to
vary. **Our comparison is fair for a six-region pipeline and says nothing about
outlier count in its intended high-dimensional setting.** Stating otherwise would
be a misrepresentation of somebody else's method.

## THE CLAIM, IN ITS FINAL FORM
Among summaries of the same normative deviations, the sign-corrected mean beats
  the brain-age gap        +0.016 AUC, +0.018 C-index
  the fitted PCA rotation  +0.003 AUC, tied on C-index
  the outlier count        +0.036 AUC, +0.030 C-index
  mean |z|                 +0.040 AUC, +0.031 C-index
with nothing to fit, nothing to transfer, and one line of code.

STILL TO VERIFY: (1) does the advantage survive scanner and cohort shift, as E48's
did - running now as E58; (2) does it hold with a richer parcellation, where
outlier count is no longer degenerate - ADNIMERGE ships six regions and no full
segmentation, so this needs data we do not have.


---

# E58 — DOES THE UNIFORM AVERAGE TRANSFER?

The average has no fitted parameters, so it cannot overfit a cohort. But the
normative model underneath it IS fitted on training controls, so the advantage
still has to be measured on held-out cohorts. Three shifts, everything fitted on
the training cohort only.

    0. random half/half        gap 0.910 | uniform 0.929 | +0.0188 [+0.0038,+0.0332] p=0.015 HOLDS
    1. ADNI1 -> ADNI2/GO/3     gap 0.926 | uniform 0.940 | +0.0138 [+0.0072,+0.0216] p=0.000 HOLDS
    2. 1.5 T -> 3 T            gap 0.930 | uniform 0.947 | +0.0170 [+0.0066,+0.0272] p=0.000 HOLDS
    3. FS 4.3/5.1 -> 6.0       gap 0.929 | uniform 0.936 | +0.0066 [-0.0025,+0.0163] p=0.170 NO

**Three of four hold.** Recruitment wave and field strength - the two shifts that
matter for deployment - are clean and the effect size is undiminished.

**The FreeSurfer version shift does not hold, and it is not simply a power
problem.** That test has the smallest AD group (60 vs 154/163/87) but its CI is no
wider than the others (half-width 0.0094 against 0.0072-0.0147); the POINT
ESTIMATE is genuinely about half the size. The likely reason is that FS 6.0
changed the segmentation of these particular structures, which changes the
correlation structure the average relies on. **Honest statement: the advantage is
established across scanner and cohort, not yet across segmentation software.**

# WHERE THE PROPOSAL STANDS

One claim, tested seven ways, surviving all but one:

    the brain-age gap weights regions by how well they predict chronological age.
    Weight them equally instead.

    +0.016 AUC diagnosis, +0.018 C-index prognosis      E56
    beats the fitted PCA rotation it replaces            E56  (+0.003 / tied)
    beats outlier count and mean |z|                     E57  (+0.036 / +0.030)
    survives image-derived volumes                       E55  (+0.020 to +0.023)
    survives recruitment wave and field strength         E58
    not established across FreeSurfer versions           E58
    one dimension is all prognosis needs                 E53b
    the healthy training population is correct           E50
    the axis is disease-specific, not generic illness    E51
    patients are ~94% along the healthy direction        E49
    and the gap is a level, not a speed                  E52 / E53a


---

# E60 — TWO SPECIFICATION ATTACKS

## A — WAS THE BRAIN-AGE MODEL A STRAW MAN?  No.
Seven age models, weak to strong, each with its CN age MAE beside its disease AUC.

    ridge + bias correction (ours)   MAE 4.196   AUC 0.914
    ridge, NO bias correction        MAE 4.196   AUC 0.914
    plain linear regression          MAE 4.196   AUC 0.911
    random forest                    MAE 4.225   AUC 0.891
    gradient boosting                MAE 4.370   AUC 0.870
    gradient boosting, deeper        MAE 4.639   AUC 0.837
    support vector regression        MAE 4.260   AUC 0.818
    UNIFORM AVERAGE (no age model)               AUC 0.928

    uniform minus the BEST of them  +0.0156 [+0.0098, +0.0216] p=0.000  HOLDS

★ **No brain-age model we could build beats the average.** Ridge was already the
strongest; every nonlinear alternative is worse. The straw-man objection is closed.
Bias correction changes the AUC by nothing (0.914 either way), because age is
already a covariate in the scoring model and the correction is linear in age.

**On the accuracy-versus-sensitivity trade-off**: corr(age MAE, disease AUC) =
-0.627 across the seven, i.e. the OPPOSITE of Schulz et al. (PLOS Biology 2025),
who report that less accurate age models detect disease better. **This is not a
refutation of them.** Seven models spanning 0.44 years of MAE is a weak test, and
our weak models are weak at both tasks for reasons that have nothing to do with
age accuracy. Their design varies accuracy deliberately; ours varies model family.
Recorded as a non-replication in a design that was not built to test it.

## B — DOES THE NORMATIVE MODEL'S FORM MATTER?  YES, IN ONE SPECIFIC WAY.

    linear in age     uniform 0.929   vs our gap +0.0166 [+0.0110,+0.0228] p=0.000 HOLDS
    quadratic (ours)  uniform 0.928   vs our gap +0.0156 [+0.0100,+0.0219] p=0.000 HOLDS
    cubic             uniform 0.929   vs our gap +0.0160 [+0.0103,+0.0225] p=0.000 HOLDS
    ICV DROPPED       uniform 0.917   vs our gap +0.0044 [-0.0072,+0.0151] p=0.440 NO
    volumes / ICV     uniform 0.912   vs our gap -0.0010 [-0.0083,+0.0068] p=0.832 NO

★★ **THE ADVANTAGE REQUIRES HEAD SIZE TO BE HANDLED AS A COVARIATE, NOT AS A
DIVISOR.** The shape of the age term is irrelevant - linear, quadratic and cubic
all give the same answer. But dropping ICV costs the advantage entirely, and
dividing volumes by ICV costs it entirely: the uniform arm falls from 0.928 to
0.912 and ties the gap.

This is a real constraint and it must travel with the claim, stated explicitly,
because **many published pipelines use ICV-normalised volumes**. The requirement
is one line in a methods section, but omitting it would make the result
irreproducible for anyone following the common convention.

Why it probably happens: dividing by ICV forces a single multiplicative head-size
correction on every region, while ICV-as-covariate lets each region have its own
head-size coefficient. The residual head-size signal that division leaves behind
is shared across regions, so averaging amplifies it rather than cancelling it -
which is exactly the mechanism that made the image-derived deviations collapse in
E54/E55. Same failure mode, different cause.


---

# E59 — THE FOUR ATTACKS. ONE LANDS, AND IT NARROWS THE CLAIM.

## A — IS IT JUST THE HIPPOCAMPUS?  On diagnosis, YES.

    AD vs CN   n=1063    gap 0.914 | hippocampus alone 0.930 | uniform 0.928
      uniform minus hippocampus  -0.0002 [-0.0120,+0.0104] p=0.990  TIED
      hippocampus minus gap      +0.0160 [+0.0054,+0.0271] p=0.000  REAL

    MCI vs CN  n=1627    gap 0.716 | hippocampus alone 0.722 | uniform 0.710
      uniform minus hippocampus  -0.0124 [-0.0258,+0.0014] p=0.085  tied, trending AGAINST

    MCI -> AD  n=783, 305 events   gap 0.731 | hippocampus 0.719 | uniform 0.750
      uniform minus hippocampus  +0.0316 [+0.0112,+0.0492] p=0.000  REAL
      hippocampus minus gap      -0.0132 [-0.0299,+0.0048] p=0.128  tied

★★★ **THE DIAGNOSIS HALF OF THE CLAIM IS DEAD.** On AD vs CN the uniform average
ties plain hippocampal volume, and on MCI vs CN it is slightly worse. Anyone
reviewing "0.928 versus 0.914" would answer it with one line of code. Withdrawn.

★★★ **THE PROGNOSIS HALF SURVIVES AND IS STRONGER THAN BEFORE.** On conversion the
average beats the gap (+0.018) AND hippocampal volume (+0.032). Since E53b showed
prognosis needs exactly one dimension, and E59-E shows uniform is within 0.005 of
a supervised fit, the average is essentially THE optimal single number for
prognosis - and it is not the hippocampus.

Side result worth its own sentence: **the brain-age gap loses to a single region on
diagnosis** (hippocampus +0.016, p=0.000).

## B — LEAVE ONE REGION OUT (AD vs CN), gap refitted on the same five

    without Hippocampus   uniform 0.904  gap 0.870   +0.0353 p=0.000 HOLDS
    without Entorhinal    uniform 0.916  gap 0.909   +0.0075 p=0.000 HOLDS
    without Fusiform      uniform 0.930  gap 0.913   +0.0181 p=0.000 HOLDS
    without MidTemp       uniform 0.924  gap 0.908   +0.0162 p=0.000 HOLDS
    without Ventricles    uniform 0.931  gap 0.914   +0.0187 p=0.000 HOLDS
    without WholeBrain    uniform 0.939  gap 0.939   +0.0002 p=0.912 NO

## C — DOES THE ADVANTAGE SCALE WITH REGION COUNT?  (AD vs CN)
    2 regions  +0.0104  positive in 73% of 15 subsets
    3 regions  +0.0139  positive in 75% of 20
    4 regions  +0.0151  positive in 80% of 15
    5 regions  +0.0151  positive in 83% of 6
    6 regions  +0.0145  positive in 100%
Magnitude plateaus after four regions; RELIABILITY rises monotonically. Expect a
hundred-region parcellation to be more consistently positive, not larger.

## E — HEADROOM AGAINST A SUPERVISED FIT ON THE SAME SIX
    AD vs CN    uniform 0.928  supervised 0.949  headroom +0.021
    MCI vs CN   uniform 0.710  supervised 0.737  headroom +0.027
    conversion  uniform 0.750  supervised 0.755  headroom +0.005

**Prognosis has nothing left on the table. Diagnosis has plenty** - and this is the
same dissociation as E53b, measured a different way.

---

# E61 — WHOLE-BRAIN VOLUME IS MOST OF THE MECHANISM

What the CN-fitted age model actually weights, standardised:

    ICV          +3.445        <- largest
    WholeBrain   -3.043
    Hippocampus  -1.451
    SEX          +0.835
    Ventricles   +0.650
    MidTemp      -0.350
    Entorhinal   -0.283
    Fusiform     -0.140        <- smallest

★ **The age model spends almost all its weight on head-size-adjusted total brain
volume, and almost none on the medial temporal regions that carry the Alzheimer
signal.** Entorhinal and Fusiform - the two structures that go first in AD - get
the two smallest coefficients of the eight. That is the misallocation, made visible.

Removing WholeBrain, on MCI -> AD conversion:

    all six regions      gap 0.731  uniform 0.750   +0.0184 [+0.0090,+0.0270] p=0.000
    WITHOUT WholeBrain   gap 0.742  uniform 0.752   +0.0094 [+0.0021,+0.0167] p=0.020

**On diagnosis, removing WholeBrain closed the deficit entirely (p=0.912). On
prognosis it closes about HALF of it and the advantage still holds (p=0.020).**
So whole-brain volume is the single biggest source of the misallocation but not
the only one; the residual comes from the ICV weight and from under-weighting
entorhinal and fusiform. Stated as measured, not rounded up to "the whole story".


---

# E62 / E62b — THE RICH ATLAS OVERTURNS THE HEADLINE, AND EXPLAINS IT

TADPOLE_D1_D2 (Kaggle: meraja99/tadpole-d1-d2final) gives the same ADNI subjects with
98 FreeSurfer regions per pipeline. The LONI-only dictionary was reconstructed:
UCSF ST codes are alphabetical over cortical+subcortical names and every right
region sits 59 codes after its left twin; six anchors verified against the
ADNIMERGE summaries in the same file (r >= 0.9965, median ratio 1.000).
Module: pilot/tadpole_atlas.py. n=1369 with all 98 regions (CN 442, MCI 684, AD 243).

## E62 — on 98 neutral regions the uniform average LOSES

    AD vs CN   gap 0.881 | uniform 0.857 | hippocampus 0.922 | outlier count 0.897 | ceiling 0.954
      uniform minus gap  -0.025 [-0.044,-0.004] p=0.007
    MCI vs CN  gap 0.688 | uniform 0.663 | hippocampus 0.751 | ceiling 0.756
    MCI->AD    gap 0.720 | uniform 0.709 | hippocampus 0.716 | ceiling 0.740   (all tied)

Same in the LONGITUDINAL pipeline (n=769) and across FS 4.3 -> 5.1 (uniform - gap
-0.019, ns). Outlier count is no longer degenerate (41% of CN score zero, 2% of AD).
**Prognosis is NOT one-dimensional on 98 regions:** k=5 minus k=1 +0.027 p=0.010,
k=20 minus k=1 +0.049 p=0.000. E53b's "exactly one dimension" was a property of the
six-region space.

## THE MECHANISM HOLDS, AND GETS SHARPER

Share of absolute weight by region class, age model vs a disease model:

    MTL 16.0% vs 33.7% | cortex 44.3 vs 42.8 | subcortical 12.6 vs 6.5 |
    callosum 9.9 vs 3.9 | ventricles 10.7 vs 7.9 | brainstem 3.4 vs 0.5

Age model's heaviest regions: brainstem, third ventricle, three callosal segments,
superior parietal, precentral, accumbens, inferior lateral ventricle, paracentral;
hippocampus only 11th. Disease model's: L/R hippocampus, L/R entorhinal, amygdala,
fusiform - all medial temporal.

★ **Rank correlation between the two weightings: -0.031. Brain age looks at the brain
in a way ORTHOGONAL to how Alzheimer's damages it.**

## E62b — SELECTION, NOT WEIGHTING (same 1369 people throughout)

Uniform weight on the medial temporal lobe: six ADNIMERGE regions 67%, 98 regions 16%.

    arm                         AD/CN   MCI/CN   MCI->AD
    gap (6)                     0.904   0.712    0.734
    uniform (6)                 0.931   0.721    0.753
    gap (98)                    0.881   0.688    0.720
    uniform (98)                0.857   0.663    0.709
    uniform, 16 MTL regions     0.940   0.728    0.742
    uniform, 82 non-MTL         0.811   0.636    0.687
    hippocampus                 0.922   0.751    0.716

    uniform(6) - gap(6)          +0.027 p=0.000 | +0.019 p=0.000   -> NOT a sample effect
    MTL uniform - gap(98)        +0.058 p=0.000 | +0.022 p=0.060
    MTL uniform - uniform(98)    +0.083 p=0.000 | +0.034 p=0.000   -> selection is the lever
    MTL uniform - hippocampus    +0.016 p=0.040 | +0.024 p=0.010
    gap(98) - gap(6)             -0.022 p=0.000 (AD/CN)

★★★ **The "uniform weighting" finding (E56-E58) was a finding about ADNI's REGION
SELECTION.** ADNIMERGE's six regions were chosen for Alzheimer's; averaging them puts
two thirds of the weight on the medial temporal lobe. On a neutral atlas the same
rule puts 16% there - exactly the share the age model gives - and loses. Restricting
the average to the 16 MTL regions recovers and exceeds the win.
**A widely used six-region ADNI summary has disease knowledge baked into its feature
choice; any "unsupervised" method evaluated on it inherits that knowledge.**

★ **Schulz et al. (PLOS Biol 2025) trade-off, measured on the same people:** 6-region
brain age MAE 3.79 y, AD/CN AUC 0.904; 98-region MAE 3.65 y, AUC 0.881. More regions
read age better and Alzheimer's worse (-0.022, p=0.000). This supersedes E60-A's
non-replication, whose design varied model family, not inputs.

---

# E63 — THE DUAL-REFERENCE MODEL (the thesis author's idea)

Model H: age model on controls, gap_H > t -> sick. Model D: age model on AD,
gap_D < t -> healthy. Agree -> decide; contradict -> flag for a physician.

    mean gap_H / gap_D:  CN -0.01 / -4.66 | MCI +2.95 / -2.77 | AD +6.89 / +0.01
    corr(gap_H, gap_D) = 0.794

★ **The core intuition is CONFIRMED:** an AD-trained model reads a healthy brain as a
young patient - healthy people land at -4.66 years, patients at 0.

AD vs CN, thresholds set on one half and scored on the other:

    literal (zero thresholds)   decides 57.4%, flags 42.6% | dual 92.4% vs single at same flag rate 93.0%
    calibrated (90%)            decides 73.8%, flags 26.2% | dual 88.3% vs single 89.8%
    bootstrap, calibrated: dual minus single -1.4 points [-4.3, +2.1] p=0.380

**Not better than one model abstaining on the same number of borderline people.**
Literal zero thresholds flag 43% because a zero cut sits at the healthy median.

MCI sorted into four groups gives a clean risk ladder: HEALTHY (agree) 22.5% convert,
FITS BOTH 30.3% (HR 1.63), FITS NEITHER 42.9% (HR 2.31), SICK (agree) 61.4% (HR 4.77);
amyloid-positive 49.5 / 56.5 / 50.0 / 76.0%. The flag behaves as a review flag
should. But in Cox the flags add nothing beyond gap_H (p=0.31, 0.99) and gap_D adds
nothing (p=0.56). Multivariate dual typicality: healthy-like 13.5% vs AD-like 64.6%
(HR 11.61) - matched by a single uniform score cut to the same group sizes (HR 11.50).
Continuous dual Mahalanobis distances are significant beyond the uniform average
(p=0.012, p<0.001) but add 0.001 of C.

PRE-REGISTERED SNAP prediction (fits-neither enriched for amyloid-negative):
**NOT supported** - 46.9% vs 39.5%, OR 1.36, p=0.36 (gap version OR 1.50, p=0.53).

★★ **Why it cannot beat one model, and this is itself the finding:** the two gaps
correlate 0.79 because patients travel ~94% along the healthy axis (E49). With one
axis, two references are two thresholds on the same ruler; a contradiction is exactly
"between the thresholds", which single-model abstention already captures.
**The value of a second reference is bounded by how OFF-AXIS a disease is.** For
Alzheimer's that is little. For conditions with orthogonal axes (autism vs ADHD in
E51, cos -0.14) the dual design should be far more informative - a prediction for
future work, not a result.

---

# E64 — WHAT DOES THE GAP ACTUALLY INDEX?

A/T/N (partial r, age+sex): in MCI the gap tracks CSF amyloid (+0.38*), amyloid PET
(+0.24*) and FDG (+0.42*) but NOT CSF p-tau (+0.07) or tau (+0.06); the hippocampus
DOES track p-tau (+0.13*) and tau (+0.13*). **Tau specificity lives in the
hippocampus, not in brain age.**

PRECLINICAL (CN, amyloid+ n=148 vs amyloid- n=303, age/sex adjusted): gap +0.17 SD
[-0.04,+0.36] AUC 0.545; hippocampus AUC 0.514; uniform AUC 0.547. **Every structural
summary is blind to preclinical Alzheimer's.**

COGNITIVE DECLINE IN HEALTHY PEOPLE (>=3 visits over >=2 y, age+sex+education):
mPACC slope - gap +0.165*, hippocampus +0.190*, uniform +0.179*; ADAS13 slope - gap
+0.220*, hippocampus +0.246*, uniform +0.229*. **The gap is NOT empty in controls:** it
predicts cognitive decline while predicting neither structural rate (E52) nor rare
conversion (E53a). Coherent reading: the gap indexes BRAIN RESERVE - a level, not a
rate of aging.

HORIZON (cumulative AUC by year, MCI): gap 0.779/0.785/0.773/0.793 at 1/2/3/5 y;
hippocampus 0.751/0.750/0.758/0.799; uniform 0.793/0.803/0.794/0.823. Point estimates
only; the hippocampus is weaker short-term and catches up by 5 years.

# E64b — TWO MECHANISMS, BOTH CONFIRMED

## THE SEX BIAS IS AN ARTEFACT OF THE STANDARD RECIPE

    standard recipe (bias correction on age only)  healthy men +0.72, women -0.58: +1.31 y [+0.86,+1.72]
    OLS instead of ridge (alpha was only 4.5)      +1.31 y [+0.85,+1.72]  -> not shrinkage
    bias correction on AGE AND SEX                 -0.01 y [-0.47,+0.40]  -> gone
    converters missed by the gap but caught by the hippocampus:
      standard recipe 24 missed, 67% women  ->  age+sex correction 26 missed, 42% women

★★ **The textbook bias correction leaves a 1.3-year sex bias that makes women's brains
look younger, and makes brain age disproportionately MISS WOMEN who go on to
convert.** E64-F found the missed converters were women with small heads (ICV 1.47 vs
1.56 L, p=0.035; male 33% vs 58%, p=0.021) who converted later (4.0 vs 2.1 y,
p<0.0001). Correcting on age AND sex removes both the bias and the female excess.
AD/CN discrimination by sex is unchanged by the correction (men 0.898, women 0.932):
that difference is separate and still to be bootstrapped.

## THE TAU PARADOX IS CSF DILUTION

In healthy people a higher gap went with LOWER CSF amyloid (looks bad) AND LOWER CSF
tau (looks good) - impossible under Alzheimer's biology. Dilution by enlarged CSF
spaces explains it, and every prediction holds:

    biomarker            gap, age+sex    gap, + ventricle size   ventricle size alone
    CSF Abeta42           +0.162*         +0.021                  +0.273*
    CSF p-tau             -0.129*         +0.018                  -0.271*
    CSF total tau         -0.137*         +0.019                  -0.287*
    p-tau / Abeta42       +0.003          +0.019                  -0.025
    t-tau / Abeta42       +0.005          +0.015                  -0.016
    amyloid PET           +0.036          +0.030                  +0.020

★★ **In cognitively normal people, the brain-age gap's association with CSF amyloid is
entirely a dilution artefact of ventricular enlargement.** Raw concentrations move
together, ratios do not, PET does not, and controlling ventricle size removes it.


---

# E65 — A DISEASE CLOCK: SAME INPUTS, TARGET = TIME TO CONVERSION

Trained on 1083 visits of 255 MCI->AD converters, target = years relative to
conversion (-6 to +2), inputs = the six CN-referenced deviations + age, out of fold
by subject. Scored on 783 baseline MCI (305 converted).

    C-index   gap 0.731 | hippocampus 0.719 | uniform (6 ADNI regions) 0.751 | DISEASE TIME 0.739
      disease time minus gap       +0.008 [-0.008,+0.023] p=0.320
      disease time minus uniform6  -0.012 [-0.021,-0.002] p=0.010   <- the plain average wins
      disease time minus hippo     +0.018 [-0.007,+0.042] p=0.140

THE UNIT (held-out converters, actual years remaining regressed on predicted):
    actual = +0.98 + 1.32 x predicted, slope 95% CI [0.89, 1.79], r = 0.35
    mean predicted remaining 1.22 y vs actual 2.60 y; median abs error 0.98 y
    HR per year: disease time 4.14 [3.42, 5.02]  vs  brain-age gap 1.20 [1.17, 1.24]

★ **Changing the target FIXES THE UNIT but not the ranking.** A predicted year of
disease time is a real year (slope CI contains 1), where a year of brain-age gap is
worth 5.6 years of age (E53a). But as a ranker it does not beat a plain average of
the same six inputs (-0.012, p=0.010): supervising on WHEN converters convert is a
different question from WHO converts.
**It inherits brain age's other flaw:** predictions are pulled toward the training
mean (predicted 1.2 y remaining vs actual 2.6 y) - the same regression-to-the-mean
bias that brain age needs a correction step for. The bias belongs to regressing any
noisy target on structure, not to chronological age specifically.


---

# E66 (half cohort) — OASIS-2 through FastSurfer: the external cohort

Kaggle `javidtheimmortal/oasis-2` raw T1, FastSurfer v2.4.2 on T4 (~128 s/scan).
Part 0 (even subjects) complete: 263 scans, 77 subjects at baseline (38 CDR>0,
32 stable CDR-0 reference - only 5 men). Part 1 re-pushed after the session reset.

    diagnosis CDR>0 vs 0:  gap(6-like) 0.749 | gap(91) 0.753 | uniform(6-like) 0.802 |
                           uniform(91) 0.787 | uniform(14 MTL) 0.838 | hippocampus 0.758
      MTL uniform - gap(91)  +0.109 [-0.016,+0.242] p=0.080   (direction of E62b, underpowered)
    MECHANISM: rank r(age weights, disease weights) = -0.009   (ADNI -0.031)  <- REPLICATES
    MMSE decline (partial r): MTL uniform +0.49, uniform(6) +0.42, gap +0.39, hippocampus +0.36
    test-retest (same-session repeat): every summary ICC >= 0.988 - not a differentiator
Orthogonality replicates in a different study, scanner and segmentation program.
Everything else waits for the full 150.

---

# E67 — THE ALIGNMENT LAW  ★★★ the unifying result

For normative deviations z with within-group covariance S and disease shift mu, any
linear summary w separates groups with d'(w)^2 = (w'mu)^2 / (w'Sw); the best is
d*^2 = mu'S^-1 mu. Their ratio eta(w) = cos^2 in the S metric between w and S^-1 mu.
Geometry from half A of ADNI (98 regions), tested on held-out half B:

    summary              eta    predicted AUC   observed AUC
    brain-age gap (98)   0.271      0.888          0.888
    uniform (98)         0.226      0.866          0.852
    uniform, MTL only    0.495      0.950          0.941
    hippocampus L+R      0.409      0.932          0.926
    first PC             0.179      0.838          0.826
    500 random region-subset summaries never seen by the law:
      corr(predicted, observed AUC) = 0.960, mean |error| 0.015 AUC

★ **Brain age uses 27% of the Alzheimer information present in the same scan.**
★ **The Schulz paradox is NOT a general law.** Across random subsets, better age
accuracy goes WITH better alignment (corr(MAE, eta) = -0.70). Accuracy is neither
necessary nor sufficient; eta alone decides. The 6-vs-98 trade-off in E62b is a
property of the AD-selected six, not of adding regions.
One quantity now explains E49 (on-axis disease), E53b/E62 (dimension), E62b
(selection), E63 (dual reference bounded by off-axis share), and the shared-factor
reason why rank-orthogonal weights still reach AUC 0.88.

# E67b — THE LAW RUNS ON SUMMARY STATISTICS
Healthy-only covariance + disease shift from:
    all 123 AD r=0.951 | 80 AD r=0.945 | 40 AD r=0.929 | 20 AD r=0.898 | 10 AD r=0.842
    published-style Cohen's d map (2 decimals): r = 0.952, mean |error| 0.020 AUC
    same map WITHOUT covariance (independent regions): r = 0.854, error 0.113
★ **No individual patient data needed.** The covariance structure is worth most of it.

# E68 — PREDICTING BRAIN AGE ACROSS DISORDERS FROM ENIGMA MAPS ALONE  ★★★

ENIGMA Toolbox d maps (68 CT + 16 subcortical; github MICA-MNI/ENIGMA). Sign
checked per file against the source papers: all patients-minus-controls except
schizophrenia (reversed). Brain-age model trained on ADNI controls (n=442, 60-90 y)
in the same 82-feature space, MAE 3.74 y.

    disorder             best d*   predicted BA d   eta_BA   hippocampus d   Kaufmann 2019 BA d
    schizophrenia          0.86        +0.37         0.19        +0.37         0.51 (ENIGMA 0.48)
    bipolar (adult)        0.65        +0.28         0.19        +0.22         0.29
    depression (adult)     0.41        +0.04         0.01        +0.16         0.10 (ENIGMA 0.14)
    autism                 0.75        +0.10         0.02        +0.03         0.07
    ADHD (adult)           0.66        -0.13         0.04        -0.07         0.06
    OCD (adult)            0.50        +0.09         0.03        +0.15         -
    epilepsy (all)         1.07        +0.40         0.14        +0.38         -
    22q11.2 deletion       3.45        +0.22         0.00        +0.93         -
    Parkinson HY1/3/4-5    0.61/1.02/1.62  +0.12/+0.43/+0.65  0.04/0.18/0.16  -/-/-

Kaufmann values extracted from the PDF text, labels matched through the reported
P-values and group sizes: ASD 0.07, ADHD 0.06, SZRISK 0.18, SZ 0.51, PSYMIX 0.21,
BD 0.29, MS 0.74, MDD 0.10, MCI 0.41, DEM 1.03.

★★★ **From regional effect maps alone - different cohorts, a late-life brain-age model,
no patient data - the law reproduces the ORDER of published brain-age effects across
five psychiatric disorders (Spearman ~0.90) and bipolar/autism almost exactly.**
Forward, falsifiable predictions for unmeasured cases: 22q11.2 deletion d~0.2
despite the largest structural difference of any disorder (brain age captures ~0% of
it, because thicker cortex runs against aging); epilepsy ~0.4; OCD ~0.1; Parkinson's
rising with Hoehn-Yahr stage (0.12 -> 0.43 -> 0.65).
Caveat: late-life covariance and model applied to mostly young-adult maps; this is a
pilot of the principle, the thesis needs lifespan normative covariance.

LITERATURE (Sept 2026): Schulz 2025 (PLOS Biol) explains the accuracy paradox
verbally via feature importance; Grødem, Smith, Vidal-Pineiro, Fjell (medRxiv May
2026) show brain age cannot measure aging rate; Snel & Schulz (arXiv 2609.07729,
Sept 2026) attribute Cohen's d to TRAINING SAMPLES via influence functions. None
derive a closed-form alignment law in feature space, none predict brain-age utility
across disorders from summary statistics, none test predictions against published
values. Open.


---

# E69 — DECOMPOSED BRAIN AGE: disease years + other years  ★★★

gap = beta*s (s = S^-1 mu discriminant score, beta from controls) + remainder; the
remainder has zero expected Alzheimer shift by construction. Fitted out of fold on
CN and AD only; everything below except AD/CN is an independent test.

    group   total gap = Alzheimer years + other years
    CN        +0.09   =   +0.13          -0.05
    MCI       +2.19   =   +1.83          +0.37
    AD        +4.74   =   +4.10          +0.64
    in healthy people only 31% of the gap's variance is Alzheimer-shaped

MCI -> AD (never used in the split): Alzheimer years C 0.744 (HR/SD 2.23), total gap
0.720 (HR 1.87), other years 0.574 (HR 1.26); difference +0.025 [-0.002,+0.049] p=0.08.
BIOLOGY in MCI (partial r): amyloid PET  Alzheimer +0.302*  other +0.011
                            APOE4        Alzheimer +0.251*  other +0.031
                            CSF p-tau    Alzheimer +0.174*  other -0.113*
★ **The tau blindness of brain age (E64) is a CANCELLATION**: the Alzheimer part
tracks p-tau positively, the rest negatively (dilution-like), and the total shows
+0.026. In CN the CSF artefact sits entirely in the other years (-0.188* vs +0.034).
COGNITIVE DECLINE in healthy people: Alzheimer years +0.216/+0.217 (mPACC/ADAS13),
other years +0.086/+0.071, difference significant for both. This REVISES E64-C's
"brain reserve" reading: the decline signal in healthy people is carried by
Alzheimer-shaped atrophy, not by a generic reserve level.
Sex difference in controls sits in the other years (+0.22 vs -0.02 y).

# E70 / E70b — THE SCHULZ CURVE, AND A DISEASE-AWARE BRAIN AGE  ★★★

A: ridge penalty swept over 7 orders of magnitude. Observed AD/CN AUC peaks at
alpha 316 (0.878); the law, from half-A geometry, puts its peak at the same place
(eta 0.279 at alpha 316-1000) but over-predicts at extreme penalties (overall r 0.62,
error 0.030). Across the curve corr(age MAE, AUC) = -0.81: in ADNI the Schulz effect is
a small local bump near the optimum, not a general trade-off.
B: relevance-weighted ridge, penalty sum w_j^2 / pi_j^gamma, pi = |d| of AD vs CN from
half A, target still chronological age in controls, output still years.
    best STANDARD at any alpha     alpha 316   MAE 3.66  AD/CN 0.878  conversion 0.721
    disease-aware alpha 100 g=2    MAE 3.62  AD/CN 0.895  conversion 0.729  <- more accurate AND more sensitive
    best disease-aware             alpha 3162 g=4  MAE 3.80  AD/CN 0.918  conversion 0.737
      vs best standard: AD/CN +0.039 [+0.016,+0.065] p=0.002; conversion +0.016 [+0.002,+0.033] p=0.027
★ **The standard accuracy/sensitivity frontier is not the real frontier.** A disease
prior moves brain age past it, and at alpha 100 improves age accuracy and disease
sensitivity together. The prior needs only an effect map, so any public map (ENIGMA)
can make a disorder-aware brain age.
C: classifier from half-A summary statistics only (w* = S^-1 mu): AD/CN 0.945 vs
logistic on individuals 0.956, MTL uniform 0.941, standard brain age 0.826.

# E71 — NON-ALZHEIMER YEARS IDENTIFY NON-ALZHEIMER NEURODEGENERATION  ★★★
MCI, top third of total gap (n=170; 53 amyloid-, 117 amyloid+):
    total gap        A- +5.24  A+ +5.69  p=0.34   <- brain age cannot tell them apart
    Alzheimer years  A- +2.29  A+ +3.64  p<0.0001
    other years      A- +2.95  A+ +2.05  p=0.001
    share 'other'    A- 0.58   A+ 0.32   p<0.0001; AUC for amyloid-negative 0.697 [0.61, 0.77]
Atrophied MCI, SAME total gap: mostly Alzheimer-shaped 73.5% convert (amyloid+ 77%),
mostly other (SNAP-like) 44.2% (amyloid+ 60%); HR 0.42 [0.29, 0.62] p<0.0001.
No effect in CN (atrophy not yet disease-shaped).
★ **The SNAP hypothesis that failed in E63 with coarse categories holds with the
continuous decomposition.** Two MCI patients who are both "+5 brain-years" have very
different prognoses, and the composition says which is which.
LITERATURE: Sci Rep 2025 decomposes morphology into NMF aging templates; the
amyloid-informed brain age (PMC12907109) adds PET. Neither splits the SAME gap into
additive disease/other years, localises amyloid/APOE/tau/dilution to the parts, or
uses the split to separate SNAP at equal gap.


# E71b — THE COMPOSITION IS NOT HIPPOCAMPAL ATROPHY IN DISGUISE
Atrophied MCI (n=149 with amyloid): amyloid-negative CV AUC hippocampus 0.599, MTL
average 0.645, composition 0.656. Logistic with both: hippocampus OR/SD 0.94 p=0.79,
composition OR/SD 1.79 p=0.016. Conversion Cox (+total gap, age): composition HR/SD
0.69 p=0.0008 with hippocampus in the model (hippocampus p=0.58); 0.78 p=0.037 with the
MTL average in the model (MTL p=0.058). Survives; overlaps partly with the MTL average.

# E72 — CONSEQUENCES OF THE LAW ACROSS ENIGMA DISORDERS
SAMPLE SIZE (n per group, 80% power): brain age vs the best linear summary d*
    schizophrenia 133 vs 21 (6x) | bipolar 219 vs 38 (6x) | depression 6,943 vs 92 (76x)
    OCD 3,285 vs 64 (52x) | ADHD 799 vs 36 (22x) | autism 4,058 vs 28 (145x)
    epilepsy 93 vs 14 (7x) | 22q11.2 186 vs 1 (141x) | Parkinson HY1 1,811 vs 42 (43x)
    Parkinson HY3 90 vs 15 (6x) | Parkinson HY4-5 45 vs 6 (8x)
★ **Brain-age studies of depression, OCD, autism and early Parkinson's are underpowered
by design**: the brain differences exist (d* 0.4-0.75) but brain age is nearly blind to
their direction. (d* is an in-sample upper bound; the ratio is indicative.)
SIMILARITY: naive cross-disorder map correlation vs operational transfer in the S
metric correlate 0.84; they disagree mainly for early Parkinson's (naive +0.40 with
schizophrenia, operational -0.06). Modest.
DISORDER-AWARE BRAIN AGE from each ENIGMA map: small gains (22q 0.29 -> 0.52,
schizophrenia 0.34 -> 0.39, autism 0.06 -> 0.12, epilepsy 0.41 -> 0.46) at +0.05-0.1 y
MAE. Large for Alzheimer's (E70), small where the disorder pattern is far from aging.

# E73 — DECOMPOSED BRAIN AGE OVER TIME
7949 visits, 1735 subjects; models fitted on baselines out of fold, applied to all visits.
Slope reliability (split-half, >=6 visits, n=450): total 0.746, Alzheimer years 0.726,
other years 0.532.
PRECLINICAL (CN, amyloid+ 79 vs amyloid- 138), slope A+ minus A-, age/sex adjusted:
    total brain age  +0.109 y/y [+0.005,+0.216] (0.28 SD)
    Alzheimer years  +0.009 [-0.048,+0.072]
    other years      +0.100 [+0.011,+0.190]
★ PREDICTION FAILED IN ITS SPECIFIC FORM, and the failure is informative: amyloid-
positive healthy people DO change faster (weakly), but NOT along the dementia-stage
Alzheimer direction. **The preclinical structural signature is not the dementia
signature.** A disease direction learned from dementia vs controls will miss it.
UNITS (Cox, HR per year of each component, age-adjusted):
    MCI (n=821, 351 events)  gap 1.198  ALZHEIMER-YEAR 1.511  other-year 1.134
    CN  (n=508, 130 events)  gap 1.148  ALZHEIMER-YEAR 1.533  other-year 1.052 (ns)
★ **An Alzheimer-year carries the same hazard in healthy and MCI people (1.51 vs
1.53).** A gap-year is a mixture of heavy and light years whose proportions differ by
population - the mechanism behind E53a's population-dependent unit.


# E74 — DOES ALZHEIMER'S POINT A DIFFERENT WAY AT EACH STAGE?  Mostly no.
Directions: preclinical rate (A+ vs A- CN slopes, 79 vs 138), prodromal rate (MCI
converters 251 vs stable 283), dementia level (AD vs CN baseline). Map reliability
0.48 / 0.88 / 0.97.
    preclinical vs prodromal r = 0.71 (ceiling-corrected ~1.0): SAME direction
    preclinical vs dementia  r = 0.39 (0.57);  prodromal vs dementia r = 0.70 (0.76)
Rate maps are ventricle-led (preclinical 25% of positive effect in ventricles,
prodromal 12%, dementia level 7%); the level map is MTL-led (33%). The difference is
largely RATE versus LEVEL, and ventricles dominate rate maps because their change is
the most precisely measured (E38) - a precision effect, not a stage rotation.
Amyloid in healthy people from slopes: every candidate AUC 0.53-0.60, overlapping CIs.
VERDICT: no stage rotation shown, no better preclinical detector. Refines E73: the
"other years" slope of amyloid+ controls is mostly ventricular expansion rate.

# E75 (half OASIS, n=77) — THE LAW ACROSS COHORT AND SOFTWARE
87 regions common to ADNI DK/FreeSurfer and OASIS DKT/FastSurfer. Geometry from
ADNI, AUC observed in OASIS (CDR>0 vs 0), 600 random summaries:
    Spearman 0.786, Pearson 0.784; raw error 0.058; after one severity factor on d'
    (c=0.79, OASIS cases milder) error 0.028.
ADNI-defined decomposition applied in OASIS: CDR>0 = +3.13 Alzheimer years + 2.24
other; MMSE decline partial r: Alzheimer years +0.457, other +0.209, gap +0.440.

# E75b — WHICH DISEASE DIRECTION IS PORTABLE?  w = (S + lam diag S)^-1 mu
    lam          ADNI held-out AUC   OASIS AUC
    0 (LDA)           0.941            0.723
    1                 0.960            0.821
    3                 0.959            0.848
    30                0.924            0.860
    map as weights    0.904            0.858
★ The raw discriminant overfits the covariance of one cohort/program and collapses
abroad; moderate diagonal shrinkage (lam ~1-3) is BETTER at home and far better
abroad. Design rule for summary-statistics markers. Recheck at full n.


---

# OASIS-2 COMPLETE — FULL EXTERNAL VALIDATION (n=150, 523 FastSurfer scans)
373 sessions (mpr-1) + 150 same-session repeats (mpr-2). Baseline: CDR 0 85 (70 stable
reference), CDR>0 65. Sanity: L/R thalamus identical in 1% of scans (the probe's
equal values were coincidence); median hippocampus L/R 3569/3624 mm3.

## E66 full
    diagnosis CDR>0 vs 0:  gap(6-like) 0.716 | gap(91) 0.716 | uniform(6-like) 0.750 |
                           uniform(91) 0.718 | uniform(14 MTL) 0.778 | hippocampus 0.719
      uniform(6-like) - gap(6-like)  +0.046 [+0.001,+0.096] p=0.048
      MTL uniform - gap(91)          +0.063 [-0.020,+0.153] p=0.128   (E62b direction, underpowered)
      uniform(91) - gap(91)          -0.006 p=0.92                    (E62 pattern: neutral atlas no gain)
    MECHANISM: rank r(age weights, disease weights) = -0.029   ADNI -0.031   <- REPLICATED
    MMSE decline partial r: MTL uniform +0.327, uniform(6) +0.281, hippocampus +0.281,
                            gap(91) +0.256, gap(6) +0.227
    age MAE 91-region 4.92 vs 6-like 5.88; disease AUC equal (0.716 vs 0.716): here more
      regions read age better at NO disease cost - the Schulz trade-off does not appear.
    sex bias: underpowered (20 men in the reference), not interpretable.
    TEST-RETEST (150 same-session pairs): gap ICC 0.997, hippocampus 0.996, uniform(91)
      0.991, MTL 0.985; single regions median 0.990 (range 0.931-1.000). Same-session
      repeatability is not what separates these summaries.

## E75 full — THE LAW ACROSS COHORT AND SOFTWARE
    600 random summaries, ADNI/FreeSurfer geometry -> OASIS/FastSurfer AUC:
      Spearman 0.855, Pearson 0.867; after one severity factor (c=0.56) error 0.018
    ADNI-defined decomposition applied in OASIS:
      CDR>0 = +2.18 Alzheimer years + 1.18 other years (total +3.36)
      AUC: Alzheimer years 0.776, gap 0.771, other years 0.641
      MMSE decline: Alzheimer years +0.349 [+0.128,+0.526], gap +0.298, other +0.110 [-0.06,+0.30]
★ The prognostic signal lives in the Alzheimer years in an independent study and program.

## E75b full — PORTABLE DIRECTION
    lam 0: ADNI 0.941 / OASIS 0.740 | lam 1: 0.960 / 0.795 | lam 3: 0.959 / 0.802 |
    lam 10: 0.944 / 0.803 | map as weights: 0.904 / 0.782
★ Rule confirmed at full n: moderate diagonal shrinkage (lam ~3) is best at home and abroad.

## E76 — DISEASE-AWARE BRAIN AGE WITH A PRIOR FROM ANOTHER STUDY
Prior = ADNI Alzheimer effect map (87 common regions); age model trained on OASIS
controls. Best standard (alpha 300): MAE 5.45, CDR AUC 0.765, decline r +0.278. Best
aware (alpha 300, gamma 2): MAE 5.64, AUC 0.794, decline r +0.305.
AUC aware - standard +0.028 [-0.007,+0.066] p=0.119. Direction replicates, underpowered;
at low penalties the prior amplifies noise and hurts - it needs adequate regularisation.


---

# ATLAS V2 (2026-10-24) — pilot/v2/
Same 7,603-person pool (77 scanners, 6-94 y). Changes, each kept only after a test on this pool:
age x sex curve (all structures), age x ICV (temporal horn, 3rd ventricle), ICV in spread (3 CSF spaces),
empirical-Bayes scanner shrinkage (BrainChart-style), left/right models + PCN's extra subcortical structures
(59 models). Spread by sex (CentileBrain-style) tested: no held-out gain, not added.

## What the published atlas missed
    men - women mean held-out z, published: total brain +0.21 (<18) ... -0.29 (60-75, 5.2 SE) -0.37 (75+);
    cortex, cerebellum, amygdala, precentral similar; CSF mirror image (+0.27..+0.37 after 40).
    Pooled-over-age sex check (mean z within 0.001) hid it: the bias flips sign across the lifespan.
    Healthy 60+ flagged as enlarged: temporal horn women 2.6% / men 3.8%; 3rd ventricle 2.0% / 3.0%.
    Published OASIS AUCs partly sex-confounded: patients more male (CDR 0.5 44% vs 28%; OASIS-2 56% vs 31%),
    sex alone AUC 0.58 / 0.52 / 0.63; within sex, v2 and published AD signatures are identical.

## Atlas v2 validation
    held-out (59 models): SD 1.009-1.024, in +-1.96 94.4-95.5%, age bands with sex bias > 2 SE: 0 in 57/59,
    small sites (n<40) in +-1.96 median 94.2% (published ~92%); |r(z, ICV)| within sex x age at noise level.
    new scanner (10 left out): median |bias| 0.40 (0 controls) 0.27 (5) 0.22 (10) 0.18 (20) 0.13 (40) 0.10 (80);
      90th pct 0.97 uncalibrated -> 0.23 with 20 controls.
    OASIS, AUC within sex (CDR 0.5 / CDR>=1 / OASIS-2 demented):
      v2 learned27 0.886 / 0.930 / 0.891   published learned17 0.843 / 0.909 / 0.878   PCN learned27 0.897 / 0.947 / 0.902
      v2 - published +0.043 [p .026] / +0.021 / +0.013;  v2 - PCN -0.011 / -0.016 / -0.011 (all n.s.)
    OASIS controls: v2 SD 1.02-1.06, PCN 0.67-0.80; |r ICV| v2 0.06, PCN 0.34-0.43.
★ A 7.6K single-pipeline atlas matches a 57K multi-pipeline atlas on discrimination and is better calibrated.

## Deep models
    neural SHASH (3-net ensemble, same folds): held-out logp worse 8/8 (-0.020/person); aging rate @70
    network/v2: entorhinal 0.25, hippocampus 0.64, temporal horn 0.64, lat. vent. 0.85, precuneus 0.82.
    residual auditors (boosted trees, MLP) on held-out z: planted 0.2 SD bias in men 60+ undetected,
    0.3 SD borderline (MLP p .013) -> black-box audits are underpowered at n = 7.6K; targeted checks are not.

## Open
    older men: heavier low tail in brainstem / cerebellum / ventral DC / total brain (brainstem 3.8% vs 2.1%),
      not in cut-off datasets or large heads (FOV ruled out); sex-specific shape untested.
    temporal horn: 3.5% of healthy 60+ beyond +1.96 in both sexes (preclinical disease in 'healthy' samples?).
    all atlases weaker in men (male controls 22-32); converters unseparated (AUC 0.54-0.63).
