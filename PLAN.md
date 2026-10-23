# PhD plan — redefining brain age as normative deviation

Last updated: 2026-10-11. Working language: English. Discussion happens in chat; this file only tracks the plan.

## Core idea

Brain age is redefined from "predicted age minus real age" to a **normative deviation framework**:

- The primary object is each region's **deviation from healthy people of the same age** (normative z-score).
- Each region gets **its own age**, but only where that region ages steeply and steadily enough for "years" to be meaningful (**age resolution**).
- A person's whole-brain deviation is split into an **aging-like part** (in years) and an **atypical part** (in SD units, with a healthy centile).
- Classic brain age is shown to be a special case: it measures only the aging-like part.

## Prior art to cite (and what we add)

| Prior work | What it did | Our addition |
|---|---|---|
| Normative modelling (Marquand; Bethlehem & Seidlitz 2022; Frangou 2024; Verdi 2023; Parker 2025) | regional z / centiles | use z as the base of brain age; regional ages with resolution |
| Klemera & Doubal 2006 (biological age) | per-marker age, precision = residual SD / slope | apply to brain regions; resolution atlas; validity rules |
| Lorenzi 2015 → Sivera 2019 → Fu 2025 | aging-parallel vs residual split of deformations | calibrated (healthy-covariance) split, age-dependent directions, uncertainty |
| Bethlehem 2022 (centile Mahalanobis distance) | total atypicality | split into aging-like + atypical |
| Irimia PNAS 2026 (voxel brain age) | local brain age by prediction | predict and test its predictability bias |

Not found anywhere: age-resolution atlas of the brain; validity rules for years; precision floor for brain age; proof that classic brain age = aging-like projection.

## Phases

### Phase 1 — Theory note (paper and pencil, ~1 week)  [status: proposed, awaiting go]
Definitions D1–D14 and propositions P1–P7 (see below), each with formula, example, purpose and prior use.
Plus a small simulation on synthetic data to check every formula.
Deliverable: theory note + list of testable predictions for Phase 3.

### Phase 2 — Healthy reference and age-resolution atlas (~2–3 weeks)
- Data: healthy participants already available (OASIS-2 FastSurfer; ADNI/TADPOLE CN 98 regions); add Cam-CAN or IXI for full lifespan (ask before downloading).
- Fit per-region normative models (age, sex, head size, scanner), including brainstem and cerebellum.
- Compute mean, SD, slope, sensitivity, resolution per region and age (bootstrap CIs); measurement resolution from OASIS same-session repeats.
- Deliverable: age-resolution atlas + validity map.

### Phase 3 — Tests (~2–3 weeks)
1. Precision floor vs best achievable brain-age error.
2. Irimia PNAS 2026 voxel model (public ONNX weights): do "older" regions match high-resolution regions; does the ranking flip around the training mean age?
3. Synthetic atrophy injection.
4. Left–right flip test.
5. Head-to-head with Fu 2025 (public code): calibrated atypicality vs their AD-specific score.

### Phase 4 — Per-person outputs (~1–2 weeks)
Regional z-map, regional age-gap map with error bars, aging-like / atypical split; check classic brain age ≈ aging-like part.

### Phase 5 — Diseases (after data access)
Alzheimer's (ADNI, OASIS; OASIS-3 if approved) → Parkinson's (PPMI) → FTD (NIFD) → Huntington's (TRACK-HD) → ALS (CALSNIC).
Question: which diseases and regions are aging-like vs atypical.

## Phase 1 contents (summary)

Definitions (region r, age a, healthy reference). **Every quantity below is a function of age a** — slopes, spreads, sensitivity, resolution, aging direction, covariance and the precision floor all change across the lifespan, so the atlas is region × age.

D7 in plain words: regional age = the age at which a healthy person's average value equals this person's value (read the healthy curve "backwards", like height age / bone age). Regional gap = regional age − real age = z × σ/k, so the same z gives different years in different regions and at different ages.

- D1 normative mean μ_r(a), spread σ_r(a)
- D2 deviation z_r = (y_r − μ_r(a)) / σ_r(a), centile Φ(z_r)
- D3 aging rate k_r(a) = dμ_r/da
- D4 aging sensitivity S_r = |k_r| / σ_r (SD per year)
- D5 age resolution R_r = σ_r / |k_r| (years)
- D6 measurement resolution M_r = ε_r / |k_r|, reliability ICC_r = 1 − ε_r²/σ_r²
- D7 regional age A_r (inverse of μ_r), regional gap G_r = z_r σ_r / k_r
- D8 validity rule for reporting years
- D9 aging direction s_r = k_r / σ_r
- D10 healthy covariance Σ of z
- D11 aging-like deviation g = sᵀΣ⁻¹z / sᵀΣ⁻¹s (years)
- D12 atypicality T² = zᵀΣ⁻¹z − (sᵀΣ⁻¹z)² / sᵀΣ⁻¹s
- D13 precision floor R_brain = 1 / √(sᵀΣ⁻¹s)
- D14 aging-likeness of a pattern δ: ρ = (sᵀΣ⁻¹δ)² / (sᵀΣ⁻¹s · δᵀΣ⁻¹δ)

Propositions:
- P1 regional gap = z × resolution (years add no information beyond z)
- P2 total² = (g / R_brain)² + T²
- P3 precision floor; shrinkage slope β = V / (V + R²)
- P4 predictability bias of voxel brain age: apparent gap_r = (β_r / β̄ − 1)(age − mean training age); sign flips at training mean
- P5 equivalence with Klemera–Doubal
- P6 bias-corrected linear brain-age gap = g exactly (blind to T)
- P7 a disease is visible to brain age in proportion to its aging-likeness ρ (Report 5 page 30)

## Phase 1 progress (2026-10-11) — first numbers on real data
Code: `pilot/p1_definitions.py`, `pilot/p1_figures.py`; outputs in `pilot/p1_out/` (tables, summary.json, 5 figures).
Data: 442 ADNI/TADPOLE healthy adults (98 FreeSurfer volumes, age 60–90); OASIS-2 150 same-session repeat pairs.
- Single-region resolution at 75: best ≈ 11.6 y (inf. lat. ventricles), hippocampus ≈ 12 y, median 23 y, brainstem ≈ 50 y, caudate >150 y.
- Resolution depends strongly on age (hippocampus 8.7 → 18.6 y from 65 to 85; putamen turning point ~80). Cross-sectional slope flattens at old age, unlike longitudinal literature → check with longitudinal slopes.
- Hippocampus −0.88 %/yr at 75 (literature longitudinal: 0.98–1.12 %/yr Fraser 2015; 1.41 %/yr Barnes 2009).
- Measurement resolution tiny (ventricles 0.3 y, hippocampus ≈ 1.4 y, ICC ≥ 0.98) → regional-age imprecision is biological between-person variation, not scanner noise.
- Whole brain: precision floor ≈ 4.2 y; held-out SD of g 5.4 y; ridge MAE 3.65 y.
- P3: per-region shrinkage β_r vs V/(V+R²): r = 0.99. Whole-brain β 0.40 observed vs 0.51–0.63 predicted.
- P4: per-region predicted age + one shared correction → apparent regional gap vs predictability r = +0.98 (older-than-mean healthy) and −0.98 (younger) — sign flip confirmed; hippocampus +9.7 / −7.8 y in the same healthy people.
- P6: classic gap vs g r = 0.82 (all), 0.77 (healthy); AD/CN AUC classic 0.865 vs g 0.766 → GLS needs regularisation.
- P2 identity exact; T² miscalibrated vs χ² (healthy median centile 0.85) → needs empirical calibration.

### Phase 1b (2026-10-11) — `pilot/p1b_long_and_reg.py`
(a) Within-person slopes, ADNI CN, scanner-consistent blocks only (same field strength + FreeSurfer version; 75 people, 294 scans):
- Mixing 1.5T→3T visits created fake tissue "growth" (cerebellum +0.9 %/yr); the consistency filter removes it.
- Hippocampus −1.1 to −1.4 %/yr (matches Fraser 2015 / Barnes 2009); entorhinal ≈ −1.8 %/yr; lateral ventricles +3.8–4.3 %/yr.
- Longitudinal rates **accelerate** with age while cross-sectional ones flatten: median k_long/k_cross 0.67 (65) → 1.39 (75) → 2.33 (85). Cross-sectional curves understate old-age aging (survivor / cohort effect).
- So resolution should be σ (between-person) / k_long (within-person). Median R: cross 15 → 41 y, long 25 → 13 y (65 → 85). Entorhinal at 75: 32 y vs ≈ 10 y.
(b) Regularised Σ_λ = (C + λI)/(1+λ): λ↑ raises AD AUC of g (0.77 → 0.85) but worsens healthy precision (SD_g 5.4 → 8.3 y) and calibration. Choose λ on healthy data only (≈ 0.25). Empirical (ECDF) T² calibration: CN median centile 0.50, MCI 0.65, AD ≈ 0.82.
(c) USC LBA model (`external/USC_LBA_estimator`, ONNX, CPU ≈ 12 s/brain): on the two shipped brains, within-brain LBA SD only 0.3–0.4 y (local maps ≈ constant). ±10 % intensity scaling shifts LBA ≈ ∓1 y.

## Data expansion (started 2026-10-11)
Goal: a large pooled healthy reference; check whether σ, k, S, R, M and the floor replicate per dataset.
| Dataset | N healthy | Ages | Design | Status |
|---|---|---|---|---|
| ADNI/TADPOLE CN | 442 (75 long.) | 60–90 | FreeSurfer tables | done |
| OASIS-2 nondemented | ~72 | 60–96 | FastSurfer, long. + same-session repeats | done (kaggle/oasis_fs_results) |
| IXI | ~580 | 20–86 | 3 sites (1.5T/3T); FastSurfer + LBA | Kaggle `ixi-lba-p0/p1` running |
| DLBS (OpenNeuro ds004856, CC0) | 464 (300 W2, 193 W3) | 21–89 | same 3T, 3 waves over ~9 y; amyloid PET; 10 repeat runs | Kaggle `dlbs-fs-p0/p1` ready |
| Kaggle ABIDE / ADHD-200 controls | many | 7–64 | youth only | optional |
Needs user application (best for aging): OASIS-3, Cam-CAN, UK Biobank, AIBL, NACC, HCP-Aging.

Run notes: Kaggle allows only 2 GPU batch sessions → DLBS runs first, `kaggle/watch_lifespan.py` pushes IXI into freed slots and fetches outputs to `kaggle/dlbs_fs_results/`, `kaggle/ixi_lba_results/`. FastSurfer 2.4.2's N4 script breaks on NumPy 2 (Kaggle has 2.1.3) → patched with sed in the IXI kernel. First IXI attempt (unpatched) was deleted before producing anything. LBA stand-in check on IXI002 (age 35.8): 58.6 (cap) / 63.7 (histogram-matched) / 59.0 (raw) → the model, not the input format, gives ~+23 y here; the cohort will show if it is systematic.
Analysis code ready: `pilot/p2_lifespan_data.py` (loaders), `pilot/p2_lifespan_atlas.py` (per-dataset atlas, floor by age, DLBS within-person rates, repeat noise), `pilot/p2_lba_p4.py` (P4 on LBA).

Quick lifespan checks with data already on disk (2026-10-11):
- OASIS-1 (316 healthy, 18–94; 20 repeats), whole-brain nWBV: R = 12 y (20), 15 (30), 26 (40), 19 (50), 9 (60), 8 (70), 10 (80), 13 (90); M ≈ 1–5 y. Brain-volume age is least informative in midlife.
- ABIDE + ADHD-200 controls (968, 6.5–56, 26 sites, FreeSurfer 5.1): development gives good resolution in adolescence (cortex 6–8 y at 12–18, WM 5–6 y at 12–14, brainstem 5–6 y at 8–12, amygdala 6 y at 12); almost every region is uninformative at 25–45 (R ≥ 20–50 y).
→ Emerging picture: age information in brain volumes is U-shaped over the lifespan (development, then aging after ~60), with a midlife "blind zone".

Consistency checks with data on disk (`pilot/p2_consistency_now.py`, same spline model everywhere, bootstrap CIs, age 75):
- ADNI CN (442, multi-site) vs OASIS-2 (72, one 1.5T): single-region R agree within CIs (hippocampus 12.1 vs 16.4; inf. lat. vent. 9.9 vs 12.8; lat. vent. 16.5 vs 14.8). OASIS-2 spread is ~15–25 % smaller (single site) but its cross-sectional slopes are also smaller.
- ADNI 1.5T (204) vs 3T (195): same spread, rates and R within CIs (hippocampus R 11.7 vs 12.8). Adding field strength as covariate barely changes σ (most for precuneus, 11.5 → 9.5 %).
- Men (213) vs women (229): no detectable difference in σ or R for the strong regions (hippocampus 12.6 vs 11.4); sex is an additive covariate; sex-specific slopes not modelled yet.
- Head size: dropping ICV inflates σ, most for thalamus (7.2 → 9.4 %, R 23 → 37 y), brainstem, cerebellum, precuneus; small for hippocampus.
- Scanner changes matter most longitudinally (fake growth) → handled by the consistency filter.
Open: vendor effects (IXI: Philips + GE), site-specific σ, sex × age slopes, ICV proxy for IXI/DLBS (FastSurfer mask volume, not eTIV), GAMLSS.

### Stratified atlas (2026-10-12) — `pilot/p2_strata.py` → `p2_out/strata_*.csv`, `headsize_methods.csv`
Data now: DLBS 967 scans (464 W1), IXI 551/581, OASIS-2; FastSurfer pool = 1072 healthy (scanners: Philips 1.5T, Philips 3T IXI, Philips 3T DLBS, GE 1.5T, Siemens 1.5T); ADNI CN 399 (sex × 1.5T/3T).
Log-volume models: slope in %/yr, spread in %, R = spread/|slope|; atlas per sex × scanner and per head-size tertile (within sex), bootstrap CIs; nested F-tests for age × factor.
- Aging RATE is essentially group-invariant: sex differences ≤ 0.1 %/yr (only hippocampus, men −0.10 %/yr faster, and total brain −0.04 reach Bonferroni with n = 1072); field strength none (ADNI); scanner none within IXI+DLBS (median 0.08 %/yr); head size ≤ 0.04 %/yr per SD. Exception: cortical slopes differ between scanners in the ≥ 60 subset (0.4–0.9 %/yr; small groups).
- SPREAD is NOT scanner-invariant: between-person σ differs by scanner (median max/min 1.19, up to 1.43; 7/17 regions p < .05; DLBS largest). Sex and head-size tertiles do not change σ.
- Head size: power-law (log y ~ log ICV) with region-specific exponent b (hippocampus 0.46–0.64, thalamus 0.73, cortex 0.88, WM ≈ 1.1, ventricles 1.6–2.7) removes ICV dependence and almost all sex differences, lowest σ. Proportion (b = 1) over-corrects (reverses sex difference). Residual method ≈ power-law.
- The FastSurfer mask proxy biases things (removes ~30 % of the aging slope, makes total-brain σ artificially small) → SynthStrip ICV running on Kaggle CPU (`kaggle/icv_synthstrip_*`, fetched by `kaggle/wait_icv.py` into `kaggle/icv_results/`).
Decision proposal: one pooled model per region: log volume ~ f(age) + sex + b·log ICV + scanner offset, with scanner-specific σ (location-scale, as in Bethlehem 2022 / HBR); keep age × sex only where it matters (hippocampus); stratified atlases kept as robustness checks.

### Real head size + lifespan results (2026-10-12 evening)
SynthStrip ICV (1921 scans; `kaggle/icv_results/`; loaders now use it as `icv`, old proxies kept as `icv_mask`, `etiv`):
- Valid: r = 0.95 with OASIS-2 eTIV; within-person change ≈ 0 in OASIS-2 (+0.05 %/yr); DLBS shows −0.15 %/yr at ALL ages, uncorrelated with brain change (r = −0.01) → scanner geometric drift between waves; corrected by rescaling each scan by person-mean ICV / scan ICV (head size as a ruler).
- Cross-sectional ICV falls with age in IXI (−0.14 %/yr) and OASIS-2 (−0.11): plausibly the birth-cohort increase in head size (cf. DeCarli 2024, JAMA Neurology — verify source quality) → uncorrected cross-sectional slopes overstate aging.
- With true ICV: head-size effect on aging slope disappears (0/17 regions; was 4/17 with the proxy). Sex: hippocampus men −0.10 %/yr faster, total brain −0.04. Scanner (IXI+DLBS): small slope differences (≤ 0.2 %/yr) in hippocampus, cortex, WM, total brain; spread differs (ratio 1.15). Power-law exponents b: hippocampus 0.69, thalamus 0.77, cortex 0.95, WM 1.13, total brain 0.99, ventricles 1.7–2.2.
DLBS within-person rates 30 → 85 (scale-drift corrected): hippocampus +0.2 → −0.85 %/yr, entorhinal 0 → −1.6, thalamus −0.1 → −0.65, total brain −0.07 → −0.68, ventricles +0.9 → +3.8. k_long/k_cross ≈ 0.9 at 30, 1.3–1.5 from 50 on. Median R: cross 24 (30) → 39 (50) → 16 (85); long 26 → 23 → 11 → midlife blind zone is partly a cross-sectional artefact.
Whole-brain floor R_brain(a), 91 features: IXI 4.5 (25) → 6.1 (50–55) → 3.4 (80); DLBS 4.0 → 6.6 (45–55) → 4.0 (85); held-out SD of g 6–11 y (Σ estimated from 70–170 people per window → too few for p = 91).
IXI cross-sectional curves agree with DLBS for thalamus, hippocampus, cortex, ventricles; disagree for putamen and young-age entorhinal (segmentation-sensitive).
PNAS LBA on IXI (536, our FastSurfer+N4 stand-in for brain.mgz): global LBA r = 0.88 with age but slope 0.36 → gap +29 y in 20s … −8 y in 80s (MAE 14 y; 6.1 y after linear recalibration). Within-brain SD 0.34 y; SD across 95 region means 0.18 y → maps ≈ one global age. Regional slopes 0.355–0.372 (almost identical) → P4 cannot be meaningfully tested on this model; sign pattern goes the predicted way (corr(β_r, regional gap) −0.74 in 20s → +0.09 in 70s) but weakly, and crossing ages scatter. NOTE: the "relative slope = β_r − β" check is a tautology, not evidence. Domain shift (no recon-all) is a caveat.

### Pooled healthy normative model (2026-10-12) — `pilot/p3_pooled.py`, `p3_spread_check.py`, `p3_figures.py` → `pilot/p3_out/`
- 1,471 healthy people, 7 scanner/software sites (ADNI FreeSurfer 1.5T/3T, DLBS Philips 3T, IXI Philips 1.5T/3T, IXI GE 1.5T, OASIS-2 Siemens 1.5T), ages 20–93; 17 regions (14 shared with ADNI); 91 hemisphere features for the floor (FastSurfer sites).
- Model per region (exact Gaussian ML): log volume = spline(age) + sex + b·log ICV + site offset; log σ = spline(age) + site offset; age × sex for hippocampus and thalamus. Scanner-specific σ chosen by LR tests (13/17 p < .05) and held-out calibration.
- Held-out (5-fold) calibration: SD(z) 1.02–1.05 on every site, 94–95 % within ±1.96, every age band; tails a bit heavy (excess kurtosis 0.3–1.2) → SHASH/t later.
- Within-person rates from DLBS + OASIS-2 (ICV-rescaled). R_long (R_cross) in years at 30 / 50 / 70 / 85: cortex 8 / 10 / 8 / 6; total brain 54 / 13 / 7 / 6; thalamus 75 / 15 / 11 / 11; lat. ventricles 36 / 18 / 12 / 9; hippocampus 35 / 115 / 16 / 11; entorhinal 185 / 92 / 15 / 9; amygdala 77 / 38 / 12 / 8; brainstem 34 / 60 / 22 / 18.
- Cross-sectional slopes understate within-person aging for entorhinal, amygdala, putamen, brainstem, precuneus, cortex after ~50 (×1.5–2.5); similar for hippocampus, thalamus, ventricles, WM.
- Measurement resolution M (OASIS-2 same-session repeats) 0.2–5 y ≪ R → biology, not scanner noise, limits regional age.
- Whole-brain floor (91 features): cross-sectional direction 3.3 (20) → 6.8 (50) → 4.2 (85) y; within-person direction 2.6 → 5.1 → 2.7 y; achieved held-out 3.6–8.6 y.
- Head-size exponents b: hippocampus 0.70, thalamus 0.78, cortex 0.99, WM 1.16, total brain 1.02, lateral ventricles 2.35. Remaining sex offsets small (|·| ≤ 2 %) except inferior lateral (+19 %) and third ventricle (+9 %, men larger).
Figures: p3_fig1_resolution_atlas, p3_fig2_rates, p3_fig3_floor_scanners.

### More data, every age, model contest (2026-10-12 night)
Data search:
- OpenNeuro (all 1,905 datasets indexed: `data/openneuro_index.json`, `openneuro_age_scan.csv`): 12 CC0 healthy-adult sets selected → `data/openneuro_jobs.csv`: 3,285 scans, 2,618 people, 1,356 aged 60+ (Precision Aging Network 968, MiND 287 [+49 MCI for later], NIMH volunteers 239, neurocognitive aging 301, GRAND 109, AgeRisk 187, lifespan sets ...). Kaggle kernels `kaggle/on_fs_p0..p3` (FastSurfer + SynthStrip on GPU + scanner metadata); `kaggle/watch_openneuro.py` pushes them when the weekly GPU quota (30 h, exhausted 2026-10-12) resets; 2-scan CPU test `on-fs-cputest`.
- OpenBHB (HuggingFace, 3,984 healthy, 6–88, raw T1 + FreeSurfer/CAT12 features; mostly 18–30): downloading implies accepting the source cohorts' DUAs → ASK USER. Only participants.tsv/README fetched.
- NKI-Rockland (1,343) and SALD (494): scans public but ages need DUA/login; CoRR ages incomplete in BIDS (CoRR is inside OpenBHB).
- Need user applications: Cam-CAN, OASIS-3, UK Biobank, HCP-A/D, AIBL, NACC, PPMI.
Every age (`pilot/p4_everyage.py` → `p4_out/everyage_table.csv`, `everyage_site_params.csv`): 2,439 people, 27 sites (adult pool + ABIDE/ADHD-200 young controls), every year 8–89; per region × sex × age: centiles 2.5–97.5 (mm³, reference head 1,409 cm³), spread, cross/within slopes, S, R, M; per site shift + spread factor + head exponent. Head-size exponent estimated per ICV method (SynthStrip / ADNI eTIV / FS5 youth eTIV — the noisy youth eTIV would otherwise dilute b). Reference spread at each age = average over the scanners that scanned that age.
- Childhood is informative: R ≈ 5–8 y for brainstem, amygdala, hippocampus, WM at 8–12; cortex 7–9 y at 12–20. Midlife (25–50) poor for most regions; aging after 60.
Model contest (`pilot/p4_models.py` → `p4_out/model_comparison.csv`; 8 regions, 5-fold held-out): SHASH ≈ Gaussian (CRPS 9.115 vs 9.119) > quantile regression (9.158) > MLP (9.169; worse age-band calibration, aging rate shrunk to ~75 % of others) ≫ boosted trees (unstable rates) > Gaussian process (no per-site spread). Keep Gaussian/SHASH; rerun contest when the OpenNeuro data arrive.

### Data processing queue (started 2026-10-12 22:30)
User approved OpenBHB download (research, non-commercial) and gave a 2nd Kaggle account (kavianasli).
Queue `data/all_jobs.csv`: 6,720 scans / 6,325 healthy people / 1,295 aged 60+ = OpenNeuro 11 CC0 sets (2,726 scans) + OpenBHB minus IXI (3,500) + SALD (494, CC BY-NC, ages from NITRC sub_information.xlsx).
`kaggle/watch_queue.py` (both accounts, 5 CPU + 2 GPU kernels each when quota allows; state `kaggle/on_fs_state.json`; results `kaggle/on_fs_results/<kernel>/`; log `kaggle/_queue_watch.log`). Kernel template `kaggle/on_fs_template.py`.
OpenBHB extra tables already local: `data/openbhb/` (participants, qc, Desikan surface features, CAT12 ROI volumes).
After processing: drop the old ABIDE FS5.1 table people (ABIDE-I controls are inside OpenBHB study 1), rebuild pooled model + every-age tables, rerun model contest.
2026-10-13: user's OASIS-1/2 DUA approved → OASIS-1 added to the queue (436 sessions incl. 100 AD + 20 reliability rescans; mpr-1 for all + mpr-2 same-session repeat; streamed per disc from download.nrg.wustl.edu inside the kernel). Queue now 7,592 scans / 6,741 people. E-DAIC (USC ICT depression interview corpus, 275 × tar.gz, 110.8 GB) is NOT MRI — asked user what it is for before downloading.
2026-10-13 evening: OASIS-1 RAW is Analyze with orient=2 (sagittal) which nibabel ignores -> first 179 OASIS-1 results were segmented on rotated heads (hippocampus ~0). Fixed loader (axis0 P->A, axis1 I->S, axis2 sagittal L->R, tag o1_orient=ASR-v1), validated locally (OAS1_0001: hippocampus 3829/3701 mm3); untagged OASIS-1 rows are ignored and re-queued by watch_queue.py. L/R direction of axis2 to be checked on the cohort (population asymmetries). Kaggle: TPU = TpuV5E8/TpuV6E8 (Tpu1VmV38 retired), 1 TPU session per account and it counts toward the 5 CPU sessions; NvidiaTeslaT4 = 2 GPUs (template now splits chunks per GPU); quotas reset Sat 17 Oct 00:00 UTC. Watcher runs detached (Start-Process) - restart it after a reboot.
2026-10-14: OASIS-1 axis2 runs R->L (cohort asymmetries reversed under ASR-v1: R-hip>L 28 % vs 72 % elsewhere) -> templates now ASL-v2; ASR-v1 rows are kept with hemispheres swapped by `pilot/on_fs_load.py` (the loader for all queue results; also QC flag). TPU via API = plain 4-CPU session (0 TPU hours charged, kaggle-cli #1197) -> TPU disabled (N_TPU=0), 5 CPU per account. Throughput: CPU ~33 scans / job / 11 h (~18 min/scan); 1,852 done on 14 Oct 10:00.
2026-10-18: queue essentially complete - 7,579 / 7,590 scans valid (6,741 people incl. all primaries, 1,795 aged 60+, ages 6-96), 42 QC failures (0.6 %); last 11 leftovers in on-fs-g179. 2 scans dropped as unavailable (OAS1_0080_MR2 not in the OASIS-1 discs; data/jobs_unavailable.csv). Lessons: local proxy outages stalled the watcher (now: 3-min socket timeout, heartbeat file, network failures retried every 10 min); 2-GPU T4 jobs worked. NEXT: pool everything (pilot/on_fs_load.py + p3/p4 pipeline), rebuild atlas, every-age tables, model contest.
2026-10-18: analysis moved to the cloud (user's laptop is short of space). Private Kaggle dataset mohammadparsakavian/brainage-normative-inputs (code + participant tables incl. ADNI [user confirmed permission] + all result JSONL, 109 MB). Full pool (pilot/p5_full_pool.py): 8,480 healthy, 77 sites, ages 6-94, 2,280 aged 60+; repeat pairs OASIS-1 425 + ds005405 192 + OASIS-2 150. Kernels p5-cloud-a (models, every-age, atlas, calibration, floor, figures) and p5-cloud-b (model contest, adults); kaggle/wait_p5.py downloads outputs to pilot/p5_out. To update code/data: re-upload a new dataset version (dataset_create_version) and re-push the kernels.
2026-10-19 cloud results (pilot/p5_out): full pool 8,480 healthy / 77 sites / ages 6-94. Held-out calibration within +-1.96: 93.5-94.8 % in every age band, SD(z) 1.01-1.13 per dataset; scanner-specific spread significant 17/17. Whole-brain floor (cross dir.) 1.7 y @10, 3.5 @20, 7-8.4 @30-55, 6.4 @70, 4.4 @85; held-out now ~= theory; within-person dir. 2.2 -> 5.9 -> 3.0. Contest on 7,296 adults: SHASH best in 7/8 regions (CRPS 8.405 vs Gaussian 8.410), QR best per-site calibration, MLP aging rate shrunk to 82 %, GBM unstable, GPR worst -> adopt SHASH as main model. Repeat noise: cerebellum cortex eps 8.7 % (suspicious, check by source). Contest crash fix: bounded log-variance in P.LS + pooling tiny adult scanner groups.
2026-10-19 data QA (user: "make sure the preprocessing and datasets are correct before anything else"):
- Matched-age agreement across datasets (ICV, brain/ICV, hippocampus, ventricles); OASIS eTIV vs SynthStrip r 0.95-0.96, nWBV vs brain/ICV r 0.94/0.77 with matching age slopes; ADNIMERGE vs TADPOLE hippocampus r 0.987. FastSurfer vs OpenBHB's own FreeSurfer/CAT12 on the same 3,479 people: r 0.77-0.96, aging rates after 45 similar (hippocampus -0.61 vs -0.78 %/yr) -> no attenuation by FastSurfer. ADNI CN hippocampus declines faster (-0.98 %/yr vs -0.23 to -0.84) = cohort, not pipeline.
- Field-of-view cut-off found: slabs that miss the bottom of the head (cerebellum / brainstem cut). Excess low-cerebellum tail only in ds005405 (5 % below robust z -3; cut-off varies by session, 17/194 sessions have runs disagreeing > 10 %) and OpenBHB (1.75 % vs 0.3 % high side); all other sources symmetric. Some scans have lower-brain labels with zero voxels (brainstem 0.9 cm3).
- Fix in p5_full_pool.fov_qc: z < -5 (cerebellum/ICV or brainstem/ICV, age spline + sex + site medians) or missing lower-brain label -> scan excluded (41 scans; 31 OpenBHB primaries); -5..-3 in OpenBHB/ds005405 -> cerebellum, brainstem, total brain set missing (55 scans); ds005405 primary = least-cut of the person's 4 scans (sub-127: cerebellum 68 -> 92 cm3). OpenBHB own-site threshold 20 -> 15 (an 18-child ABIDE site with 0.86x1.5x0.86 mm voxels had head-size masks ~8 % large inside the pooled "other sites"). Flags: p5_out/qc_fov_flags.csv. Pool now 8,450 healthy / 86 sites.
- Visual QC kernels mohammadparsakavian/viz-check-seg (23 scans, one per source + suspects) and viz-check-seg2 (10 suspects, with SynthStrip outline) -> pilot/viz_check*. Kaggle GPU allowance now 6 h/week per account (kavianasli exhausted until 24 Oct).
- p5-cloud-a/b re-run on the corrected pool (dataset version "Field-of-view QC ...").
- Visual review (CPU kernels kavianasli/viz-cpu-*; panels re-oriented by pilot/viz_fix/reorient.py; sheets pilot/viz_fix/sheet1/2): typical scans of IXI, DLBS, OASIS-2, OASIS-1 25/55/80, OpenBHB, SALD, ds006395, ds007522, ds007831 segment correctly; cut-off scans confirmed (slab edge through posterior fossa); ds005405 bad runs = signal fall-off at the bottom of the head (cortex also -12 % in that run); OpenBHB poor-contrast scans under-segment cortex; two extreme atrophy cases are genuine (kept). ds007831 inputs are pre-skull-stripped but CSF kept: brain/ICV 0.747, aging slope and flat ICV-age like the others -> no action. Same for ds004711 (cortex/ICV mid-range, cortex -0.35 %/yr after 40). All 8 remaining OpenNeuro typical scans (sheet3) segment correctly -> every source has passed visual review.
- Image-quality QC (p5_full_pool.image_qc): OpenBHB CAT12 IQR > 3 (worse than grade C; low-cortex outliers 12-20 % vs < 1 %) -> 36 excluded; BAD_SITES = OpenBHB s1 site53 (17 children, Euler median -103, cortex under-segmented). Flags p5_out/qc_image_flags.csv. Final pool 8,401 healthy / 85 sites / 6-94 y / 1,139 under 18 / 2,281 aged 60+. Final cloud rerun pushed 2026-10-19 21:10 (dataset version "Image-quality QC ...").
2026-10-19/20 FINAL cloud run (SHASH, cleaned pool; outputs pilot/p5_out, pre-cleaning Gaussian run kept in pilot/p5_out_gauss):
- 8,401 healthy / 85 sites; every-age table ages 7-89 x 17 regions x 2 sexes, all centiles ordered, no gaps. Held-out calibration 94.3 % within +-1.96 (bands 93.1 % <18 ... 95.0 % 75+); per dataset 92-95 % (loosest: small or pooled sites, ds005026, ds006395, OpenBHB children's sites). Site-specific spread significant 17/17.
- SHASH: most regions slightly left-skewed (nu -0.05..-0.2) with heavier tails (tau 0.87-1.0); ventricles right-skewed; cerebellum/brainstem now ~Gaussian (gain 0.6 / 3.6) -> their old heavy low tail was the cut-off scans. Repeat noise ds005405 cerebellum 8.7 % -> 2.8 % (OASIS-1 0.55 %).
- vs pre-cleaning Gaussian run: aging rates unchanged (<= 0.08 %/yr), ventricle medians -2-3 % (skew), spreads slightly narrower; floor cross 1.7 (10) / 3.3 (20) / 8.1-8.3 (40-55) / 6.7 (70) / 4.4 (85) y, within-person 1.9 / 2.5 / 5.0-5.5 / 4.6 / 2.7 y; held-out 3-20 % above theory, up to 49 % at 85 (small windows, 85 features).
- Model contest: 7,262 adults, 59 sites: SHASH best (CRPS 8.335 vs Gaussian 8.339), MLP rate 0.83x, GBM unstable, GPR worst. Fig 3 title was hard-coded "1,471 people" -> now from p5_out/contest_meta.json (written by part B).
- Site corrections: median 3.6 %, 95th pct 18 %; > 20 % almost only inf. lat. and third ventricle (scanner-dependent tiny CSF spaces) and ADHD-200 (FreeSurfer 5: putamen +21-32 %, cortex spread x2.6) -> treat those regions/sources as least portable.
2026-10-20 checks (user questions):
- Region alignment: 7,603 / 8,401 people = identical FastSurfer 2.4.2 labels (aseg + DKT). ADNI = UCSF FreeSurfer 4.3/5.1, Desikan atlas, 14/17 regions, names partly reconstructed (6 anchors exact) -> all 14 within 0.83-1.28x of FastSurfer per ICV, same aging sign (mislabel would be ~2x off); official dictionary from LONI still wanted. ADHD-200 = FreeSurfer 5 tables, 11/17 regions, 9-44 % larger per ICV. Site offsets absorb constant differences only. TODO options: reprocess ADHD-200 raw (INDI, public) + ADNI raw (LONI) with FastSurfer.
- Ventricles verified: within one scanner / sex / 5-yr band the 95th/5th centile ratio is 3.3-4.6x in all 7 groups checked (SD of log 38-49 %, head adjustment barely changes it); same-session repeat difference median 0.5 % -> real anatomy.
- ADNI CN (399; ADNIMERGE 2026, follow-up median 6.5 y): amyloid+ 44 % of measured (146/329), later MCI/dementia 25 % (101; dementia 28), APOE4 27 %; preclinical vs clean: hippocampus -1.8 %, ventricles +12 %. BUT the steep ADNI hippocampal slope is NOT preclinical AD (clean subset -0.87 vs all -0.80 %/yr): FastSurfer reweighted to ADNI's ages gives -0.73 vs ADNI -0.89 (amygdala -0.80 vs -0.82, ventricles equal); residual gap mostly cortical parcels (precuneus -0.40 vs -0.68) = older FreeSurfer. Earlier "ADNI steeper = preclinical AD" claim was wrong -> mostly age mix. Status file kaggle/_adni_cn_status.csv.
- Within-person rates rest on DLBS 314 people (2-3 scans, median 8 y, ages 21-97) + OASIS-2 72 nondemented (2-5 visits, median 3 y; converters already excluded).
- Uncalibrated new scanner: site-shift SD 2-10 % -> z error 0.24-0.79 SD (cortex worst) -> ~20+ local healthy controls needed.
2026-10-20 evening - SINGLE-PIPELINE ATLAS (user decisions): ADNI dropped from the healthy atlas (kept for disease phase); ADHD-200 FS5 tables dropped; ADHD-200 raw reprocessing (545 TDC, data/adhd200_raw/adhd200_tdc_jobs.csv, INDI S3) prepared, started, then STOPPED by the user (GPU allowance gone -> ~11 h on CPU); add later. all_jobs.csv restored to 7,590.
- Pool = 7,603 healthy, 17 sources, 77 scanners, ages 6-94 (871 under 18, 1,784 aged 60+), 4,201 women; one head-size method (SynthStrip), reference head 1,538 cm3.
- New outputs (p5_full_pool part A): slope_long / R_long CIs; trajectories.csv (each DLBS/OASIS-2 repeat scan mapped to the reference scale by its own centile); individual_rates.csv (per-person %/yr, true person-to-person SD after removing visit-to-visit noise); noise_budget.csv (same-session vs longitudinal noise; DLBS ~2x OASIS-2); eps_long_pct / M_long in the every-age table. Figures p5_fig4_trajectories, p5_fig5_normalization_noise.
- Longitudinal noise >> same-session: hippocampus 2.6 % vs 0.85 %, cortex 2.2 vs 0.43, ventricles 4.4 vs 0.64 (DLBS scanner scale drift -0.2 %/yr in ICV, corrected by ICV rescaling). True person-to-person spread of aging speed resolvable only for ventricles (SD 1.2-1.9 %/yr) and hippocampus 70+ (0.4-1.2 %/yr).
- Cloud run p5-cloud-a/b pushed 2026-10-20 ~19:40 (dataset version "Single pipeline ...").
2026-10-21 LITERATURE CROSS-CHECK (all verified from PubMed/PMC abstracts or full text):
- Agree: lifespan shapes (Bethlehem 2022 Nature; Dima 2022 HBM ENIGMA: putamen early peak + linear decline, hippocampus/amygdala/thalamus stable to 6th decade, variability rises with age); healthy hippocampal atrophy (Fraser 2015 NeuroImage: 0.38 / 0.98 / 1.12 %/yr for <55 / 55-70 / 70+, automated -0.47) vs ours ~0 / 0.25-0.5 / 0.5-0.8; whole brain (Hedman 2012 HBM 0.2 -> >0.5 %/yr; Resnick 2003 J Neurosci 5.4 cm3/yr, ventricles +1.4 cm3/yr); cortex ~0.5 %/yr (Fjell 2009); allometry order WM>1>cortex>deep GM (de Jong 2017 HBM: WM 1.14, cortex 0.75-0.79, MTL 0.60, thalamus 0.59, cerebellum 0.55; ours higher for deep GM, sample-dependent per authors); sex beyond size ~1 % variance (Eliot 2021); ventricles vary between days (Maclaren 2014); stable individual differences before 60 (Grødem 2026; Vidal-Pineiro 2021; Korbmacher 2025).
- Explained differences: our charts are head-size-adjusted -> WM peak 46 (raw WM in our data peaks 30-40 = Bethlehem 28.7), total brain peak ~8 (raw keeps growing to late teens; children's skulls grow +5 %, older cohorts' heads ~4 % smaller).
- CORRECTION: within-person cortical rates are inflated by DLBS (cortex -0.76 DLBS vs -0.40 OASIS-2 vs -0.37 cross-sectional at 70-82; precuneus -0.90 / -0.44 / -0.39) -> DLBS scanner drift; tissue-rank agreement cross vs within only rho 0.28 (Fjell 2013: 0.91). Treat R_long for cortex/parcels as provisional; medial temporal within-person rates faster in OASIS-2 (hippocampus -1.0, entorhinal -1.8) = likely preclinical AD admixture (Fjell 2009).
- Brain-age bridge: healthy volumetric brain-age SD (our floor, held-out) 8.4 / 7.3 / 6.4 y at 60/70/80; AD BAG +6-10 y (Franke & Gaser 2019) = d ~0.9-1.4 (Bethlehem AD GMV d 0.88) -> group-real, individually weak (Tan 2025 AUC 0.70; our OASIS pilot 0.709). Hippocampus AD -24 % / MCI -12 % (Shi 2009) = z -3.5 / -1.6 at 70 (= 53 / 25 "hippocampal years": years unit breaks for disease). Rates: AD 4.66 vs controls 1.41 %/yr (Barnes 2009); visit noise makes 1-yr individual change unreadable (Elliott 2026), 2-3 yr ok for hippocampus; total-brain monitoring resolution 3.3 y per scan at 70 vs AD +1.5 brain-yr/yr.
2026-10-21 PHASE 6 DISEASE TEST (pilot/p6_disease_test.py -> pilot/p6_out; 5-fold: controls scored by atlas models refitted without them):
- OASIS-1 CDR 0.5 (n=68) / CDR>=1 (n=25) vs 114 controls >=55; OASIS-2 demented 64 / converted 14 (baseline) vs 72 nondemented. AUC (sensitivity at 95 % specificity):
  conventional ridge brain-age gap (healthy SD 8.6-9 y): 0.58 (9 %) / 0.72 (24 %) / O2 dem 0.64 (12 %); median patient gap +4.5 / +8.6 y
  atlas brain-age gap (GLS on z, healthy SD 7-8 y): 0.79 (25 %) / 0.91 (60 %) / 0.83 (44 %); median +7.4 / +15.0 / +8.1 y
  hippocampus z: 0.76 (44 %) / 0.83 (52 %) / 0.76 (44 %); median z only -1.2 / -1.5 / -1.3 (31-40 % beyond -1.96) -> literature-based "-3.5" estimate was too optimistic
  AD-signature mean z: 0.81 (50 %) / 0.91 (60 %) / 0.83 (55 %); learned multi-region: 0.84 / 0.90 / 0.87; outlier count median 1 / 4 / 2 regions
  converters at baseline: nothing separates (AUC 0.39-0.58, n=14); their within-person hippocampal rate does a bit (-1.8 vs -1.0 %/yr, AUC 0.69; amygdala 0.67)
  OASIS-2 demented rates: hippocampus -2.4 vs -1.0 %/yr (AUC 0.75), amygdala -3.9 vs -1.1 (0.75), entorhinal -5.9 vs -1.8 (0.70)
- Consistent with Tan 2025 (scalar gap AUC 0.70 vs direct 0.92). Caveats: single scanner per cohort, clinical diagnosis without amyloid, small n (CIs in summary.csv).
Applications (user): NACC, OASIS-3, ADNI+AIBL(+A4/LEARN) via LONI, Cam-CAN, PPMI, HABS-HD, NKI-RS DUA, HCP-A/D (NDA), UK Biobank.

Next: pull IXI + DLBS results → pooled normative models (FastSurfer space: IXI + DLBS + OASIS-2) → per-dataset replication of the atlas → P4 test on IXI LBA maps → DLBS lifespan within-person slopes.

## Decisions pending
- Go / no-go for Phase 1.
- Gaussian vs GAMLSS normative model; covariates.
- Threshold for reporting years (e.g., R ≤ 10 years?).
- Feature set for Phase 2 (volumes, thickness, brainstem/cerebellum; later diffusion, iron).
