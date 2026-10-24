# Head-to-head with PCN braincharts (Rutherford et al. 2022, eLife)

Scores the same 357 OASIS people as `pilot/p6_disease_test.py`, with the same published folds, using PCN's
`lifespan_57K_82sites` model. PCN learns the OASIS scanner from exactly the controls our atlas used for it
(that cohort's pool controls minus the fold being scored).

## Setup (once)

1. Clone the pretrained models (≈ 20 MB, no LFS):
   `git clone --depth 1 https://github.com/predictive-clinical-neuroscience/braincharts external/pcn_braincharts`
   (or point `PCN_BRAINCHARTS` at an existing clone).
2. PCN's toolkit needs Python ≤ 3.12. Make a separate environment, e.g. with `uv`:
   ```
   uv venv -p python3.11 pcnenv
   VIRTUAL_ENV=pcnenv uv pip install --no-deps "pcntoolkit==0.35"
   VIRTUAL_ENV=pcnenv uv pip install "bspline>=0.1.1,<0.2" "nibabel>=5.3.1,<6" six "scikit-learn>=1.5.2,<2" "scipy>=1.12,<2" \
        "matplotlib>=3.9.2,<4" "seaborn>=0.13.2,<0.14" "numba>=0.60,<0.61" "nutpie>=0.13.2,<0.14" "pymc>=5.18,<6" pandas
   ```
   torch is deliberately **not** installed (the CUDA build is ~5 GB): the braincharts models are Bayesian linear
   regressions that never use it, but pcntoolkit imports it eagerly. `torchstub/` is a placeholder that lets the
   import succeed and raises `TorchPlaceholderUsed` if anything ever calls it, so a completed run proves torch
   was not needed.
3. pcntoolkit 0.35's `predict()` looks for `NM_0_*_fit.pkl`; the braincharts files are `NM_0_0_estimate.pkl`,
   so `b_pcn_score.py` passes `inputsuffix="estimate"`.

## Run

```
python pilot/v2/pcn/a_export.py                                                  # main env: people.csv, adapt_pool.csv
PYTHONPATH=pilot/v2/pcn/torchstub pcnenv/bin/python pilot/v2/pcn/b_pcn_score.py selftest   # PCN on its own example
PYTHONPATH=pilot/v2/pcn/torchstub pcnenv/bin/python pilot/v2/pcn/b_pcn_score.py oasis      # pcn_z.csv
python pilot/v2/pcn/c_compare.py && python pilot/v2/pcn/d_extra.py              # main env: cmp_auc.csv, cmp_calibration.csv
```

Caveat: PCN was trained on FreeSurfer output and is fed FastSurfer volumes here; its scanner adaptation absorbs
constant offsets but not region-specific age-dependent differences (this works against PCN).
