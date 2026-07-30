> 4AM Productions presents...

# HYPERVIEW2 Challenge

HYPERVIEW2 is a (now-closed) ESA Φ-Lab / KP Labs challenge - part of the EASi workshop at ECAI 2025 - that scores participants on both prediction accuracy (the HYPERVIEW Score) and quality of the explainability analysis.

The repository here contains a machine learning pipeline for predicting six soil properties from multi-modal **satellite and airborne hyperspectral imagery**, built for the [HYPERVIEW2 Challenge](https://www.igik.edu.pl/en/hyperview-challenge).

---

## Table of Contents

- [Problem](#problem)
- [Data](#data)
- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
- [Approach](#approach)
  - [Feature Engineering](#feature-engineering)
  - [Model](#model)
  - [Evaluation](#evaluation)
  - [Explainability (XAI)](#explainability-xai)
- [Configuration](#configuration)
- [Usage](#usage)
- [Results](#results)
- [Development Notes](#development-notes)
- [Future Work (& potential improvements to current approach)](#future-work--potential-improvements-to-current-approach)

---

## Problem

The challenge was to predict six soil chemical properties from hyperspectral and multispectral satellite/airborne imagery over agricultural fields:

| Target | Unit |
|--------|------|
| Boron (B) | mg/kg |
| Copper (Cu) | mg/kg |
| Zinc (Zn) | mg/kg |
| Iron (Fe) | mg/kg |
| Sulfur (S) | mg/kg |
| Manganese (Mn) | mg/kg |

The competition metric is the **HYPERVIEW Score**: the average normalized mean squared error (MSE) across all six targets, where normalization is relative to a naïve baseline (always predicting the training mean).

```
HYPERVIEW Score = mean( MSE_i / MSE_baseline_i )   for i in {B, Cu, Zn, Fe, S, Mn}
```

---

## Data

The raw imagery and ground-truth CSV used by this pipeline are hosted on [EOTDL](https://www.eotdl.com/datasets/HYPERVIEW2). Download the data from there into data/raw/ following the layout in [project structure](#project-structure) below.

The dataset provides three modalities per agricultural field:

| Modality | Sensor | Bands | Available |
|----------|--------|-------|-----------|
| `msi_satellite` | Sentinel-2 | 12 | Train + Test |
| `hsi_satellite` | PRISMA | 230 | Train + Test |
| `hsi_airborne` | airborne HSI | 430 | Train Only |

Each field is stored as a masked `.npz` file. The mask identifies valid soil pixels - the pipeline extracts only those pixels and takes the mean per band, reducing each field to a single reflectance vector.

In addition, the dataset provides a ground-truth CSV (`train_gt.csv`) of the six soil trace-element concentrations.

> **Note:** ~10% of fields have zero valid soil pixels when pooled across all three modalities. However, it is concentrated almost entirely in `msi_satellite` (Sentinel-2), which is missing valid soil pixels in ~31% of fields, while `hsi_satellite` and `hsi_airborne` have full coverage. These are handled by the mean falling back to zeros with a logged warning.

---

## Project Structure

```
├── config.yaml                  # all experiment settings
├── train_pipeline.py            # main training script
├── generate_submission.py       # inference on test set into submission.csv
│
├── src/
│   ├── data/
│   │   └── loaders.py           # NPZ loading, soil pixel extraction, train/val split
│   ├── features/
│   │   ├── pca.py               # variance filter → correlation filter → scaler → PCA
│   │   └── dropout.py           # modality dropout (training augmentation)
│   ├── models/
│   │   └── train.py             # per-target model training (Random Forest or XGBoost)
│   ├── evaluation/
│   │   ├── metrics.py           # HYPERVIEW score + MAE / RMSE / R²
│   │   └── xai.py               # SHAP (TreeExplainer) + spectral band importance
│   ├── validation/
│   │   └── validate_data.py     # pre-training data integrity checks
│   └── utils/
│       ├── logger.py            # dual file + console logger
│       └── reproducibility.py  # global seed setting
│
├── legacy/
│   ├── random_forest_pipeline.py  # original script (RF, adjustable dropout)
│   └── xgboost_pipeline.py        # original script (XGBoost, adjustable dropout)
│
├── data/
│   ├── raw/                     # download train/test datasets from eotdl site
│   │   ├── train/               # {msi_satellite, hsi_satellite, hsi_airborne}/*.npz
│   │   ├── test/                # {msi_satellite, hsi_satellite}/*.npz
│   │   └── train_gt.csv         # ground truth labels
│   └── wavelengths.json         # band center wavelengths per modality
│
├── features/                    # saved preprocessors and models (auto-created)
│   └── pca/                     # preprocessor_{modality}.pkl
│
├── logs/                        # training logs (auto-created, not uploaded here)
└── xai_results/                 # XAI plots and CSVs (auto-created)
```

### Legacy Scripts

The `legacy/` folder contains the original scripts written before the pipeline was refactored. They are kept for reference to show the progression of the work.

Both scripts share the same core logic as the refactored pipeline (soil pixel extraction, PCA, per-target modelling, HYPERVIEW scoring) but lack configurability, reproducibility, data validation, model persistence, and the feature engineering additions introduced during refactoring.

---

## Pipeline Overview

```
Raw .npz files
      │
      ▼
Soil pixel extraction → mean per band
      │
      ▼
Train / Val split (70/30)
      │
      ▼
[Optional] log1p target transform
      │
      ▼
Per-modality feature engineering:
  Variance filter → Correlation filter → StandardScaler → PCA
      │
      ▼
Modality dropout (airborne, training only)
      │
      ▼
Concatenate modalities → flat feature vector
      │
      ▼
Train one model per soil property (RF or XGBoost)
      │
      ▼
Evaluate (HYPERVIEW Score, MAE, RMSE, R²)
      │
      ▼
[Optional] XAI analysis
```

---

## Approach

### Feature Engineering

The core feature engineering is in `src/features/pca.py`. Each modality is processed independently through four (optional) steps fitted on training data and applied to validation/test:

#### 1. Variance Filter
Drops features in the bottom N% by variance across training samples. Low-variance bands carry little discriminative signal and can destabilise PCA. For example:

```yaml
use_variance_filter: true
variance_threshold: 5.0    # drop bottom 5% by variance
```

#### 2. Correlation Filter
Greedy pairwise correlation removal: iterates through features in order and drops the later feature of any pair with absolute correlation above the threshold. Hyperspectral data has many adjacent bands that are near-perfectly correlated; this removes the redundancy before it enters PCA. For example:

```yaml
use_correlation_filter: true
correlation_threshold: 0.99
```

> At 0.95, HSI satellite reduces from 230 → ~6 features and airborne from 430 → ~3, making PCA redundant. The current default is 0.99 which preserves more spectral information. Both filters are off by default - enable them experimentally.

#### 3. StandardScaler
Zero-mean, unit-variance scaling applied after filtering. PCA is sensitive to feature scale, so this is applied before it. Can be disabled to reproduce the original PCA-only baseline.

#### 4. PCA
Reduces each modality to a fixed number of components (configured per modality). All four preprocessing artefacts - variance mask, correlation mask, scaler, PCA model - are saved together as a single `preprocessor_{modality}.pkl` and reloaded for inference.

To restore the original pipeline → set all three flags to false:
```yaml
use_variance_filter: false
use_correlation_filter: false
use_scaler: false
```
This gives identical behaviour to a plain PCA-only baseline.

#### Modality Dropout
During training only, airborne HSI features are randomly zeroed out with a configurable probability. Since airborne data is unavailable at test time, this teaches the model to function without it and prevents over-reliance on the highest-quality modality. For example:

```yaml
dropout:
  airborne: 0.05    # 5% of training samples have airborne zeroed out
```

At inference, missing airborne features are filled with zeros, which would be consistent with what the model saw during dropout.

### Model

One independent regressor is trained per soil property, all on the same concatenated feature vector. Two model types are supported and switchable via config:

- **Random Forest** (`sklearn.ensemble.RandomForestRegressor`)
- **XGBoost** (`xgboost.XGBRegressor`) with Tweedie objective, suitable for right-skewed targets like soil concentrations

#### Log Transform
Soil properties have right-skewed distributions. Enabling `log_transform: true` applies `log1p` to targets before training, which compresses outliers and can improve fit. Predictions are inverse-transformed with `expm1` before scoring and submission. For example:

```yaml
train:
  log_transform: true    # true → log1p train, expm1 at inference
```

The HYPERVIEW score is always reported on the original scale, comparable to the leaderboards.

### Evaluation

The HYPERVIEW Score normalises each target's MSE against a naive mean-prediction baseline, then averages across targets. This makes scores comparable across soil properties with very different absolute scales (e.g. Fe in hundreds vs B in single digits).

Additional metrics (mean absolute error (MAE), root mean squared error (RMSE), R²) are logged per target for diagnostic purposes.

### Explainability (XAI)

SHAP analysis uses a *shortcut approach* to produce physically interpretable spectral band importances without the computational cost of running SHAP through the full preprocessing pipeline:

1. TreeExplainer runs on the PCA-compressed features - this is fast and exact for tree-based models
2. SHAP values are projected back to band space via the linear PCA mapping:
   ```
   band_importance = |shap_values_pca @ pca.components_|
   ```
   Since PCA is a linear transformation, this is mathematically valid and requires no additional model calls
3. The variance and correlation masks stored in the preprocessor are applied to the wavelength list so band labels on the x-axis correctly correspond to the surviving features

This produces physically interpretable spectral plots at a fraction of the compute cost of running SHAP on raw inputs. An `errors_all_targets.csv` file is also generated with the residuals for all targets.

**Plots Produced:**

| File | Description |
|------|-------------|
| `shap_spectral_bands.png` | 2×3 grid - mean \|SHAP\| vs wavelength (nm) per target, one line per modality |
| `shap_modality_importance.png` | grouped bar chart - % SHAP contribution of MSI / HSI-sat / HSI-airborne per target |
| `pred_vs_actual_grid.png` | scatter plots, predicted vs actual, all targets |
| `spatial_predictions_grid.png` | sequential prediction vs actual trace, all targets |

---

## Configuration

All settings are in `config.yaml`. The key options include:

```yaml
train:
  log_transform: true       # log1p targets before training
  val_split: 0.7            # 70% train, 30% val

feature_selection:
  use_variance_filter: false
  variance_threshold: 5.0
  use_correlation_filter: false
  correlation_threshold: 0.99
  use_scaler: true

pca:
  msi: 10
  hsi: 10
  airborne: 30

model:
  type: random_forest       # or xgboost

xai:
  enabled: false            # set true to run SHAP analysis
  wavelengths_path: data/wavelengths.json
```

---

## Usage

**Train:**
```bash
python train_pipeline.py
```

**Generate submission:**
```bash
python generate_submission.py
```

Trained models and preprocessors are saved to `features/`. Logs are written to `logs/train_pipeline.log`. If XAI is enabled, results are saved to `xai_results/`.

---

## Results

| Target | Normalized MSE | R² |
|--------|----------|-----|
| B  | 0.320 | 0.680 |
| Cu | 0.553 | 0.446 |
| Zn | 0.425 | 0.574 |
| Fe | 0.399 | 0.601 |
| S  | 0.583 | 0.417 |
| Mn | 0.447 | 0.553 |
| **Mean** | **0.4543** | — |

[Leaderboards](https://platform.ai4eo.eu/hyperview2/leaderboard).

> Scores are on the validation set (30% holdout), current shipped config: filters off, StandardScaler + PCA (10/10/30), XGBoost, `log_transform: false`. Reproduced directly from `train_pipeline.log`.

---

## Development Notes

- The original (legacy) scripts were written initially. The refactoring and extension into the current modular pipeline were done iteratively via Claude chat at first, and then via Claude Code (for practice with coding agents).
- The scoring discrepancy between log-space and original-space evaluation was identified during development - the pipeline now always reports scores in original scale for leaderboard comparability.
- The XAI shortcut (TreeExplainer + PCA projection) was chosen over a full PermutationExplainer approach for speed, at the cost of approximating the filtering step's effect on band attributions.

---

## Future Work (& potential improvements to current approach)

**Feature Engineering**
- Tune the correlation threshold per modality rather than applying a single global value - for example, HSI satellite and airborne lose too much information at 0.95
- Based on literature, (a) experiment with atmospheric band removal before filtering, which may improve HSI feature quality, and (b) investigate per-target feature selection to exploit the fact that different soil properties respond to different spectral regions

**Modelling**
- Implement per-target hyperparameter tuning (currently all six targets share the same model config)
- Experiment with: (a) ensemble methods combining RF and XGBoost predictions, (b) `ExtraTreesRegressor` as a drop-in replacement
- Test whether log-transform training actually helps each target individually rather than applying it globally (it is not recommended for current implementation)

**Pipeline**
- Implement cross-validation instead of a single holdout split for more robust score estimation
- Check stratified splitting for balanced train/val distributions across soil property ranges

**XAI**
- Try upgrading to PermutationExplainer with a Partition masker to fully account for the filtering and scaling steps in band attribution, rather than the current PCA projection shortcut
