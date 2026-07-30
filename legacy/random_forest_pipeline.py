# %%
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import random

# %% Dataset paths
DATASET_DIR = Path('.')

HSI_AIRBORNE_DIR = DATASET_DIR / 'train' / 'hsi_airborne'
HSI_SATELLITE_TRAIN_DIR = DATASET_DIR / 'train' / 'hsi_satellite'
HSI_SATELLITE_TEST_DIR = DATASET_DIR / 'test' / 'hsi_satellite'
MSI_SATELLITE_TRAIN_DIR = DATASET_DIR / 'train' / 'msi_satellite'
MSI_SATELLITE_TEST_DIR = DATASET_DIR / 'test' / 'msi_satellite'
GT_TRAIN_CSV_PATH = DATASET_DIR / 'train_gt.csv'

# %% Load the ground truth measurements
gt_train_df = pd.read_csv(GT_TRAIN_CSV_PATH)
y_train_all = gt_train_df[['Fe', 'Zn', 'B', 'Cu', 'S', 'Mn']].values

# %% Load and preprocess data from a directory
def load_masked_mean(directory):
    files = sorted(directory.glob('*.npz'))
    data = []
    for file in files:
        with np.load(file) as npz:
            arr = np.ma.MaskedArray(**npz)
            soil_pixels = arr[:, arr.mask[0] == False]  # Use mask from first band
            mean_pixel = np.mean(soil_pixels, axis=1) if soil_pixels.size > 0 else np.zeros(arr.shape[0])
            data.append(mean_pixel)
    return np.array(data)

# %% Load and reduce HSI with PCA
def fit_pca_from_dir(directory, n_components):
    raw_data = load_masked_mean(directory)
    pca = PCA(n_components=n_components)
    features = pca.fit_transform(raw_data)
    return features, pca

def transform_with_pca_from_dir(directory, pca):
    raw_data = load_masked_mean(directory)
    return pca.transform(raw_data)

# %% Apply modality dropout
def apply_modality_dropout(X_airborne, X_satellite_hsi, X_satellite_msi, dropout_prob):
    X_combined = []
    for i in range(X_airborne.shape[0]):
        include_airborne = random.random() > dropout_prob
        airborne = X_airborne[i] if include_airborne else np.zeros_like(X_airborne[i])
        x = np.concatenate([airborne, X_satellite_hsi[i], X_satellite_msi[i]])
        X_combined.append(x)
    return np.array(X_combined)

# %% Load all training data
print("Loading and processing training data...")
X_airborne_raw = load_masked_mean(HSI_AIRBORNE_DIR)
X_hsi_satellite_raw = load_masked_mean(HSI_SATELLITE_TRAIN_DIR)
X_msi_satellite_raw = load_masked_mean(MSI_SATELLITE_TRAIN_DIR)

X_airborne, pca_air = fit_pca_from_dir(HSI_AIRBORNE_DIR, n_components=10)
X_hsi_satellite, pca_hsi = fit_pca_from_dir(HSI_SATELLITE_TRAIN_DIR, n_components=10)
X_msi_satellite = X_msi_satellite_raw  # Already 13 bands

# %% Combine and split data
X_combined = apply_modality_dropout(X_airborne, X_hsi_satellite, X_msi_satellite, dropout_prob=0)

X_train = X_combined[:1300]
y_train = y_train_all[:1300]
X_val = X_combined[1300:]
y_val = y_train_all[1300:]

# %% Train one model per target using XGBoost
print("Training separate XGBoost models per target...")
columns = ['Fe', 'Zn', 'B', 'Cu', 'S', 'Mn']
models = {}
y_preds = []

for i, col in enumerate(columns):
    print(f"Training model for {col}...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train[:, i])
    models[col] = model
    y_pred = model.predict(X_val)
    y_preds.append(y_pred)

# %% Evaluate
print("Evaluating...")
y_preds = np.stack(y_preds, axis=1)
baseline_pred = np.tile(np.mean(y_train, axis=0), (y_val.shape[0], 1))

mse_model = mean_squared_error(y_val, y_preds, multioutput='raw_values')
mse_baseline = mean_squared_error(y_val, baseline_pred, multioutput='raw_values')
scores = mse_model / mse_baseline

for name, score in zip(columns, scores):
    print(f"{name} score: {score:.4f}")

print(f"Final score using RF with PCA with dropout (mean over all): {np.mean(scores):.4f}")

# %% Load and predict on test set
print("Preparing test set...")
X_hsi_satellite_test = transform_with_pca_from_dir(HSI_SATELLITE_TEST_DIR, pca=pca_hsi)
X_msi_satellite_test = load_masked_mean(MSI_SATELLITE_TEST_DIR)
X_airborne_dummy = np.zeros((X_msi_satellite_test.shape[0], X_airborne.shape[1]))
X_test_final = np.concatenate([X_airborne_dummy, X_hsi_satellite_test, X_msi_satellite_test], axis=1)

print("Predicting test set...")
y_test_pred = []
for col in columns:
    pred = models[col].predict(X_test_final)
    y_test_pred.append(pred)

y_test_pred = np.stack(y_test_pred, axis=1)
submission = pd.DataFrame(data=y_test_pred, columns=columns)
submission_filename = "submission.csv"
submission.to_csv(submission_filename, index_label="sample_index")
print(f"Submission file saved as {submission_filename}")
