# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 00:20:45 2026

@author: Hachem
"""

import time
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor

from src.data.loaders import soil_masks, load_labels
from src.features.pca import fit_PCA, transform_with_PCA
from src.features.dropout import apply_dropout
from src.evaluation.metrics import calculate_mse_scores, calculate_baseline_mse
from src.evaluation.xai import run_permutation_importance, run_shap_analysis, run_error_analysis
from scripts import validate_data  # Import the validation functions

def train_pipeline(config, dataset_dir):
    start_time = time.time()

    # Directories
    MSI_SAT_TRAIN = dataset_dir / 'data' / 'raw' / 'msi_satellite'
    HSI_SAT_TRAIN = dataset_dir / 'data' / 'raw' / 'hsi_satellite'
    HSI_AIR_TRAIN = dataset_dir / 'data' / 'raw' / 'hsi_airborne'
    TRAIN_GT = dataset_dir / 'data' / 'raw' / 'train_gt.csv'

    # ============================
    # Run data validation first
    # ============================
    print("\n=== Running Data Validation ===")
    validate_data.check_ground_truth(TRAIN_GT)
    validate_data.check_file_alignment(MSI_SAT_TRAIN, HSI_SAT_TRAIN, HSI_AIR_TRAIN)
    validate_data.check_mask_assumptions(MSI_SAT_TRAIN)
    validate_data.check_mask_assumptions(HSI_SAT_TRAIN)
    validate_data.check_mask_assumptions(HSI_AIR_TRAIN)
    validate_data.check_zero_soil_pixels(MSI_SAT_TRAIN)
    validate_data.check_zero_soil_pixels(HSI_SAT_TRAIN)
    validate_data.check_zero_soil_pixels(HSI_AIR_TRAIN)
    print("=== Data Validation Completed ===\n")

    # ============================
    # Load data
    # ============================
    y_all = load_labels(TRAIN_GT)
    X_msi_sat = soil_masks(MSI_SAT_TRAIN)
    X_hsi_sat = soil_masks(HSI_SAT_TRAIN)
    X_hsi_air = soil_masks(HSI_AIR_TRAIN)

    # Split train/validation
    split_idx = int(len(y_all) * 0.7)
    X_msi_train, X_msi_val = X_msi_sat[:split_idx], X_msi_sat[split_idx:]
    X_hsi_train, X_hsi_val = X_hsi_sat[:split_idx], X_hsi_sat[split_idx:]
    X_air_train, X_air_val = X_hsi_air[:split_idx], X_hsi_air[split_idx:]

    y_train, y_val = y_all[:split_idx], y_all[split_idx:]
    y_train = np.log1p(y_train)
    y_val = np.log1p(y_val)

    # PCA
    X_msi_train, pca_msi = fit_PCA(X_msi_train, config['pca_msi_components'])
    X_hsi_train, pca_hsi = fit_PCA(X_hsi_train, config['pca_hsi_components'])
    X_air_train, pca_air = fit_PCA(X_air_train, config['pca_air_components'])

    # Apply dropout
    X_train = apply_dropout(X_msi_train, X_hsi_train, X_air_train, dropout_prob=config['dropout_prob'])

    # Transform validation
    X_msi_val = transform_with_PCA(X_msi_val, pca_msi)
    X_hsi_val = transform_with_PCA(X_hsi_val, pca_hsi)
    X_air_val = transform_with_PCA(X_air_val, pca_air)
    X_val = np.concatenate([X_msi_val, X_hsi_val, X_air_val], axis=1)

    # Train per target
    columns = ['B', 'Cu', 'Zn', 'Fe', 'S', 'Mn']
    models = {}
    y_preds = []

    for i, col in enumerate(columns):
        print(f"Training model for {col}")
        t0 = time.time()
        model = XGBRegressor(
            objective="reg:tweedie",
            tweedie_variance_power=1.5,
            n_estimators=config['n_estimators'],
            learning_rate=config['learning_rate'],
            random_state=config['seed'],
            n_jobs=2
        )
        model.fit(X_train, y_train[:, i])
        models[col] = model
        y_pred = model.predict(X_val)
        y_preds.append(y_pred)
        print(f"{col} training completed in {time.time() - t0:.2f}s")

    y_preds = np.stack(y_preds, axis=1)

    # ============================
    # Evaluation
    # ============================
    model_mse = calculate_mse_scores(y_val, y_preds)
    baseline_pred = np.tile(np.mean(y_train, axis=0), (y_val.shape[0], 1))
    baseline_mse = calculate_baseline_mse(y_val, baseline_pred)
    scores = model_mse / baseline_mse

    print("\nValidation Scores:")
    for col, score in zip(columns, scores):
        print(f"{col}: {score:.4f}")
    print(f"Average Score: {np.mean(scores):.4f}")

    # ============================
    # Run XAI
    # ============================
    print("\n=== Running XAI ===")
    feature_names = [f"f{i}" for i in range(X_train.shape[1])]
    xai_output_dir = "experiments/xai_outputs"

    for i, col in enumerate(columns):
        print(f"XAI for {col}")
        run_permutation_importance(models[col], X_val, y_val[:, i], feature_names, xai_output_dir, col)
        run_shap_analysis(models[col], X_val, feature_names, xai_output_dir, col)
        run_error_analysis(y_val[:, i], np.squeeze(np.array([y_preds[:, i]]).T), xai_output_dir, col)

    elapsed = time.time() - start_time
    print(f"\nTotal training pipeline runtime: {elapsed:.2f}s")

    return models, (pca_msi, pca_hsi, pca_air)
