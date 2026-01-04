# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 00:21:59 2026

@author: Hachem
"""

"""
HYPERVIEW2 Challenge - Generate Submission

Author: Hachem
Created: Sat Jan 3 2026
"""

import yaml
from pathlib import Path
import numpy as np
import logging

# Utilities
from src.utils.reproducibility import set_seed, get_random_state
from src.utils.logger import setup_logger, log_config

# Data
from src.data.loaders import load_all_modalities

# Features
from src.features.pca import load_pca_models, transform_with_PCA
from src.features.dropout import concat_modalities_dict

# Models
from src.models.train import load_models, predict_all_targets


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_submission():
    """Generate submission predictions for test set."""
    
    # ------------------------------
    # Setup
    # ------------------------------
    config = load_config()
    seed = get_random_state(config)
    set_seed(seed)

    logger = setup_logger(
        log_dir=config["validation"]["log_dir"],
        log_file="generate_submission.log",
        name="hyperview2"
    )
    log_config(config, logger)

    # ------------------------------
    # Load trained models
    # ------------------------------
    experiment_name = config.get("experiment", {}).get("name", "baseline")
    model_dir = Path(config["train"]["model_dir"])

    models = load_models(
        load_dir=model_dir,
        experiment_name=experiment_name,
        target_names=config["targets"]["columns"]
    )

    # ------------------------------
    # Load PCA models
    # ------------------------------
    pca_save_dir = model_dir / "pca"
    modalities = ['msi_satellite', 'hsi_satellite', 'hsi_airborne']
    pca_models = load_pca_models(pca_save_dir, modalities)

    # ------------------------------
    # Load test data
    # ------------------------------
    test_dir = Path(config["data"]["test_dir"])
    X_test_dict = load_all_modalities(test_dir, config["data"]["modalities"]["test"])

    # ------------------------------
    # Transform test data using PCA
    # ------------------------------
    X_test_pca_dict = {}
    for modality, X_test_mod in X_test_dict.items():
        if modality in pca_models and pca_models[modality] is not None:
            X_test_pca_dict[modality] = transform_with_PCA(
                X_test_mod,
                pca_models[modality],
                modality_name=modality
            )
        else:
            X_test_pca_dict[modality] = X_test_mod
    
    for modality in pca_models:  # ensure all training modalities exist
        if modality not in X_test_pca_dict:
            n_features = pca_models[modality].n_components_
            n_samples = next(iter(X_test_pca_dict.values())).shape[0]
            logger.info(f"[PCA] Adding missing modality '{modality}' as zeros for test set")
            X_test_pca_dict[modality] = np.zeros((n_samples, n_features))
    # ------------------------------
    # Concatenate test features
    # ------------------------------
    X_test = concat_modalities_dict(X_test_pca_dict)
    logger.info(f"Test features shape: {X_test.shape}")

    # ------------------------------
    # Generate predictions
    # ------------------------------
    y_test_pred = predict_all_targets(models, X_test, target_names=config["targets"]["columns"])
    logger.info(f"Predictions generated: {y_test_pred.shape}")

    # ------------------------------
    # Save submission CSV
    # ------------------------------
    submission_file = Path(config["data"]["root_dir"]) / "submission.csv"
    import pandas as pd
    df_sub = pd.DataFrame(y_test_pred, columns=config["targets"]["columns"])
    df_sub.to_csv(submission_file, index=False)
    logger.info(f"Submission saved: {submission_file}")


if __name__ == "__main__":
    generate_submission()
