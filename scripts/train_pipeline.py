# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 00:20:27 2026

@author: Hachem
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HYPERVIEW2 Challenge - Main Training Pipeline

This script orchestrates the complete training workflow:
1. Load and validate data
2. Apply preprocessing (PCA, dropout)
3. Train models for all soil parameters
4. Evaluate on validation set
5. Run XAI analysis
6. Save models and results

Usage:
    python train_pipeline.py

Author: Hachem
Created: Sat Jan 3 2026
"""

import time
from pathlib import Path
import yaml
import numpy as np

# Utilities
from src.utils.reproducibility import set_seed, get_random_state
from src.utils.logger import setup_logger, log_config

# Data
from src.data.loaders import load_labels, load_all_modalities, train_val_split

# Features
from src.features.pca import apply_pca, save_pca_models, transform_with_PCA
from src.features.dropout import apply_dropout_dict, concat_modalities_dict

# Models
from src.models.train import train_and_save, predict_all_targets

# Evaluation
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.xai import run_full_xai_analysis

# Validation
from src.validation.validate_data import run_validation


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Main training pipeline."""
    start_time = time.time()

    # -------------------------
    # Load config
    # -------------------------
    config = load_config()

    # -------------------------
    # Setup logger
    # -------------------------
    log_dir = Path(config["validation"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        log_dir=log_dir,
        log_file="train_pipeline.log",
        name="hyperview2"
    )
    logger.info("=" * 80)
    logger.info("HYPERVIEW2 TRAINING PIPELINE")
    logger.info("=" * 80)
    log_config(config, logger)

    # -------------------------
    # Set random seed
    # -------------------------
    seed = get_random_state(config)
    set_seed(seed)

    # -------------------------
    # DATA VALIDATION
    # -------------------------
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: DATA VALIDATION")
    logger.info("=" * 80)

    dataset_dir = Path(config["data"]["root_dir"])
    validation_passed = run_validation(dataset_dir, config, is_train=True)

    if not validation_passed and config["experiment"]["strict_validation"]:
        logger.error("Validation failed and strict_validation is enabled. Exiting.")
        return

    # -------------------------
    # DATA LOADING
    # -------------------------
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: DATA LOADING")
    logger.info("=" * 80)

    gt_csv = Path(config["data"]["gt_csv"])
    y_all = load_labels(gt_csv)
    logger.info(f"Loaded {len(y_all)} ground truth labels")

    train_dir = Path(config["data"]["train_dir"])
    modalities = config["data"]["modalities"]["train"]
    X_dict = load_all_modalities(train_dir, modalities)

    # -------------------------
    # TRAIN/VALIDATION SPLIT
    # -------------------------
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: TRAIN/VALIDATION SPLIT")
    logger.info("=" * 80)

    X_train_dict, X_val_dict, y_train, y_val = train_val_split(
        X_dict,
        y_all,
        val_split=config["train"]["val_split"],
        shuffle=config["train"]["shuffle"],
        random_seed=seed
    )

    if config["train"]["log_transform"]:
        logger.info("[TRANSFORM] Applying log1p transform to targets")
        y_train_transformed = np.log1p(y_train)
        y_val_transformed = np.log1p(y_val)
    else:
        y_train_transformed = y_train.copy()
        y_val_transformed = y_val.copy()

    # -------------------------
    # FEATURE ENGINEERING: PCA
    # -------------------------
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: PCA DIMENSIONALITY REDUCTION")
    logger.info("=" * 80)

    pca_config = {
        'msi_satellite': config["pca"]["msi"],
        'hsi_satellite': config["pca"]["hsi"],
        'hsi_airborne': config["pca"]["airborne"]
    }

    # Apply PCA to training data
    X_train_pca_dict, pca_models = apply_pca(
        X_train_dict, pca_config, fit=True
    )

    # Transform validation data
    X_val_pca_dict = {}
    for modality, X_val_modality in X_val_dict.items():
        if modality in pca_models and pca_models[modality] is not None:
            X_val_pca_dict[modality] = transform_with_PCA(
                X_val_modality, pca_models[modality], modality_name=modality
            )
        else:
            X_val_pca_dict[modality] = X_val_modality

    # Save PCA models
    pca_save_dir = Path(config["train"]["model_dir"]) / "pca"
    save_pca_models(pca_models, pca_save_dir)

    # -------------------------
    # FEATURE ENGINEERING: MODALITY DROPOUT
    # -------------------------
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: MODALITY DROPOUT (Training Augmentation)")
    logger.info("=" * 80)

    modality_order = ['msi_satellite', 'hsi_satellite', 'hsi_airborne']
    X_train_pca_list = [X_train_pca_dict[m] for m in modality_order]
    X_val_pca_list   = [X_val_pca_dict[m] for m in modality_order]

    dropout_config = {'hsi_airborne': config['dropout']['airborne']}

    X_train = apply_dropout_dict(
        X_train_pca_dict,
        dropout_config,
        seed=seed
    )
    
    X_val = concat_modalities_dict(X_val_pca_dict)

    logger.info(f"Training features shape: {X_train.shape}")
    logger.info(f"Validation features shape: {X_val.shape}")

    # -------------------------
    # MODEL TRAINING
    # -------------------------
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: MODEL TRAINING")
    logger.info("=" * 80)

    save_models_dir = Path(config["train"]["model_dir"]) if config["train"]["save_models"] else None
    models = train_and_save(X_train, y_train_transformed, config, save_dir=save_models_dir)

    # -------------------------
    # VALIDATION EVALUATION
    # -------------------------
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: VALIDATION EVALUATION (Normalized MSE)")
    logger.info("=" * 80)
    
    # Generate predictions
    y_val_pred = predict_all_targets(models, X_val)
    
    target_names = config["targets"]["columns"]
    
    # Use metrics.py evaluate_predictions function
    scores_dict = evaluate_predictions(
        y_true=y_val_transformed,
        y_pred=y_val_pred,
        y_train=y_train_transformed,
        target_names=target_names,
        log_results=True
    )

    # -------------------------
    # XAI ANALYSIS
    # -------------------------
    logger.info("\n" + "=" * 80)
    logger.info("STEP 8: XAI ANALYSIS")
    logger.info("=" * 80)
    
    if config["xai"]["enabled"]:
        logger.info("[XAI] Running explainability analysis...")
        feature_names = [f"f{i}" for i in range(X_val.shape[1])]
        xai_output_dir = Path(config["xai"]["output_dir"])
        xai_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            run_full_xai_analysis(
                models=models,
                X_val=X_val,
                y_val=y_val_transformed,
                feature_names=feature_names,
                output_dir=str(xai_output_dir),
                config=config,
                target_names=target_names
            )
            logger.info(f"[XAI] Analysis complete. Results saved to {xai_output_dir}")
        except Exception as e:
            logger.error(f"[XAI] XAI analysis failed: {e}")
    else:
        logger.info("[XAI] XAI analysis disabled in config")

    # -------------------------
    # COMPLETION
    # -------------------------
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING PIPELINE COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Total runtime: {elapsed:.2f}s ({elapsed/60:.2f} minutes)")
    logger.info(f"HYPERVIEW Mean Score: {scores_dict['hyperview_score']:.4f}")
    logger.info("=" * 80 + "\n")

    return models, pca_models, scores_dict


if __name__ == "__main__":
    main()