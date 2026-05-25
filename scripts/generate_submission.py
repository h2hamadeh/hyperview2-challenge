
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.utils.logger import log_config, setup_logger
from src.utils.reproducibility import get_random_state, set_seed
from src.data.loaders import load_all_modalities
from src.features.pca import apply_pca, load_pca_models
from src.features.dropout import concat_modalities_dict
from src.models.train import load_models, predict_all_targets


def load_config(config_path: str = "config.yaml") -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_submission():
    """Generate submission predictions for the test set."""

    # ──────────────────────────────────────────
    # Setup
    # ──────────────────────────────────────────
    config = load_config()
    seed   = get_random_state(config)
    set_seed(seed)

    logger = setup_logger(
        log_dir=config["validation"]["log_dir"],
        log_file="generate_submission.log",
        name="hyperview2",
    )
    log_config(config, logger)

    # ──────────────────────────────────────────
    # Load Trained Models
    # ──────────────────────────────────────────
    experiment_name = config.get("experiment", {}).get("name", "baseline")
    model_dir       = Path(config["train"]["model_dir"])

    models = load_models(
        load_dir=model_dir,
        experiment_name=experiment_name,
        target_names=config["targets"]["columns"],
    )

    # ──────────────────────────────────────────
    # Load Preprocessors  (var/corr masks + scaler + PCA per modality)
    # ──────────────────────────────────────────
    pca_save_dir = model_dir / "pca"
    all_train_modalities = ['msi_satellite', 'hsi_satellite', 'hsi_airborne']
    preprocessors = load_pca_models(pca_save_dir, all_train_modalities)

    # ──────────────────────────────────────────
    # Load Test Data
    # ──────────────────────────────────────────
    test_dir   = Path(config["data"]["test_dir"])
    test_mods  = config["data"]["modalities"]["test"]   # airborne absent at test time
    X_test_dict = load_all_modalities(test_dir, test_mods)

    # ──────────────────────────────────────────
    # Apply Preprocessing (filter → scale → PCA)
    # ──────────────────────────────────────────
    pca_config = {
        'msi_satellite': config["pca"]["msi"],
        'hsi_satellite': config["pca"]["hsi"],
        'hsi_airborne':  config["pca"]["airborne"],
    }

    X_test_pca_dict, _ = apply_pca(
        X_test_dict,
        pca_config,
        fit=False,
        preprocessors=preprocessors,
    )

    n_test = next(iter(X_test_pca_dict.values())).shape[0]
    for mod in all_train_modalities:
        if mod not in X_test_pca_dict:
            prep = preprocessors.get(mod)
            if prep is not None and prep.get('pca') is not None:
                n_feat = prep['pca'].n_components_
            else:
                n_feat = pca_config.get(mod, 0)
            logger.info(
                f"[SUBMIT] Modality '{mod}' absent at test time — "
                f"filling with zeros ({n_test} × {n_feat})"
            )
            X_test_pca_dict[mod] = np.zeros((n_test, n_feat))

    # ──────────────────────────────────────────
    # Concatenate Features
    # ──────────────────────────────────────────
    X_test = concat_modalities_dict(X_test_pca_dict)
    logger.info(f"Test features shape: {X_test.shape}")

    # ──────────────────────────────────────────
    # Predict
    # ──────────────────────────────────────────
    y_test_pred = predict_all_targets(
        models, X_test, target_names=config["targets"]["columns"]
    )
    logger.info(f"Predictions shape: {y_test_pred.shape}")

    # ──────────────────────────────────────────
    # (optional) apply inverse-transform if log1p was applied during training
    # ──────────────────────────────────────────
    log_transform = config["train"].get("log_transform", False)
    if log_transform:
        logger.info("[SUBMIT] Applying expm1 inverse transform to predictions")
        y_test_pred = np.expm1(y_test_pred)

    # ──────────────────────────────────────────
    # Save Submission CSV
    # ──────────────────────────────────────────
    submission_file = Path(config["data"]["root_dir"]) / "submission.csv"
    df_sub = pd.DataFrame(y_test_pred, columns=config["targets"]["columns"])
    df_sub.to_csv(submission_file, index=False)
    logger.info(f"Submission saved → {submission_file}")


if __name__ == "__main__":
    generate_submission()
