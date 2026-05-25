
import time
from pathlib import Path

import numpy as np
import yaml

from src.utils.logger import log_config, setup_logger
from src.utils.reproducibility import get_random_state, set_seed
from src.data.loaders import load_all_modalities, load_labels, train_val_split
from src.features.pca import apply_pca, save_pca_models
from src.features.dropout import apply_dropout_dict, concat_modalities_dict
from src.models.train import predict_all_targets, train_and_save
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.xai import run_full_xai_analysis
from src.validation.validate_data import run_validation


def load_config(config_path: str = "config.yaml") -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    start_time = time.time()

    config = load_config()

    log_dir = Path(config["validation"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir=log_dir, log_file="train_pipeline.log", name="hyperview2")

    logger.info("=" * 80)
    logger.info("HYPERVIEW2 TRAINING PIPELINE")
    logger.info("=" * 69)
    logger.info("CONFIGURATION")
    logger.info("=" * 69)
    log_config(config, logger)

    seed = get_random_state(config)
    set_seed(seed)

    # ──────────────────────────────────────────
    # STEP 1: DATA VALIDATION
    # ──────────────────────────────────────────
    logger.info("=" * 80)
    logger.info("STEP 1: DATA VALIDATION")
    logger.info("=" * 69)

    dataset_dir = Path(config["data"]["root_dir"])
    validation_passed = run_validation(dataset_dir, config, is_train=True)

    if not validation_passed and config["experiment"]["strict_validation"]:
        logger.error("Validation failed and strict_validation is enabled. Exiting.")
        return

    # ──────────────────────────────────────────
    # STEP 2: DATA LOADING
    # ──────────────────────────────────────────
    logger.info("=" * 80)
    logger.info("STEP 2: DATA LOADING")
    logger.info("=" * 69)

    gt_csv   = Path(config["data"]["gt_csv"])
    y_all    = load_labels(gt_csv)
    logger.info(f"Loaded {len(y_all)} ground truth labels")

    train_dir  = Path(config["data"]["train_dir"])
    modalities = config["data"]["modalities"]["train"]
    X_dict     = load_all_modalities(train_dir, modalities)

    # ──────────────────────────────────────────
    # STEP 3: TRAIN / VAL SPLIT
    # ──────────────────────────────────────────
    logger.info("=" * 80)
    logger.info("STEP 3: TRAIN/VALIDATION SPLIT")
    logger.info("=" * 69)

    X_train_dict, X_val_dict, y_train, y_val = train_val_split(
        X_dict,
        y_all,
        val_split=config["train"]["val_split"],
        shuffle=config["train"]["shuffle"],
        random_seed=seed,
    )

    # (optional)
    log_transform = config["train"].get("log_transform", False)
    if log_transform:
        logger.info("[TRANSFORM] Applying log1p to targets (train + val)")
        y_train_t = np.log1p(y_train)
        y_val_t   = np.log1p(y_val)
    else:
        y_train_t = y_train.copy()
        y_val_t   = y_val.copy()

    # ──────────────────────────────────────────
    # STEP 4: FEATURE ENGINEERING
    # ──────────────────────────────────────────
    logger.info("=" * 80)
    logger.info("STEP 4: FEATURE ENGINEERING (filter → scale → PCA)")
    logger.info("=" * 69)

    pca_config = {
        'msi_satellite': config["pca"]["msi"],
        'hsi_satellite': config["pca"]["hsi"],
        'hsi_airborne':  config["pca"]["airborne"],
    }

    feature_selection_config = config.get("feature_selection", {})

    # fit on training data
    X_train_pca_dict, preprocessors = apply_pca(
        X_train_dict,
        pca_config,
        feature_selection_config=feature_selection_config,
        fit=True,
    )

    # apply to validation data
    X_val_pca_dict, _ = apply_pca(
        X_val_dict,
        pca_config,
        feature_selection_config=feature_selection_config,
        fit=False,
        preprocessors=preprocessors,
    )

    # save preprocessors
    pca_save_dir = Path(config["train"]["model_dir"]) / "pca"
    save_pca_models(preprocessors, pca_save_dir)

    # ──────────────────────────────────────────
    # STEP 5: MODALITY DROPOUT
    # ──────────────────────────────────────────
    logger.info("=" * 80)
    logger.info("STEP 5: MODALITY DROPOUT")
    logger.info("=" * 69)

    dropout_config = {'hsi_airborne': config['dropout']['airborne']}

    X_train = apply_dropout_dict(X_train_pca_dict, dropout_config, seed=seed)
    X_val   = concat_modalities_dict(X_val_pca_dict)

    logger.info(f"Training features shape : {X_train.shape}")
    logger.info(f"Validation features shape: {X_val.shape}")

    # ──────────────────────────────────────────
    # STEP 6: MODEL TRAINING
    # ──────────────────────────────────────────
    logger.info("=" * 80)
    logger.info("STEP 6: MODEL TRAINING")
    logger.info("=" * 69)

    save_models_dir = Path(config["train"]["model_dir"]) if config["train"]["save_models"] else None
    models = train_and_save(X_train, y_train_t, config, save_dir=save_models_dir)

    # ──────────────────────────────────────────
    # STEP 7: VALIDATION EVALUATION
    # ──────────────────────────────────────────
    logger.info("=" * 80)
    logger.info("STEP 7: VALIDATION EVALUATION")
    logger.info("=" * 69)

    y_val_pred_t = predict_all_targets(models, X_val, target_names=config["targets"]["columns"])

    # (optional)
    if log_transform:
        logger.info("[TRANSFORM] Inverse log1p on predictions before scoring")
        y_val_pred_eval = np.expm1(y_val_pred_t)
        y_val_eval      = y_val          # original scale
        y_train_eval    = y_train        # original scale baseline
    else:
        y_val_pred_eval = y_val_pred_t
        y_val_eval      = y_val_t
        y_train_eval    = y_train_t

    target_names = config["targets"]["columns"]
    scores_dict  = evaluate_predictions(
        y_true=y_val_eval,
        y_pred=y_val_pred_eval,
        y_train=y_train_eval,
        target_names=target_names,
        log_results=True,
    )

    # ──────────────────────────────────────────
    # STEP 8: XAI ANALYSIS
    # ──────────────────────────────────────────
    logger.info("=" * 80)
    logger.info("STEP 8: XAI ANALYSIS")
    logger.info("=" * 69)

    if config["xai"].get("enabled", False):
        logger.info("[XAI] Running explainability analysis...")

        xai_output_dir = Path(config["xai"]["output_dir"])
        xai_output_dir.mkdir(parents=True, exist_ok=True)

        # ── Load wavelengths for spectral plots ──
        wavelengths_per_modality = {}
        wl_path = Path(config["xai"].get("wavelengths_path", "wavelengths.json"))
        if wl_path.exists():
            import pandas as pd
            wl_df = pd.read_json(wl_path)
            wavelengths_per_modality = {
                'msi_satellite': wl_df['msi_satellite_wavelengths'].dropna().tolist(),
                'hsi_satellite': wl_df['hsi_satellite_wavelengths'].dropna().tolist(),
                'hsi_airborne':  wl_df['hsi_aerial_wavelengths'].dropna().tolist(),
            }
            logger.info(f"[XAI] Wavelengths loaded from {wl_path}")
        else:
            logger.warning(f"[XAI] wavelengths.json not found at {wl_path}, band labels will be indices")

        # ── Modality order and PCA component counts (must match concat order) ──
        modality_order = ['msi_satellite', 'hsi_satellite', 'hsi_airborne']
        pca_components_per_modality = [
            preprocessors[m]['pca'].n_components_
            if preprocessors.get(m) and preprocessors[m].get('pca') is not None
            else 0
            for m in modality_order
        ]

        feature_names = [f"f{i}" for i in range(X_val.shape[1])]

        try:
            run_full_xai_analysis(
                models=models,
                X_val=X_val,
                y_val=y_val_eval,
                feature_names=feature_names,
                output_dir=str(xai_output_dir),
                config=config,
                target_names=target_names,
                preprocessors=preprocessors,
                wavelengths_per_modality=wavelengths_per_modality,
                modality_order=modality_order,
                pca_components_per_modality=pca_components_per_modality,
            )
            logger.info(f"[XAI] Analysis complete → {xai_output_dir}")
        except Exception as e:
            logger.error(f"[XAI] Analysis failed: {e}", exc_info=True)
    else:
        logger.info("[XAI] Disabled in config")

    # ──────────────────────────────────────────
    # DONE
    # ──────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info("=" * 80)
    logger.info("TRAINING PIPELINE COMPLETED")
    logger.info("=" * 69)
    logger.info(f"Total runtime : {elapsed:.1f}s  ({elapsed / 60:.2f} min)")
    logger.info(f"HYPERVIEW Score: {scores_dict['hyperview_score']:.4f}")
    logger.info("=" * 80)

    return models, preprocessors, scores_dict


if __name__ == "__main__":
    main()
