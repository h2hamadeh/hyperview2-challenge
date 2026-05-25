# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 00:19:59 2026

@author: Hachem
"""

# -*- coding: utf-8 -*-
"""
Model training utilities for HYPERVIEW2 challenge.

Author: Hachem
"""

import time
import numpy as np
from pathlib import Path
import logging
import joblib
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Optional, Union

logger = logging.getLogger("hyperview2")


DEFAULT_TARGETS = ['B', 'Cu', 'Zn', 'Fe', 'S', 'Mn']


def train_single_model(
    X: np.ndarray,
    y: np.ndarray,
    model_cfg: dict
) -> Union[XGBRegressor, RandomForestRegressor]:
    """
    Train a single regressor (XGBoost or Random Forest based on config).
    
    Parameters
    ----------
    X : np.ndarray
        Training features
    y : np.ndarray
        Training target for single soil parameter
    model_cfg : dict
        Model configuration containing 'type' and model-specific parameters
        
    Returns
    -------
    Union[XGBRegressor, RandomForestRegressor]
        Trained model instance
    """
    model_type = model_cfg.get("type", "xgboost").lower()
    
    if model_type == "xgboost":
        logger.debug("[MODEL] Using XGBoost")
        xgb_cfg = model_cfg.get("xgboost", model_cfg)  # Fallback for backwards compatibility
        model = XGBRegressor(
            objective=xgb_cfg.get("objective", "reg:tweedie"),
            tweedie_variance_power=xgb_cfg.get("tweedie_variance_power", 1.5),
            n_estimators=xgb_cfg.get("n_estimators", 100),
            learning_rate=xgb_cfg.get("learning_rate", 0.1),
            max_depth=xgb_cfg.get("max_depth", 6),
            random_state=xgb_cfg.get("seed", 42),
            n_jobs=xgb_cfg.get("n_jobs", 2)
        )
    elif model_type == "random_forest":
        logger.debug("[MODEL] Using Random Forest")
        rf_cfg = model_cfg.get("random_forest", {})
        model = RandomForestRegressor(
            n_estimators=rf_cfg.get("n_estimators", 100),
            max_depth=rf_cfg.get("max_depth", 15),
            min_samples_split=rf_cfg.get("min_samples_split", 5),
            min_samples_leaf=rf_cfg.get("min_samples_leaf", 2),
            random_state=rf_cfg.get("seed", 42),
            n_jobs=rf_cfg.get("n_jobs", 2)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X, y)
    return model


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict,
    target_names: Optional[list] = None
) -> Dict[str, Union[XGBRegressor, RandomForestRegressor]]:
    """
    Train one model per soil parameter.
    """
    targets = target_names or DEFAULT_TARGETS
    model_cfg = config.get("model", {})
    
    model_type = model_cfg.get("type", "random_forest")
    
    logger.info("=" * 80)
    logger.info(f"TRAINING MODELS ({model_type.upper()})")
    logger.info("=" * 80)

    models = {}
    times = {}

    for i, target in enumerate(targets):
        t0 = time.time()
        logger.info(f"[TRAIN] {target}")

        models[target] = train_single_model(
            X_train,
            y_train[:, i],
            model_cfg
        )

        times[target] = time.time() - t0
        logger.info(f"[DONE] {target} in {times[target]:.2f}s")

    total = sum(times.values())
    logger.info(f"[TRAIN] Total: {total:.2f}s | Avg: {total / len(targets):.2f}s")

    return models


def save_models(
    models: Dict[str, Union[XGBRegressor, RandomForestRegressor]],
    save_dir: Path,
    experiment_name: str
):
    """
    Save trained models to disk.
    """
    out_dir = Path(save_dir) / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for target, model in models.items():
        path = out_dir / f"model_{target}.pkl"
        joblib.dump(model, path)
        logger.info(f"[SAVE] {target} → {path}")


def load_models(
    load_dir: Path,
    experiment_name: str,
    target_names: Optional[list] = None
) -> Dict[str, Optional[Union[XGBRegressor, RandomForestRegressor]]]:
    """
    Load trained models from disk.
    """
    targets = target_names or DEFAULT_TARGETS
    in_dir = Path(load_dir) / experiment_name

    models = {}
    for target in targets:
        path = in_dir / f"model_{target}.pkl"
        if path.exists():
            models[target] = joblib.load(path)
            logger.info(f"[LOAD] {target}")
        else:
            logger.warning(f"[LOAD] Missing model: {path}")
            models[target] = None

    return models


def predict_all_targets(
    models: Dict[str, Union[XGBRegressor, RandomForestRegressor]],
    X: np.ndarray,
    target_names: Optional[list] = None
) -> np.ndarray:
    """
    Predict all soil parameters.
    """
    targets = target_names or DEFAULT_TARGETS
    preds = []

    for target in targets:
        model = models.get(target)
        if model is None:
            logger.error(f"[PREDICT] Missing model for {target}")
            preds.append(np.zeros(X.shape[0]))
        else:
            preds.append(model.predict(X))

    return np.stack(preds, axis=1)


def train_and_save(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict,
    save_dir: Optional[Path] = None
) -> Dict[str, Union[XGBRegressor, RandomForestRegressor]]:
    """
    Train models and optionally save them.
    """
    models = train_models(X_train, y_train, config)

    if save_dir and config.get("train", {}).get("save_models", False):
        exp_name = config.get("experiment", {}).get("name", "baseline")
        save_models(models, save_dir, exp_name)

    return models
