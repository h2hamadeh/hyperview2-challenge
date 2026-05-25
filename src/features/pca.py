# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 00:20:45 2026

@author: Hachem
"""

# -*- coding: utf-8 -*-
"""
PCA dimensionality reduction utilities.

Author: Hachem
"""

from pathlib import Path
import numpy as np
import logging
import joblib
from sklearn.decomposition import PCA

logger = logging.getLogger("hyperview2")


def fit_pca(X: np.ndarray, n_components: int, name: str) -> tuple:
    """
    Fit PCA on training data and return transformed data + model.
    """
    if X.size == 0:
        logger.warning(f"[PCA] {name}: empty data, skipping")
        return X, None

    if X.shape[1] < n_components:
        logger.warning(
            f"[PCA] {name}: requested {n_components}, "
            f"using {X.shape[1]} instead"
        )
        n_components = X.shape[1]

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    var = pca.explained_variance_ratio_.sum() * 100
    logger.info(f"[PCA] {name}: {X.shape[1]} → {n_components} ({var:.2f}%)")

    return X_pca, pca


def transform_with_PCA(X, pca_model, modality_name=None):
    """
    Apply a pre-fitted PCA model.
    """
    if modality_name is not None:
        logger.info(f"[PCA] Transforming {modality_name} with PCA")
    return pca_model.transform(X)


def save_pca_models(pca_dict: dict, save_dir: Path):
    """
    Save PCA models to disk.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for name, pca in pca_dict.items():
        if pca is None:
            continue
        path = save_dir / f"pca_{name}.pkl"
        joblib.dump(pca, path)
        logger.info(f"[PCA] Saved {name} PCA → {path}")


def load_pca_models(load_dir: Path, modalities: list) -> dict:
    """
    Load PCA models from disk.
    """
    load_dir = Path(load_dir)
    pca_dict = {}

    for name in modalities:
        path = load_dir / f"pca_{name}.pkl"
        if path.exists():
            pca_dict[name] = joblib.load(path)
            logger.info(f"[PCA] Loaded {name} PCA")
        else:
            logger.warning(f"[PCA] Missing PCA for {name}")
            pca_dict[name] = None

    return pca_dict


def apply_pca(
    X_dict: dict,
    pca_config: dict,
    pca_dict: dict = None,
    fit: bool = True
) -> tuple:
    """
    Apply PCA per modality.

    fit=True  → fit PCA on X_dict (training only)
    fit=False → use provided pca_dict (validation / test)
    """
    X_out = {}
    pcas = {} if fit else pca_dict

    for name, X in X_dict.items():
        if name not in pca_config or X.size == 0:
            X_out[name] = X
            if fit:
                pcas[name] = None
            continue

        if fit:
            X_pca, pca = fit_pca(X, pca_config[name], name)
            X_out[name] = X_pca
            pcas[name] = pca
        else:
            X_out[name] = transform_with_PCA(X, pca_dict.get(name), name)

    return X_out, pcas
