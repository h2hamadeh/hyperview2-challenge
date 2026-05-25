# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 00:20:45 2026

@author: Hachem
"""

# -*- coding: utf-8 -*-
"""
Data loading utilities for HYPERVIEW2 challenge.

Author: Hachem
"""

from pathlib import Path
import numpy as np
import pandas as pd
import logging
from typing import Tuple, List

logger = logging.getLogger("hyperview2")


def soil_masks(directory: Path, strict: bool = False) -> np.ndarray:
    """
    Load NPZ files and extract mean per-band soil pixel values.

    Returns
    -------
    np.ndarray
        Shape (n_samples, n_bands)
    """
    files = sorted(directory.glob("*.npz"))

    if not files:
        msg = f"No .npz files found in {directory}"
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
        return np.empty((0,))

    outputs = []
    zero_soil = 0

    for file in files:
        try:
            npz = np.load(file)
            if "data" not in npz or "mask" not in npz:
                logger.warning(f"{file.name} missing 'data' or 'mask', skipping")
                continue

            arr = np.ma.MaskedArray(**npz)
            soil_pixels = arr[:, ~arr.mask[0]]

            if soil_pixels.size == 0:
                zero_soil += 1
                outputs.append(np.zeros(arr.shape[0]))
            else:
                outputs.append(soil_pixels.mean(axis=1))

        except Exception as e:
            logger.error(f"Failed loading {file.name}: {e}")

    if zero_soil > 0:
        logger.warning(f"{zero_soil}/{len(files)} samples have zero soil pixels")

    return np.array(outputs)


def load_labels(csv_path: Path) -> np.ndarray:
    """
    Load ground truth CSV labels.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"GT CSV not found: {csv_path}")

    soil_cols = ['B', 'Cu', 'Zn', 'Fe', 'S', 'Mn']
    df = pd.read_csv(csv_path)

    missing = [c for c in soil_cols if c not in df.columns]
    if missing:
        raise ValueError(f"GT CSV missing columns: {missing}")

    logger.info(f"Loaded {len(df)} labels from {csv_path.name}")
    return df[soil_cols].values


def load_all_modalities(
    train_dir: Path,
    modalities: List[str],
    strict: bool = False
) -> dict:
    """
    Load all requested modalities into a dictionary.
    """
    data = {}

    for modality in modalities:
        mod_dir = train_dir / modality
        if not mod_dir.exists():
            msg = f"Modality directory not found: {mod_dir}"
            if strict:
                raise FileNotFoundError(msg)
            logger.warning(msg)
            data[modality] = np.empty((0,))
            continue

        X = soil_masks(mod_dir, strict=strict)
        data[modality] = X
        logger.info(f"Loaded {modality}: {X.shape}")

    return data


def train_val_split(
    X_dict: dict,
    y: np.ndarray,
    val_split: float = 0.7,
    shuffle: bool = True,
    random_seed: int = 42
) -> Tuple[dict, dict, np.ndarray, np.ndarray]:
    """
    Split data into train and validation sets.
    """
    n = len(y)
    idx = np.arange(n)

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(idx)

    split = int(n * val_split)
    train_idx = idx[:split]
    val_idx = idx[split:]

    X_train, X_val = {}, {}
    for m, X in X_dict.items():
        if X.size > 0:
            X_train[m] = X[train_idx]
            X_val[m] = X[val_idx]
        else:
            X_train[m] = np.empty((0,))
            X_val[m] = np.empty((0,))

    return X_train, X_val, y[train_idx], y[val_idx]
