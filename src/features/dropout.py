# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 00:20:45 2026

@author: Hachem
"""

# -*- coding: utf-8 -*-
"""
Modality dropout for training robustness.

Author: Hachem
"""

import numpy as np
import random
import logging

logger = logging.getLogger("hyperview2")


def apply_dropout_dict(
    X_dict: dict,
    dropout_config: dict,
    seed: int = 42,
    enable: bool = True
) -> np.ndarray:
    """
    Apply modality dropout (training only) and concatenate features.

    dropout_config example:
        {'hsi_airborne': 0.1}
    """
    if not enable or not dropout_config:
        return concat_modalities_dict(X_dict)

    np.random.seed(seed)
    random.seed(seed)

    num_samples = len(next(iter(X_dict.values())))
    outputs = []
    drop_counts = {m: 0 for m in dropout_config}

    for i in range(num_samples):
        parts = []

        for modality, X in X_dict.items():
            if X.size == 0:
                continue

            if modality in dropout_config and random.random() <= dropout_config[modality]:
                parts.append(np.zeros_like(X[i]))
                drop_counts[modality] += 1
            else:
                parts.append(X[i])

        outputs.append(np.concatenate(parts))

    for m, c in drop_counts.items():
        logger.info(
            f"[DROPOUT] {m}: {c}/{num_samples} samples "
            f"({c / num_samples * 100:.1f}%) dropped"
        )

    return np.array(outputs)


def concat_modalities_dict(X_dict: dict) -> np.ndarray:
    """
    Concatenate all available modalities without dropout.
    """
    arrays = [X for X in X_dict.values() if X.size > 0]
    if not arrays:
        return np.empty((0,))
    return np.concatenate(arrays, axis=1)
