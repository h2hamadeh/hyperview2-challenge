# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 00:18:09 2026

@author: Hachem
"""

# -*- coding: utf-8 -*-
"""
Data validation utilities for HYPERVIEW2 challenge.

Author: Hachem
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict

logger = logging.getLogger("hyperview2")
DEFAULT_TARGETS = ['B', 'Cu', 'Zn', 'Fe', 'S', 'Mn']


def validate_gt_csv(gt_file: Path, config: dict) -> bool:
    if not gt_file.exists():
        logger.error(f"[ERROR] GT CSV not found: {gt_file}")
        return False
    try:
        df = pd.read_csv(gt_file)
        cols_check = config.get("validation", {}).get("checks", {}).get("gt_columns", True)
        missing_check = config.get("validation", {}).get("checks", {}).get("gt_missing", True)
        negative_check = config.get("validation", {}).get("checks", {}).get("gt_negative", True)

        # Columns check
        if cols_check:
            if all(c in df.columns for c in DEFAULT_TARGETS):
                logger.info("[OK] GT CSV columns aligned")
            else:
                logger.error(f"[ERROR] GT CSV columns mismatch: expected {DEFAULT_TARGETS}, got {list(df.columns)}")
                return False

        # Missing values
        if missing_check:
            missing = df[DEFAULT_TARGETS].isnull().sum().sum()
            if missing == 0:
                logger.info("[OK] No missing values")
            else:
                logger.error(f"[ERROR] {missing} missing values in GT CSV")
                return False

        # Negative values
        if negative_check:
            negative = (df[DEFAULT_TARGETS] < 0).sum().sum()
            if negative == 0:
                logger.info("[OK] No negative values")
            else:
                logger.warning(f"[WARN] {negative} negative values in GT CSV")

        logger.info(f"[OK] GT CSV validated: {len(df)} samples")
        return True

    except Exception as e:
        logger.error(f"[ERROR] Failed to validate GT CSV: {e}")
        return False


def validate_modality_alignment(directories: Dict[str, Path], config: dict) -> bool:
    if not config.get("validation", {}).get("checks", {}).get("modality_alignment", True):
        return True

    counts = {}
    for mod, dir_ in directories.items():
        if dir_.exists():
            n = len(list(dir_.glob("*.npz")))
            counts[mod] = n
            logger.info(f"[INFO] {mod}: {n} files")
        else:
            counts[mod] = 0
            logger.warning(f"[WARN] Directory not found: {dir_}")

    non_zero = [c for c in counts.values() if c > 0]
    if len(set(non_zero)) <= 1:
        logger.info("[OK] Modalities aligned")
        return True
    else:
        logger.error(f"[ERROR] Modality counts mismatch: {counts}")
        return False


def validate_masks_and_soil_pixels(directories: Dict[str, Path], config: dict) -> bool:
    if not config.get("validation", {}).get("checks", {}).get("mask_consistency", True):
        return True

    zero_files, total_files = [], 0
    for mod, dir_ in directories.items():
        if not dir_.exists():
            continue
        for file in dir_.glob("*.npz"):
            total_files += 1
            try:
                with np.load(file) as npz:
                    if "data" in npz and "mask" in npz:
                        arr = np.ma.MaskedArray(**npz)
                        soil = arr[:, ~arr.mask[0]]
                        if soil.size == 0:
                            zero_files.append((mod, file.name))
                    else:
                        logger.warning(f"[WARN] {file.name} missing 'data' or 'mask'")
            except Exception as e:
                logger.error(f"[ERROR] Failed loading {file.name}: {e}")

    threshold = config.get("validation", {}).get("thresholds", {}).get("max_zero_soil_fraction", 0.5)
    frac = len(zero_files) / total_files if total_files > 0 else 0
    if frac > threshold:
        logger.warning(f"[WARN] {len(zero_files)}/{total_files} files ({frac*100:.1f}%) zero soil pixels (threshold {threshold*100:.0f}%)")
    else:
        logger.info(f"[OK] {len(zero_files)}/{total_files} files ({frac*100:.1f}%) zero soil pixels")

    if zero_files:
        for mod, fname in zero_files[:10]:
            logger.info(f"  - {mod}/{fname}")
        if len(zero_files) > 10:
            logger.info(f"  ... and {len(zero_files)-10} more")

    return True


def run_validation(dataset_dir: Path, config: dict, is_train: bool = True) -> bool:
    if not config.get("validation", {}).get("enabled", True):
        logger.info("[SKIP] Validation disabled")
        return True

    logger.info("="*80)
    logger.info("DATA VALIDATION")
    logger.info("="*80)

    all_pass = True
    data_dir = Path(config["data"]["train_dir"] if is_train else config["data"]["test_dir"])
    gt_file = Path(config["data"]["gt_csv"]) if is_train else None
    modalities = config["data"]["modalities"]["train" if is_train else "test"]

    if gt_file and is_train:
        logger.info("\n--- Validating GT CSV ---")
        if not validate_gt_csv(gt_file, config):
            all_pass = False

    dirs = {mod: data_dir/mod for mod in modalities}
    logger.info("\n--- Validating Modality Alignment ---")
    if not validate_modality_alignment(dirs, config):
        all_pass = False

    logger.info("\n--- Validating Masks and Soil Pixels ---")
    if not validate_masks_and_soil_pixels(dirs, config):
        all_pass = False

    logger.info("\n" + "="*80)
    if all_pass:
        logger.info("✓ ALL VALIDATION CHECKS PASSED")
    else:
        logger.warning("✗ SOME VALIDATION CHECKS FAILED")
    logger.info("="*80 + "\n")

    return all_pass


def quick_validation(dataset_dir: Path, config: dict) -> bool:
    gt_csv = Path(config["data"]["gt_csv"])
    train_dir = Path(config["data"]["train_dir"])

    if not gt_csv.exists():
        logger.error(f"[ERROR] GT CSV not found: {gt_csv}")
        return False
    if not train_dir.exists():
        logger.error(f"[ERROR] Training dir not found: {train_dir}")
        return False

    logger.info("[OK] Quick validation passed")
    return True
