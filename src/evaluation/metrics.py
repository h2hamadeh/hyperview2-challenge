# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 00:20:45 2026

@author: Hachem
"""

# -*- coding: utf-8 -*-
"""
Evaluation metrics for HYPERVIEW2 challenge.

Author: Hachem
Created: Fri Jan 2 00:20:45 2026
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from typing import Dict, Tuple

logger = logging.getLogger("hyperview2")


def calculate_mse_scores(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute per-target MSE scores.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values, shape (n_samples, n_targets)
    y_pred : np.ndarray
        Predicted values, shape (n_samples, n_targets)
        
    Returns
    -------
    np.ndarray
        MSE for each target, shape (n_targets,)
    """
    return mean_squared_error(y_true, y_pred, multioutput='raw_values')


def calculate_baseline_mse(y_true: np.ndarray, baseline_pred: np.ndarray) -> np.ndarray:
    """
    Compute baseline MSE (for normalization).
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values, shape (n_samples, n_targets)
    baseline_pred : np.ndarray
        Baseline predictions (usually mean of training set), shape (n_samples, n_targets)
        
    Returns
    -------
    np.ndarray
        Baseline MSE for each target, shape (n_targets,)
    """
    return mean_squared_error(y_true, baseline_pred, multioutput='raw_values')


def calculate_hyperview_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calculate HYPERVIEW Score as defined in the competition.
    
    The HYPERVIEW Score is:
    Score = (1/|Ψ|) * Σ(MSE_i / MSE_baseline_i)
    
    Where:
    - |Ψ| is the number of test samples
    - MSE_i is the MSE for the i-th soil parameter
    - MSE_baseline_i is the baseline MSE (predicting training mean)
    
    Lower scores are better, with 0 being perfect.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values, shape (n_samples, n_targets)
    y_pred : np.ndarray
        Predicted values, shape (n_samples, n_targets)
    y_train : np.ndarray
        Training set values for baseline calculation, shape (n_train_samples, n_targets)
        
    Returns
    -------
    hyperview_score : float
        Overall HYPERVIEW score (average of normalized MSEs)
    normalized_mses : np.ndarray
        Normalized MSE for each target (MSE / baseline_MSE)
    baseline_mses : np.ndarray
        Baseline MSE for each target
        
    Example
    -------
    >>> score, norm_mses, base_mses = calculate_hyperview_score(y_val, y_pred, y_train)
    >>> print(f"HYPERVIEW Score: {score:.4f}")
    """
    # Calculate model MSE for each target
    model_mse = calculate_mse_scores(y_true, y_pred)
    
    # Calculate baseline MSE (predicting mean of training set)
    train_means = np.mean(y_train, axis=0)
    baseline_pred = np.tile(train_means, (y_true.shape[0], 1))
    baseline_mse = calculate_baseline_mse(y_true, baseline_pred)
    
    # Normalize MSEs by baseline
    normalized_mses = model_mse / baseline_mse
    
    # HYPERVIEW score is the average of normalized MSEs
    hyperview_score = np.mean(normalized_mses)
    
    return hyperview_score, normalized_mses, baseline_mse


def calculate_additional_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calculate additional evaluation metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values, shape (n_samples, n_targets)
    y_pred : np.ndarray
        Predicted values, shape (n_samples, n_targets)
        
    Returns
    -------
    dict
        Dictionary containing MAE, RMSE, and R² for each target
    """
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred, multioutput='raw_values'),
        'rmse': np.sqrt(calculate_mse_scores(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred, multioutput='raw_values')
    }
    return metrics


def log_evaluation_results(
    hyperview_score: float,
    normalized_mses: np.ndarray,
    baseline_mses: np.ndarray,
    additional_metrics: Dict[str, np.ndarray] = None,
    target_names: list = None
):
    """
    Log comprehensive evaluation results.
    
    Parameters
    ----------
    hyperview_score : float
        Overall HYPERVIEW score
    normalized_mses : np.ndarray
        Normalized MSE for each target
    baseline_mses : np.ndarray
        Baseline MSE for each target
    additional_metrics : dict, optional
        Additional metrics (MAE, RMSE, R²)
    target_names : list, optional
        Names of target variables (default: ['B', 'Cu', 'Zn', 'Fe', 'S', 'Mn'])
    """
    if target_names is None:
        target_names = ['B', 'Cu', 'Zn', 'Fe', 'S', 'Mn']
    
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    
    # Overall HYPERVIEW score
    logger.info(f"\n{'HYPERVIEW Score:':<30} {hyperview_score:.4f}")
    logger.info(f"{'(Lower is better, 0 is perfect)':<30}")
    
    # Per-target scores
    logger.info("\n" + "-" * 80)
    logger.info(f"{'Target':<10} {'Norm MSE':<12} {'Baseline MSE':<15}")
    logger.info("-" * 80)
    
    for i, target in enumerate(target_names):
        logger.info(
            f"{target:<10} {normalized_mses[i]:<12.4f} {baseline_mses[i]:<15.2f}"
        )
    
    # Additional metrics
    if additional_metrics is not None:
        logger.info("\n" + "-" * 80)
        logger.info(f"{'Target':<10} {'MAE':<12} {'RMSE':<12} {'R²':<12}")
        logger.info("-" * 80)
        
        for i, target in enumerate(target_names):
            logger.info(
                f"{target:<10} "
                f"{additional_metrics['mae'][i]:<12.4f} "
                f"{additional_metrics['rmse'][i]:<12.4f} "
                f"{additional_metrics['r2'][i]:<12.4f}"
            )
    
    logger.info("=" * 80 + "\n")


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    target_names: list = None,
    log_results: bool = True
) -> Dict:
    """
    Comprehensive evaluation of predictions.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values
    y_pred : np.ndarray
        Predicted values
    y_train : np.ndarray
        Training set values (for baseline)
    target_names : list, optional
        Names of target variables
    log_results : bool, default=True
        Whether to log results
        
    Returns
    -------
    dict
        Dictionary containing all evaluation metrics
    """
    # Calculate HYPERVIEW score
    hyperview_score, normalized_mses, baseline_mses = calculate_hyperview_score(
        y_true, y_pred, y_train
    )
    
    # Calculate additional metrics
    additional_metrics = calculate_additional_metrics(y_true, y_pred)
    
    # Log results
    if log_results:
        log_evaluation_results(
            hyperview_score,
            normalized_mses,
            baseline_mses,
            additional_metrics,
            target_names
        )
    
    # Return all metrics
    return {
        'hyperview_score': hyperview_score,
        'normalized_mses': normalized_mses,
        'baseline_mses': baseline_mses,
        'mae': additional_metrics['mae'],
        'rmse': additional_metrics['rmse'],
        'r2': additional_metrics['r2']
    }