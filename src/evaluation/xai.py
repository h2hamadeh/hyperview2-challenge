# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 00:19:09 2026

@author: Hachem
"""

# -*- coding: utf-8 -*-
"""
Explainable AI (XAI) utilities for HYPERVIEW2 challenge.

Author: Hachem
Created: Fri Jan 2 00:20:45 2026
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import permutation_importance
import logging

logger = logging.getLogger("hyperview2")

# -----------------------------
# PERMUTATION IMPORTANCE
# -----------------------------
def run_permutation_importance(model, X, y, feature_names, output_dir, target_name):
    """Compute permutation importance and save plot and CSV."""
    result = permutation_importance(
        model, X, y, n_repeats=10, random_state=9000, n_jobs=2
    )
    importances = result.importances_mean
    indices = np.argsort(importances)[::-1]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.title(f"Permutation Importance - {target_name}")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"perm_importance_{target_name}.png"))
    plt.close()

    # Save CSV
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df.to_csv(os.path.join(output_dir, f"perm_importance_{target_name}.csv"), index=False)
    logger.info(f"[XAI] Permutation importance saved for {target_name}")


# -----------------------------
# SHAP ANALYSIS
# -----------------------------
def run_shap_analysis(model, X, feature_names, output_dir, target_name):
    """Compute SHAP values and save summary plot and CSV."""
    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        # Summary Plot
        plt.figure(figsize=(12, 6))
        shap.summary_plot(shap_values, features=X, feature_names=feature_names, show=False)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"shap_summary_{target_name}.png"))
        plt.close()
        plt.close('all')  # Force close all remaining figures
        
        # Save SHAP values as CSV
        shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
        shap_df.to_csv(os.path.join(output_dir, f"shap_values_{target_name}.csv"), index=False)
        logger.info(f"[XAI] SHAP analysis saved for {target_name}")
    except Exception as e:
        logger.warning(f"[XAI] SHAP analysis failed for {target_name}: {e}")


# -----------------------------
# RESIDUAL / ERROR ANALYSIS
# -----------------------------
def run_error_analysis(y_true, y_pred, output_dir, target_names=None):
    """
    Plot predicted vs actual and spatial predictions as multi-target grids (2x3).
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values, shape (n_samples, n_targets)
    y_pred : np.ndarray
        Predicted values, shape (n_samples, n_targets)
    output_dir : str
        Output directory for saving plots and CSV
    target_names : list, optional
        Names of targets (default: ['B', 'Cu', 'Zn', 'Fe', 'S', 'Mn'])
    """
    if target_names is None:
        target_names = ['B', 'Cu', 'Zn', 'Fe', 'S', 'Mn']
    
    os.makedirs(output_dir, exist_ok=True)
    n_targets = y_true.shape[1]
    
    # ===== PLOT 1: Predicted vs Actual (2x3 grid) =====
    plt.figure(figsize=(18, 10))
    for i, target in enumerate(target_names):
        plt.subplot(2, 3, i + 1)
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.6, edgecolor='k')
        plt.plot(
            [y_true[:, i].min(), y_true[:, i].max()],
            [y_true[:, i].min(), y_true[:, i].max()],
            'r--',
            linewidth=2
        )
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{target}')
        plt.grid(True, alpha=0.3)
    plt.suptitle('Predicted vs Actual - All Targets', fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pred_vs_actual_grid.png"), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"[XAI] Pred vs Actual grid plot saved")
    
    # ===== PLOT 2: Spatial/Sequential Predictions (2x3 grid) =====
    plt.figure(figsize=(18, 12))
    for i, target in enumerate(target_names):
        plt.subplot(2, 3, i + 1)
        plt.plot(y_true[:, i], label='Actual', color='red', linewidth=2)
        plt.plot(y_pred[:, i], label='Predicted', color='blue', alpha=0.7, linewidth=2)
        plt.title(f'{target}')
        plt.xlabel('Spatial Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.suptitle('Spatial/Sequential Predictions - All Targets', fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spatial_predictions_grid.png"), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"[XAI] Spatial predictions grid plot saved")

    ## ===== SAVE ERROR CSV =====
    residuals = y_true - y_pred
    errors_data = {}
    
    for i, target in enumerate(target_names):
        errors_data[f'{target}_actual'] = y_true[:, i]
        errors_data[f'{target}_predicted'] = y_pred[:, i]
        errors_data[f'{target}_residual'] = residuals[:, i]
    
    df = pd.DataFrame(errors_data)
    df.to_csv(os.path.join(output_dir, "errors_all_targets.csv"), index=False)
    logger.info(f"[XAI] Error analysis CSV saved for all targets")


# -----------------------------
# FULL XAI ANALYSIS WRAPPER
# -----------------------------
def run_full_xai_analysis(models, X_val, y_val, feature_names, output_dir, config=None, target_names=None):
    """
    Run XAI analysis for all models.
    Saves permutation importance, SHAP values, and residual/error plots/CSVs.
    
    Parameters
    ----------
    models : dict
        Dictionary of trained models (target -> model)
    X_val : np.ndarray
        Validation features
    y_val : np.ndarray
        Validation targets, shape (n_samples, n_targets)
    feature_names : list
        Names of features
    output_dir : str
        Output directory for XAI results
    config : dict, optional
        Configuration dictionary
    target_names : list, optional
        Names of targets
    """
    os.makedirs(output_dir, exist_ok=True)

    if target_names is None:
        target_names = list(models.keys())

    # Generate predictions for all targets
    y_pred = np.stack([models[t].predict(X_val) for t in target_names], axis=1)
    
    # Per-target analyses (Permutation Importance & SHAP)
    for i, target in enumerate(target_names):
        y_true_target = y_val[:, i]
        y_pred_target = y_pred[:, i]

        # Permutation Importance
        run_permutation_importance(models[target], X_val, y_true_target, feature_names, output_dir, target)

        # SHAP Analysis
        run_shap_analysis(models[target], X_val, feature_names, output_dir, target)
    
    # Multi-target error analysis (creates 2x3 grid plots)
    run_error_analysis(y_val, y_pred, output_dir, target_names=target_names)