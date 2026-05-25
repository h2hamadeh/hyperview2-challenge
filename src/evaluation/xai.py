
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import logging

logger = logging.getLogger("hyperview2")


# ─────────────────────────────────────────────────────────────
# HELPERS: map PCA SHAP to band importances
# ─────────────────────────────────────────────────────────────
def _shap_pca_to_bands(shap_values_pca: np.ndarray, pca_model) -> np.ndarray:
    band_shap = np.abs(shap_values_pca @ pca_model.components_)
    return band_shap.mean(axis=0)


def _apply_filters_to_wavelengths(wavelengths: list, var_mask, corr_mask) -> list:
    wl = np.array(wavelengths)
    if var_mask is not None:
        wl = wl[var_mask]
    if corr_mask is not None:
        wl = wl[corr_mask]
    return wl.tolist()


# ─────────────────────────────────────────────────────────────
# SHAP ON PCA FEATURES + MAP BACK TO BANDS
# ─────────────────────────────────────────────────────────────
def run_spectral_shap(
    models: dict,
    X_val_pca: np.ndarray,
    preprocessors: dict,
    wavelengths_per_modality: dict,
    modality_order: list,
    pca_components_per_modality: list,
    output_dir: str,
    target_names: list,
):
    
    os.makedirs(output_dir, exist_ok=True)

    # build index slices to split X_val_pca back per modality
    slices = {}
    start = 0
    for mod, n_comp in zip(modality_order, pca_components_per_modality):
        slices[mod] = slice(start, start + n_comp)
        start += n_comp

    # per-modality: filtered wavelengths
    filtered_wavelengths = {}
    for mod in modality_order:
        prep = preprocessors.get(mod)
        wl_raw = wavelengths_per_modality.get(mod, [])
        if prep is None or not wl_raw:
            filtered_wavelengths[mod] = wl_raw
        else:
            filtered_wavelengths[mod] = _apply_filters_to_wavelengths(
                wl_raw,
                prep.get('var_mask'),
                prep.get('corr_mask'),
            )

    # compute SHAP for each target
    all_band_importances = {}

    for target in target_names:
        model = models.get(target)
        if model is None:
            logger.warning(f"[XAI] No model for {target}, skipping SHAP")
            continue

        logger.info(f"[XAI] Running TreeExplainer SHAP for {target}")
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_val_pca)   # (n_samples, n_pca_features)
        except Exception as e:
            logger.warning(f"[XAI] TreeExplainer failed for {target}: {e}")
            continue

        band_imp_per_mod = {}
        for mod in modality_order:
            prep = preprocessors.get(mod)
            if prep is None or prep.get('pca') is None:
                continue
            pca_model = prep['pca']
            shap_mod = shap_vals[:, slices[mod]]           # (n_samples, n_comp_mod)
            band_imp = _shap_pca_to_bands(shap_mod, pca_model)
            band_imp_per_mod[mod] = band_imp

        all_band_importances[target] = band_imp_per_mod

        rows = []
        for mod, imp in band_imp_per_mod.items():
            wl = filtered_wavelengths.get(mod, [f"f{i}" for i in range(len(imp))])
            for w, v in zip(wl, imp):
                rows.append({'modality': mod, 'wavelength': w, 'importance': v})
        pd.DataFrame(rows).to_csv(
            os.path.join(output_dir, f"shap_band_importance_{target}.csv"), index=False
        )

    if not all_band_importances:
        logger.warning("[XAI] No SHAP results to plot")
        return

    _plot_spectral_importance(
        all_band_importances, filtered_wavelengths, modality_order,
        target_names, output_dir
    )

    _plot_modality_importance(
        all_band_importances, modality_order, target_names, output_dir
    )


def _plot_spectral_importance(
    all_band_importances, filtered_wavelengths, modality_order, target_names, output_dir
):

    n_targets = len(target_names)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    mod_colors = {
        'msi_satellite':  '#e41a1c',
        'hsi_satellite':  '#377eb8',
        'hsi_airborne':   '#4daf4a',
    }
    mod_labels = {
        'msi_satellite':  'MSI (Sentinel-2)',
        'hsi_satellite':  'HSI Satellite (PRISMA)',
        'hsi_airborne':   'HSI Airborne',
    }

    for ax_idx, target in enumerate(target_names):
        ax = axes[ax_idx]
        band_imp = all_band_importances.get(target, {})

        for mod in modality_order:
            if mod not in band_imp:
                continue
            imp = band_imp[mod]
            wl  = filtered_wavelengths.get(mod, list(range(len(imp))))

            # Use numeric wavelengths if available, else integer index
            try:
                x = [float(w) for w in wl]
            except (TypeError, ValueError):
                x = list(range(len(imp)))

            ax.plot(x, imp,
                    color=mod_colors.get(mod, 'grey'),
                    label=mod_labels.get(mod, mod),
                    linewidth=1.5, alpha=0.85)

        ax.set_title(target, fontsize=13, fontweight='bold')
        ax.set_xlabel('Wavelength (nm)', fontsize=10)
        ax.set_ylabel('Mean |SHAP|', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for i in range(n_targets, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Spectral Band Importance (SHAP → Band Space)', fontsize=16, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'shap_spectral_bands.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"[XAI] Spectral band importance plot saved → {out_path}")


def _plot_modality_importance(
    all_band_importances, modality_order, target_names, output_dir
):

    mod_labels = {
        'msi_satellite':  'MSI',
        'hsi_satellite':  'HSI Sat',
        'hsi_airborne':   'HSI Air',
    }
    mod_colors = {
        'msi_satellite':  '#e41a1c',
        'hsi_satellite':  '#377eb8',
        'hsi_airborne':   '#4daf4a',
    }

    records = []
    for target in target_names:
        band_imp = all_band_importances.get(target, {})
        totals = {mod: float(band_imp[mod].sum()) for mod in modality_order if mod in band_imp}
        grand = sum(totals.values())
        if grand == 0:
            continue
        row = {'target': target}
        for mod in modality_order:
            row[mod] = totals.get(mod, 0.0) / grand * 100
        records.append(row)

    if not records:
        logger.warning("[XAI] No modality importance data to plot")
        return

    df = pd.DataFrame(records).set_index('target')
    df.to_csv(os.path.join(output_dir, 'shap_modality_importance.csv'))

    x = np.arange(len(df))
    width = 0.25
    n_mods = len(modality_order)
    offsets = np.linspace(-(n_mods - 1) / 2, (n_mods - 1) / 2, n_mods) * width

    fig, ax = plt.subplots(figsize=(10, 5))
    for mod, offset in zip(modality_order, offsets):
        if mod not in df.columns:
            continue
        ax.bar(
            x + offset,
            df[mod],
            width=width,
            label=mod_labels.get(mod, mod),
            color=mod_colors.get(mod, 'grey'),
            alpha=0.85,
            edgecolor='white',
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, fontsize=11)
    ax.set_ylabel('% SHAP Contribution', fontsize=11)
    ax.set_title('Modality Importance per Soil Property', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'shap_modality_importance.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"[XAI] Modality importance plot saved → {out_path}")


# ─────────────────────────────────────────────────────────────
# ERROR / RESIDUAL ANALYSIS
# ─────────────────────────────────────────────────────────────

def run_error_analysis(y_true, y_pred, output_dir, target_names=None):
    if target_names is None:
        target_names = ['B', 'Cu', 'Zn', 'Fe', 'S', 'Mn']

    os.makedirs(output_dir, exist_ok=True)

    # PLOT 1: Predicted vs Actual
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (i, target) in zip(axes.flatten(), enumerate(target_names)):
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.6, edgecolor='k', s=20)
        mn = min(y_true[:, i].min(), y_pred[:, i].min())
        mx = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(target)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Predicted vs Actual – All Targets', fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pred_vs_actual_grid.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("[XAI] Pred vs Actual grid saved")

    # PLOT 2: Spatial / Sequential
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for ax, (i, target) in zip(axes.flatten(), enumerate(target_names)):
        ax.plot(y_true[:, i], label='Actual',    color='red',  linewidth=2)
        ax.plot(y_pred[:, i], label='Predicted', color='blue', linewidth=2, alpha=0.7)
        ax.set_title(target)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle('Sequential Predictions – All Targets', fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spatial_predictions_grid.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("[XAI] Spatial predictions grid saved")

    # ── CSV ──
    residuals = y_true - y_pred
    rows = {}
    for i, target in enumerate(target_names):
        rows[f'{target}_actual']    = y_true[:, i]
        rows[f'{target}_predicted'] = y_pred[:, i]
        rows[f'{target}_residual']  = residuals[:, i]
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, 'errors_all_targets.csv'), index=False)
    logger.info("[XAI] Error analysis CSV saved")


# ─────────────────────────────────────────────────────────────
# XAI WRAPPER
# ─────────────────────────────────────────────────────────────

def run_full_xai_analysis(
    models: dict,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list,
    output_dir: str,
    config: dict = None,
    target_names: list = None,
    preprocessors: dict = None,
    wavelengths_per_modality: dict = None,
    modality_order: list = None,
    pca_components_per_modality: list = None,
):

    os.makedirs(output_dir, exist_ok=True)

    if target_names is None:
        target_names = list(models.keys())

    y_pred = np.stack([models[t].predict(X_val) for t in target_names], axis=1)

    spectral_ready = all([
        preprocessors is not None,
        wavelengths_per_modality is not None,
        modality_order is not None,
        pca_components_per_modality is not None,
    ])

    if spectral_ready:
        run_spectral_shap(
            models=models,
            X_val_pca=X_val,
            preprocessors=preprocessors,
            wavelengths_per_modality=wavelengths_per_modality,
            modality_order=modality_order,
            pca_components_per_modality=pca_components_per_modality,
            output_dir=output_dir,
            target_names=target_names,
        )
    else:
        logger.warning(
            "[XAI] Spectral SHAP skipped: preprocessors / wavelengths / "
            "modality_order / pca_components not provided"
        )

    run_error_analysis(y_val, y_pred, output_dir, target_names=target_names)
