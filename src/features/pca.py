
from pathlib import Path
import numpy as np
import logging
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("hyperview2")


# ─────────────────────────────────────────────
# VARIANCE FILTER
# ─────────────────────────────────────────────
def fit_variance_filter(X: np.ndarray, threshold_percentile: float, name: str) -> np.ndarray:
    variances = np.var(X, axis=0)
    cutoff = np.percentile(variances, threshold_percentile)
    mask = variances >= cutoff
    n_removed = int((~mask).sum())
    logger.info(
        f"[VAR FILTER] {name}: removed {n_removed}/{X.shape[1]} features "
    )
    return mask


def apply_variance_filter(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return X[:, mask]


# ─────────────────────────────────────────────
# CORRELATION FILTER
# ─────────────────────────────────────────────
def fit_correlation_filter(X: np.ndarray, threshold: float, name: str) -> np.ndarray:
    if X.shape[1] < 2:
        return np.ones(X.shape[1], dtype=bool)

    corr = np.corrcoef(X, rowvar=False)
    np.fill_diagonal(corr, 0.0) # to ignore self-correlation
    corr = np.abs(corr)

    n = X.shape[1]
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if keep[j] and corr[i, j] > threshold:
                keep[j] = False # to drop the later feature of the pair

    n_removed = int((~keep).sum())
    logger.info(
        f"[CORR FILTER] {name}: removed {n_removed}/{n} features "
    )
    return keep


def apply_correlation_filter(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return X[:, mask]


# ─────────────────────────────────────────────
# PCA
# ─────────────────────────────────────────────

def fit_pca(X: np.ndarray, n_components: int, name: str) -> tuple:
    
    if X.size == 0:
        logger.warning(f"[PCA] {name}: empty data, skipping")
        return X, None

    n_components = min(n_components, X.shape[1], X.shape[0])
    if n_components != n_components:      # was clamped
        logger.warning(f"[PCA] {name}: clamped n_components to {n_components}")

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    var = pca.explained_variance_ratio_.sum() * 100
    logger.info(f"[PCA] [TRAIN] {name}: {X.shape[1]} → {n_components} components ({var:.2f}% variance explained)")
    return X_pca, pca


def transform_with_PCA(X: np.ndarray, pca_model, modality_name: str = None) -> np.ndarray:
    if modality_name:
        logger.info(f"[PCA] [VAL] Transforming {modality_name}")
    return pca_model.transform(X)


# ─────────────────────────────────────────────
# PER-MODALITY PREPROCESSING
# ─────────────────────────────────────────────

def fit_modality_preprocessor(
    X: np.ndarray,
    n_components: int,
    name: str,
    var_threshold: float = 5.0,
    corr_threshold: float = 0.95,
    use_variance_filter: bool = True,
    use_correlation_filter: bool = True,
    use_scaler: bool = True,
) -> tuple:
    
    preprocessor = {
        'var_mask': None,
        'corr_mask': None,
        'scaler': None,
        'pca': None,
    }

    # variance filter
    if use_variance_filter:
        var_mask = fit_variance_filter(X, var_threshold, name)
        X = apply_variance_filter(X, var_mask)
        preprocessor['var_mask'] = var_mask
        if X.shape[1] == 0:
            logger.warning(f"[PREPROC] {name}: all features removed by variance filter")
            return X, preprocessor

    # correlation filter
    if use_correlation_filter:
        corr_mask = fit_correlation_filter(X, corr_threshold, name)
        X = apply_correlation_filter(X, corr_mask)
        preprocessor['corr_mask'] = corr_mask
        if X.shape[1] == 0:
            logger.warning(f"[PREPROC] {name}: all features removed by correlation filter")
            return X, preprocessor

    logger.info(f"[PREPROC] {name}: {X.shape[1]} features remaining after filtering")

    # standardScaler
    if use_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        preprocessor['scaler'] = scaler
    else:
        logger.info(f"[PREPROC] {name}: StandardScaler disabled")

    # PCA
    X, pca = fit_pca(X, n_components, name)
    preprocessor['pca'] = pca

    return X, preprocessor


def transform_modality(
    X: np.ndarray,
    preprocessor: dict,
    name: str = None,
) -> np.ndarray:
   
    tag = f"[PREPROC] {name}" if name else "[PREPROC]"

    if preprocessor['var_mask'] is not None:
        X = apply_variance_filter(X, preprocessor['var_mask'])

    if preprocessor['corr_mask'] is not None:
        X = apply_correlation_filter(X, preprocessor['corr_mask'])

    if preprocessor['scaler'] is not None:
        X = preprocessor['scaler'].transform(X)

    if preprocessor['pca'] is not None:
        X = transform_with_PCA(X, preprocessor['pca'], modality_name=name)

    return X


# ─────────────────────────────────────────────
# BATCH APPLY TO ALL MODALITIES
# ─────────────────────────────────────────────

def apply_pca(
    X_dict: dict,
    pca_config: dict,
    feature_selection_config: dict = None,
    fit: bool = True,
    preprocessors: dict = None,
) -> tuple:
    
    fs = feature_selection_config or {}
    use_var    = fs.get('use_variance_filter', True)
    var_thr    = fs.get('variance_threshold', 5.0)
    use_corr   = fs.get('use_correlation_filter', True)
    corr_thr   = fs.get('correlation_threshold', 0.95)
    use_scaler = fs.get('use_scaler', True)

    X_out = {}
    fitted_preprocessors = {} if fit else preprocessors

    for name, X in X_dict.items():
        if X.size == 0:
            X_out[name] = X
            if fit:
                fitted_preprocessors[name] = None
            continue

        if name not in pca_config:
            X_out[name] = X
            if fit:
                fitted_preprocessors[name] = None
            continue

        n_components = pca_config[name]

        if fit:
            X_transformed, preprocessor = fit_modality_preprocessor(
                X, n_components, name,
                var_threshold=var_thr,
                corr_threshold=corr_thr,
                use_variance_filter=use_var,
                use_correlation_filter=use_corr,
                use_scaler=use_scaler,
            )
            X_out[name] = X_transformed
            fitted_preprocessors[name] = preprocessor
        else:
            prep = preprocessors.get(name)
            if prep is None:
                logger.warning(f"[PREPROC] No preprocessor found for {name}, using raw data")
                X_out[name] = X
            else:
                X_out[name] = transform_modality(X, prep, name=name)

    return X_out, fitted_preprocessors


# ─────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────

def save_pca_models(preprocessors: dict, save_dir: Path):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for name, prep in preprocessors.items():
        if prep is None:
            continue
        path = save_dir / f"preprocessor_{name}.pkl"
        joblib.dump(prep, path)
        logger.info(f"[PREPROC] Saved {name} preprocessor → {path}")


def load_pca_models(load_dir: Path, modalities: list) -> dict:
    load_dir = Path(load_dir)
    preprocessors = {}

    for name in modalities:
        path = load_dir / f"preprocessor_{name}.pkl"
        if path.exists():
            preprocessors[name] = joblib.load(path)
            logger.info(f"[PREPROC] Loaded {name} preprocessor")
        else:
            logger.warning(f"[PREPROC] Missing preprocessor for {name} at {path}")
            preprocessors[name] = None

    return preprocessors
