
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from typing import Dict, Tuple

logger = logging.getLogger("hyperview2")


def calculate_mse_scores(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    
    return mean_squared_error(y_true, y_pred, multioutput='raw_values')


def calculate_baseline_mse(y_true: np.ndarray, baseline_pred: np.ndarray) -> np.ndarray:
    
    return mean_squared_error(y_true, baseline_pred, multioutput='raw_values')


def calculate_hyperview_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    
    model_mse = calculate_mse_scores(y_true, y_pred)
    
    train_means = np.mean(y_train, axis=0)
    baseline_pred = np.tile(train_means, (y_true.shape[0], 1))
    baseline_mse = calculate_baseline_mse(y_true, baseline_pred)
    
    normalized_mses = model_mse / baseline_mse
    
    hyperview_score = np.mean(normalized_mses)
    
    return hyperview_score, normalized_mses, baseline_mse


def calculate_additional_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    
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
    
    if target_names is None:
        target_names = ['B', 'Cu', 'Zn', 'Fe', 'S', 'Mn']

    logger.info("=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 69)
    logger.info(f"{'HYPERVIEW Score:':<30} {hyperview_score:.4f}")

    logger.info("-" * 68)
    logger.info(f"{'Target':<10} {'Norm MSE':<12} {'Baseline MSE':<15}")
    logger.info("-" * 68)

    for i, target in enumerate(target_names):
        logger.info(
            f"{target:<10} {normalized_mses[i]:<12.4f} {baseline_mses[i]:<15.2f}"
        )

    if additional_metrics is not None:
        logger.info("-" * 68)
        logger.info(f"{'Target':<10} {'MAE':<12} {'RMSE':<12} {'R²':<12}")
        logger.info("-" * 68)

        for i, target in enumerate(target_names):
            logger.info(
                f"{target:<10} "
                f"{additional_metrics['mae'][i]:<12.4f} "
                f"{additional_metrics['rmse'][i]:<12.4f} "
                f"{additional_metrics['r2'][i]:<12.4f}"
            )


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    target_names: list = None,
    log_results: bool = True
) -> Dict:

    hyperview_score, normalized_mses, baseline_mses = calculate_hyperview_score(
        y_true, y_pred, y_train
    )
    
    additional_metrics = calculate_additional_metrics(y_true, y_pred)
    
    if log_results:
        log_evaluation_results(
            hyperview_score,
            normalized_mses,
            baseline_mses,
            additional_metrics,
            target_names
        )

    return {
        'hyperview_score': hyperview_score,
        'normalized_mses': normalized_mses,
        'baseline_mses': baseline_mses,
        'mae': additional_metrics['mae'],
        'rmse': additional_metrics['rmse'],
        'r2': additional_metrics['r2']
    }