"""The five regression metrics used in the base paper (MAE, MSE, RMSE, R2, MAPE)."""
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

METRIC_NAMES = ["MAE", "MSE", "RMSE", "R2", "MAPE"]


def compute_metrics(y_true, y_pred):
    """MAPE is returned as a fraction (0.14 = 14%), which is how the paper reports it."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
    }
