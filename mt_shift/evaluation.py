import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
    }


def compare_methods(data: pd.DataFrame) -> pd.DataFrame:
    targets = {
        "Rho": ("Rho_real", "Rho", "Rho_filtered", "Rho_ml"),
        "Zxx": ("absZxx_real", "absZxx", "Zxx_filtered", "Zxx_ml"),
        "Zxy": ("absZxy_real", "absZxy", "Zxy_filtered", "Zxy_ml"),
        "Zyx": ("absZyx_real", "absZyx", "Zyx_filtered", "Zyx_ml"),
        "Zyy": ("absZyy_real", "absZyy", "Zyy_filtered", "Zyy_ml"),
    }
    method_names = (
        "inhomogeneous near-surface layer",
        "spatial filtering",
        "gradient boosting",
    )
    rows = []

    for target, columns in targets.items():
        reference = data[columns[0]]
        for method, column in zip(method_names, columns[1:]):
            row = regression_metrics(reference, data[column])
            row["target"] = target
            row["method"] = method
            rows.append(row)

    return pd.DataFrame(rows)[["target", "method", "MSE", "RMSE", "MAE", "R2", "MAPE"]]
