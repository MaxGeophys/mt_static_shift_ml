from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib

from config import (
    FILTER_HALF_WIDTH_STEPS,
    FILTER_PERIOD,
    FILTER_RADIUS_STEPS,
    FILTER_STEEPNESS,
    ML_FEATURE_RADIUS_STEPS,
)
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from mt_shift.features import prepare_features
from mt_shift.filtering import spatial_filter
from mt_shift.io import load_tabular_data, save_model_data
from mt_shift.model import (
    RHO_FEATURES,
    Zxx_FEATURES,
    Zxy_FEATURES,
    Zyx_FEATURES,
    Zyy_FEATURES,
    XGB_RHO_PARAMS,
    XGB_Z_ADD_PARAMS,
    XGB_Z_MAIN_PARAMS,
    create_model,
)

DATA_PATH = PROJECT_ROOT / "data" / "Train_model.txt"
MODEL_PATHS = {
    "Rho": PROJECT_ROOT / "models" / "XGBR_Rho.joblib",
    "Zxx": PROJECT_ROOT / "models" / "XGBR_Zxx.joblib",
    "Zxy": PROJECT_ROOT / "models" / "XGBR_Zxy.joblib",
    "Zyx": PROJECT_ROOT / "models" / "XGBR_Zyx.joblib",
    "Zyy": PROJECT_ROOT / "models" / "XGBR_Zyy.joblib",
}
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def validation(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    prediction = model.predict(X_test)
    return {
        "MSE": mean_squared_error(y_test, prediction),
        "RMSE": root_mean_squared_error(y_test, prediction),
        "MAE": mean_absolute_error(y_test, prediction),
        "R2": r2_score(y_test, prediction),
    }


def fit_model(data: pd.DataFrame, features: list[str], target: str, params: dict):
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    model = create_model(params)
    model.fit(X_train, y_train)
    return model, validation(model, X_test, y_test)


def train_models(data: pd.DataFrame):
    data_train = data[data["Rho"] <= 6000].copy()
    models = {}
    metrics = {}

    models["Rho"], metrics["Rho"] = fit_model(
        data_train, RHO_FEATURES, "KR", XGB_RHO_PARAMS
    )
    models["Zxy"], metrics["Zxy"] = fit_model(
        data_train, Zxy_FEATURES, "Kxy", XGB_Z_MAIN_PARAMS
    )
    models["Zyx"], metrics["Zyx"] = fit_model(
        data_train, Zyx_FEATURES, "Kyx", XGB_Z_MAIN_PARAMS
    )

    data_train["Kxy_predicted"] = models["Zxy"].predict(data_train[Zxy_FEATURES])
    data_train["Kyx_predicted"] = models["Zyx"].predict(data_train[Zyx_FEATURES])

    models["Zxx"], metrics["Zxx"] = fit_model(
        data_train, Zxx_FEATURES, "Kxx", XGB_Z_ADD_PARAMS
    )
    models["Zyy"], metrics["Zyy"] = fit_model(
        data_train, Zyy_FEATURES, "Kyy", XGB_Z_ADD_PARAMS
    )

    return models, metrics, data_train


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "models").mkdir(parents=True, exist_ok=True)

    data = prepare_features(
        load_tabular_data(DATA_PATH),
        radius_steps=ML_FEATURE_RADIUS_STEPS,
    )
    filtered_data = spatial_filter(
        data,
        filter_period=FILTER_PERIOD,
        radius_steps=FILTER_RADIUS_STEPS,
        half_width_steps=FILTER_HALF_WIDTH_STEPS,
        steepness=FILTER_STEEPNESS,
    )
    save_model_data(
        filtered_data[["X", "Y", "T", "Rho", "Rho_mean", "Rho_filtered", "KR"]],
        OUTPUT_DIR / "training_spatial_filter_results.txt",
    )

    models, metrics, data_train = train_models(data)

    for name, model in models.items():
        joblib.dump(model, MODEL_PATHS[name])

    save_model_data(
        data_train[RHO_FEATURES + ["KR"]],
        OUTPUT_DIR / "MT_Data_train.txt",
    )

    print("\nValidation")
    for predictor in ("Rho", "Zxx", "Zxy", "Zyx", "Zyy"):
        values = " | ".join(
            f"{name}: {value:.6f}" for name, value in metrics[predictor].items()
        )
        print(f"{predictor}: {values}")


if __name__ == "__main__":
    main()
