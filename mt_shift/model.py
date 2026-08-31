from __future__ import annotations

from xgboost import XGBRegressor




XGB_RHO_PARAMS = {
    "max_depth": 6,
    "n_estimators": 1000,
    "objective": "reg:pseudohubererror",
    "learning_rate": 0.01,
    "reg_alpha": 0,
    "reg_lambda": 15,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "n_jobs": -1,
    "verbosity": 0,
}

RHO_FEATURES = [
    "Rho_mean_attitude",
    "Rho",
    "absZxy",
    "imagZxy",
]

XGB_Z_MAIN_PARAMS = {
    'max_depth': 7,
    'n_estimators': 2000,
    'objective' : "reg:pseudohubererror",
    'learning_rate': 0.01,
    'reg_alpha': 0,
    'reg_lambda': 15,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'n_jobs' : -1,
}

Zxy_FEATURES = [
    "Zxy_mean_attitude",
    "Rho",
    "absZxy",
    "absZxx",
    "imagZxy",
    "AlphaEgg2"
]

Zyx_FEATURES = [
    "Zyx_mean_attitude",
    "Rho",
    "absZyx",
    "absZyy",
    "imagZyx",
    "AlphaEgg2"
]

XGB_Z_ADD_PARAMS = {
    'max_depth': 7,
    'n_estimators': 2000,
    'objective' : "reg:pseudohubererror",
    'learning_rate': 0.01,
    'reg_alpha': 0,
    'reg_lambda': 15,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'n_jobs' : -1
}

Zxx_FEATURES = [
    "Zxx_mean_attitude",
    "absZxx",
    "imagZxx",
    "Kxy_predicted",
    "AlphaEgg2"
]

Zyy_FEATURES = [
    "Zyy_mean_attitude",
    "absZyy",
    "imagZyy",
    "Kyx_predicted",
    "AlphaEgg2"
]


def create_model(params: dict) -> XGBRegressor:
    return XGBRegressor(**params)
