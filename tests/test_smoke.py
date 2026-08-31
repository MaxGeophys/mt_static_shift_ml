from pathlib import Path

from mt_shift.features import prepare_features
from mt_shift.io import load_tabular_data
from mt_shift.model import RHO_FEATURES


def test_training_data_has_required_features():
    root = Path(__file__).resolve().parents[1]
    data = load_tabular_data(root / "data" / "Train_model.txt")
    prepared = prepare_features(data)
    assert all(column in prepared.columns for column in RHO_FEATURES)
    assert "KR" in prepared.columns


def test_exponential_spatial_weights_for_one_step_neighbourhood():
    import numpy as np

    from mt_shift.filtering import exponential_spatial_weights

    weights = exponential_spatial_weights(
        radius_steps=1,
        half_width_steps=1.0,
        steepness=3.0,
    )

    assert weights.shape == (3, 3)
    assert np.isclose(weights[1, 1], 1.0)
    assert np.isclose(weights[0, 1], np.exp(-1.0))
    assert np.isclose(weights[0, 0], np.exp(-(np.sqrt(2.0) ** 3)))
    assert weights[0, 0] < weights[0, 1] < weights[1, 1]


def test_ml_feature_radius_is_independent_from_spatial_filter_radius():
    import pandas as pd

    from config import ML_FEATURE_RADIUS_STEPS
    from mt_shift.features import prepare_features
    from mt_shift.filtering import spatial_filter

    root = Path(__file__).resolve().parents[1]
    raw = load_tabular_data(root / "data" / "Test_model.txt")

    prepared = prepare_features(raw, radius_steps=ML_FEATURE_RADIUS_STEPS)
    ml_feature_columns = [
        "Rho_mean",
        "Zxx_mean",
        "Zxy_mean",
        "Zyx_mean",
        "Zyy_mean",
        "Rho_mean_attitude",
        "Zxx_mean_attitude",
        "Zxy_mean_attitude",
        "Zyx_mean_attitude",
        "Zyy_mean_attitude",
        "WA",
        "WA_Zxy",
        "WA_Zyx",
        "WD",
    ]
    before = prepared[ml_feature_columns].copy()

    # Changing only the baseline filter radius must not mutate/recalculate
    # the ML feature table.
    _ = spatial_filter(
        prepared,
        filter_period=100.0,
        radius_steps=2,
        half_width_steps=1.0,
        steepness=3.0,
    )

    pd.testing.assert_frame_equal(before, prepared[ml_feature_columns])
