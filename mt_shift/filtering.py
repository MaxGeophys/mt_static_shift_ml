from __future__ import annotations

import numpy as np
import pandas as pd


def _pivot_xy(
    data: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    return data.pivot_table(
        index="Y",
        columns="X",
        values=value_column,
    )


def _pivot_ty(
    data: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    return data.pivot_table(
        index=["T", "Y"],
        columns="X",
        values=value_column,
    )


def exponential_spatial_weights(
    radius_steps: int,
    half_width_steps: float,
    steepness: float,
) -> np.ndarray:
    """Build the spatial weights W = exp(-(|R| / R0) ** q).

    Distances are expressed in observation-grid steps. The neighbourhood is
    square: radius_steps=1 produces a 3x3 window. Euclidean distance is used
    only for the weight itself, so orthogonal neighbours have R=1 and diagonal
    neighbours have R=sqrt(2).
    """
    if radius_steps < 0:
        raise ValueError("radius_steps must be >= 0")
    if half_width_steps <= 0:
        raise ValueError("half_width_steps must be > 0")
    if steepness <= 0:
        raise ValueError("steepness must be > 0")

    offsets = np.arange(-radius_steps, radius_steps + 1, dtype=float)
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    distance_steps = np.hypot(dx, dy)

    return np.exp(-((distance_steps / half_width_steps) ** steepness))


def _calculate_filter_coefficient(
    data_pivot: pd.DataFrame,
    weight_pivot: pd.DataFrame,
    additional_weight_pivot: pd.DataFrame,
    radius_steps: int,
    half_width_steps: float,
    steepness: float,
) -> pd.DataFrame:
    spatial_weights = exponential_spatial_weights(
        radius_steps=radius_steps,
        half_width_steps=half_width_steps,
        steepness=steepness,
    )

    rows, cols = data_pivot.shape
    window_size = 2 * radius_steps + 1
    filtered_data = np.full(data_pivot.shape, np.nan, dtype=float)

    # Preserve the original behaviour at the grid boundary: only points with a
    # complete neighbourhood are filtered here. Boundary values are filled from
    # the local-mean fallback later in the pipeline.
    for i in range(radius_steps, rows - radius_steps):
        for j in range(radius_steps, cols - radius_steps):
            row_slice = slice(i - radius_steps, i + radius_steps + 1)
            col_slice = slice(j - radius_steps, j + radius_steps + 1)

            values = data_pivot.iloc[row_slice, col_slice].to_numpy(dtype=float)
            local_weights = weight_pivot.iloc[row_slice, col_slice].to_numpy(dtype=float)
            additional_weights = additional_weight_pivot.iloc[
                row_slice, col_slice
            ].to_numpy(dtype=float)

            if values.shape != (window_size, window_size):
                continue

            weights = spatial_weights * local_weights * additional_weights
            valid = np.isfinite(values) & np.isfinite(weights) & (values > 0)

            if not np.any(valid):
                continue

            weights = np.where(valid, weights, 0.0)
            weight_sum = np.sum(weights)

            if weight_sum == 0 or not np.isfinite(weight_sum):
                continue

            # Weighted geometric mean. This preserves the multiplicative
            # filtering logic used by the original implementation.
            normalized_weights = weights / weight_sum
            filtered_value = np.exp(np.sum(normalized_weights[valid] * np.log(values[valid])))
            filtered_data[i, j] = filtered_value

    return pd.DataFrame(
        filtered_data / data_pivot.to_numpy(),
        index=data_pivot.index,
        columns=data_pivot.columns,
    )


def _apply_coefficient(
    full_pivot: pd.DataFrame,
    coefficient: pd.DataFrame,
    fallback_pivot: pd.DataFrame,
) -> pd.DataFrame:
    coefficient_expanded = coefficient.reindex(
        full_pivot.index.get_level_values("Y")
    )
    coefficient_expanded.index = full_pivot.index

    filtered = full_pivot * coefficient_expanded
    return filtered.fillna(fallback_pivot)


def _stack_result(
    pivot: pd.DataFrame,
    value_name: str,
) -> pd.DataFrame:
    result = pivot.stack().reset_index()
    result.columns = [
        "T",
        "Y",
        "X",
        value_name,
    ]
    return result


def spatial_filter(
    data: pd.DataFrame,
    filter_period: float = 100.0,
    radius_steps: int = 1,
    half_width_steps: float = 1.0,
    steepness: float = 3.0,
) -> pd.DataFrame:
    rho = _pivot_ty(data, "Rho")
    zxx = _pivot_ty(data, "absZxx")
    zxy = _pivot_ty(data, "absZxy")
    zyx = _pivot_ty(data, "absZyx")
    zyy = _pivot_ty(data, "absZyy")

    rho_mean = _pivot_ty(data, "Rho_mean")
    zxx_mean = _pivot_ty(data, "Zxx_mean")
    zxy_mean = _pivot_ty(data, "Zxy_mean")
    zyx_mean = _pivot_ty(data, "Zyx_mean")
    zyy_mean = _pivot_ty(data, "Zyy_mean")

    data_period = data[data["T"] == filter_period]

    if data_period.empty:
        available_periods = sorted(data["T"].unique())
        raise ValueError(
            f"Filter period T={filter_period} is not present in the dataset. "
            f"Available periods: {available_periods}"
        )

    rho_period = _pivot_xy(data_period, "Rho")
    zxy_period = _pivot_xy(data_period, "absZxy")
    zyx_period = _pivot_xy(data_period, "absZyx")

    wa_rho = _pivot_xy(data_period, "WA")
    wa_zxy = _pivot_xy(data_period, "WA_Zxy")
    wa_zyx = _pivot_xy(data_period, "WA_Zyx")

    # This is the pre-existing additional weight based on the tensor geometry.
    wd = _pivot_xy(data_period, "WD")

    filter_kwargs = {
        "radius_steps": radius_steps,
        "half_width_steps": half_width_steps,
        "steepness": steepness,
    }

    k_rho = _calculate_filter_coefficient(
        data_pivot=rho_period,
        weight_pivot=wa_rho,
        additional_weight_pivot=wd,
        **filter_kwargs,
    )

    k_zxy = _calculate_filter_coefficient(
        data_pivot=zxy_period,
        weight_pivot=wa_zxy,
        additional_weight_pivot=wd,
        **filter_kwargs,
    )

    k_zyx = _calculate_filter_coefficient(
        data_pivot=zyx_period,
        weight_pivot=wa_zyx,
        additional_weight_pivot=wd,
        **filter_kwargs,
    )

    rho_filtered = _apply_coefficient(
        full_pivot=rho,
        coefficient=k_rho,
        fallback_pivot=rho_mean,
    )
    zxy_filtered = _apply_coefficient(
        full_pivot=zxy,
        coefficient=k_zxy,
        fallback_pivot=zxy_mean,
    )
    zyx_filtered = _apply_coefficient(
        full_pivot=zyx,
        coefficient=k_zyx,
        fallback_pivot=zyx_mean,
    )

    zxx_filtered = _apply_coefficient(
        full_pivot=zxx,
        coefficient=k_zxy,
        fallback_pivot=zxx_mean,
    )
    zyy_filtered = _apply_coefficient(
        full_pivot=zyy,
        coefficient=k_zyx,
        fallback_pivot=zyy_mean,
    )

    result = _stack_result(rho_filtered, "Rho_filtered")
    result = result.merge(
        _stack_result(zxx_filtered, "Zxx_filtered"),
        on=["T", "Y", "X"],
    )
    result = result.merge(
        _stack_result(zxy_filtered, "Zxy_filtered"),
        on=["T", "Y", "X"],
    )
    result = result.merge(
        _stack_result(zyx_filtered, "Zyx_filtered"),
        on=["T", "Y", "X"],
    )
    result = result.merge(
        _stack_result(zyy_filtered, "Zyy_filtered"),
        on=["T", "Y", "X"],
    )

    return data.merge(
        result,
        on=["T", "Y", "X"],
        how="left",
    )
