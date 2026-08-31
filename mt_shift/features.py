from __future__ import annotations

import numpy as np
import pandas as pd


def processing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()


    df["Zxx"] = df["realZxx"] + 1j * df["imagZxx"]
    df["Zxy"] = df["realZxy"] + 1j * df["imagZxy"]
    df["Zyx"] = df["realZyx"] + 1j * df["imagZyx"]
    df["Zyy"] = df["realZyy"] + 1j * df["imagZyy"]


    df["Zxx_real"] = df["realZxx_real"] + 1j * df["imagZxx_real"]
    df["Zxy_real"] = df["realZxy_real"] + 1j * df["imagZxy_real"]
    df["Zyx_real"] = df["realZyx_real"] + 1j * df["imagZyx_real"]
    df["Zyy_real"] = df["realZyy_real"] + 1j * df["imagZyy_real"]

    for component in ("xx", "xy", "yx", "yy"):
        df[f"absZ{component}"] = np.abs(df[f"Z{component}"])
        df[f"absZ{component}_real"] = np.abs(df[f"Z{component}_real"])



    df["Kxx"] = df["absZxx"] / df["absZxx_real"]
    df["Kxy"] = df["absZxy"] / df["absZxy_real"]
    df["Kyx"] = df["absZyx"] / df["absZyx_real"]
    df["Kyy"] = df["absZyy"] / df["absZyy_real"]

    real_zxx = df["realZxx"].values
    imag_zxx = df["imagZxx"].values
    real_zxy = df["realZxy"].values
    imag_zxy = df["imagZxy"].values
    real_zyx = df["realZyx"].values
    imag_zyx = df["imagZyx"].values
    real_zyy = df["realZyy"].values
    imag_zyy = df["imagZyy"].values


    det = real_zxx * real_zyy - real_zxy * imag_zyx

    mask = det != 0
    for name in ("Fxx", "Fxy", "Fyx", "Fyy"):
        df[name] = 0.0

    df.loc[mask, "Fxx"] = (
        real_zyy[mask] * imag_zxx[mask]
        - real_zxy[mask] * imag_zyx[mask]
    ) / det[mask]
    df.loc[mask, "Fxy"] = (
        real_zyy[mask] * imag_zxy[mask]
        - real_zxy[mask] * imag_zyy[mask]
    ) / det[mask]
    df.loc[mask, "Fyx"] = (
        real_zxx[mask] * imag_zyx[mask]
        - real_zyx[mask] * imag_zxx[mask]
    ) / det[mask]
    df.loc[mask, "Fyy"] = (
        real_zxx[mask] * imag_zyy[mask]
        - real_zyx[mask] * imag_zxy[mask]
    ) / det[mask]

    fxy = df["Fxy"].values
    fyx = df["Fyx"].values
    fxx = df["Fxx"].values
    fyy = df["Fyy"].values

    denominator = fxx - fyy
    alpha_cbb = np.full_like(denominator, np.nan)
    mask_denom = denominator != 0
    alpha_cbb[mask_denom] = np.mod(
        np.rad2deg(
            np.arctan((fxy[mask_denom] + fyx[mask_denom]) / denominator[mask_denom]) / 2
        ),
        90,
    )
    df["AlphaCBB"] = alpha_cbb

    zxx = df["Zxx"].values
    zxy = df["Zxy"].values
    zyx = df["Zyx"].values
    zyy = df["Zyy"].values

    zxx_real = df["Zxx_real"].values
    zxy_real = df["Zxy_real"].values
    zyx_real= df["Zyx_real"].values
    zyy_real = df["Zyy_real"].values

    zr = (zxy - zyx) / 2
    det_z = zxx * zyy - zxy * zyx
    det_z_real = zxx_real * zyy_real - zxy_real * zyx_real
    tmp = np.sqrt(zr**2 - det_z)

    mu0 = 4 * np.pi / 10000000
    Omega = 2*np.pi/df['T']
    df['Rho'] = (np.abs(np.sqrt(det_z))**2)/(mu0*Omega)
    df['Rho_real'] = (np.abs(np.sqrt(det_z_real)) ** 2) / (mu0 * Omega)

    zp_pl = zr + tmp
    zp_mi = zr - tmp

    pe_pl = np.full_like(zp_pl, np.nan, dtype=np.complex128)
    pe_mi = np.full_like(zp_mi, np.nan, dtype=np.complex128)

    mask_pl = (zp_pl + zxx + zyx) != 0
    mask_mi = (zp_mi + zxx + zyx) != 0

    pe_pl[mask_pl] = (
        zxy[mask_pl] - zp_pl[mask_pl] + zyy[mask_pl]
    ) / (zp_pl[mask_pl] + zxx[mask_pl] + zyx[mask_pl])
    pe_mi[mask_mi] = (
        zxy[mask_mi] - zp_mi[mask_mi] + zyy[mask_mi]
    ) / (zp_mi[mask_mi] + zxx[mask_mi] + zyx[mask_mi])

    tp_pl = np.mod(
        np.degrees(np.arctan2(2 * np.real(pe_pl), 1 - np.abs(pe_pl) ** 2) / 2),
        90,
    )
    tp_mi = np.mod(
        np.degrees(np.arctan2(2 * np.real(pe_mi), 1 - np.abs(pe_mi) ** 2) / 2),
        90,
    )

    df["Eggers1"] = tp_pl
    df["Eggers2"] = tp_mi
    df["AlphaEgg1"] = np.abs(df["Eggers1"] - df["AlphaCBB"]) / 90
    df["AlphaEgg2"] = np.abs(df["Eggers2"] - df["AlphaCBB"]) / 90

    return df


def calculate_moving_average(
    pivot_multi: pd.DataFrame,
    radius_steps: int = 1,
) -> pd.DataFrame:
    result_df = pivot_multi.copy()

    for period in pivot_multi.index.get_level_values("T").unique():
        period_data = pivot_multi.xs(period, level="T")
        values = period_data.values
        rows, cols = values.shape
        period_result = np.full_like(values, np.nan, dtype=float)

        for i in range(rows):
            for j in range(cols):
                row_start = max(0, i - radius_steps)
                row_end = min(rows, i + radius_steps + 1)
                col_start = max(0, j - radius_steps)
                col_end = min(cols, j + radius_steps + 1)

                window = values[row_start:row_end, col_start:col_end]
                window_mean = np.nanmean(window)

                if not np.isnan(window_mean) and window_mean != 0:
                    period_result[i, j] = window_mean

        period_result_df = pd.DataFrame(
            period_result,
            index=period_data.index,
            columns=period_data.columns,
        )

        for y_idx in period_data.index:
            result_df.loc[(period, y_idx), :] = period_result_df.loc[y_idx, :].values

    return result_df


def add_spatial_features(
    df: pd.DataFrame,
    radius_steps: int = 1,
) -> pd.DataFrame:
    df = df.copy()

    pivots = {
        "Rho": df.pivot_table(index=["T", "Y"], columns="X", values="Rho"),
        "Zxx": df.pivot_table(index=["T", "Y"], columns="X", values="absZxx"),
        "Zxy": df.pivot_table(index=["T", "Y"], columns="X", values="absZxy"),
        "Zyx": df.pivot_table(index=["T", "Y"], columns="X", values="absZyx"),
        "Zyy": df.pivot_table(index=["T", "Y"], columns="X", values="absZyy"),
    }

    mean_frames = []
    for name, pivot in pivots.items():
        smoothed = calculate_moving_average(pivot, radius_steps=radius_steps)
        frame = smoothed.stack().reset_index()
        frame.columns = ["T", "Y", "X", f"{name}_mean"]
        mean_frames.append(frame)

    spatial_means = mean_frames[0]
    for frame in mean_frames[1:]:
        spatial_means = spatial_means.merge(frame)

    df = df.merge(spatial_means)

    df["Rho_mean_attitude"] = df["Rho"] / df["Rho_mean"]
    df["Zxx_mean_attitude"] = df["absZxx"] / df["Zxx_mean"]
    df["Zxy_mean_attitude"] = df["absZxy"] / df["Zxy_mean"]
    df["Zyx_mean_attitude"] = df["absZyx"] / df["Zyx_mean"]
    df["Zyy_mean_attitude"] = df["absZyy"] / df["Zyy_mean"]


    df["lgRho"] = np.log10(df["Rho"])
    df["lgRho_mean"] = np.log10(df["Rho_mean"])
    df["WA"] = df["Rho_mean_attitude"].apply(lambda x: 1 / x if x > 1 else x)
    df["WA_Zxy"] = df["Zxy_mean_attitude"].apply(lambda x: 1 / x if x > 1 else x)
    df["WA_Zyx"] = df["Zyx_mean_attitude"].apply(lambda x: 1 / x if x > 1 else x)
    df["WD"] = (45 - df["AlphaEgg1"]) / 45

    return df


def prepare_features(
    df: pd.DataFrame,
    radius_steps: int = 1,
) -> pd.DataFrame:
    return add_spatial_features(processing(df), radius_steps=radius_steps)
