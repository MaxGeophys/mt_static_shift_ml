from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

from scripts.evaluate import evaluate

DATA_PATH = PROJECT_ROOT / "data" / "Test_model.txt"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "txt"
MAPS_DIR = OUTPUT_ROOT / "maps"
PROFILES_DIR = OUTPUT_ROOT / "profiles"
CURVES_DIR = OUTPUT_ROOT / "curves"
COMPONENTS = ("Zxx", "Zxy", "Zyx", "Zyy")
METHODS = ("hmg", "anomal", "filtered", "ml")


def period_name(period: float) -> str:
    return f"{period:g}".replace(".", "p")


def coordinate_name(value: float) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p")


def phase_deg(real: pd.Series, imag: pd.Series) -> np.ndarray:
    return np.degrees(np.arctan2(imag.to_numpy(dtype=float), real.to_numpy(dtype=float)))


def build_export_table(data: pd.DataFrame) -> pd.DataFrame:
    export = pd.DataFrame(index=data.index)
    export["X"] = data["X"]
    export["Y"] = data["Y"]
    export["T"] = data["T"]

    export["Rho_hmg"] = data["Rho_real"]
    export["Rho_anomal"] = data["Rho"]
    export["Rho_filtered"] = data["Rho_filtered"]
    export["Rho_ml"] = data["Rho_ml"]

    for component in COMPONENTS:
        export[f"abs{component}_hmg"] = data[f"abs{component}_real"]
        export[f"phase{component}_hmg"] = phase_deg(
            data[f"real{component}_real"], data[f"imag{component}_real"]
        )
        export[f"abs{component}_anomal"] = data[f"abs{component}"]
        export[f"phase{component}_anomal"] = phase_deg(
            data[f"real{component}"], data[f"imag{component}"]
        )
        export[f"abs{component}_filtered"] = data[f"{component}_filtered"]
        export[f"phase{component}_filtered"] = export[f"phase{component}_anomal"]
        export[f"abs{component}_ml"] = data[f"{component}_ml"]
        export[f"phase{component}_ml"] = export[f"phase{component}_anomal"]

    for period in sorted(data["T"].unique()):
        mask = data["T"] == period
        period_data = data.loc[mask]
        export.loc[mask, "RMSE_Rho_anomal"] = root_mean_squared_error(
            period_data["Rho_real"], period_data["Rho"]
        )
        export.loc[mask, "RMSE_Rho_filtered"] = root_mean_squared_error(
            period_data["Rho_real"], period_data["Rho_filtered"]
        )
        export.loc[mask, "RMSE_Rho_ml"] = root_mean_squared_error(
            period_data["Rho_real"], period_data["Rho_ml"]
        )

        for component in COMPONENTS:
            reference = period_data[f"abs{component}_real"]
            export.loc[mask, f"RMSE_{component}_anomal"] = root_mean_squared_error(
                reference, period_data[f"abs{component}"]
            )
            export.loc[mask, f"RMSE_{component}_filtered"] = root_mean_squared_error(
                reference, period_data[f"{component}_filtered"]
            )
            export.loc[mask, f"RMSE_{component}_ml"] = root_mean_squared_error(
                reference, period_data[f"{component}_ml"]
            )

    ordered = ["X", "Y", "T", "Rho_hmg", "Rho_anomal", "Rho_filtered", "Rho_ml"]
    ordered += ["RMSE_Rho_anomal", "RMSE_Rho_filtered", "RMSE_Rho_ml"]

    for component in COMPONENTS:
        for method in METHODS:
            ordered += [f"abs{component}_{method}", f"phase{component}_{method}"]
        ordered += [
            f"RMSE_{component}_anomal",
            f"RMSE_{component}_filtered",
            f"RMSE_{component}_ml",
        ]

    return export[ordered]


def save_table(data: pd.DataFrame, path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, sep="\t", index=False, columns=columns, float_format="%.10g")


def write_maps(data: pd.DataFrame) -> int:
    columns = [column for column in data.columns if column != "T"]
    count = 0
    for period in sorted(data["T"].unique()):
        period_data = data[data["T"] == period].sort_values(["Y", "X"])
        save_table(period_data, MAPS_DIR / f"T_{period_name(period)}s.txt", columns)
        count += 1
    return count


def write_profiles(data: pd.DataFrame) -> int:
    columns = list(data.columns)
    count = 0

    for y in sorted(data["Y"].unique()):
        profile = data[data["Y"] == y].sort_values(["T", "X"])
        save_table(profile, PROFILES_DIR / f"Y_{coordinate_name(y)}m.txt", columns)
        count += 1

    for x in sorted(data["X"].unique()):
        profile = data[data["X"] == x].sort_values(["T", "Y"])
        save_table(profile, PROFILES_DIR / f"X_{coordinate_name(x)}m.txt", columns)
        count += 1

    return count


def write_curves(data: pd.DataFrame) -> int:
    columns = [column for column in data.columns if column not in ("X", "Y")]
    rmse_columns = [column for column in columns if column.startswith("RMSE_")]
    count = 0

    for (x, y), site in data.groupby(["X", "Y"], sort=True):
        site = site.sort_values("T").copy()


        curve_rmse = {
            "RMSE_Rho_anomal": root_mean_squared_error(site["Rho_hmg"], site["Rho_anomal"]),
            "RMSE_Rho_filtered": root_mean_squared_error(site["Rho_hmg"], site["Rho_filtered"]),
            "RMSE_Rho_ml": root_mean_squared_error(site["Rho_hmg"], site["Rho_ml"]),
        }
        for component in COMPONENTS:
            reference = site[f"abs{component}_hmg"]
            curve_rmse[f"RMSE_{component}_anomal"] = root_mean_squared_error(
                reference, site[f"abs{component}_anomal"]
            )
            curve_rmse[f"RMSE_{component}_filtered"] = root_mean_squared_error(
                reference, site[f"abs{component}_filtered"]
            )
            curve_rmse[f"RMSE_{component}_ml"] = root_mean_squared_error(
                reference, site[f"abs{component}_ml"]
            )

        site.loc[:, rmse_columns] = np.nan
        first_row = site.index[0]
        for column, value in curve_rmse.items():
            site.loc[first_row, column] = value

        filename = f"X_{coordinate_name(x)}m_Y_{coordinate_name(y)}m.txt"
        save_table(site, CURVES_DIR / filename, columns)
        count += 1

    return count


def main() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)

    data = build_export_table(evaluate(DATA_PATH))
    map_count = write_maps(data)
    profile_count = write_profiles(data)
    curve_count = write_curves(data)

    print(f"Created {map_count} map files in {MAPS_DIR}")
    print(f"Created {profile_count} profile files in {PROFILES_DIR}")
    print(f"Created {curve_count} curve files in {CURVES_DIR}")


if __name__ == "__main__":
    main()
