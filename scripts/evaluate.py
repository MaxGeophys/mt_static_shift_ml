import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd

from config import (
    FILTER_HALF_WIDTH_STEPS,
    FILTER_PERIOD,
    FILTER_RADIUS_STEPS,
    FILTER_STEEPNESS,
    ML_FEATURE_RADIUS_STEPS,
    RHO_K_PERIODS,
)
from mt_shift.evaluation import compare_methods
from mt_shift.features import prepare_features
from mt_shift.filtering import spatial_filter
from mt_shift.io import load_tabular_data, save_model_data
from mt_shift.model import RHO_FEATURES, Zxx_FEATURES, Zxy_FEATURES, Zyx_FEATURES, Zyy_FEATURES

MODEL_PATHS = {
    "Rho": PROJECT_ROOT / "models" / "XGBR_Rho.joblib",
    "Zxx": PROJECT_ROOT / "models" / "XGBR_Zxx.joblib",
    "Zxy": PROJECT_ROOT / "models" / "XGBR_Zxy.joblib",
    "Zyx": PROJECT_ROOT / "models" / "XGBR_Zyx.joblib",
    "Zyy": PROJECT_ROOT / "models" / "XGBR_Zyy.joblib",
}
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def evaluate(
    data_path: Path,
    filter_period: float | None = None,
    filter_radius_steps: int | None = None,
    filter_half_width_steps: float | None = None,
    filter_steepness: float | None = None,
) -> pd.DataFrame:
    if filter_period is None:
        filter_period = FILTER_PERIOD
    if filter_radius_steps is None:
        filter_radius_steps = FILTER_RADIUS_STEPS
    if filter_half_width_steps is None:
        filter_half_width_steps = FILTER_HALF_WIDTH_STEPS
    if filter_steepness is None:
        filter_steepness = FILTER_STEEPNESS

    missing_models = [path for path in MODEL_PATHS.values() if not path.exists()]
    if missing_models:
        raise FileNotFoundError("Model not found. Run scripts/train.py first.")

    data = prepare_features(
        load_tabular_data(data_path),
        radius_steps=ML_FEATURE_RADIUS_STEPS,
    )
    data = spatial_filter(
        data,
        filter_period=filter_period,
        radius_steps=filter_radius_steps,
        half_width_steps=filter_half_width_steps,
        steepness=filter_steepness,
    )
    models = {name: joblib.load(path) for name, path in MODEL_PATHS.items()}

    data["K_Rho_predicted"] = models["Rho"].predict(data[RHO_FEATURES])
    data["Kxy_predicted"] = models["Zxy"].predict(data[Zxy_FEATURES])
    data["Kyx_predicted"] = models["Zyx"].predict(data[Zyx_FEATURES])
    data["Kxx_predicted"] = models["Zxx"].predict(data[Zxx_FEATURES])
    data["Kyy_predicted"] = models["Zyy"].predict(data[Zyy_FEATURES])

    K_pivot = (
        data[data["T"].isin(RHO_K_PERIODS)]
        .groupby(["Y", "X"], as_index=False)["K_Rho_predicted"]
        .mean()
        .pivot(index="Y", columns="X", values="K_Rho_predicted")
    )

    pivot_Rho = data.pivot_table(index=["T", "Y"], columns="X", values="Rho")
    pivot = pivot_Rho.div(K_pivot, level="Y")

    pivot_df = pivot.stack().reset_index()
    pivot_df.columns = ["T", "Y", "X", "Rho_ml"]

    data = data.merge(
        pivot_df,
        on=["T", "Y", "X"],
        how="left",
    )

    data["Zxx_ml"] = data["absZxx"] / data["Kxx_predicted"]
    data["Zxy_ml"] = data["absZxy"] / data["Kxy_predicted"]
    data["Zyx_ml"] = data["absZyx"] / data["Kyx_predicted"]
    data["Zyy_ml"] = data["absZyy"] / data["Kyy_predicted"]

    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "Test_model.txt",
    )
    parser.add_argument(
        "--filter-period",
        type=float,
        default=None,
        help=f"Spatial-filter reference period in seconds (default from config.py: {FILTER_PERIOD:g}).",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = evaluate(args.data, filter_period=args.filter_period)
    comparison = compare_methods(data)
    comparison.to_csv(OUTPUT_DIR / "comparison_metrics.csv", index=False)
    save_model_data(data, OUTPUT_DIR / "test_results.txt")

    print("\nComparison with the homogeneous near-surface reference model")
    for target in comparison["target"].drop_duplicates():
        target_comparison = comparison[comparison["target"] == target]
        print(f"\n--- {target} ---")
        print(target_comparison.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
