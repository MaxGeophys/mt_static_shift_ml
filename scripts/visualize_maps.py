from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import numpy as np
from sklearn.metrics import root_mean_squared_error

from scripts.evaluate import evaluate

DATA_PATH = PROJECT_ROOT / "data" / "Test_model.txt"
MAPS_DIR = PROJECT_ROOT / "visualization" / "Maps"

TARGETS = {
    "Rho": {
        "columns": ("Rho_real", "Rho", "Rho_filtered", "Rho_ml"),
        "scale": "rho",
        "label": r"$\rho_{eff}$, Ohm $\cdot$ m",
    },
    "Zxx": {
        "columns": ("absZxx_real", "absZxx", "Zxx_filtered", "Zxx_ml"),
        "scale": "diag",
        "label": r"$|Z_{xx}|$ / $|Z_{yy}|$",
    },
    "Zxy": {
        "columns": ("absZxy_real", "absZxy", "Zxy_filtered", "Zxy_ml"),
        "scale": "offdiag",
        "label": r"$|Z_{xy}|$ / $|Z_{yx}|$",
    },
    "Zyx": {
        "columns": ("absZyx_real", "absZyx", "Zyx_filtered", "Zyx_ml"),
        "scale": "offdiag",
        "label": r"$|Z_{xy}|$ / $|Z_{yx}|$",
    },
    "Zyy": {
        "columns": ("absZyy_real", "absZyy", "Zyy_filtered", "Zyy_ml"),
        "scale": "diag",
        "label": r"$|Z_{xx}|$ / $|Z_{yy}|$",
    },
}

PANEL_LABELS = ("a", "b", "c", "d")
PANEL_TITLES = (
    "Homogeneous near-surface layer",
    "Multiple near-surface inhomogeneities",
    "Spatial filtering",
    "Gradient boosting",
)


def scale_values(data, targets):
    values = []
    for target in targets:
        for column in TARGETS[target]["columns"]:
            values.append(data[column].to_numpy(dtype=float))
    return np.concatenate(values)


def build_norms(data):
    rho = scale_values(data, ["Rho"])
    offdiag = scale_values(data, ["Zxy", "Zyx"])
    diag = scale_values(data, ["Zxx", "Zyy"])

    rho_positive = rho[np.isfinite(rho) & (rho > 0)]
    offdiag_positive = offdiag[np.isfinite(offdiag) & (offdiag > 0)]
    diag_finite = diag[np.isfinite(diag)]
    diag_nonzero = np.abs(diag_finite[diag_finite != 0])

    return {
        "rho": LogNorm(vmin=rho_positive.min(), vmax=rho_positive.max()),
        "offdiag": LogNorm(vmin=offdiag_positive.min(), vmax=offdiag_positive.max()),
        "diag": SymLogNorm(
            linthresh=np.quantile(diag_nonzero, 0.05),
            vmin=diag_finite.min(),
            vmax=diag_finite.max(),
            base=10,
        ),
    }


def period_name(period):
    return f"{period:g}".replace(".", "p")


def map_grid(period_data, column):
    pivot = period_data.pivot(index="X", columns="Y", values=column).sort_index().sort_index(axis=1)
    return pivot.columns.to_numpy(), pivot.index.to_numpy(), pivot.to_numpy()


def norm_levels(norm, count):
    return norm.inverse(np.linspace(0.0, 1.0, count))


def create_map_figure(period_data, target, period, norm):
    config = TARGETS[target]
    reference = period_data[config["columns"][0]]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.5), constrained_layout=True)
    fill_levels = norm_levels(norm, 25)
    line_levels = norm_levels(norm, 13)
    contour = None

    for index, (ax, column, panel, title) in enumerate(
        zip(axes.flat, config["columns"], PANEL_LABELS, PANEL_TITLES)
    ):
        y, x, values = map_grid(period_data, column)
        contour = ax.contourf(y, x, values, levels=fill_levels, norm=norm, cmap="turbo")
        ax.contour(y, x, values, levels=line_levels, norm=norm, colors="black", linewidths=0.35, alpha=0.45)
        ax.set_aspect("equal")
        ax.set_xlabel("Y, m")
        ax.set_ylabel("X, m")
        ax.tick_params(direction="out")
        ax.text(0.015, 0.985, panel, transform=ax.transAxes, ha="left", va="top", fontsize=16, fontweight="bold")
        if index == 0:
            ax.set_title(title, fontsize=11)
        else:
            rmse = root_mean_squared_error(reference, period_data[column])
            ax.set_title(f"{title}\nRMSE = {rmse:.4g}", fontsize=11)

    colorbar = fig.colorbar(contour, ax=axes, shrink=0.86, pad=0.035)
    ticks = norm.inverse(np.linspace(0.05, 0.95, 7))
    colorbar.set_ticks(ticks)
    if config["scale"] == "rho":
        colorbar.set_ticklabels([f"{value:.3g}" for value in ticks])
    else:
        colorbar.set_ticklabels([f"{value:.2e}" for value in ticks])
    colorbar.set_label(config["label"], fontsize=12)
    fig.suptitle(f"{target}, T = {period:g} s", fontsize=15)
    return fig


def save_map(task):
    period_data, target, period, norm = task
    fig = create_map_figure(period_data, target, period, norm)
    output_path = MAPS_DIR / f"{target}_T_{period_name(period)}s.png"
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def main():
    if MAPS_DIR.exists():
        shutil.rmtree(MAPS_DIR)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)

    data = evaluate(DATA_PATH)
    norms = build_norms(data)
    periods = sorted(data["T"].unique())
    tasks = []

    for period in periods:
        period_data = data[data["T"] == period]
        for target, config in TARGETS.items():
            tasks.append((period_data, target, period, norms[config["scale"]]))

    workers = min(4, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        list(executor.map(save_map, tasks))

    print(f"Created {len(tasks)} maps in {MAPS_DIR}")


if __name__ == "__main__":
    main()
