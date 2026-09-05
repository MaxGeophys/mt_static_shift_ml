from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import evaluate






OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "edi"

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "Test_model.txt"

OUTPUT_DIRECTORIES = {
    "reference": OUTPUT_ROOT / "reference model",
    "anomalous": OUTPUT_ROOT / "inhomogeneous near-surface layer",
    "filtered": OUTPUT_ROOT / "filtration result",
    "ml": OUTPUT_ROOT / "ML result",
}


IMPEDANCE_SCALE = np.sqrt(
    0.2 * 2.0 * np.pi * 4.0 * np.pi * 1e-7
)

EMPTY_VALUE = 1.0e32


COMPONENTS = ("Zxx", "Zxy", "Zyx", "Zyy")


def original_impedance(
    data: pd.DataFrame,
    component: str,
) -> np.ndarray:

    return (
        data[f"real{component}"].to_numpy(dtype=float)
        + 1j * data[f"imag{component}"].to_numpy(dtype=float)
    )


def homogeneous_impedance(
    data: pd.DataFrame,
    component: str,
) -> np.ndarray:

    return (
        data[f"real{component}_real"].to_numpy(dtype=float)
        + 1j * data[f"imag{component}_real"].to_numpy(dtype=float)
    )


def corrected_impedance(
    data: pd.DataFrame,
    component: str,
    amplitude_column: str,
) -> np.ndarray:

    original = original_impedance(data, component)

    amplitude = data[amplitude_column].to_numpy(dtype=float)

    return amplitude * np.exp(1j * np.angle(original))






def write_values(
    file,
    values: np.ndarray,
    per_line: int = 5,
    *,
    float_format: str = "13.6g",
) -> None:


    for start in range(0, len(values), per_line):
        chunk = values[start:start + per_line]
        file.write("".join(format(float(value), float_format) for value in chunk))
        file.write("\n")

    file.write("\n")


def write_section(
    file,
    name: str,
    values: np.ndarray,
) -> None:
    values = np.asarray(values)
    file.write(f">{name} // {len(values)}\n")
    write_values(file, values)


def write_frequency_section(
    file,
    frequencies: np.ndarray,
) -> None:


    frequencies = np.asarray(frequencies, dtype=float)
    file.write(f">FREQ NFREQ={len(frequencies)} // {len(frequencies)}\n")
    write_values(file, frequencies, float_format="13.6f")


def write_constant_section(
    file,
    name: str,
    value: float,
    size: int,
) -> None:

    write_section(
        file,
        name,
        np.full(size, value, dtype=float),
    )






def decimal_to_dms(
    value: float,
) -> tuple[int, int, float]:

    degrees = int(np.floor(value))

    minutes_float = (value - degrees) * 60.0
    minutes = int(np.floor(minutes_float))

    seconds = (
        (minutes_float - minutes) * 60.0
    )

    return degrees, minutes, seconds


def geographic_coordinates(
    x: float,
    y: float,
) -> tuple[float, float]:

    latitude = (
        55
        + 42 / 60.0
        + 4.25 / 3600.0
        + x / 31.0 / 3600.0
    )

    longitude = (
        37
        + 31 / 60.0
        + 39.25 / 3600.0
        + y / 17.5 / 3600.0
    )

    return latitude, longitude






def coordinate_name(value: float) -> str:


    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p")


def station_numbers(
    data: pd.DataFrame,
) -> dict[tuple[float, float], tuple[int, int]]:

    x_values = np.sort(data["X"].unique())
    y_values = np.sort(data["Y"].unique())

    expected_ids_x = 15 + 5 * np.arange(len(x_values))
    expected_ids_y = 15 + 5 * np.arange(len(y_values))

    x_ids = dict(zip(x_values, expected_ids_x))
    y_ids = dict(zip(y_values, expected_ids_y))

    return {
        (x, y): (
            int(x_ids[x]),
            int(y_ids[y]),
        )
        for x in x_values
        for y in y_values
    }






def write_edi(
    filename: Path,
    site: pd.DataFrame,
    x: float,
    y: float,
    station_x: int,
    station_y: int,
    impedance: dict[str, np.ndarray],
) -> None:

    site = site.sort_values("T")

    periods = site["T"].to_numpy(dtype=float)
    frequencies = 1.0 / periods

    nfreq = len(periods)

    latitude, longitude = geographic_coordinates(x, y)

    lat_deg, lat_min, lat_sec = decimal_to_dms(latitude)
    lon_deg, lon_min, lon_sec = decimal_to_dms(longitude)


    impedance = {
        component: values / IMPEDANCE_SCALE
        for component, values in impedance.items()
    }

    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    today = datetime.now()

    with filename.open(
        "w",
        encoding="ascii",
    ) as file:





        file.write(">HEAD\n\n")

        file.write(
            f'    DATAID="nx={station_x}_ny={station_y}"\n'
        )

        file.write('    ACQBY="MT3DFwd"\n')
        file.write('    FILEBY="MTDA"\n')

        file.write(
            f'    ACQDATE="{today.year}/{today.month}/{today.day}"\n'
        )

        file.write(
            f'    FILEDATE="{today.year}/{today.month}/{today.day}"\n'
        )

        file.write(
            f"    LAT={lat_deg}:{lat_min}:{lat_sec:.2f}\n"
        )

        file.write(
            f"    LONG={lon_deg}:{lon_min}:{lon_sec:.2f}\n"
        )

        file.write("    ELEV=0\n")
        file.write("    UNITS=M\n")
        file.write('    STDVERS="SEG 1.0"\n')
        file.write('    PROGVERS="2023/02/23"\n')
        file.write('    PROGDATE="2023/02/23"\n')
        file.write("    EMPTY=1.0E32\n\n")





        file.write(">INFO\n\n")
        file.write(
            "    Result of 3D MT modeling using MT3DFwd software\n"
        )
        file.write(
            "    This file was created by MT static-shift ML project\n"
        )

        file.write(
            f"    Site coordinates: nx={station_x} "
            f"(x={x:.0f} m) "
            f"ny={station_y} "
            f"(y={y:.0f} m)\n\n"
        )





        file.write(">=DEFINEMEAS\n\n")

        file.write(
            ">!****THE X,Y OFFSETS ARE RELATIVE TO THIS REFERENCE****!\n"
        )

        file.write(
            f"    REFLAT={lat_deg}:{lat_min}:{lat_sec:.2f}\n"
        )

        file.write(
            f"    REFLONG={lon_deg}:{lon_min}:{lon_sec:.2f}\n"
        )

        file.write("    REFELEV=0\n\n")





        file.write(">=MTSECT\n\n")

        write_frequency_section(
            file,
            frequencies,
        )






        variance_source = {
            "Zxx": "Zxy",
            "Zxy": "Zxy",
            "Zyx": "Zyx",
            "Zyy": "Zyx",
        }

        for component in COMPONENTS:
            z = impedance[component]

            write_section(
                file,
                f"{component.upper()}R",
                np.real(z),
            )

            write_section(
                file,
                f"{component.upper()}I",
                -np.imag(z),
            )

            write_section(
                file,
                f"{component.upper()}.VAR",
                np.abs(impedance[variance_source[component]]) / 1600.0,
            )





        zxy = impedance["Zxy"]
        zyx = impedance["Zyx"]

        rho_xy = 0.2 * periods * np.abs(zxy)
        rho_yx = 0.2 * periods * np.abs(zyx)

        write_section(
            file,
            "RHOXY",
            rho_xy,
        )

        write_section(
            file,
            "RHOXY.ERR",
            (rho_xy / 20.0) ** 2,
        )

        write_section(
            file,
            "RHOYX",
            rho_yx,
        )

        write_section(
            file,
            "RHOYX.ERR",
            (rho_yx / 20.0) ** 2,
        )





        write_section(
            file,
            "PHSXY",
            -np.angle(zxy, deg=True),
        )

        write_constant_section(
            file,
            "PHSXY.ERR",
            2.5,
            nfreq,
        )

        write_section(
            file,
            "PHSYX",
            -np.angle(zyx, deg=True),
        )

        write_constant_section(
            file,
            "PHSYX.ERR",
            2.5,
            nfreq,
        )












        file.write("\n>END\n")






def build_impedance_sets(
    site: pd.DataFrame,
) -> dict[str, dict[str, np.ndarray]]:

    original = {
        component: original_impedance(site, component)
        for component in COMPONENTS
    }

    reference = {
        component: homogeneous_impedance(site, component)
        for component in COMPONENTS
    }

    filtered = {
        component: corrected_impedance(
            site,
            component,
            f"{component}_filtered",
        )
        for component in COMPONENTS
    }

    ml = {
        component: corrected_impedance(
            site,
            component,
            f"{component}_ml",
        )
        for component in COMPONENTS
    }

    return {
        "reference": reference,
        "anomalous": original,
        "filtered": filtered,
        "ml": ml,
    }






def main() -> None:

    parser = argparse.ArgumentParser(
        description="Write EDI files for the MT static-shift experiment."
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=(
            "Input model TXT file. "
            "Default: data/Test_model.txt"
        ),
    )

    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Input data file not found:\n{args.data}"
        )

    print(f"Evaluating data: {args.data}")

    data = evaluate(args.data)

    required_columns = [
        "X",
        "Y",
        "T",
        "realZxx",
        "imagZxx",
        "realZxy",
        "imagZxy",
        "realZyx",
        "imagZyx",
        "realZyy",
        "imagZyy",
        "realZxx_real",
        "imagZxx_real",
        "realZxy_real",
        "imagZxy_real",
        "realZyx_real",
        "imagZyx_real",
        "realZyy_real",
        "imagZyy_real",
        "Zxx_filtered",
        "Zxy_filtered",
        "Zyx_filtered",
        "Zyy_filtered",
        "Zxx_ml",
        "Zxy_ml",
        "Zyx_ml",
        "Zyy_ml",
    ]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "The evaluation dataframe is missing columns:\n"
            + "\n".join(f"  {column}" for column in missing)
        )


    station_map = station_numbers(data)

    for directory in OUTPUT_DIRECTORIES.values():
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        for edi_file in directory.glob("*.edi"):
            edi_file.unlink()

    sites = (
        data[["X", "Y"]]
        .drop_duplicates()
        .sort_values(["X", "Y"])
    )

    print(f"Found {len(sites)} observation points.")

    for _, row in sites.iterrows():

        x = row["X"]
        y = row["Y"]

        site = data[
            (data["X"] == x)
            & (data["Y"] == y)
        ].copy()

        station_x, station_y = station_map[(x, y)]

        filename = (
            f"X_{coordinate_name(x)}m_"
            f"Y_{coordinate_name(y)}m.edi"
        )

        impedance_sets = build_impedance_sets(site)

        for name, impedance in impedance_sets.items():

            write_edi(
                filename=(
                    OUTPUT_DIRECTORIES[name]
                    / filename
                ),
                site=site,
                x=x,
                y=y,
                station_x=station_x,
                station_y=station_y,
                impedance=impedance,
            )

    print("\nEDI files successfully written:")

    for name, directory in OUTPUT_DIRECTORIES.items():
        count = len(
            list(directory.glob("*.edi"))
        )

        print(
            f"  {name:12s}: "
            f"{count:3d} files -> {directory}"
        )


if __name__ == "__main__":
    main()
