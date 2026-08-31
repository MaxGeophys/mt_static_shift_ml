from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_tabular_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", index_col=False)


def save_model_data(data: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, sep="\t", index=False, encoding="utf-8-sig")
