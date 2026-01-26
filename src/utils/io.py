"""I/O utilities for reading and writing pipeline artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import json
import pickle

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def read_table(path: str | Path) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Input not found: {path_obj}")

    if path_obj.suffix == ".csv":
        return pd.read_csv(path_obj)
    if path_obj.suffix == ".parquet":
        return pd.read_parquet(path_obj)
    raise ValueError(f"Unsupported table format: {path_obj.suffix}")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)

    if path_obj.suffix == ".csv":
        df.to_csv(path_obj, index=False)
        return
    if path_obj.suffix == ".parquet":
        df.to_parquet(path_obj, index=False)
        return
    raise ValueError(f"Unsupported table format: {path_obj.suffix}")


def read_json(path: str | Path) -> Dict[str, Any]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"JSON not found: {path_obj}")
    return json.loads(path_obj.read_text())


def write_json(data: Dict[str, Any], path: str | Path) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    path_obj.write_text(json.dumps(data, indent=2, sort_keys=True))


def write_pickle(obj: Any, path: str | Path) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("wb") as handle:
        pickle.dump(obj, handle)


def read_pickle(path: str | Path) -> Any:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Pickle not found: {path_obj}")
    with path_obj.open("rb") as handle:
        return pickle.load(handle)
