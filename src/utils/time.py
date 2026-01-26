"""Time helpers for week alignment."""
from __future__ import annotations

import pandas as pd


def to_week_start(series: pd.Series, week_start_day: int = 0) -> pd.Series:
    """Convert timestamps to week start dates (default Monday=0)."""
    dt = pd.to_datetime(series, errors="coerce")
    day_offset = (dt.dt.dayofweek - week_start_day) % 7
    return (dt - pd.to_timedelta(day_offset, unit="D")).dt.normalize()


def add_week_start_date(df: pd.DataFrame, date_col: str, week_start_day: int = 0) -> pd.DataFrame:
    df = df.copy()
    df["week_start_date"] = to_week_start(df[date_col], week_start_day)
    return df


def sort_by_sku_week(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["sku_id", "week_start_date"]).reset_index(drop=True)
