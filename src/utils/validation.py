"""Validation and sanity checks for RCA pipeline."""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.config import FeatureConfig, PathsConfig
from src.utils.io import write_json
from src.utils.time import sort_by_sku_week


DEFAULT_ASSUMPTIONS: List[str] = [
    "Weekly alignment uses week_start_date as the canonical key (Monday start by default).",
    "Missing promo and external signals imply no activity (filled with 0).",
    "Inventory and pricing are forward-filled within SKU when gaps exist.",
    "Forecast vs actual anomaly inputs are authoritative for anomaly detection.",
]

DEFAULT_FAILURE_MODES: List[str] = [
    "Inventory data can be stale or unreliable (e.g., zero on-hand with positive sales).",
    "Pricing data gaps may hide true promo effects or price shocks.",
    "External signals may be incomplete, leading to demand attribution misses.",
    "Cold-start SKUs lack history, reducing confidence for temporal features.",
    "Label noise can bias model evaluation or SHAP validation.",
]


def validate_unique_key(df: pd.DataFrame, key_cols: List[str]) -> Dict[str, Any]:
    duplicate_mask = df.duplicated(subset=key_cols, keep=False)
    duplicate_count = int(duplicate_mask.sum())
    return {
        "duplicate_count": duplicate_count,
        "total_rows": int(len(df)),
        "duplicate_pct": float(duplicate_count / max(len(df), 1)),
    }


def compute_join_coverage(df: pd.DataFrame, coverage_map: Dict[str, List[str]]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for name, columns in coverage_map.items():
        available = [col for col in columns if col in df.columns]
        if not available:
            results[name] = {"coverage": 0.0, "missing_columns": columns}
            continue
        coverage = float(df[available].notna().any(axis=1).mean())
        results[name] = {"coverage": coverage, "missing_columns": []}
    return results


def _get_sales_series(df: pd.DataFrame) -> pd.Series:
    if "actual_sales" in df.columns:
        return df["actual_sales"].fillna(0)
    if "sales_units" in df.columns:
        return df["sales_units"].fillna(0)
    return pd.Series([0] * len(df))


def sanity_checks(df: pd.DataFrame, config: FeatureConfig) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    df = sort_by_sku_week(df)
    sales = _get_sales_series(df)

    if "on_hand" in df.columns:
        inventory_zero_sales_positive = (df["on_hand"].fillna(0) <= 0) & (sales > 0)
        sku_counts = inventory_zero_sales_positive.groupby(df["sku_id"]).sum()
        flagged_skus = sku_counts[sku_counts >= 2].index.tolist()
        report["inventory_unreliability"] = {
            "flagged_sku_count": int(len(flagged_skus)),
            "example_skus": flagged_skus[:10],
        }
    else:
        report["inventory_unreliability"] = {"flagged_sku_count": 0, "example_skus": []}

    if "promo_flag" in df.columns and "price" in df.columns:
        df["price_lag1"] = df.groupby("sku_id")["price"].shift(1)
        price_change_pct = (df["price"] - df["price_lag1"]) / df["price_lag1"].replace(0, pd.NA)
        no_price_change = price_change_pct.abs().fillna(0) < config.promo_price_change_threshold
        promo_no_price_change = (df["promo_flag"].fillna(0) == 1) & no_price_change
        report["promo_price_mismatch"] = {
            "count": int(promo_no_price_change.sum()),
            "pct": float(promo_no_price_change.mean()),
        }
    else:
        report["promo_price_mismatch"] = {"count": 0, "pct": 0.0}

    if "on_hand" in df.columns:
        df["on_hand_lag1"] = df.groupby("sku_id")["on_hand"].shift(1)
        with pd.option_context("mode.use_inf_as_na", True):
            ratio = df["on_hand"] / df["on_hand_lag1"]
        sudden_spike = (df["on_hand_lag1"].fillna(0) > 0) & (ratio > config.inventory_spike_multiplier)
        report["inventory_spikes"] = {
            "count": int(sudden_spike.sum()),
            "pct": float(sudden_spike.mean()),
        }
    else:
        report["inventory_spikes"] = {"count": 0, "pct": 0.0}

    return report


def run_validation_checks(df: pd.DataFrame, config: FeatureConfig) -> Dict[str, Any]:
    key_report = validate_unique_key(df, ["sku_id", "week_start_date"])
    coverage_map = {
        "sales": ["actual_sales", "sales_units"],
        "pricing": ["price"],
        "promotions": ["promo_flag"],
        "inventory": ["on_hand", "fill_rate", "lead_time"],
        "external": ["external_event_flag", "external_event_intensity"],
    }
    coverage = compute_join_coverage(df, coverage_map)
    sanity = sanity_checks(df, config)
    return {"key": key_report, "coverage": coverage, "sanity": sanity}


def log_assumptions_and_failure_modes(paths: PathsConfig) -> None:
    write_json({"assumptions": DEFAULT_ASSUMPTIONS}, paths.assumptions_path)
    write_json({"failure_modes": DEFAULT_FAILURE_MODES}, paths.failure_modes_path)
