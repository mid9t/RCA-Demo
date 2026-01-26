"""Build SKU-week aligned base table with validation checks."""
from __future__ import annotations

import argparse
from typing import Dict, Optional, Tuple

import pandas as pd

from src.config import PipelineConfig, load_config
from src.utils.io import read_table, write_table, write_json
from src.utils.time import add_week_start_date, sort_by_sku_week
from src.utils.validation import log_assumptions_and_failure_modes, run_validation_checks


def _standardize_sku(df: pd.DataFrame, sku_col: str = "sku_id") -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = df.copy()
    original = df[sku_col].astype(str)
    standardized = (
        original.str.strip()
        .str.upper()
        .str.replace(r"[^A-Z0-9_-]", "", regex=True)
    )
    df[sku_col] = standardized
    changed = (original != standardized).mean()
    return df, {"pct_changed": float(changed)}


def _apply_sku_mapping(df: pd.DataFrame, mapping: Optional[pd.DataFrame]) -> Dict[str, float]:
    if mapping is None:
        return {"pct_unmapped": 0.0}

    required_cols = {"sku_alias", "sku_id"}
    if not required_cols.issubset(mapping.columns):
        raise ValueError("SKU mapping must contain sku_alias and sku_id columns")

    alias_map = dict(zip(mapping["sku_alias"].astype(str), mapping["sku_id"].astype(str)))
    mapped = df["sku_id"].map(alias_map)
    df["sku_id"] = mapped.fillna(df["sku_id"])
    unmapped = float(mapped.isna().mean())
    pct_mapped = float(mapped.notna().mean())
    return {"pct_unmapped": unmapped, "pct_mapped": pct_mapped}


def _prepare_anomalies(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    df = df.copy()
    if "week_start_date" not in df.columns:
        if "date" not in df.columns:
            raise ValueError("Anomalies table must include week_start_date or date")
        df = add_week_start_date(df, "date", config.features.week_start_day)
    if "actual_sales" not in df.columns:
        if "units" in df.columns:
            df = df.rename(columns={"units": "actual_sales"})
        elif "sales_units" in df.columns:
            df = df.rename(columns={"sales_units": "actual_sales"})
        else:
            raise ValueError("Anomalies table must include actual_sales or units")
    return df


def _prepare_sales(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    df = df.copy()
    if "date" not in df.columns:
        raise ValueError("Sales table must include date")
    if "units" in df.columns:
        df = df.rename(columns={"units": "sales_units"})
    df = add_week_start_date(df, "date", config.features.week_start_day)
    agg = df.groupby(["sku_id", "week_start_date"], as_index=False).agg(
        sales_units=("sales_units", "sum"),
        sales_revenue=("revenue", "sum") if "revenue" in df.columns else ("sales_units", "sum"),
    )
    return agg


def _prepare_pricing(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    df = df.copy()
    if "date" not in df.columns:
        raise ValueError("Pricing table must include date")
    if "price" not in df.columns:
        raise ValueError("Pricing table must include price")
    df = add_week_start_date(df, "date", config.features.week_start_day)
    agg_func = "mean" if config.features.price_agg == "mean" else "last"
    agg = df.groupby(["sku_id", "week_start_date"], as_index=False).agg(
        price=("price", agg_func),
    )
    return agg


def _prepare_promotions(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    df = df.copy()
    if "date" not in df.columns:
        raise ValueError("Promotions table must include date")
    df = add_week_start_date(df, "date", config.features.week_start_day)
    if "promo_flag" not in df.columns:
        df["promo_flag"] = 1
    agg = df.groupby(["sku_id", "week_start_date"], as_index=False).agg(
        promo_flag=("promo_flag", "max"),
        promo_days=("promo_flag", "sum"),
    )
    return agg


def _prepare_inventory(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    df = df.copy()
    if "date" not in df.columns:
        raise ValueError("Inventory table must include date")
    df = add_week_start_date(df, "date", config.features.week_start_day)
    agg_func = "mean" if config.features.inventory_agg == "mean" else "last"
    agg_map = {}
    if "on_hand" in df.columns:
        agg_map["on_hand"] = ("on_hand", agg_func)
    if "on_order" in df.columns:
        agg_map["on_order"] = ("on_order", agg_func)
    if "lead_time" in df.columns:
        agg_map["lead_time"] = ("lead_time", agg_func)
    if "fill_rate" in df.columns:
        agg_map["fill_rate"] = ("fill_rate", agg_func)
    if not agg_map:
        raise ValueError("Inventory table must include at least on_hand")
    agg = df.groupby(["sku_id", "week_start_date"], as_index=False).agg(**agg_map)
    for col in ["on_hand", "on_order", "lead_time", "fill_rate"]:
        if col not in agg.columns:
            agg[col] = pd.NA
    return agg


def _prepare_external(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    df = df.copy()
    if "date" not in df.columns:
        raise ValueError("External signals table must include date")
    df = add_week_start_date(df, "date", config.features.week_start_day)
    if "event_flag" in df.columns and "external_event_flag" not in df.columns:
        df = df.rename(columns={"event_flag": "external_event_flag"})
    if "event_intensity" in df.columns and "external_event_intensity" not in df.columns:
        df = df.rename(columns={"event_intensity": "external_event_intensity"})
    if "external_event_flag" not in df.columns:
        raise ValueError("External signals table must include external_event_flag or event_flag")
    if "external_event_intensity" not in df.columns:
        df["external_event_intensity"] = df["external_event_flag"].astype(float)
    agg = df.groupby(["sku_id", "week_start_date"], as_index=False).agg(
        external_event_flag=("external_event_flag", "max"),
        external_event_intensity=("external_event_intensity", "mean"),
    )
    return agg


def _forward_fill_by_sku(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    df[columns] = df.groupby("sku_id")[columns].ffill()
    return df


def build_base_table(config: PipelineConfig) -> Tuple[pd.DataFrame, Dict[str, float]]:
    anomalies = read_table(config.paths.anomalies_path)
    sales = read_table(config.paths.sales_path)
    pricing = read_table(config.paths.pricing_path)
    promotions = read_table(config.paths.promotions_path)
    inventory = read_table(config.paths.inventory_path)

    external = None
    if config.paths.external_signals_path:
        try:
            external = read_table(config.paths.external_signals_path)
        except FileNotFoundError:
            external = None

    mapping = None
    if config.paths.sku_alias_path:
        mapping = read_table(config.paths.sku_alias_path)

    anomalies, sku_stats = _standardize_sku(anomalies)
    sales, _ = _standardize_sku(sales)
    pricing, _ = _standardize_sku(pricing)
    promotions, _ = _standardize_sku(promotions)
    inventory, _ = _standardize_sku(inventory)
    if external is not None:
        external, _ = _standardize_sku(external)

    sku_map_stats = _apply_sku_mapping(anomalies, mapping)
    for df in [sales, pricing, promotions, inventory]:
        _apply_sku_mapping(df, mapping)
    if external is not None:
        _apply_sku_mapping(external, mapping)

    anomalies = _prepare_anomalies(anomalies, config)
    sales_agg = _prepare_sales(sales, config)
    pricing_agg = _prepare_pricing(pricing, config)
    promotions_agg = _prepare_promotions(promotions, config)
    inventory_agg = _prepare_inventory(inventory, config)
    external_agg = _prepare_external(external, config) if external is not None else None

    base = anomalies.copy()
    for table in [sales_agg, pricing_agg, promotions_agg, inventory_agg]:
        base = base.merge(table, on=["sku_id", "week_start_date"], how="left")
    if external_agg is not None:
        base = base.merge(external_agg, on=["sku_id", "week_start_date"], how="left")

    for col in ["promo_flag", "promo_days", "external_event_flag", "external_event_intensity"]:
        if col in base.columns:
            base[col] = base[col].fillna(0)

    inventory_cols = [col for col in ["on_hand", "on_order", "lead_time", "fill_rate"] if col in base.columns]
    for col in inventory_cols:
        base[f"{col}_missing_flag"] = base[col].isna().astype(int)
    if inventory_cols:
        base = _forward_fill_by_sku(base, inventory_cols)
        if "on_hand" in base.columns:
            base["on_hand"] = base["on_hand"].clip(lower=0)
        if "on_order" in base.columns:
            base["on_order"] = base["on_order"].clip(lower=0)
        if "lead_time" in base.columns:
            base["lead_time"] = base["lead_time"].clip(lower=0)
        if "fill_rate" in base.columns:
            base["fill_rate"] = base["fill_rate"].clip(lower=0, upper=1)

    if "price" in base.columns:
        base["price_missing_flag"] = base["price"].isna().astype(int)
        base = _forward_fill_by_sku(base, ["price"])

    base = sort_by_sku_week(base)

    stats = {**sku_stats, **sku_map_stats}
    return base, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SKU-week base table.")
    parser.add_argument("--config", default=None, help="Path to YAML/JSON config")
    args = parser.parse_args()

    config = load_config(args.config)
    base, sku_stats = build_base_table(config)

    validation_report = run_validation_checks(base, config.features)
    validation_report["sku_standardization"] = sku_stats

    write_table(base, config.paths.base_table_path)
    write_json(validation_report, config.paths.validation_report_path)
    log_assumptions_and_failure_modes(config.paths)

    print(f"Wrote base table to {config.paths.base_table_path}")


if __name__ == "__main__":
    main()
