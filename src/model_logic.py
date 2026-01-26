"""Feature engineering, baseline attribution, and model training logic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from src.config import FeatureConfig, ModelConfig
from src.utils.time import sort_by_sku_week


@dataclass
class FeatureMetadata:
    name: str
    feature_type: str
    description: str
    expected_directionality: Optional[str] = None


def _add_metadata(
    metadata: List[FeatureMetadata],
    name: str,
    feature_type: str,
    description: str,
    expected_directionality: Optional[str] = None,
) -> None:
    if any(entry.name == name for entry in metadata):
        return
    metadata.append(
        FeatureMetadata(
            name=name,
            feature_type=feature_type,
            description=description,
            expected_directionality=expected_directionality,
        )
    )


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    def slope(values: np.ndarray) -> float:
        if len(values) < 2 or np.all(np.isnan(values)):
            return np.nan
        x = np.arange(len(values))
        y = values.astype(float)
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return np.nan
        x = x[mask]
        y = y[mask]
        coef = np.polyfit(x, y, 1)
        return float(coef[0])

    return series.rolling(window=window, min_periods=2).apply(slope, raw=True)


def _add_lags(df: pd.DataFrame, col: str, lags: List[int]) -> pd.DataFrame:
    for lag in lags:
        df[f"{col}_lag{lag}"] = df.groupby("sku_id")[col].shift(lag)
    return df


def _add_rolling(df: pd.DataFrame, col: str, windows: List[int]) -> pd.DataFrame:
    for window in windows:
        df[f"{col}_roll{window}_mean"] = df.groupby("sku_id")[col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f"{col}_roll{window}_std"] = df.groupby("sku_id")[col].transform(
            lambda x: x.rolling(window=window, min_periods=2).std()
        )
    return df


def build_features(base_df: pd.DataFrame, config: FeatureConfig) -> Tuple[pd.DataFrame, List[FeatureMetadata]]:
    df = sort_by_sku_week(base_df).copy()
    metadata: List[FeatureMetadata] = []

    if "forecast" in df.columns:
        df["sales_delta"] = df["actual_sales"] - df["forecast"]
        df["sales_delta_pct"] = df["sales_delta"] / df["forecast"].replace(0, np.nan)
        _add_metadata(metadata, "sales_delta", "temporal", "Actual minus forecast units")
        _add_metadata(metadata, "sales_delta_pct", "temporal", "Forecast deviation percent")

    df = _add_lags(df, "actual_sales", config.lags)
    df = _add_rolling(df, "actual_sales", config.rolling_windows)
    if "actual_sales_lag1" not in df.columns:
        df["actual_sales_lag1"] = df.groupby("sku_id")["actual_sales"].shift(1)
        _add_metadata(metadata, "actual_sales_lag1", "temporal", "Sales lag 1")
    for window in config.rolling_windows:
        mean_col = f"actual_sales_roll{window}_mean"
        std_col = f"actual_sales_roll{window}_std"
        if mean_col in df.columns:
            _add_metadata(metadata, mean_col, "temporal", f"Sales {window}-week rolling mean")
        if std_col in df.columns:
            _add_metadata(metadata, std_col, "temporal", f"Sales {window}-week rolling std dev")
    if "actual_sales_roll4_mean" in df.columns:
        df["trend_adjusted_sales_delta"] = df["actual_sales"] - df["actual_sales_roll4_mean"]
        _add_metadata(metadata, "trend_adjusted_sales_delta", "demand", "Sales vs 4-week baseline")
    df["sales_wow_pct"] = (
        df["actual_sales"] - df["actual_sales_lag1"]
    ) / df["actual_sales_lag1"].replace(0, np.nan)
    _add_metadata(metadata, "sales_wow_pct", "temporal", "Week-over-week sales percent change")

    df["sales_slope_rate"] = df.groupby("sku_id")["actual_sales"].transform(
        lambda x: _rolling_slope(x, config.slope_window)
    )
    _add_metadata(metadata, "sales_slope_rate", "temporal", "Slope of sales over recent weeks")

    if "price" in df.columns:
        df = _add_lags(df, "price", config.lags)
        df = _add_rolling(df, "price", config.rolling_windows)
        if "price_lag1" not in df.columns:
            df["price_lag1"] = df.groupby("sku_id")["price"].shift(1)
            _add_metadata(metadata, "price_lag1", "demand", "Price lag 1")
        df["price_change_pct"] = (df["price"] - df["price_lag1"]) / df["price_lag1"].replace(0, np.nan)
        if "price_roll4_mean" not in df.columns:
            df["price_roll4_mean"] = df.groupby("sku_id")["price"].transform(
                lambda x: x.rolling(window=4, min_periods=1).mean()
            )
            _add_metadata(metadata, "price_roll4_mean", "demand", "Price 4-week rolling mean")
        df["price_index_roll4"] = df["price"] / df["price_roll4_mean"].replace(0, np.nan)
        _add_metadata(metadata, "price_change_pct", "demand", "Week-over-week price change")
        _add_metadata(metadata, "price_index_roll4", "demand", "Price vs 4-week average")
        for window in config.rolling_windows:
            mean_col = f"price_roll{window}_mean"
            std_col = f"price_roll{window}_std"
            if mean_col in df.columns:
                _add_metadata(metadata, mean_col, "demand", f"Price {window}-week rolling mean")
            if std_col in df.columns:
                _add_metadata(metadata, std_col, "demand", f"Price {window}-week rolling std dev")

    if "promo_flag" in df.columns:
        df["promo_normalized_sales"] = df["actual_sales"] / (1 + df["promo_flag"].fillna(0))
        _add_metadata(metadata, "promo_flag", "demand", "Promo active flag")
        _add_metadata(metadata, "promo_normalized_sales", "demand", "Sales normalized by promo activity")

    if "category_sales" in df.columns:
        df["sku_vs_category_sales"] = df["actual_sales"] / df["category_sales"].replace(0, np.nan)
        _add_metadata(metadata, "sku_vs_category_sales", "demand", "SKU sales vs category sales")

    if "brand_sales" in df.columns:
        df["sku_vs_brand_sales"] = df["actual_sales"] / df["brand_sales"].replace(0, np.nan)
        _add_metadata(metadata, "sku_vs_brand_sales", "demand", "SKU sales vs brand sales")

    if "on_hand" in df.columns:
        df["stockout_flag"] = (df["on_hand"].fillna(0) <= config.stockout_threshold).astype(int)
        _add_metadata(metadata, "stockout_flag", "supply", "On-hand below stockout threshold")

        roll_mean_col = "actual_sales_roll4_mean"
        if roll_mean_col not in df.columns:
            df[roll_mean_col] = df.groupby("sku_id")["actual_sales"].transform(
                lambda x: x.rolling(window=4, min_periods=1).mean()
            )
            _add_metadata(metadata, roll_mean_col, "temporal", "Sales 4-week rolling mean")
        df["inventory_coverage"] = df["on_hand"] / df[roll_mean_col].replace(0, np.nan)
        _add_metadata(metadata, "inventory_coverage", "supply", "On-hand coverage vs 4-week sales")

        df["inventory_zero_sales_flag"] = ((df["on_hand"] <= 0) & (df["actual_sales"] > 0)).astype(int)
        df["inventory_zero_sales_weeks"] = df.groupby("sku_id")["inventory_zero_sales_flag"].transform(
            lambda x: x.rolling(window=4, min_periods=1).sum()
        )
        _add_metadata(metadata, "inventory_zero_sales_flag", "supply", "On-hand is zero while sales positive")
        _add_metadata(metadata, "inventory_zero_sales_weeks", "supply", "Count of recent inventory reliability flags")

        df["coverage_below_threshold"] = (df["inventory_coverage"] < config.inventory_coverage_threshold).astype(int)
        _add_metadata(metadata, "coverage_below_threshold", "supply", "Coverage below threshold")

        df["coverage_weeks_below"] = df.groupby("sku_id")["coverage_below_threshold"].transform(
            lambda x: x.rolling(window=4, min_periods=1).sum()
        )
        _add_metadata(metadata, "coverage_weeks_below", "supply", "Count of recent weeks below coverage")

        df["inventory_slope_rate"] = df.groupby("sku_id")["on_hand"].transform(
            lambda x: _rolling_slope(x, config.slope_window)
        )
        _add_metadata(metadata, "inventory_slope_rate", "supply", "Slope of on-hand inventory")

    if "fill_rate" in df.columns:
        df["fill_rate_roll4"] = df.groupby("sku_id")["fill_rate"].transform(
            lambda x: x.rolling(window=4, min_periods=1).mean()
        )
        df["fill_rate_change"] = df["fill_rate"] - df["fill_rate_roll4"]
        _add_metadata(metadata, "fill_rate_roll4", "supply", "Fill-rate 4-week rolling mean")
        _add_metadata(metadata, "fill_rate_change", "supply", "Fill-rate deviation from 4-week mean")

    if "lead_time" in df.columns:
        df["lead_time_roll4"] = df.groupby("sku_id")["lead_time"].transform(
            lambda x: x.rolling(window=4, min_periods=1).mean()
        )
        df["lead_time_change"] = df["lead_time"] - df["lead_time_roll4"]
        _add_metadata(metadata, "lead_time_roll4", "supply", "Lead-time 4-week rolling mean")
        _add_metadata(metadata, "lead_time_change", "supply", "Lead-time deviation from 4-week mean")

    if "external_event_flag" in df.columns:
        df = _add_lags(df, "external_event_flag", config.lags)
        _add_metadata(metadata, "external_event_flag", "demand", "External event flag")

    if "external_event_intensity" in df.columns:
        df = _add_lags(df, "external_event_intensity", config.lags)
        _add_metadata(metadata, "external_event_intensity", "demand", "External event intensity")

    if "inventory_coverage" in df.columns:
        df = _add_lags(df, "inventory_coverage", config.lags)
    if "promo_flag" in df.columns:
        df = _add_lags(df, "promo_flag", config.lags)

    for lag in config.lags:
        if f"actual_sales_lag{lag}" in df.columns:
            _add_metadata(metadata, f"actual_sales_lag{lag}", "temporal", f"Sales lag {lag}")
        if "price" in df.columns and f"price_lag{lag}" in df.columns:
            _add_metadata(metadata, f"price_lag{lag}", "demand", f"Price lag {lag}")
        if "inventory_coverage" in df.columns and f"inventory_coverage_lag{lag}" in df.columns:
            _add_metadata(metadata, f"inventory_coverage_lag{lag}", "supply", f"Coverage lag {lag}")
        if "promo_flag" in df.columns and f"promo_flag_lag{lag}" in df.columns:
            _add_metadata(metadata, f"promo_flag_lag{lag}", "demand", f"Promo flag lag {lag}")
        if "external_event_flag" in df.columns and f"external_event_flag_lag{lag}" in df.columns:
            _add_metadata(metadata, f"external_event_flag_lag{lag}", "demand", f"External event flag lag {lag}")
        if "external_event_intensity" in df.columns and f"external_event_intensity_lag{lag}" in df.columns:
            _add_metadata(metadata, f"external_event_intensity_lag{lag}", "demand", f"External event intensity lag {lag}")

    df["actual_sales_lag52"] = df.groupby("sku_id")["actual_sales"].shift(config.yoy_lag)
    df["sales_yoy_pct"] = (
        df["actual_sales"] - df["actual_sales_lag52"]
    ) / df["actual_sales_lag52"].replace(0, np.nan)
    _add_metadata(metadata, "sales_yoy_pct", "temporal", "Year-over-year sales change")

    return df, metadata


def _evidence_list(row: pd.Series) -> List[str]:
    evidence: List[str] = []
    if row.get("stockout_flag", 0) == 1:
        evidence.append("Stockout flag triggered")
    if row.get("inventory_coverage", np.inf) < row.get("coverage_threshold", np.inf):
        evidence.append("Low inventory coverage")
    if row.get("inventory_zero_sales_weeks", 0) >= 2:
        evidence.append("Inventory unreliability flagged")
    if row.get("fill_rate_change", 0) < 0:
        evidence.append("Fill-rate drop")
    if row.get("lead_time_change", 0) > 0:
        evidence.append("Lead-time increase")
    if row.get("price_change_pct", 0) > 0.05:
        evidence.append("Price increase")
    if row.get("promo_flag", 0) == 1 and row.get("sales_wow_pct", 0) < 0:
        evidence.append("Promo active but sales down")
    if row.get("external_event_flag", 0) == 1:
        evidence.append("External event flagged")
    return evidence[:5]


def run_baseline_attribution(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    df = df.copy()
    df["coverage_threshold"] = config.inventory_coverage_threshold
    sales_drop = df.get("sales_wow_pct", pd.Series(index=df.index, data=0)) <= config.sales_drop_pct_threshold
    supply_trigger = (
        (df.get("stockout_flag", 0) == 1) | (df.get("inventory_coverage", np.inf) < config.inventory_coverage_threshold)
    ) & sales_drop
    demand_trigger = (
        (df.get("inventory_coverage", np.inf) >= config.inventory_coverage_threshold)
        & (df.get("external_event_flag", 0) == 1)
        & sales_drop
    )

    df["baseline_label"] = "UNCERTAIN"
    df.loc[supply_trigger & demand_trigger, "baseline_label"] = "MIXED"
    df.loc[supply_trigger & ~demand_trigger, "baseline_label"] = "SUPPLY"
    df.loc[demand_trigger & ~supply_trigger, "baseline_label"] = "DEMAND"

    df["baseline_confidence"] = 0.0
    df.loc[supply_trigger, "baseline_confidence"] += 0.6
    df.loc[demand_trigger, "baseline_confidence"] += 0.6
    df.loc[sales_drop, "baseline_confidence"] += 0.2
    df["baseline_confidence"] = df["baseline_confidence"].clip(0, 1)
    data_warning = df.get("inventory_zero_sales_weeks", 0) >= 2
    if isinstance(data_warning, pd.Series):
        df.loc[data_warning, "baseline_confidence"] *= 0.5

    evidence = df.apply(_evidence_list, axis=1)
    df["baseline_evidence"] = evidence
    return df


def evaluate_baseline(df: pd.DataFrame, label_col: str = "label") -> Dict[str, Any]:
    if label_col not in df.columns:
        return {"status": "skipped", "reason": "No labels available"}

    valid = df.dropna(subset=[label_col])
    if valid.empty:
        return {"status": "skipped", "reason": "No labeled rows"}

    y_true = valid[label_col]
    y_pred = valid["baseline_label"]
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    coverage = float((y_pred != "UNCERTAIN").mean())
    return {"status": "ok", "coverage": coverage, "report": report}


def _time_split(df: pd.DataFrame, validation_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = sort_by_sku_week(df)
    weeks = sorted(df["week_start_date"].dropna().unique())
    if not weeks:
        raise ValueError("No week_start_date values for time split")
    cutoff = int(len(weeks) * (1 - validation_size))
    cutoff = max(1, min(cutoff, len(weeks) - 1))
    train_weeks = set(weeks[:cutoff])
    train = df[df["week_start_date"].isin(train_weeks)]
    valid = df[~df["week_start_date"].isin(train_weeks)]
    return train, valid


def _get_feature_columns(metadata: List[FeatureMetadata]) -> List[str]:
    return [entry.name for entry in metadata]


def train_random_forest(
    df: pd.DataFrame,
    metadata: List[FeatureMetadata],
    config: ModelConfig,
    label_col: str = "label",
) -> Dict[str, Any]:
    if label_col not in df.columns:
        return {"status": "skipped", "reason": "No labels available"}

    labeled = df.dropna(subset=[label_col]).copy()
    if labeled.empty:
        return {"status": "skipped", "reason": "No labeled rows"}

    feature_cols = _get_feature_columns(metadata)
    feature_cols = [col for col in feature_cols if col in labeled.columns]
    if not feature_cols:
        return {"status": "skipped", "reason": "No feature columns found"}

    train_df, valid_df = _time_split(labeled, config.validation_size)
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[label_col]
    X_valid = valid_df[feature_cols].fillna(0)
    y_valid = valid_df[label_col]

    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        random_state=config.random_state,
        class_weight=config.class_weight,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    proba = model.predict_proba(X_valid)
    y_conf = proba.max(axis=1)

    report = classification_report(y_valid, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_valid, y_pred).tolist()
    false_supply = float(((y_pred == "SUPPLY") & (y_valid != "SUPPLY")).mean())
    false_demand = float(((y_pred == "DEMAND") & (y_valid != "DEMAND")).mean())

    shap_summary: Optional[pd.DataFrame] = None
    shap_status = {"status": "skipped", "reason": "SHAP not available"}

    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_valid)
        class_names = model.classes_.tolist()
        rows = []
        for class_idx, class_name in enumerate(class_names):
            class_values = shap_values[class_idx] if isinstance(shap_values, list) else shap_values
            mean_abs = np.abs(class_values).mean(axis=0)
            for feat, val in zip(feature_cols, mean_abs):
                rows.append({"class": class_name, "feature": feat, "mean_abs_shap": float(val)})
        shap_summary = pd.DataFrame(rows)
        shap_status = {"status": "ok"}
    except Exception as exc:  # pragma: no cover - optional
        shap_status = {"status": "skipped", "reason": str(exc)}

    trusted_flag = {
        "trusted": False,
        "reason": "SHAP unavailable",
    }

    if shap_summary is not None:
        metadata_map = {entry.name: entry.feature_type for entry in metadata}
        def _top_matches(target_type: str) -> bool:
            if shap_summary is None:
                return False
            subset = shap_summary[shap_summary["class"].str.upper() == target_type]
            if subset.empty:
                return False
            top = subset.sort_values("mean_abs_shap", ascending=False).head(5)
            matches = sum(1 for feat in top["feature"] if metadata_map.get(feat) == target_type.lower())
            return matches >= 3

        supply_ok = _top_matches("SUPPLY")
        demand_ok = _top_matches("DEMAND")
        trusted_flag = {
            "trusted": bool(supply_ok and demand_ok),
            "reason": "SHAP alignment check",
            "supply_ok": supply_ok,
            "demand_ok": demand_ok,
            "shap_status": shap_status,
        }

    metrics = {
        "status": "ok",
        "classification_report": report,
        "confusion_matrix": cm,
        "false_supply_rate": false_supply,
        "false_demand_rate": false_demand,
        "validation_size": float(config.validation_size),
        "mean_validation_confidence": float(y_conf.mean()) if len(y_conf) > 0 else 0.0,
    }

    return {
        "status": "ok",
        "model": model,
        "metrics": metrics,
        "feature_columns": feature_cols,
        "shap_summary": shap_summary,
        "trusted_flag": trusted_flag,
        "shap_status": shap_status,
    }


def predict_with_model(
    model: RandomForestClassifier,
    feature_df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    X = feature_df[feature_cols].fillna(0)
    pred = model.predict(X)
    proba = model.predict_proba(X)
    conf = proba.max(axis=1)
    output = feature_df[["sku_id", "week_start_date"]].copy()
    output["rf_label"] = pred
    output["rf_confidence"] = conf
    return output
