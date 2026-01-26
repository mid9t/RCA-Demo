"""Configuration models for the RCA pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


@dataclass
class PathsConfig:
    anomalies_path: str = "data/anomalies.csv"
    sales_path: str = "data/sales.csv"
    pricing_path: str = "data/pricing.csv"
    promotions_path: str = "data/promotions.csv"
    inventory_path: str = "data/inventory.csv"
    external_signals_path: str = "data/external_signals.csv"
    sku_alias_path: Optional[str] = None
    output_dir: str = "artifacts"
    base_table_path: str = "artifacts/base_sku_week.parquet"
    feature_table_path: str = "artifacts/feature_table.parquet"
    feature_metadata_path: str = "artifacts/feature_metadata.json"
    baseline_predictions_path: str = "artifacts/baseline_predictions.parquet"
    baseline_metrics_path: str = "artifacts/baseline_metrics.json"
    model_path: str = "artifacts/trained_model.pkl"
    model_metrics_path: str = "artifacts/model_metrics.json"
    shap_summary_path: str = "artifacts/shap_summary.csv"
    trusted_model_flag_path: str = "artifacts/trusted_model_flag.json"
    rca_output_path: str = "artifacts/rca_outputs.json"
    validation_report_path: str = "artifacts/validation_report.json"
    stress_test_report_path: str = "artifacts/stress_test_report.json"
    assumptions_path: str = "artifacts/assumptions.json"
    failure_modes_path: str = "artifacts/failure_modes.json"


@dataclass
class FeatureConfig:
    week_start_day: int = 0  # Monday
    price_agg: str = "mean"  # "mean" or "last"
    inventory_agg: str = "mean"  # "mean" or "last"
    stockout_threshold: float = 1.0
    inventory_coverage_threshold: float = 2.0
    sales_drop_pct_threshold: float = -0.2
    promo_price_change_threshold: float = 0.01
    inventory_spike_multiplier: float = 3.0
    lags: List[int] = field(default_factory=lambda: [1, 2, 4])
    rolling_windows: List[int] = field(default_factory=lambda: [4, 8])
    slope_window: int = 4
    yoy_lag: int = 52
    eps: float = 1e-6


@dataclass
class ModelConfig:
    n_estimators: int = 200
    max_depth: Optional[int] = 6
    min_samples_leaf: int = 5
    random_state: int = 42
    class_weight: str = "balanced"
    validation_size: float = 0.2


@dataclass
class ThresholdConfig:
    internal_confidence_threshold: float = 0.6
    external_confidence_threshold: float = 0.6
    rf_confidence_threshold: float = 0.65
    baseline_confidence_threshold: float = 0.55
    mixed_confidence_threshold: float = 0.5


@dataclass
class PipelineConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _merge_dicts(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and key in base:
            base[key] = _merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Optional[str] = None) -> PipelineConfig:
    if not path:
        return PipelineConfig()

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    raw: Dict[str, Any]
    if config_path.suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("PyYAML is required to load YAML configs.")
        raw = yaml.safe_load(config_path.read_text()) or {}
    elif config_path.suffix == ".json":
        raw = json.loads(config_path.read_text())
    else:
        raise ValueError("Config must be .yaml, .yml, or .json")

    base = PipelineConfig().to_dict()
    merged = _merge_dicts(base, raw)
    return PipelineConfig(
        paths=PathsConfig(**merged.get("paths", {})),
        features=FeatureConfig(**merged.get("features", {})),
        model=ModelConfig(**merged.get("model", {})),
        thresholds=ThresholdConfig(**merged.get("thresholds", {})),
    )
