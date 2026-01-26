"""End-to-end orchestration for RCA outputs."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from src.agents.external_agent import analyze_external
from src.agents.internal_agent import analyze_internal
from src.config import PipelineConfig
from src.model_logic import predict_with_model, run_baseline_attribution
from src.utils.time import sort_by_sku_week


def _unique_evidence(items: List[str], limit: int = 5) -> List[str]:
    seen = set()
    output = []
    for item in items:
        if item not in seen:
            output.append(item)
            seen.add(item)
        if len(output) >= limit:
            break
    return output


def _shap_top_features(shap_summary: Optional[pd.DataFrame], label: str, top_n: int = 5) -> List[str]:
    if shap_summary is None or shap_summary.empty:
        return []
    subset = shap_summary[shap_summary["class"].str.upper() == label.upper()]
    if subset.empty:
        return []
    top = subset.sort_values("mean_abs_shap", ascending=False).head(top_n)
    return top["feature"].tolist()


def run_orchestrator(
    anomalies_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    config: PipelineConfig,
    model: Optional[Any] = None,
    feature_cols: Optional[List[str]] = None,
    trusted_flag: Optional[Dict[str, Any]] = None,
    shap_summary: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    keys = anomalies_df[["sku_id", "week_start_date"]].drop_duplicates()
    df = keys.merge(feature_df, on=["sku_id", "week_start_date"], how="left")
    df = sort_by_sku_week(df)

    baseline_df = run_baseline_attribution(df, config.features)

    rf_output = None
    rf_trusted = bool(trusted_flag and trusted_flag.get("trusted"))
    if model is not None and feature_cols and rf_trusted:
        rf_output = predict_with_model(model, baseline_df, feature_cols)
        baseline_df = baseline_df.merge(rf_output, on=["sku_id", "week_start_date"], how="left")

    outputs: List[Dict[str, Any]] = []

    for _, row in baseline_df.iterrows():
        internal = analyze_internal(row, config.features)
        external = None
        if internal["internal_confidence"] < config.thresholds.internal_confidence_threshold or internal[
            "internal_label"
        ] == "UNCERTAIN":
            external = analyze_external(row)
        else:
            external = {
                "external_label": "UNCERTAIN",
                "ranked_causes": [],
                "evidence": [],
                "external_score": 0.0,
                "external_confidence": 0.0,
                "external_summary": "External agent skipped",
                "keywords": [],
            }

        internal_high = internal["internal_confidence"] >= config.thresholds.internal_confidence_threshold
        external_high = external["external_confidence"] >= config.thresholds.external_confidence_threshold

        final_label = row.get("baseline_label", "UNCERTAIN")
        confidence = float(row.get("baseline_confidence", 0.0))
        model_used = False

        if internal_high and external_high:
            final_label = "MIXED"
            confidence = min(internal["internal_confidence"], external["external_confidence"], config.thresholds.mixed_confidence_threshold)
        elif rf_trusted and row.get("rf_confidence", 0) >= config.thresholds.rf_confidence_threshold:
            final_label = row.get("rf_label", final_label)
            confidence = float(row.get("rf_confidence", confidence))
            model_used = True
        else:
            final_label = row.get("baseline_label", "UNCERTAIN")
            confidence = float(row.get("baseline_confidence", 0.0))

        evidence = []
        evidence.extend(internal.get("evidence", []))
        evidence.extend(external.get("evidence", []))
        evidence.extend(row.get("baseline_evidence", []))
        top_evidence = _unique_evidence([str(item) for item in evidence], limit=5)

        shap_features = _shap_top_features(shap_summary, final_label) if model_used else []

        recommended_checks = []
        if final_label == "UNCERTAIN":
            recommended_checks = [
                "Validate inventory feeds and missingness flags",
                "Confirm promo flags and pricing alignment",
                "Review external signal coverage for the week",
            ]

        outputs.append(
            {
                "sku_id": row.get("sku_id"),
                "week_start_date": str(row.get("week_start_date")),
                "final_label": final_label,
                "confidence": confidence,
                "top_evidence": top_evidence,
                "internal_summary": internal.get("internal_summary"),
                "external_summary": external.get("external_summary"),
                "shap_top_features": shap_features,
                "recommended_checks": recommended_checks,
            }
        )

    return outputs
