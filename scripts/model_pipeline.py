"""Generate features, baseline attribution, and train RF model."""
from __future__ import annotations

import argparse
from dataclasses import asdict

import pandas as pd

from src.config import load_config
from src.model_logic import build_features, evaluate_baseline, run_baseline_attribution, train_random_forest
from src.utils.io import read_table, write_json, write_pickle, write_table
from src.utils.validation import log_assumptions_and_failure_modes


def main() -> None:
    parser = argparse.ArgumentParser(description="Run feature engineering and model training.")
    parser.add_argument("--config", default=None, help="Path to YAML/JSON config")
    args = parser.parse_args()

    config = load_config(args.config)
    base_df = read_table(config.paths.base_table_path)

    feature_df, metadata = build_features(base_df, config.features)
    write_table(feature_df, config.paths.feature_table_path)

    metadata_payload = [asdict(entry) for entry in metadata]
    write_json({"features": metadata_payload}, config.paths.feature_metadata_path)

    baseline_df = run_baseline_attribution(feature_df, config.features)
    baseline_pred = baseline_df[
        ["sku_id", "week_start_date", "baseline_label", "baseline_confidence", "baseline_evidence"]
    ]
    write_table(baseline_pred, config.paths.baseline_predictions_path)
    baseline_metrics = evaluate_baseline(baseline_df)
    write_json(baseline_metrics, config.paths.baseline_metrics_path)

    train_out = train_random_forest(feature_df, metadata, config.model)
    if train_out.get("status") == "ok":
        write_pickle(train_out["model"], config.paths.model_path)
        write_json(train_out["metrics"], config.paths.model_metrics_path)
        write_json(train_out["trusted_flag"], config.paths.trusted_model_flag_path)

        shap_summary = train_out.get("shap_summary")
        if shap_summary is None:
            shap_summary = pd.DataFrame(columns=["class", "feature", "mean_abs_shap"])
        shap_summary.to_csv(config.paths.shap_summary_path, index=False)
    else:
        write_json(train_out, config.paths.model_metrics_path)
        write_json({"trusted": False, "reason": train_out.get("reason")}, config.paths.trusted_model_flag_path)
        pd.DataFrame(columns=["class", "feature", "mean_abs_shap"]).to_csv(
            config.paths.shap_summary_path, index=False
        )

    log_assumptions_and_failure_modes(config.paths)
    print("Wrote feature table and model artifacts")


if __name__ == "__main__":
    main()
