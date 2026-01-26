"""Run end-to-end RCA orchestration."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.orchestrator import run_orchestrator
from src.utils.io import read_json, read_pickle, read_table, write_json


def _safe_read_json(path: str):
    try:
        return read_json(path)
    except FileNotFoundError:
        return None


def _safe_read_pickle(path: str):
    try:
        return read_pickle(path)
    except FileNotFoundError:
        return None


def _safe_read_csv(path: str) -> pd.DataFrame | None:
    if not Path(path).exists():
        return None
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RCA orchestrator.")
    parser.add_argument("--config", default=None, help="Path to YAML/JSON config")
    args = parser.parse_args()

    config = load_config(args.config)
    anomalies_df = read_table(config.paths.anomalies_path)
    feature_df = read_table(config.paths.feature_table_path)

    trusted_flag = _safe_read_json(config.paths.trusted_model_flag_path)
    model = _safe_read_pickle(config.paths.model_path)
    shap_summary = _safe_read_csv(config.paths.shap_summary_path)

    feature_cols = None
    metadata = _safe_read_json(config.paths.feature_metadata_path)
    if metadata and "features" in metadata:
        feature_cols = [item["name"] for item in metadata["features"]]

    outputs = run_orchestrator(
        anomalies_df=anomalies_df,
        feature_df=feature_df,
        config=config,
        model=model,
        feature_cols=feature_cols,
        trusted_flag=trusted_flag,
        shap_summary=shap_summary,
    )

    write_json({"outputs": outputs}, config.paths.rca_output_path)
    print(f"Wrote RCA outputs to {config.paths.rca_output_path}")


if __name__ == "__main__":
    main()
