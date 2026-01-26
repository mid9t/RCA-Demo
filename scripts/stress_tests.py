"""Stress test harness for RCA pipeline."""
from __future__ import annotations

import argparse
from typing import Dict, List

import pandas as pd

from src.config import PipelineConfig, load_config
from src.model_logic import build_features
from src.orchestrator import run_orchestrator
from src.utils.io import write_json
from src.utils.validation import run_validation_checks


def _build_synthetic_cases() -> pd.DataFrame:
    weeks = pd.date_range("2024-01-01", periods=6, freq="W-MON")
    records: List[Dict] = []

    def add_weekly_rows(sku_id: str, sales: List[float], on_hand: List[float],
                        external_flags: List[int], forecast: List[float], brand_sales: List[float] | None = None,
                        category_sales: List[float] | None = None) -> None:
        for idx, week in enumerate(weeks):
            if idx >= len(sales):
                continue
            records.append(
                {
                    "sku_id": sku_id,
                    "week_start_date": week,
                    "actual_sales": sales[idx],
                    "forecast": forecast[idx],
                    "anomaly_score": (sales[idx] - forecast[idx]),
                    "price": 10.0,
                    "promo_flag": 0,
                    "on_hand": on_hand[idx],
                    "on_order": 0,
                    "lead_time": 7.0,
                    "fill_rate": 0.95,
                    "external_event_flag": external_flags[idx],
                    "external_event_intensity": float(external_flags[idx]),
                    "brand_sales": brand_sales[idx] if brand_sales else None,
                    "category_sales": category_sales[idx] if category_sales else None,
                }
            )

    # Mixed cause: stockout + external event
    add_weekly_rows(
        "SKU_MIX",
        sales=[100, 100, 100, 95, 90, 60],
        on_hand=[200, 180, 160, 120, 80, 0],
        external_flags=[0, 0, 0, 0, 0, 1],
        forecast=[100, 100, 100, 100, 100, 100],
    )

    # Phantom inventory: zero on-hand but sales positive
    add_weekly_rows(
        "SKU_PHANTOM",
        sales=[90, 85, 80, 80, 80, 75],
        on_hand=[50, 20, 0, 0, 0, 0],
        external_flags=[0, 0, 0, 0, 0, 0],
        forecast=[100, 100, 100, 100, 100, 100],
    )

    # Cannibalization proxy: SKU down but brand stable
    add_weekly_rows(
        "SKU_CAN",
        sales=[110, 105, 100, 95, 90, 60],
        on_hand=[200, 200, 200, 200, 200, 180],
        external_flags=[0, 0, 0, 0, 0, 0],
        forecast=[110, 110, 110, 110, 110, 110],
        brand_sales=[1000, 1000, 1000, 1000, 1000, 1000],
        category_sales=[2000, 2000, 2000, 2000, 2000, 2000],
    )

    # Slow supply degradation
    add_weekly_rows(
        "SKU_SUPPLY",
        sales=[120, 115, 110, 100, 85, 60],
        on_hand=[300, 240, 180, 120, 80, 40],
        external_flags=[0, 0, 0, 0, 0, 0],
        forecast=[120, 120, 120, 120, 120, 120],
    )

    # External noise without sales change
    add_weekly_rows(
        "SKU_NOISE",
        sales=[100, 100, 100, 100, 100, 100],
        on_hand=[200, 200, 200, 200, 200, 200],
        external_flags=[0, 0, 0, 0, 0, 1],
        forecast=[100, 100, 100, 100, 100, 100],
    )

    # Cold start SKU with limited history
    records.append(
        {
            "sku_id": "SKU_NEW",
            "week_start_date": weeks[-1],
            "actual_sales": 70,
            "forecast": 100,
            "anomaly_score": -30,
            "price": 10.0,
            "promo_flag": 0,
            "on_hand": 50,
            "on_order": 0,
            "lead_time": 7.0,
            "fill_rate": 0.95,
            "external_event_flag": 0,
            "external_event_intensity": 0.0,
            "brand_sales": None,
            "category_sales": None,
        }
    )

    return pd.DataFrame.from_records(records)


def _extract_output(outputs: List[Dict], sku_id: str) -> Dict:
    for row in outputs:
        if row["sku_id"] == sku_id:
            return row
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RCA stress tests.")
    parser.add_argument("--config", default=None, help="Path to YAML/JSON config")
    args = parser.parse_args()

    config = load_config(args.config)
    base_df = _build_synthetic_cases()
    feature_df, _ = build_features(base_df, config.features)

    anomalies_df = base_df.drop_duplicates(subset=["sku_id", "week_start_date"]).copy()
    last_week = anomalies_df.groupby("sku_id")["week_start_date"].transform("max")
    anomalies_df = anomalies_df[anomalies_df["week_start_date"] == last_week]

    outputs = run_orchestrator(anomalies_df, feature_df, config)

    validation_report = run_validation_checks(base_df, config.features)
    inventory_flags = validation_report["sanity"]["inventory_unreliability"]["flagged_sku_count"]

    tests = []

    mix_out = _extract_output(outputs, "SKU_MIX")
    tests.append(
        {
            "name": "mixed_cause",
            "passed": mix_out.get("final_label") == "MIXED",
            "details": mix_out,
        }
    )

    phantom_out = _extract_output(outputs, "SKU_PHANTOM")
    phantom_ok = phantom_out.get("final_label") != "SUPPLY" or phantom_out.get("confidence", 1) < 0.6
    tests.append(
        {
            "name": "phantom_inventory",
            "passed": phantom_ok and inventory_flags > 0,
            "details": {"output": phantom_out, "inventory_flagged": inventory_flags},
        }
    )

    cannibal_out = _extract_output(outputs, "SKU_CAN")
    cannibal_ok = cannibal_out.get("final_label") in {"UNCERTAIN", "DEMAND"} and cannibal_out.get(
        "confidence", 1
    ) < 0.7
    tests.append(
        {
            "name": "cannibalization_proxy",
            "passed": cannibal_ok,
            "details": cannibal_out,
        }
    )

    supply_out = _extract_output(outputs, "SKU_SUPPLY")
    tests.append(
        {
            "name": "slow_supply_degradation",
            "passed": supply_out.get("final_label") == "SUPPLY",
            "details": supply_out,
        }
    )

    noise_out = _extract_output(outputs, "SKU_NOISE")
    tests.append(
        {
            "name": "external_noise",
            "passed": noise_out.get("final_label") != "DEMAND",
            "details": noise_out,
        }
    )

    cold_out = _extract_output(outputs, "SKU_NEW")
    tests.append(
        {
            "name": "cold_start",
            "passed": cold_out.get("confidence", 1) < 0.6,
            "details": cold_out,
        }
    )

    passed = sum(1 for test in tests if test["passed"])
    report = {
        "passed": passed,
        "total": len(tests),
        "tests": tests,
    }

    write_json(report, config.paths.stress_test_report_path)
    print(f"Wrote stress test report to {config.paths.stress_test_report_path}")


if __name__ == "__main__":
    main()
