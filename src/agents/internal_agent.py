"""Internal operational RCA agent."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.config import FeatureConfig


def analyze_internal(row: pd.Series, config: FeatureConfig) -> Dict[str, Any]:
    causes: List[Dict[str, Any]] = []
    evidence: List[str] = []
    score = 0.0

    stockout = int(row.get("stockout_flag", 0) == 1)
    coverage = row.get("inventory_coverage", np.nan)
    coverage_low = False
    if pd.notna(coverage):
        coverage_low = coverage < config.inventory_coverage_threshold

    if stockout:
        causes.append({"cause": "stockout", "weight": 0.4})
        evidence.append(f"stockout_flag=1")
        score += 0.4

    if coverage_low:
        causes.append({"cause": "low_inventory_coverage", "weight": 0.3})
        evidence.append(f"inventory_coverage={coverage:.2f}")
        score += 0.3

    coverage_weeks = row.get("coverage_weeks_below", 0)
    if pd.notna(coverage_weeks) and coverage_weeks >= 2:
        causes.append({"cause": "sustained_low_inventory", "weight": 0.2})
        evidence.append(f"coverage_weeks_below={coverage_weeks:.0f}")
        score += 0.2

    inventory_warning = row.get("inventory_zero_sales_weeks", 0)
    if pd.notna(inventory_warning) and inventory_warning >= 2:
        evidence.append("inventory_unreliability_flag")
        score *= 0.5

    fill_rate_change = row.get("fill_rate_change", np.nan)
    if pd.notna(fill_rate_change) and fill_rate_change < -0.05:
        causes.append({"cause": "fill_rate_drop", "weight": 0.15})
        evidence.append(f"fill_rate_change={fill_rate_change:.2f}")
        score += 0.15

    lead_time_change = row.get("lead_time_change", np.nan)
    if pd.notna(lead_time_change) and lead_time_change > 0:
        causes.append({"cause": "lead_time_increase", "weight": 0.15})
        evidence.append(f"lead_time_change={lead_time_change:.2f}")
        score += 0.15

    price_change = row.get("price_change_pct", np.nan)
    if pd.notna(price_change) and abs(price_change) >= 0.05:
        causes.append({"cause": "price_change", "weight": 0.1})
        evidence.append(f"price_change_pct={price_change:.2f}")
        score += 0.1

    promo_flag = row.get("promo_flag", 0)
    sales_wow_pct = row.get("sales_wow_pct", np.nan)
    if promo_flag == 1 and pd.notna(sales_wow_pct) and sales_wow_pct < 0:
        causes.append({"cause": "promo_mismatch", "weight": 0.1})
        evidence.append("promo active but sales down")
        score += 0.1

    causes = sorted(causes, key=lambda x: x["weight"], reverse=True)
    confidence = min(score, 1.0)

    label = "SUPPLY" if score >= 0.4 else "UNCERTAIN"
    summary = "Supply-side signals detected" if label == "SUPPLY" else "Internal signals inconclusive"

    return {
        "internal_label": label,
        "candidate_causes": [c["cause"] for c in causes],
        "evidence": evidence[:5],
        "internal_score": score,
        "internal_confidence": confidence,
        "internal_summary": summary,
    }
