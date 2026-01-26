"""External market and competitor RCA agent (stubbed)."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def derive_keywords(row: pd.Series) -> List[str]:
    keywords = []
    for field in ["sku_id", "brand", "category"]:
        value = row.get(field)
        if pd.notna(value):
            keywords.append(str(value))
    return keywords


def analyze_external(row: pd.Series) -> Dict[str, Any]:
    event_flag = int(row.get("external_event_flag", 0) == 1)
    intensity = row.get("external_event_intensity", np.nan)

    causes: List[str] = []
    evidence: List[str] = []
    score = 0.0

    if event_flag:
        causes.append("external_event")
        evidence.append("external_event_flag=1")
        score += 0.4

    if pd.notna(intensity) and intensity > 0:
        causes.append("external_event_intensity")
        evidence.append(f"external_event_intensity={intensity:.2f}")
        score += 0.2

    if not causes:
        evidence.append("No external signal triggers")

    confidence = min(score, 1.0)
    summary = "External demand signals detected" if score > 0 else "External signals inconclusive"

    return {
        "external_label": "DEMAND" if score >= 0.4 else "UNCERTAIN",
        "ranked_causes": causes,
        "evidence": evidence[:5],
        "external_score": score,
        "external_confidence": confidence,
        "external_summary": summary,
        "keywords": derive_keywords(row),
    }
