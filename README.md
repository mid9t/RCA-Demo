# Automated RCA for SKU-Week Sales Anomalies (Demo)

This repository is a portfolio-style mock of an Automated Root Cause Analysis (RCA) system I built during a previous internship. It is designed to highlight technical skills, system design thinking, and a pragmatic, interpretability-first ML workflow.

## Project overview
- **Goal:** Explain sales anomalies at SKU-week grain with evidence, not just a label.
- **Input:** Anomaly signal from forecast vs actual sales at SKU-week.
- **Output:** Attribution of anomaly cause at minimum supply-driven vs demand-driven, plus evidence and confidence.
- **Data sources:**
  - Internal: sales, pricing, promotions, inventory/fulfillment
  - External: market/competitor/event signals (keyword-driven; mocked)
- **Modeling:**
  - Baseline heuristic attribution
  - Interpretable Random Forest classifier
  - SHAP explainability with trust gating
- **Principles:** Prioritize data correctness, interpretability, and failure-safe outputs over marginal accuracy.

## System architecture
The system is modular and aligned around SKU-week contracts.

```
Raw Tables
  |  sales / pricing / promos / inventory / external / anomalies
  v
Data Preprocessing (SKU-week alignment + validation)
  v
Feature Engineering (separate supply vs demand + temporal features)
  v
Baseline Heuristics  ----->  RF Classifier + SHAP (trusted gating)
  v                                |
Internal Agent  <-------------------
External Agent (stubbed)
  v
Orchestrator (decision logic + confidence gating)
  v
RCA Output (label, confidence, evidence, summaries)
```

### Component roles
- **Data preprocessing:** Standardizes SKUs, aggregates to SKU-week, joins sources, handles missing data, and runs sanity checks.
- **Feature engineering:** Creates supply and demand features separately, adds temporal dynamics (lags, rolling windows, slope), and maintains feature metadata.
- **Baseline attribution:** Rule-based, transparent benchmark that serves as guardrails.
- **RF + SHAP:** Interpretable classifier with SHAP-based trust checks; falls back to baseline if SHAP contradicts domain logic.
- **Agents:**
  - Internal agent focuses on supply, pricing, promo, and fulfillment signals.
  - External agent summarizes event signals (stubbed interface).
- **Orchestrator:** Combines agents and models, applies confidence gating, and outputs evidence with safe defaults.

## Key files to review
- `scripts/data_preprocessing.py` - SKU-week alignment, joins, missing handling, validation report.
- `src/model_logic.py` - Feature engineering, baseline heuristics, RF training, SHAP checks.
- `src/agents/internal_agent.py` - Supply-first RCA logic and evidence scoring.
- `src/agents/external_agent.py` - External demand analysis interface (mocked).
- `src/orchestrator.py` - End-to-end decision logic and confidence gating.
- `scripts/model_pipeline.py` - Feature + baseline + RF/SHAP pipeline entry point.
- `scripts/run_orchestrator.py` - Produces RCA outputs for anomalies.
- `scripts/stress_tests.py` - Synthetic stress tests for failure modes.
- `src/utils/validation.py` - Join coverage and sanity checks, assumption/failure mode logging.
- `src/config.py` - Central configuration for data paths, feature windows, and thresholds.

## Outputs and artifacts
All artifacts are written to `artifacts/` by default:
- `base_sku_week.*` (clean SKU-week table)
- `feature_table.*` (engineered features)
- `baseline_predictions.*`
- `trained_model.pkl`, `model_metrics.json`, `shap_summary.csv`, `trusted_model_flag.json`
- `rca_outputs.json`
- `validation_report.json`, `stress_test_report.json`

## Running the demo
Example (CSV outputs, no parquet engine required):

```
PYTHONPATH=. python scripts/data_preprocessing.py --config config.json
PYTHONPATH=. python scripts/model_pipeline.py --config config.json
PYTHONPATH=. python scripts/run_orchestrator.py --config config.json
PYTHONPATH=. python scripts/stress_tests.py --config config.json
```

## Notes on scope
This project stays within a strict, internship-style scope: no deep learning, no production infra, no extra data sources beyond internal sales/pricing/promos/inventory plus external event signals.
