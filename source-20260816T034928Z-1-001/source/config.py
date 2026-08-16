"""Shared paths and schema for the cleaned rain_final package."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLIT_DIR = DATA_DIR / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"
PREDICTION_DIR = RESULTS_DIR / "predictions"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURE_DIR = RESULTS_DIR / "figures"
REPORT_DIR = PROJECT_ROOT / "report"
LOG_DIR = PROJECT_ROOT / "logs"

SEED = 42
HORIZONS = (12.0, 24.0, 36.0, 60.0)
TIME_GRID = HORIZONS

ID_COL = "patient_id"
TIME_COL = "time"
EVENT_COL = "event"
COHORT_COL = "dataset"

FEATURES = (
    "age",
    "sex",
    "grade",
    "idh_mutant",
    "codel_1p19q",
    "idh_codel_subtype",
    "mgmt_methylated",
    "age_missing",
    "grade_missing",
    "idh_mutant_missing",
    "codel_1p19q_missing",
    "idh_codel_subtype_missing",
    "mgmt_methylated_missing",
)

MODEL_ORDER = (
    "cox_residual_kg_attention_nohazard",
    "xgboost_aft",
    "random_survival_forest",
    "survival_svm",
    "linear_regression",
    "regular_neural_network",
)

MODEL_LABELS = {
    "cox_residual_kg_attention_nohazard": "CoxRes-KGA",
    "xgboost_aft": "XGBoost AFT",
    "random_survival_forest": "Random survival forest",
    "survival_svm": "Survival SVM",
    "linear_regression": "Linear regression",
    "regular_neural_network": "Regular neural network",
}


def ensure_dirs() -> None:
    for path in (PREDICTION_DIR, METRICS_DIR, FIGURE_DIR, REPORT_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)
