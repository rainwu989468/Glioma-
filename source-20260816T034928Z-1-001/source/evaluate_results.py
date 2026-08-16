"""Create a consistent six-model summary from corrected-data predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import HORIZONS, METRICS_DIR, MODEL_ORDER, PREDICTION_DIR, ensure_dirs


def concordance_index(time, event, risk) -> float:
    """Comparable-pair C-index; higher risk means shorter survival."""
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    r = np.asarray(risk, dtype=float)
    valid = np.isfinite(t) & np.isfinite(r) & (t > 0)
    t, e, r = t[valid], e[valid], r[valid]
    comparable = 0
    concordant = 0.0
    for index in range(len(t)):
        if e[index] != 1:
            continue
        later = t > t[index]
        comparable += int(later.sum())
        concordant += float((r[index] > r[later]).sum())
        concordant += 0.5 * float((r[index] == r[later]).sum())
    return float(concordant / comparable) if comparable else float("nan")


def censoring_survival(time: np.ndarray, event: np.ndarray):
    order = np.argsort(time)
    ordered_time = time[order]
    censored = (1 - event[order]).astype(int)
    unique = np.unique(ordered_time)
    values = []
    current = 1.0
    for value in unique:
        at_risk = int(np.sum(ordered_time >= value))
        n_censored = int(np.sum((ordered_time == value) & (censored == 1)))
        current *= max(0.0, 1.0 - n_censored / at_risk) if at_risk else 1.0
        values.append(max(current, 1e-3))
    values = np.asarray(values, dtype=float)

    def evaluate(query):
        query = np.asarray(query, dtype=float)
        locations = np.searchsorted(unique, query, side="right") - 1
        output = np.ones_like(query, dtype=float)
        valid = locations >= 0
        output[valid] = values[locations[valid]]
        return np.clip(output, 1e-3, 1.0)

    return evaluate


def brier_score(time, event, survival, horizon: float) -> float:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    survival = np.clip(np.asarray(survival, dtype=float), 1e-5, 1 - 1e-5)
    estimate_g = censoring_survival(time, event)
    event_before = (time <= horizon) & (event == 1)
    survived = time > horizon
    included = event_before | survived
    weights = np.zeros(len(time), dtype=float)
    weights[event_before] = 1.0 / estimate_g(time[event_before])
    weights[survived] = 1.0 / estimate_g(np.full(int(survived.sum()), horizon))
    outcome = survived.astype(float)
    denominator = float(weights[included].sum())
    if denominator <= 0:
        return float("nan")
    return float(np.sum(weights[included] * (outcome[included] - survival[included]) ** 2) / denominator)


def dynamic_auc(time, event, risk, horizon: float) -> float:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    case_risk = risk[(time <= horizon) & (event == 1)]
    control_risk = risk[time > horizon]
    if len(case_risk) == 0 or len(control_risk) == 0:
        return float("nan")
    score = 0.0
    for value in case_risk:
        score += float((value > control_risk).sum())
        score += 0.5 * float((value == control_risk).sum())
    return float(score / (len(case_risk) * len(control_risk)))


def summarize(frame: pd.DataFrame) -> dict:
    time = frame["time"].to_numpy(dtype=float)
    event = frame["event"].to_numpy(dtype=int)
    risk = frame["risk_score"].to_numpy(dtype=float)
    row = {
        "n": int(len(frame)),
        "events": int(event.sum()),
        "c_index": concordance_index(time, event, risk),
    }
    aucs, briers = [], []
    for horizon in HORIZONS:
        auc = dynamic_auc(time, event, risk, horizon)
        brier = brier_score(time, event, frame[f"survival_{int(horizon)}"], horizon)
        row[f"auc_{int(horizon)}"] = auc
        row[f"brier_{int(horizon)}"] = brier
        if np.isfinite(auc):
            aucs.append(auc)
        if np.isfinite(brier):
            briers.append(brier)
    row["auc_mean"] = float(np.mean(aucs)) if aucs else float("nan")
    # Predictions are available at four prespecified horizons, so the reported
    # IBS is their mean rather than numerical integration over a dense grid.
    row["ibs"] = float(np.mean(briers)) if briers else float("nan")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=PREDICTION_DIR / "all_predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=METRICS_DIR)
    args = parser.parse_args()
    ensure_dirs()
    predictions = pd.read_csv(args.predictions, dtype={"fold": str})

    fold_rows = []
    for (strategy, fold, model), frame in predictions.groupby(["strategy", "fold", "model"], sort=True):
        row = summarize(frame)
        row.update({
            "strategy": strategy,
            "fold": fold,
            "model": model,
            "train_cohort": frame["train_cohort"].iloc[0],
            "test_cohort": frame["test_cohort"].iloc[0],
        })
        fold_rows.append(row)
    fold_metrics = pd.DataFrame(fold_rows)

    summary_rows = []
    for model in MODEL_ORDER:
        internal = predictions[(predictions["strategy"] == "combined_cv") & (predictions["model"] == model)]
        row = summarize(internal)
        row.update({
            "validation": "Internal combined 5-fold CV",
            "strategy": "combined_cv",
            "train_cohort": "CGGA+TCGA",
            "test_cohort": "CGGA+TCGA",
            "model": model,
        })
        summary_rows.append(row)
        for strategy, train_cohort, test_cohort, label in (
            ("train_CGGA_test_TCGA", "CGGA", "TCGA", "External CGGA to TCGA"),
            ("train_TCGA_test_CGGA", "TCGA", "CGGA", "External TCGA to CGGA"),
        ):
            external = predictions[(predictions["strategy"] == strategy) & (predictions["model"] == model)]
            row = summarize(external)
            row.update({
                "validation": label,
                "strategy": strategy,
                "train_cohort": train_cohort,
                "test_cohort": test_cohort,
                "model": model,
            })
            summary_rows.append(row)
    performance = pd.DataFrame(summary_rows)
    group = performance.groupby("validation")
    performance["rank_c_index"] = group["c_index"].rank(ascending=False, method="min")
    performance["rank_auc"] = group["auc_mean"].rank(ascending=False, method="min")
    performance["rank_ibs"] = group["ibs"].rank(ascending=True, method="min")
    performance = performance.sort_values(["validation", "rank_c_index", "model"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    performance.to_csv(args.output_dir / "performance_summary.csv", index=False)

    audit = {
        "metric_definition": (
            "C-index is the requested comparable-pair concordance statistic. AUC and censoring-adjusted "
            "Brier scores are evaluated at 12, 24, 36, and 60 months."
        ),
        "ibs_definition": (
            "Mean of censoring-adjusted Brier scores at 12, 24, 36, and 60 months; reported as the "
            "four-horizon IBS convention used consistently throughout this experiment."
        ),
        "prediction_origin": "All six models refitted from corrected raw-data cohorts.",
        "prediction_rows": int(len(predictions)),
        "models": list(MODEL_ORDER),
    }
    (args.output_dir / "metric_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Wrote {len(fold_metrics)} fold rows and {len(performance)} summary rows")


if __name__ == "__main__":
    main()
