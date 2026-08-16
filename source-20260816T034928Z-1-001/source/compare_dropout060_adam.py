"""Compare dropout 0.60 with dropout 0.40 while holding CoxRes-KGA settings fixed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap_analysis import (
    METRICS,
    PROPOSED_MODEL,
    VALIDATION_LABELS,
    bootstrap_indices,
    percentile_interval,
    summarize,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results" / "sensitivity" / "dropout040_adam" / "predictions" / "all_predictions.csv"
CANDIDATE = ROOT / "results" / "sensitivity" / "dropout060_adam" / "predictions" / "all_predictions.csv"
OUTPUT = ROOT / "results" / "sensitivity" / "dropout060_adam" / "metrics"
KEYS = ["dataset", "patient_id", "time", "event"]
VALUES = ["risk_score", "survival_12", "survival_24", "survival_36", "survival_60"]


def model_frame(path: Path, strategy: str) -> pd.DataFrame:
    predictions = pd.read_csv(path, dtype={"fold": str})
    return (
        predictions.loc[
            predictions["strategy"].eq(strategy) & predictions["model"].eq(PROPOSED_MODEL),
            KEYS + VALUES,
        ]
        .sort_values(["dataset", "patient_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def main() -> None:
    rng = np.random.default_rng(42)
    rows = []
    replicate_rows = []
    for strategy, validation in VALIDATION_LABELS.items():
        baseline = model_frame(BASELINE, strategy)
        candidate = model_frame(CANDIDATE, strategy)
        if not baseline[KEYS].equals(candidate[KEYS]):
            raise ValueError(f"Patient alignment differs for {strategy}")
        baseline_point = summarize(baseline)
        candidate_point = summarize(candidate)
        current_replicates = []
        for replicate in range(1, 1001):
            indices = bootstrap_indices(baseline, rng)
            baseline_sample = summarize(baseline.iloc[indices])
            candidate_sample = summarize(candidate.iloc[indices])
            row = {
                "strategy": strategy,
                "validation": validation,
                "replicate": replicate,
                **{
                    f"delta_{metric}": candidate_sample[metric] - baseline_sample[metric]
                    for metric in METRICS
                },
            }
            replicate_rows.append(row)
            current_replicates.append(row)
        current = pd.DataFrame(current_replicates)
        summary = {"strategy": strategy, "validation": validation}
        for metric in METRICS:
            low, high = percentile_interval(current[f"delta_{metric}"])
            summary.update(
                {
                    f"dropout040_{metric}": baseline_point[metric],
                    f"dropout060_{metric}": candidate_point[metric],
                    f"delta_{metric}": candidate_point[metric] - baseline_point[metric],
                    f"delta_{metric}_low": low,
                    f"delta_{metric}_high": high,
                }
            )
        rows.append(summary)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT / "dropout040_vs_dropout060.csv", index=False)
    pd.DataFrame(replicate_rows).to_csv(OUTPUT / "dropout040_vs_dropout060_replicates.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
