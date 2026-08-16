"""Compare the dropout-0.40/Adam CoxRes-KGA sensitivity with the retained model."""

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
PRIMARY = ROOT / "results" / "predictions" / "all_predictions.csv"
SENSITIVITY = (
    ROOT / "results" / "sensitivity" / "dropout040_adam" / "predictions" / "all_predictions.csv"
)
OUTPUT = ROOT / "results" / "sensitivity" / "dropout040_adam" / "metrics"
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
        primary = model_frame(PRIMARY, strategy)
        sensitivity = model_frame(SENSITIVITY, strategy)
        if not primary[KEYS].equals(sensitivity[KEYS]):
            raise ValueError(f"Patient alignment differs for {strategy}")
        primary_point = summarize(primary)
        sensitivity_point = summarize(sensitivity)
        for replicate in range(1, 1001):
            indices = bootstrap_indices(primary, rng)
            primary_sample = summarize(primary.iloc[indices])
            sensitivity_sample = summarize(sensitivity.iloc[indices])
            replicate_rows.append(
                {
                    "strategy": strategy,
                    "validation": validation,
                    "replicate": replicate,
                    **{
                        f"delta_{metric}": sensitivity_sample[metric] - primary_sample[metric]
                        for metric in METRICS
                    },
                }
            )
        replicates = pd.DataFrame(replicate_rows)
        current = replicates.loc[replicates["strategy"].eq(strategy)]
        row = {"strategy": strategy, "validation": validation}
        for metric in METRICS:
            low, high = percentile_interval(current[f"delta_{metric}"])
            row.update(
                {
                    f"primary_{metric}": primary_point[metric],
                    f"sensitivity_{metric}": sensitivity_point[metric],
                    f"delta_{metric}": sensitivity_point[metric] - primary_point[metric],
                    f"delta_{metric}_low": low,
                    f"delta_{metric}_high": high,
                }
            )
        rows.append(row)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT / "retained_comparison.csv", index=False)
    pd.DataFrame(replicate_rows).to_csv(OUTPUT / "retained_comparison_replicates.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
