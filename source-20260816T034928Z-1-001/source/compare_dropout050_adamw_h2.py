"""Compare the dropout-0.50/two-head/AdamW variant with prior CoxRes-KGA models."""

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
RUNS = {
    "Retained 0.12/AdamW/1h": ROOT / "results" / "predictions" / "all_predictions.csv",
    "Best-rank 0.40/Adam/1h": ROOT / "results" / "sensitivity" / "dropout040_adam" / "predictions" / "all_predictions.csv",
    "Candidate 0.50/AdamW/2h": ROOT / "results" / "sensitivity" / "dropout050_adamw_h2" / "predictions" / "all_predictions.csv",
}
CANDIDATE = "Candidate 0.50/AdamW/2h"
BASELINES = ("Retained 0.12/AdamW/1h", "Best-rank 0.40/Adam/1h")
OUTPUT = ROOT / "results" / "sensitivity" / "dropout050_adamw_h2" / "metrics"
KEYS = ["dataset", "patient_id", "time", "event"]
VALUES = ["risk_score", "survival_12", "survival_24", "survival_36", "survival_60"]


def load_frame(path: Path, strategy: str) -> pd.DataFrame:
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
    point_rows = []
    replicate_rows = []
    for strategy, validation in VALIDATION_LABELS.items():
        frames = {name: load_frame(path, strategy) for name, path in RUNS.items()}
        reference = frames[CANDIDATE][KEYS]
        if any(not frame[KEYS].equals(reference) for frame in frames.values()):
            raise ValueError(f"Patient alignment differs for {strategy}")
        points = {name: summarize(frame) for name, frame in frames.items()}
        point_rows.extend(
            {"strategy": strategy, "validation": validation, "configuration": name, **metrics}
            for name, metrics in points.items()
        )
        for replicate in range(1, 1001):
            indices = bootstrap_indices(frames[CANDIDATE], rng)
            sampled = {name: summarize(frame.iloc[indices]) for name, frame in frames.items()}
            for baseline in BASELINES:
                replicate_rows.append(
                    {
                        "strategy": strategy,
                        "validation": validation,
                        "baseline": baseline,
                        "replicate": replicate,
                        **{
                            f"delta_{metric}": sampled[CANDIDATE][metric] - sampled[baseline][metric]
                            for metric in METRICS
                        },
                    }
                )

    points = pd.DataFrame(point_rows)
    replicates = pd.DataFrame(replicate_rows)
    summary_rows = []
    for (strategy, validation, baseline), group in replicates.groupby(
        ["strategy", "validation", "baseline"], sort=False
    ):
        candidate = points.loc[
            points["strategy"].eq(strategy) & points["configuration"].eq(CANDIDATE)
        ].iloc[0]
        reference = points.loc[
            points["strategy"].eq(strategy) & points["configuration"].eq(baseline)
        ].iloc[0]
        row = {"strategy": strategy, "validation": validation, "baseline": baseline}
        for metric in METRICS:
            low, high = percentile_interval(group[f"delta_{metric}"])
            row.update(
                {
                    f"baseline_{metric}": float(reference[metric]),
                    f"candidate_{metric}": float(candidate[metric]),
                    f"delta_{metric}": float(candidate[metric] - reference[metric]),
                    f"delta_{metric}_low": low,
                    f"delta_{metric}_high": high,
                }
            )
        summary_rows.append(row)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    points.to_csv(OUTPUT / "three_configuration_points.csv", index=False)
    replicates.to_csv(OUTPUT / "paired_configuration_replicates.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT / "paired_configuration_differences.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
