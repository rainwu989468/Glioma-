"""Paired patient-bootstrap comparison of CoxRes-KGA architecture variants."""

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
    "32d/1h": ROOT / "results" / "sensitivity" / "rank_tuning" / "predictions" / "all_predictions.csv",
    "16d/1h": ROOT / "results" / "predictions" / "all_predictions.csv",
    "16d/2h": ROOT / "results" / "sensitivity" / "d16_h2" / "predictions" / "all_predictions.csv",
}
BASELINE = "32d/1h"
KEYS = ["dataset", "patient_id", "time", "event"]
VALUE_COLUMNS = ["risk_score", "survival_12", "survival_24", "survival_36", "survival_60"]


def load_frames(strategy: str) -> dict[str, pd.DataFrame]:
    frames = {}
    reference = None
    for architecture, path in RUNS.items():
        predictions = pd.read_csv(path, dtype={"fold": str})
        frame = predictions.loc[
            predictions["strategy"].eq(strategy) & predictions["model"].eq(PROPOSED_MODEL),
            KEYS + VALUE_COLUMNS,
        ].sort_values(["dataset", "patient_id"], kind="mergesort").reset_index(drop=True)
        if reference is None:
            reference = frame[KEYS]
        elif not frame[KEYS].equals(reference):
            raise ValueError(f"Patient alignment differs for {strategy}: {architecture}")
        frames[architecture] = frame
    return frames


def main() -> None:
    rng = np.random.default_rng(42)
    point_rows = []
    replicate_rows = []
    for strategy, validation in VALIDATION_LABELS.items():
        frames = load_frames(strategy)
        points = {architecture: summarize(frame) for architecture, frame in frames.items()}
        for architecture, metrics in points.items():
            point_rows.append({"strategy": strategy, "validation": validation, "architecture": architecture, **metrics})
        for replicate in range(1, 1001):
            indices = bootstrap_indices(frames[BASELINE], rng)
            sampled = {architecture: summarize(frame.iloc[indices]) for architecture, frame in frames.items()}
            for architecture in ("16d/1h", "16d/2h"):
                replicate_rows.append(
                    {
                        "strategy": strategy,
                        "validation": validation,
                        "replicate": replicate,
                        "architecture": architecture,
                        **{
                            f"delta_{metric}": sampled[architecture][metric] - sampled[BASELINE][metric]
                            for metric in METRICS
                        },
                    }
                )

    points = pd.DataFrame(point_rows)
    replicates = pd.DataFrame(replicate_rows)
    summary_rows = []
    for (strategy, validation, architecture), group in replicates.groupby(
        ["strategy", "validation", "architecture"], sort=False
    ):
        alternative = points.loc[
            points["strategy"].eq(strategy) & points["architecture"].eq(architecture)
        ].iloc[0]
        baseline = points.loc[
            points["strategy"].eq(strategy) & points["architecture"].eq(BASELINE)
        ].iloc[0]
        row = {"strategy": strategy, "validation": validation, "architecture": architecture}
        for metric in METRICS:
            column = f"delta_{metric}"
            low, high = percentile_interval(group[column])
            row.update(
                {
                    f"point_{metric}": float(alternative[metric]),
                    column: float(alternative[metric] - baseline[metric]),
                    f"{column}_low": low,
                    f"{column}_high": high,
                }
            )
        summary_rows.append(row)

    output_dir = ROOT / "results" / "sensitivity" / "architecture_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    points.to_csv(output_dir / "bootstrap_points.csv", index=False)
    replicates.to_csv(output_dir / "bootstrap_replicates.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "paired_bootstrap_differences.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
