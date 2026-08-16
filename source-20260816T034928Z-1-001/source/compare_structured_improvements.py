"""Compare leakage-safe CoxRes-KGA structured-improvement variants."""

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
    "Baseline 16d/1h": ROOT / "results" / "predictions" / "all_predictions.csv",
    "A: nonlinear Cox": ROOT / "results" / "sensitivity" / "structured_A_nonlinear" / "predictions" / "all_predictions.csv",
    "B: nonlinear + crossfit": ROOT / "results" / "sensitivity" / "structured_B_crossfit" / "predictions" / "all_predictions.csv",
    "C: nonlinear + crossfit + orthogonal": ROOT / "results" / "sensitivity" / "structured_C_orthogonal" / "predictions" / "all_predictions.csv",
}
BASELINE = "Baseline 16d/1h"
KEYS = ["dataset", "patient_id", "time", "event"]
VALUE_COLUMNS = ["risk_score", "survival_12", "survival_24", "survival_36", "survival_60"]


def load_frames(strategy: str) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    reference = None
    for variant, path in RUNS.items():
        predictions = pd.read_csv(path, dtype={"fold": str})
        frame = predictions.loc[
            predictions["strategy"].eq(strategy) & predictions["model"].eq(PROPOSED_MODEL),
            KEYS + VALUE_COLUMNS,
        ].sort_values(["dataset", "patient_id"], kind="mergesort").reset_index(drop=True)
        if reference is None:
            reference = frame[KEYS]
        elif not frame[KEYS].equals(reference):
            raise ValueError(f"Patient alignment differs for {strategy}: {variant}")
        frames[variant] = frame
    return frames


def main() -> None:
    rng = np.random.default_rng(42)
    point_rows: list[dict] = []
    replicate_rows: list[dict] = []
    for strategy, validation in VALIDATION_LABELS.items():
        frames = load_frames(strategy)
        points = {variant: summarize(frame) for variant, frame in frames.items()}
        for variant, metrics in points.items():
            point_rows.append({"strategy": strategy, "validation": validation, "variant": variant, **metrics})
        for replicate in range(1, 1001):
            indices = bootstrap_indices(frames[BASELINE], rng)
            sampled = {variant: summarize(frame.iloc[indices]) for variant, frame in frames.items()}
            for variant in RUNS:
                if variant == BASELINE:
                    continue
                replicate_rows.append(
                    {
                        "strategy": strategy,
                        "validation": validation,
                        "replicate": replicate,
                        "variant": variant,
                        **{
                            f"delta_{metric}": sampled[variant][metric] - sampled[BASELINE][metric]
                            for metric in METRICS
                        },
                    }
                )

    points = pd.DataFrame(point_rows)
    replicates = pd.DataFrame(replicate_rows)
    summary_rows: list[dict] = []
    for (strategy, validation, variant), group in replicates.groupby(
        ["strategy", "validation", "variant"], sort=False
    ):
        alternative = points.loc[
            points["strategy"].eq(strategy) & points["variant"].eq(variant)
        ].iloc[0]
        baseline = points.loc[
            points["strategy"].eq(strategy) & points["variant"].eq(BASELINE)
        ].iloc[0]
        row = {"strategy": strategy, "validation": validation, "variant": variant}
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

    output_dir = ROOT / "results" / "sensitivity" / "structured_improvement_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    points.to_csv(output_dir / "points.csv", index=False)
    replicates.to_csv(output_dir / "bootstrap_replicates.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "paired_bootstrap_differences.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
