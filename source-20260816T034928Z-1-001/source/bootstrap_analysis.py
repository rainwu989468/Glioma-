"""Paired patient-bootstrap uncertainty analysis for corrected-cohort predictions.

The implementation deliberately reproduces the metric conventions in
``evaluate_results.py``.  It uses faster, algebraically equivalent ranking
calculations so that 1,000 paired resamples are practical without changing the
reported estimands.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import HORIZONS, METRICS_DIR, MODEL_ORDER, PREDICTION_DIR, SEED, ensure_dirs


VALIDATION_LABELS = {
    "combined_cv": "Internal combined 5-fold CV",
    "train_CGGA_test_TCGA": "External CGGA to TCGA",
    "train_TCGA_test_CGGA": "External TCGA to CGGA",
}
PROPOSED_MODEL = "cox_residual_kg_attention_nohazard"
METRICS = ("c_index", "auc_mean", "ibs")


class FenwickTree:
    """Frequency table supporting prefix counts in logarithmic time."""

    def __init__(self, size: int) -> None:
        self.values = np.zeros(size + 1, dtype=np.int64)

    def add(self, index: int) -> None:
        index += 1
        while index < len(self.values):
            self.values[index] += 1
            index += index & -index

    def prefix(self, index: int) -> int:
        if index < 0:
            return 0
        total = 0
        index += 1
        while index:
            total += int(self.values[index])
            index -= index & -index
        return total


def concordance_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    """Comparable-pair concordance, exactly matching evaluate_results.py."""

    valid = np.isfinite(time) & np.isfinite(risk) & (time > 0)
    time = np.asarray(time[valid], dtype=float)
    event = np.asarray(event[valid], dtype=int)
    risk = np.asarray(risk[valid], dtype=float)
    if len(time) == 0:
        return float("nan")

    risk_levels = np.unique(risk)
    risk_index = np.searchsorted(risk_levels, risk)
    order = np.argsort(-time, kind="mergesort")
    tree = FenwickTree(len(risk_levels))
    comparable = 0
    concordant = 0.0
    later_count = 0
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and time[order[end]] == time[order[position]]:
            end += 1
        group = order[position:end]
        for row_index in group[event[group] == 1]:
            coordinate = int(risk_index[row_index])
            lower = tree.prefix(coordinate - 1)
            at_or_below = tree.prefix(coordinate)
            concordant += lower + 0.5 * (at_or_below - lower)
            comparable += later_count
        for row_index in group:
            tree.add(int(risk_index[row_index]))
            later_count += 1
        position = end
    return float(concordant / comparable) if comparable else float("nan")


def dynamic_auc(time: np.ndarray, event: np.ndarray, risk: np.ndarray, horizon: float) -> float:
    cases = np.asarray(risk[(time <= horizon) & (event == 1)], dtype=float)
    controls = np.sort(np.asarray(risk[time > horizon], dtype=float))
    if len(cases) == 0 or len(controls) == 0:
        return float("nan")
    lower = np.searchsorted(controls, cases, side="left")
    at_or_below = np.searchsorted(controls, cases, side="right")
    return float(np.sum(lower + 0.5 * (at_or_below - lower)) / (len(cases) * len(controls)))


def censoring_survival(time: np.ndarray, event: np.ndarray):
    order = np.argsort(time, kind="mergesort")
    ordered_time = time[order]
    ordered_event = event[order]
    unique, first, counts = np.unique(ordered_time, return_index=True, return_counts=True)
    at_risk = len(time) - np.cumsum(np.r_[0, counts[:-1]])
    censored = np.add.reduceat((1 - ordered_event).astype(int), first)
    factors = np.where(at_risk > 0, 1.0 - censored / at_risk, 1.0)
    values = np.maximum(np.cumprod(factors), 1e-3)

    def evaluate(query: np.ndarray) -> np.ndarray:
        query = np.asarray(query, dtype=float)
        locations = np.searchsorted(unique, query, side="right") - 1
        output = np.ones_like(query, dtype=float)
        valid = locations >= 0
        output[valid] = values[locations[valid]]
        return np.clip(output, 1e-3, 1.0)

    return evaluate


def brier_score(
    time: np.ndarray,
    event: np.ndarray,
    survival: np.ndarray,
    horizon: float,
) -> float:
    survival = np.clip(np.asarray(survival, dtype=float), 1e-5, 1 - 1e-5)
    estimate_g = censoring_survival(time, event)
    event_before = (time <= horizon) & (event == 1)
    survived = time > horizon
    included = event_before | survived
    weights = np.zeros(len(time), dtype=float)
    weights[event_before] = 1.0 / estimate_g(time[event_before])
    weights[survived] = 1.0 / estimate_g(np.full(int(survived.sum()), horizon))
    denominator = float(weights[included].sum())
    if denominator <= 0:
        return float("nan")
    outcome = survived.astype(float)
    return float(np.sum(weights[included] * (outcome[included] - survival[included]) ** 2) / denominator)


def summarize(frame: pd.DataFrame) -> dict[str, float]:
    time = frame["time"].to_numpy(dtype=float)
    event = frame["event"].to_numpy(dtype=int)
    risk = frame["risk_score"].to_numpy(dtype=float)
    aucs = [dynamic_auc(time, event, risk, horizon) for horizon in HORIZONS]
    briers = [
        brier_score(time, event, frame[f"survival_{int(horizon)}"].to_numpy(dtype=float), horizon)
        for horizon in HORIZONS
    ]
    return {
        "c_index": concordance_index(time, event, risk),
        "auc_mean": float(np.nanmean(aucs)),
        "ibs": float(np.nanmean(briers)),
    }


def aligned_frames(predictions: pd.DataFrame, strategy: str) -> dict[str, pd.DataFrame]:
    columns = [
        "dataset", "patient_id", "time", "event", "risk_score",
        *[f"survival_{int(horizon)}" for horizon in HORIZONS],
    ]
    frames = {}
    reference = None
    for model in MODEL_ORDER:
        frame = predictions[
            (predictions["strategy"] == strategy) & (predictions["model"] == model)
        ][columns].sort_values(["dataset", "patient_id"], kind="mergesort").reset_index(drop=True)
        keys = frame[["dataset", "patient_id", "time", "event"]]
        if reference is None:
            reference = keys
        elif not keys.equals(reference):
            raise ValueError(f"Patient alignment differs for {strategy}: {model}")
        frames[model] = frame
    return frames


def bootstrap_indices(reference: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Resample patients within cohort, preserving the combined-CV cohort mix."""

    sampled = []
    for cohort in reference["dataset"].drop_duplicates():
        indices = np.flatnonzero(reference["dataset"].to_numpy() == cohort)
        sampled.append(rng.choice(indices, size=len(indices), replace=True))
    return np.concatenate(sampled)


def percentile_interval(values: pd.Series) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.empty:
        return float("nan"), float("nan")
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=PREDICTION_DIR / "all_predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=METRICS_DIR)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")

    ensure_dirs()
    predictions = pd.read_csv(args.predictions, dtype={"fold": str})
    point_rows = []
    replicate_rows = []
    rng = np.random.default_rng(args.seed)

    for strategy, validation in VALIDATION_LABELS.items():
        frames = aligned_frames(predictions, strategy)
        reference = frames[PROPOSED_MODEL]
        point = {model: summarize(frame) for model, frame in frames.items()}
        for model, metrics in point.items():
            point_rows.append({"strategy": strategy, "validation": validation, "model": model, **metrics})

        for replicate in range(1, args.repeats + 1):
            indices = bootstrap_indices(reference, rng)
            for model, frame in frames.items():
                metrics = summarize(frame.iloc[indices])
                replicate_rows.append({
                    "strategy": strategy,
                    "validation": validation,
                    "replicate": replicate,
                    "model": model,
                    **metrics,
                })

    points = pd.DataFrame(point_rows)
    replicates = pd.DataFrame(replicate_rows)
    summary_rows = []
    for (strategy, validation, model), group in replicates.groupby(
        ["strategy", "validation", "model"], sort=False
    ):
        point = points[(points["strategy"] == strategy) & (points["model"] == model)].iloc[0]
        for metric in METRICS:
            low, high = percentile_interval(group[metric])
            summary_rows.append({
                "strategy": strategy,
                "validation": validation,
                "model": model,
                "metric": metric,
                "estimate": float(point[metric]),
                "ci_low": low,
                "ci_high": high,
                "valid_replicates": int(np.isfinite(group[metric]).sum()),
            })

    paired_rows = []
    for (strategy, validation), group in replicates.groupby(["strategy", "validation"], sort=False):
        point_group = points[points["strategy"] == strategy].set_index("model")
        for comparator in MODEL_ORDER:
            if comparator == PROPOSED_MODEL:
                continue
            proposed_reps = group[group["model"] == PROPOSED_MODEL].set_index("replicate")
            comparator_reps = group[group["model"] == comparator].set_index("replicate")
            for metric in METRICS:
                if metric == "ibs":
                    advantage = comparator_reps[metric] - proposed_reps[metric]
                    estimate = point_group.loc[comparator, metric] - point_group.loc[PROPOSED_MODEL, metric]
                else:
                    advantage = proposed_reps[metric] - comparator_reps[metric]
                    estimate = point_group.loc[PROPOSED_MODEL, metric] - point_group.loc[comparator, metric]
                low, high = percentile_interval(advantage)
                finite = advantage[np.isfinite(advantage)]
                paired_rows.append({
                    "strategy": strategy,
                    "validation": validation,
                    "comparator": comparator,
                    "metric": metric,
                    "advantage": float(estimate),
                    "ci_low": low,
                    "ci_high": high,
                    "probability_advantage_gt_zero": float((finite > 0).mean()),
                    "valid_replicates": int(len(finite)),
                })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "bootstrap_performance.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(args.output_dir / "paired_bootstrap_advantage.csv", index=False)
    replicates.to_csv(args.output_dir / "bootstrap_replicates.csv", index=False)
    print(
        f"Wrote {len(replicates):,} bootstrap model replicates "
        f"({args.repeats} paired resamples per validation setting)"
    )


if __name__ == "__main__":
    main()
