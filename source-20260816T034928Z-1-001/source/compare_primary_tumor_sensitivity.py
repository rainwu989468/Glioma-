"""Compare full-cohort and primary-tumor-only performance without conflating estimands."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap_analysis import bootstrap_indices, percentile_interval, summarize
from config import SEED

ROOT = Path(__file__).resolve().parents[1]
MODEL = "cox_residual_kg_attention_nohazard"
SENSITIVITY = ROOT / "results" / "sensitivity" / "primary_tumor_only"


def main(repeats: int = 1000) -> None:
    full = pd.read_csv(ROOT / "results" / "metrics" / "performance_summary.csv")
    primary = pd.read_csv(SENSITIVITY / "metrics" / "performance_summary.csv")
    keys = ["validation", "model"]
    columns = keys + ["c_index", "rank_c_index", "auc_mean", "rank_auc", "ibs", "rank_ibs"]
    comparison = full[columns].merge(primary[columns], on=keys, suffixes=("_full", "_primary"))
    for metric in ("c_index", "auc_mean", "ibs"):
        comparison[f"delta_{metric}"] = comparison[f"{metric}_primary"] - comparison[f"{metric}_full"]
    comparison.to_csv(SENSITIVITY / "metrics" / "full_vs_primary_summary.csv", index=False)

    strategy = "train_CGGA_test_TCGA"
    full_predictions = pd.read_csv(ROOT / "results" / "predictions" / "all_predictions.csv")
    primary_predictions = pd.read_csv(SENSITIVITY / "predictions" / "all_predictions.csv")
    select = lambda frame: frame[(frame.strategy == strategy) & (frame.model == MODEL)].sort_values("patient_id").reset_index(drop=True)
    full_model, primary_model = select(full_predictions), select(primary_predictions)
    identity = ["patient_id", "dataset", "time", "event"]
    if not full_model[identity].equals(primary_model[identity]):
        raise ValueError("CGGA-to-TCGA comparison does not contain identical destination patients")

    point_full, point_primary = summarize(full_model), summarize(primary_model)
    rng = np.random.default_rng(SEED)
    rows = []
    for replicate in range(1, repeats + 1):
        indices = bootstrap_indices(full_model, rng)
        full_metrics = summarize(full_model.iloc[indices])
        primary_metrics = summarize(primary_model.iloc[indices])
        rows.append({"replicate": replicate, **{
            f"delta_{metric}": primary_metrics[metric] - full_metrics[metric]
            for metric in ("c_index", "auc_mean", "ibs")
        }})
    replicates = pd.DataFrame(rows)
    summary = {}
    for metric in ("c_index", "auc_mean", "ibs"):
        low, high = percentile_interval(replicates[f"delta_{metric}"])
        summary.update({
            f"delta_{metric}": point_primary[metric] - point_full[metric],
            f"delta_{metric}_low": low,
            f"delta_{metric}_high": high,
        })
    pd.DataFrame([{"validation": "External CGGA to TCGA", **summary}]).to_csv(
        SENSITIVITY / "metrics" / "paired_training_restriction.csv", index=False
    )
    replicates.to_csv(SENSITIVITY / "metrics" / "paired_training_restriction_replicates.csv", index=False)


if __name__ == "__main__":
    main()
