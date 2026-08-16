"""Audit an isolated repeat of the retained CoxRes-KGA configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results" / "sensitivity" / "dropout040_adam"
REPEAT = ROOT / "results" / "reproducibility" / "dropout040_adam_repeat"
OUTPUT = REPEAT / "audit"
PROPOSED = "cox_residual_kg_attention_nohazard"
KEYS = ["strategy", "fold", "dataset", "patient_id", "time", "event"]
PREDICTION_COLUMNS = [
    "risk_score",
    "survival_12",
    "survival_24",
    "survival_36",
    "survival_60",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def proposed_predictions(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / "predictions" / "all_predictions.csv", dtype={"fold": str})
    return (
        frame.loc[frame["model"].eq(PROPOSED), KEYS + PREDICTION_COLUMNS]
        .sort_values(KEYS, kind="mergesort")
        .reset_index(drop=True)
    )


def sorted_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"fold": str})
    return frame.sort_values(columns, kind="mergesort").reset_index(drop=True)


def manifest_projection(path: Path) -> list[dict]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    projection = []
    for job in sorted(manifest, key=lambda item: item["job_id"]):
        details = job["model_details"]
        architecture = details["architecture"]
        projection.append(
            {
                "job_id": job["job_id"],
                "selected_config": details["selected_config"],
                "seeds": details["seeds"],
                "token_dimension": architecture["token_dimension"],
                "attention_heads": architecture["attention_heads"],
                "attention_layers": architecture["attention_layers"],
                "dropout": architecture["dropout"],
                "optimizer": architecture["optimizer"],
                "learning_rate": architecture["learning_rate"],
                "weight_decay": architecture["weight_decay"],
                "selected_epochs": [member["selected_epoch"] for member in details["members"]],
            }
        )
    return projection


def main() -> None:
    baseline = proposed_predictions(BASELINE)
    repeat = proposed_predictions(REPEAT)
    keys_equal = baseline[KEYS].equals(repeat[KEYS])
    if not keys_equal:
        raise ValueError("Patient-level prediction keys do not align")

    prediction_rows = []
    for column in PREDICTION_COLUMNS:
        original = baseline[column].to_numpy(dtype=float)
        reproduced = repeat[column].to_numpy(dtype=float)
        difference = reproduced - original
        prediction_rows.append(
            {
                "quantity": column,
                "n": len(original),
                "exactly_equal": bool(np.array_equal(original, reproduced)),
                "allclose_at_1e-12": bool(np.allclose(original, reproduced, rtol=0.0, atol=1e-12)),
                "max_absolute_difference": float(np.max(np.abs(difference))),
                "mean_absolute_difference": float(np.mean(np.abs(difference))),
                "correlation": float(np.corrcoef(original, reproduced)[0, 1]),
            }
        )
    prediction_audit = pd.DataFrame(prediction_rows)

    baseline_performance = sorted_csv(
        BASELINE / "metrics" / "performance_summary.csv", ["validation", "model"]
    )
    repeat_performance = sorted_csv(
        REPEAT / "metrics" / "performance_summary.csv", ["validation", "model"]
    )
    baseline_folds = sorted_csv(
        BASELINE / "metrics" / "fold_metrics.csv", ["strategy", "fold", "model"]
    )
    repeat_folds = sorted_csv(
        REPEAT / "metrics" / "fold_metrics.csv", ["strategy", "fold", "model"]
    )

    metric_rows = []
    for validation in baseline_performance["validation"].drop_duplicates():
        original = baseline_performance[
            baseline_performance["validation"].eq(validation)
            & baseline_performance["model"].eq(PROPOSED)
        ].iloc[0]
        reproduced = repeat_performance[
            repeat_performance["validation"].eq(validation)
            & repeat_performance["model"].eq(PROPOSED)
        ].iloc[0]
        metric_rows.append(
            {
                "validation": validation,
                **{
                    f"original_{metric}": float(original[metric])
                    for metric in ("c_index", "auc_mean", "ibs")
                },
                **{
                    f"repeat_{metric}": float(reproduced[metric])
                    for metric in ("c_index", "auc_mean", "ibs")
                },
                **{
                    f"delta_{metric}": float(reproduced[metric] - original[metric])
                    for metric in ("c_index", "auc_mean", "ibs")
                },
                "original_c_index_rank": int(original["rank_c_index"]),
                "repeat_c_index_rank": int(reproduced["rank_c_index"]),
                "original_auc_rank": int(original["rank_auc"]),
                "repeat_auc_rank": int(reproduced["rank_auc"]),
                "original_ibs_rank": int(original["rank_ibs"]),
                "repeat_ibs_rank": int(reproduced["rank_ibs"]),
            }
        )
    metric_audit = pd.DataFrame(metric_rows)

    baseline_manifest = manifest_projection(BASELINE / "metrics" / "training_manifest.json")
    repeat_manifest = manifest_projection(REPEAT / "metrics" / "training_manifest.json")
    compared_files = [
        "predictions/all_predictions.csv",
        "metrics/fold_metrics.csv",
        "metrics/performance_summary.csv",
        "metrics/bootstrap_performance.csv",
        "metrics/paired_bootstrap_advantage.csv",
        "metrics/bootstrap_replicates.csv",
    ]
    file_hashes = {
        relative: {
            "original_sha256": sha256(BASELINE / relative),
            "repeat_sha256": sha256(REPEAT / relative),
            "identical": sha256(BASELINE / relative) == sha256(REPEAT / relative),
        }
        for relative in compared_files
    }

    exact_predictions = bool(prediction_audit["exactly_equal"].all())
    exact_performance = baseline_performance.equals(repeat_performance)
    exact_folds = baseline_folds.equals(repeat_folds)
    exact_manifest_projection = baseline_manifest == repeat_manifest
    exact_ranks = bool(
        (
            metric_audit[
                ["original_c_index_rank", "original_auc_rank", "original_ibs_rank"]
            ].to_numpy()
            == metric_audit[
                ["repeat_c_index_rank", "repeat_auc_rank", "repeat_ibs_rank"]
            ].to_numpy()
        ).all()
    )
    summary = {
        "configuration": {
            "token_dimension": 16,
            "attention_heads": 1,
            "attention_layers": 1,
            "dropout": 0.4,
            "optimizer": "adam",
            "learning_rate": 3e-4,
            "weight_decay": 1e-3,
            "max_epochs": 170,
            "ensemble_members": 5,
        },
        "patient_keys_equal": keys_equal,
        "exact_patient_level_predictions": exact_predictions,
        "exact_fold_metrics": exact_folds,
        "exact_performance_summary": exact_performance,
        "exact_metric_ranks": exact_ranks,
        "exact_training_decisions": exact_manifest_projection,
        "all_compared_files_byte_identical": all(item["identical"] for item in file_hashes.values()),
        "stable_and_reproducible": all(
            (
                keys_equal,
                exact_predictions,
                exact_folds,
                exact_performance,
                exact_ranks,
                exact_manifest_projection,
            )
        ),
        "file_hashes": file_hashes,
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    prediction_audit.to_csv(OUTPUT / "prediction_reproducibility.csv", index=False)
    metric_audit.to_csv(OUTPUT / "metric_reproducibility.csv", index=False)
    (OUTPUT / "training_decisions_original.json").write_text(
        json.dumps(baseline_manifest, indent=2), encoding="utf-8"
    )
    (OUTPUT / "training_decisions_repeat.json").write_text(
        json.dumps(repeat_manifest, indent=2), encoding="utf-8"
    )
    (OUTPUT / "reproducibility_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
