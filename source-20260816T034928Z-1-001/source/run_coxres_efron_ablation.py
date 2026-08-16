"""Run the training-only Efron/knowledge-bias CoxRes-KGA sensitivity analysis."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import HORIZONS, MODEL_ORDER, PREDICTION_DIR, SEED, SPLIT_DIR
from models import fit_coxres_ensemble
from run_experiments import jobs, prediction_frame


PROPOSED_MODEL = "cox_residual_kg_attention_nohazard"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "sensitivity" / "efron_ablation",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    prediction_dir = args.output_root / "predictions"
    metrics_dir = args.output_root / "metrics"
    job_dir = prediction_dir / ("smoke_jobs" if args.smoke else "refit_jobs")
    job_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    horizons = np.asarray(HORIZONS, dtype=float)

    for job in jobs(SPLIT_DIR):
        output = job_dir / f"{job['job_id']}__{PROPOSED_MODEL}.csv"
        detail_path = job_dir / f"{job['job_id']}__{PROPOSED_MODEL}.json"
        if output.exists() and detail_path.exists() and not args.force:
            continue
        started = time.time()
        prediction = fit_coxres_ensemble(
            job["train"], job["test"], horizons, SEED, args.smoke, run_ablation=True
        )
        frame = prediction_frame(job["test"], prediction, job, PROPOSED_MODEL)
        frame["result_origin"] = "sensitivity_efron_ablation"
        frame.to_csv(output, index=False)
        detail_path.write_text(
            json.dumps(
                {
                    "job_id": job["job_id"],
                    "strategy": job["strategy"],
                    "fold": str(job["fold"]),
                    "model": PROPOSED_MODEL,
                    "train_n": len(job["train"]),
                    "test_n": len(job["test"]),
                    "seed": SEED,
                    "elapsed_seconds": time.time() - started,
                    "model_details": prediction.details,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if args.smoke:
        return

    proposed = pd.concat(
        [
            pd.read_csv(path, dtype={"fold": str})
            for path in sorted(job_dir.glob(f"*__{PROPOSED_MODEL}.csv"))
        ],
        ignore_index=True,
    )
    primary = pd.read_csv(PREDICTION_DIR / "all_predictions.csv", dtype={"fold": str})
    comparators = primary[primary["model"] != PROPOSED_MODEL]
    combined = pd.concat([comparators, proposed], ignore_index=True)
    if set(combined["model"]) != set(MODEL_ORDER):
        raise ValueError("Sensitivity output must contain the same six-model scope as the primary analysis")
    combined.to_csv(prediction_dir / "all_predictions.csv", index=False)
    manifest = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(job_dir.glob(f"*__{PROPOSED_MODEL}.json"))
    ]
    (metrics_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("evaluate_results.py")),
            "--predictions",
            str(prediction_dir / "all_predictions.csv"),
            "--output-dir",
            str(metrics_dir),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("bootstrap_analysis.py")),
            "--predictions",
            str(prediction_dir / "all_predictions.csv"),
            "--output-dir",
            str(metrics_dir),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
