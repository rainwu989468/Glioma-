"""Run leakage-safe Cox-penalty and loss-balance tuning for CoxRes-KGA."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import HORIZONS, MODEL_ORDER, PREDICTION_DIR, SEED, SPLIT_DIR
from models import RANK_TUNING_CONFIGS, fit_coxres_ensemble
from run_experiments import jobs, prediction_frame


PROPOSED_MODEL = "cox_residual_kg_attention_nohazard"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "sensitivity" / "rank_tuning",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--token-dimension", type=int, default=16)
    parser.add_argument("--attention-heads", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adamw")
    parser.add_argument("--max-epochs", type=int, default=170)
    parser.add_argument("--nonlinear-cox", action="store_true")
    parser.add_argument("--ensemble-mode", choices=("seed", "crossfit"), default="seed")
    parser.add_argument("--orthogonality-penalty", type=float, default=0.0)
    args = parser.parse_args()
    if args.token_dimension < 1 or args.attention_heads < 1 or args.token_dimension % args.attention_heads:
        parser.error("--token-dimension must be positive and divisible by --attention-heads")
    if args.orthogonality_penalty < 0:
        parser.error("--orthogonality-penalty must be nonnegative")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in the interval [0, 1)")
    if args.max_epochs < 70:
        parser.error("--max-epochs must be at least 70")
    candidates = tuple(
        replace(
            config,
            name=(
                f"{config.name}_orth{args.orthogonality_penalty:g}"
                if args.orthogonality_penalty > 0
                else config.name
            ),
            orthogonality_penalty=args.orthogonality_penalty,
        )
        for config in RANK_TUNING_CONFIGS
    )

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
            print(f"SKIP {job['job_id']}", flush=True)
            continue
        started = time.time()
        print(f"START {job['job_id']} train={len(job['train'])} test={len(job['test'])}", flush=True)
        prediction = fit_coxres_ensemble(
            job["train"],
            job["test"],
            horizons,
            SEED,
            args.smoke,
            run_ablation=True,
            candidate_configs=candidates,
            selection_label="Cox penalty and loss balance selected within outer training data",
            token_dimension=args.token_dimension,
            attention_heads=args.attention_heads,
            dropout=args.dropout,
            optimizer_name=args.optimizer,
            nonlinear_cox=args.nonlinear_cox,
            ensemble_mode=args.ensemble_mode,
            max_epochs=args.max_epochs,
        )
        frame = prediction_frame(job["test"], prediction, job, PROPOSED_MODEL)
        epoch_suffix = "" if args.max_epochs == 170 else f"_maxep{args.max_epochs}"
        frame["result_origin"] = (
            f"structured_improvement_d{args.token_dimension}_h{args.attention_heads}"
            f"_nonlinear{int(args.nonlinear_cox)}_{args.ensemble_mode}"
            f"_orth{args.orthogonality_penalty:g}_drop{args.dropout:g}_{args.optimizer}{epoch_suffix}"
        )
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
        print(f"DONE {job['job_id']} seconds={time.time() - started:.1f}", flush=True)

    if args.smoke:
        return

    tuned = pd.concat(
        [
            pd.read_csv(path, dtype={"fold": str})
            for path in sorted(job_dir.glob(f"*__{PROPOSED_MODEL}.csv"))
        ],
        ignore_index=True,
    )
    primary = pd.read_csv(PREDICTION_DIR / "all_predictions.csv", dtype={"fold": str})
    comparators = primary[primary["model"] != PROPOSED_MODEL]
    combined = pd.concat([comparators, tuned], ignore_index=True)
    if set(combined["model"]) != set(MODEL_ORDER):
        raise ValueError("Rank-tuning output must contain the same six-model scope as the primary analysis")
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
