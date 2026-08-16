"""Refit all six models on corrected cohorts with resumable job outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import FEATURES, HORIZONS, ID_COL, MODEL_ORDER, PREDICTION_DIR, SEED, SPLIT_DIR, ensure_dirs
from models import fit_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jobs(split_root: Path):
    combined = split_root / "combined_cv"
    folders = [path for path in combined.glob("fold_*") if path.is_dir()]
    for folder in sorted(folders, key=lambda path: int(path.name.split("_")[-1])):
        fold = folder.name.split("_")[-1]
        yield {
            "job_id": f"combined_cv_fold_{fold}", "strategy": "combined_cv", "fold": fold,
            "train_cohort": "CGGA+TCGA", "test_cohort": "CGGA+TCGA",
            "train": pd.read_csv(folder / "train.csv"), "test": pd.read_csv(folder / "test.csv"),
        }
    external = split_root / "external_validation"
    for train_cohort, test_cohort in (("CGGA", "TCGA"), ("TCGA", "CGGA")):
        strategy = f"train_{train_cohort}_test_{test_cohort}"
        folder = external / strategy
        yield {
            "job_id": strategy, "strategy": strategy, "fold": "external",
            "train_cohort": train_cohort, "test_cohort": test_cohort,
            "train": pd.read_csv(folder / "train.csv"), "test": pd.read_csv(folder / "test.csv"),
        }


def prediction_frame(test: pd.DataFrame, prediction, job: dict, model: str) -> pd.DataFrame:
    columns = [
        ID_COL, "dataset", "age", "sex", "grade", "idh_mutant", "codel_1p19q",
        "idh_codel_subtype", "mgmt_methylated", "time", "event",
    ]
    frame = test[columns].copy()
    frame["strategy"] = job["strategy"]
    frame["fold"] = str(job["fold"])
    frame["model"] = model
    frame["original_model_id"] = model
    frame["train_cohort"] = job["train_cohort"]
    frame["test_cohort"] = job["test_cohort"]
    frame["risk_score"] = np.asarray(prediction.risk, dtype=float)
    for index, horizon in enumerate(HORIZONS):
        frame[f"survival_{int(horizon)}"] = np.asarray(prediction.survival[:, index], dtype=float)
    frame["result_origin"] = "refit_corrected_raw"
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", type=Path, default=SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=PREDICTION_DIR)
    parser.add_argument("--models", nargs="*", choices=MODEL_ORDER, default=list(MODEL_ORDER))
    parser.add_argument("--result-origin", default="refit_corrected_raw")
    parser.add_argument("--analysis-label", default="Full six-model refit on corrected outcome-eligible cohorts")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    metrics_dir = args.output_dir.parent / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    partial = args.output_dir / ("smoke_jobs" if args.smoke else "refit_jobs")
    partial.mkdir(parents=True, exist_ok=True)
    detail_rows = []
    horizons = np.asarray(HORIZONS, dtype=float)

    for job in jobs(args.split_root):
        train, test = job["train"], job["test"]
        for model in args.models:
            output = partial / f"{job['job_id']}__{model}.csv"
            detail_path = partial / f"{job['job_id']}__{model}.json"
            if output.exists() and detail_path.exists() and not args.force:
                print(f"SKIP {job['job_id']} {model}", flush=True)
                continue
            started = time.time()
            print(f"START {job['job_id']} {model} train={len(train)} test={len(test)}", flush=True)
            prediction = fit_model(model, train, test, horizons, SEED, args.smoke)
            frame = prediction_frame(test, prediction, job, model)
            frame["result_origin"] = args.result_origin
            frame.to_csv(output, index=False)
            details = {
                "job_id": job["job_id"], "strategy": job["strategy"], "fold": str(job["fold"]),
                "model": model, "train_n": len(train), "test_n": len(test), "seed": SEED,
                "elapsed_seconds": time.time() - started, "model_details": prediction.details,
            }
            detail_path.write_text(json.dumps(details, indent=2, default=str), encoding="utf-8")
            detail_rows.append(details)
            print(f"DONE {job['job_id']} {model} seconds={details['elapsed_seconds']:.1f}", flush=True)

    if args.smoke:
        return
    expected = {f"{job['job_id']}__{model}.csv" for job in jobs(args.split_root) for model in MODEL_ORDER}
    present = {path.name for path in partial.glob("*.csv")}
    missing = sorted(expected - present)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} experiment outputs: {missing[:5]}")
    frames = [pd.read_csv(partial / name, dtype={"fold": str}) for name in sorted(expected)]
    combined = pd.concat(frames, ignore_index=True)
    key = [ID_COL, "dataset", "strategy", "fold", "model"]
    if combined.duplicated(key).any():
        raise ValueError("Duplicate patient/model predictions detected")
    combined.to_csv(args.output_dir / "all_predictions.csv", index=False)
    details = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(partial.glob("*.json"))]
    (metrics_dir / "training_manifest.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )
    provenance = {
        "analysis": args.analysis_label,
        "result_origin": args.result_origin,
        "raw_files": {
            name: {
                "path": str(Path(__file__).resolve().parents[1] / "data" / "raw" / name),
                "sha256": sha256(Path(__file__).resolve().parents[1] / "data" / "raw" / name),
            }
            for name in ("CGGA_raw.xlsx", "TCGA_raw.xlsx")
        },
        "models": list(MODEL_ORDER),
        "validation": ["combined five-fold cross-validation", "CGGA to TCGA", "TCGA to CGGA"],
        "prediction_rows": int(len(combined)),
        "job_count": int(len(details)),
        "seed": SEED,
    }
    (metrics_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(combined)} corrected-data predictions", flush=True)


if __name__ == "__main__":
    main()
