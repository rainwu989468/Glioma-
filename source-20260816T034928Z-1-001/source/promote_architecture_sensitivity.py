"""Promote a completed CoxRes-KGA architecture sensitivity run to primary results."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL = "cox_residual_kg_attention_nohazard"
SOURCE = ROOT / "results" / "sensitivity" / "d16_h1"
PRIMARY_PREDICTIONS = ROOT / "results" / "predictions"
PRIMARY_METRICS = ROOT / "results" / "metrics"


def main() -> None:
    source_jobs = SOURCE / "predictions" / "refit_jobs"
    target_jobs = PRIMARY_PREDICTIONS / "refit_jobs"
    for source in sorted(source_jobs.glob(f"*__{MODEL}.*")):
        target = target_jobs / source.name
        if source.suffix == ".csv":
            frame = pd.read_csv(source, dtype={"fold": str})
            frame["result_origin"] = "refit_corrected_raw"
            frame.to_csv(target, index=False)
        else:
            shutil.copy2(source, target)

    primary = pd.read_csv(PRIMARY_PREDICTIONS / "all_predictions.csv", dtype={"fold": str})
    alternative = pd.read_csv(SOURCE / "predictions" / "all_predictions.csv", dtype={"fold": str})
    proposed = alternative.loc[alternative["model"].eq(MODEL)].copy()
    proposed["result_origin"] = "refit_corrected_raw"
    combined = pd.concat([primary.loc[~primary["model"].eq(MODEL)], proposed], ignore_index=True)
    combined.to_csv(PRIMARY_PREDICTIONS / "all_predictions.csv", index=False)

    manifest_path = PRIMARY_METRICS / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    alternative_manifest = json.loads(
        (SOURCE / "metrics" / "training_manifest.json").read_text(encoding="utf-8")
    )
    replacements = {(job["job_id"], job["model"]): job for job in alternative_manifest}
    promoted = [replacements.get((job["job_id"], job["model"]), job) for job in manifest]
    if sum(job["model"] == MODEL for job in promoted) != 7:
        raise ValueError("Primary manifest must contain seven promoted CoxRes-KGA jobs")
    manifest_path.write_text(json.dumps(promoted, indent=2), encoding="utf-8")
    print("Promoted 16d/1-head CoxRes-KGA predictions and seven training manifests")


if __name__ == "__main__":
    main()
