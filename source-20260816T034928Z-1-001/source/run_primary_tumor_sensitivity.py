"""Run the six-model analysis after restricting both cohorts to primary tumors."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from config import EVENT_COL, FEATURES, ID_COL, RAW_DIR, TIME_COL
from prepare_data import clean_cgga, clean_tcga, normalize_missing, summarize, write_splits

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
DEFAULT_OUTPUT = ROOT / "results" / "sensitivity" / "primary_tumor_only"


def primary_tcga_ids(path: Path) -> set[str]:
    """Return patients whose TCGA barcode denotes a primary solid tumor (code 01)."""

    raw = normalize_missing(pd.read_excel(path, usecols=["Patient ID", "Sample ID"]))
    sample_type = raw["Sample ID"].astype("string").str.split("-").str[3].str[:2]
    unexpected = sorted(sample_type.dropna().loc[~sample_type.isin(["01"])].unique())
    if unexpected:
        raise ValueError(f"Unexpected TCGA sample-type codes: {unexpected}")
    return set(raw.loc[sample_type.eq("01"), "Patient ID"].dropna().astype(str))


def prepare_subset(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create an audited harmonized subset without modifying primary analysis data."""

    cgga, _ = clean_cgga(RAW_DIR / "CGGA_raw.xlsx")
    tcga, _ = clean_tcga(RAW_DIR / "TCGA_raw.xlsx")

    prs = cgga["prs_type"].astype("string").str.strip().str.lower()
    cgga_primary = cgga.loc[prs.eq("primary")].copy()
    tcga_ids = primary_tcga_ids(RAW_DIR / "TCGA_raw.xlsx")
    tcga_primary = tcga.loc[tcga[ID_COL].astype(str).isin(tcga_ids)].copy()

    if len(cgga_primary) != 404:
        raise ValueError(f"Expected 404 eligible primary CGGA tumors, found {len(cgga_primary)}")
    if len(tcga_primary) != len(tcga):
        raise ValueError("Not every eligible TCGA record has primary-solid-tumor sample code 01")

    data_dir = output_root / "data"
    split_dir = output_root / "splits"
    data_dir.mkdir(parents=True, exist_ok=True)
    harmonized_columns = [ID_COL, "dataset", *FEATURES, TIME_COL, EVENT_COL]
    harmonized = pd.concat(
        [cgga_primary[harmonized_columns], tcga_primary[harmonized_columns]], ignore_index=True
    )
    harmonized.to_csv(data_dir / "glioma_primary_tumor_harmonized.csv", index=False)
    pd.DataFrame([summarize(cgga_primary), summarize(tcga_primary)]).to_csv(
        data_dir / "dataset_summary.csv", index=False
    )
    pd.DataFrame([
        {"dataset": "CGGA", "stage": "Outcome-eligible records", "n": len(cgga)},
        {"dataset": "CGGA", "stage": "Excluded recurrent tumors", "n": len(cgga) - len(cgga_primary)},
        {"dataset": "CGGA", "stage": "Primary-tumor sensitivity records", "n": len(cgga_primary)},
        {"dataset": "TCGA", "stage": "Outcome-eligible records", "n": len(tcga)},
        {"dataset": "TCGA", "stage": "Excluded non-primary sample types", "n": len(tcga) - len(tcga_primary)},
        {"dataset": "TCGA", "stage": "Primary-tumor sensitivity records", "n": len(tcga_primary)},
    ]).to_csv(data_dir / "eligibility_audit.csv", index=False)
    (data_dir / "provenance.json").write_text(json.dumps({
        "analysis": "Primary-tumor-only sensitivity analysis",
        "cgga_rule": "PRS_type equals Primary after trimming and case normalization",
        "tcga_rule": "TCGA barcode sample-type code equals 01 (primary solid tumor)",
        "rows": {"CGGA": len(cgga_primary), "TCGA": len(tcga_primary)},
    }, indent=2), encoding="utf-8")
    write_splits(harmonized, split_dir)
    return cgga_primary, tcga_primary


def run(arguments: list[str]) -> None:
    print("+", " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args()

    prepare_subset(args.output_root)
    predictions = args.output_root / "predictions"
    metrics = args.output_root / "metrics"
    experiment_args = [
        sys.executable, str(SOURCE / "run_experiments.py"),
        "--split-root", str(args.output_root / "splits"),
        "--output-dir", str(predictions),
        "--result-origin", "primary_tumor_only_sensitivity",
        "--analysis-label", "Six-model primary-tumor-only sensitivity analysis",
    ]
    if args.force:
        experiment_args.append("--force")
    run(experiment_args)
    run([sys.executable, str(SOURCE / "evaluate_results.py"),
         "--predictions", str(predictions / "all_predictions.csv"), "--output-dir", str(metrics)])
    run([sys.executable, str(SOURCE / "bootstrap_analysis.py"),
         "--predictions", str(predictions / "all_predictions.csv"), "--output-dir", str(metrics),
         "--repeats", str(args.bootstrap_repeats)])
    run([sys.executable, str(SOURCE / "compare_primary_tumor_sensitivity.py")])


if __name__ == "__main__":
    main()
