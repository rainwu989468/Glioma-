"""Rebuild the corrected-cohort analysis from raw data through final report."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
def run(arguments: list[str]) -> None:
    print("+", " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=ROOT, check=True)


def main() -> None:
    run([sys.executable, str(SOURCE / "prepare_data.py")])
    run([sys.executable, str(SOURCE / "run_experiments.py")])
    run([sys.executable, str(SOURCE / "evaluate_results.py")])
    run([sys.executable, str(SOURCE / "bootstrap_analysis.py")])
    run([
        sys.executable, str(SOURCE / "run_coxres_rank_tuning.py"),
        "--output-root", str(ROOT / "results" / "sensitivity" / "rank_tuning"),
        "--token-dimension", "32", "--attention-heads", "1", "--force",
    ])
    run([
        sys.executable, str(SOURCE / "run_coxres_rank_tuning.py"),
        "--output-root", str(ROOT / "results" / "sensitivity" / "d16_h2"),
        "--token-dimension", "16", "--attention-heads", "2", "--force",
    ])
    run([sys.executable, str(SOURCE / "compare_architecture_sensitivity.py")])
    run([sys.executable, str(SOURCE / "bootstrap_architecture_sensitivity.py")])
    run([
        sys.executable, str(SOURCE / "run_coxres_rank_tuning.py"),
        "--output-root", str(ROOT / "results" / "sensitivity" / "structured_A_nonlinear"),
        "--nonlinear-cox", "--force",
    ])
    run([
        sys.executable, str(SOURCE / "run_coxres_rank_tuning.py"),
        "--output-root", str(ROOT / "results" / "sensitivity" / "structured_B_crossfit"),
        "--nonlinear-cox", "--ensemble-mode", "crossfit", "--force",
    ])
    run([
        sys.executable, str(SOURCE / "run_coxres_rank_tuning.py"),
        "--output-root", str(ROOT / "results" / "sensitivity" / "structured_C_orthogonal"),
        "--nonlinear-cox", "--ensemble-mode", "crossfit",
        "--orthogonality-penalty", "0.01", "--force",
    ])
    run([sys.executable, str(SOURCE / "compare_structured_improvements.py")])
    run([
        sys.executable, str(SOURCE / "run_primary_tumor_sensitivity.py"),
        "--force",
    ])
    run([
        sys.executable, str(SOURCE / "run_coxres_rank_tuning.py"),
        "--output-root", str(ROOT / "results" / "sensitivity" / "dropout040_adam"),
        "--token-dimension", "16", "--attention-heads", "1",
        "--dropout", "0.4", "--optimizer", "adam", "--force",
    ])
    run([sys.executable, str(SOURCE / "compare_dropout_adam_sensitivity.py")])
    run([
        sys.executable, str(SOURCE / "run_coxres_rank_tuning.py"),
        "--output-root", str(ROOT / "results" / "sensitivity" / "dropout050_adamw_h2"),
        "--token-dimension", "16", "--attention-heads", "2",
        "--dropout", "0.5", "--optimizer", "adamw", "--force",
    ])
    run([sys.executable, str(SOURCE / "compare_dropout050_adamw_h2.py")])
    run([
        sys.executable, str(SOURCE / "run_coxres_rank_tuning.py"),
        "--output-root", str(ROOT / "results" / "sensitivity" / "dropout060_adam"),
        "--token-dimension", "16", "--attention-heads", "1",
        "--dropout", "0.6", "--optimizer", "adam", "--force",
    ])
    run([sys.executable, str(SOURCE / "compare_dropout060_adam.py")])
    run([
        sys.executable, str(SOURCE / "run_coxres_rank_tuning.py"),
        "--output-root", str(ROOT / "results" / "sensitivity" / "dropout050_adamw_h1"),
        "--token-dimension", "16", "--attention-heads", "1",
        "--dropout", "0.5", "--optimizer", "adamw", "--force",
    ])
    run([sys.executable, str(SOURCE / "compare_dropout050_adamw_h1.py")])
    run([
        sys.executable, str(SOURCE / "run_coxres_rank_tuning.py"),
        "--output-root", str(ROOT / "results" / "sensitivity" / "d32_e120_adam"),
        "--token-dimension", "32", "--attention-heads", "1",
        "--dropout", "0.4", "--optimizer", "adam", "--max-epochs", "120", "--force",
    ])
    run([sys.executable, str(SOURCE / "compare_d32_e120_adam.py")])
    run([sys.executable, str(SOURCE / "summarize_efron_ablation.py")])
    run(["/usr/local/bin/Rscript", str(SOURCE / "generate_figures.R")])
    run([sys.executable, str(SOURCE / "generate_report.py")])


if __name__ == "__main__":
    main()
