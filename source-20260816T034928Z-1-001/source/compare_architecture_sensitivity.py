"""Compare the prespecified CoxRes-KGA architecture sensitivity runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL = "cox_residual_kg_attention_nohazard"
RUNS = {
    "32d/1h": ROOT / "results" / "sensitivity" / "rank_tuning" / "metrics",
    "16d/1h": ROOT / "results" / "metrics",
    "16d/2h": ROOT / "results" / "sensitivity" / "d16_h2" / "metrics",
}


def main() -> None:
    summary_rows = []
    fold_rows = []
    for architecture, metrics_dir in RUNS.items():
        performance = pd.read_csv(metrics_dir / "performance_summary.csv")
        proposed = performance.loc[performance["model"].eq(MODEL)].copy()
        proposed.insert(0, "architecture", architecture)
        summary_rows.append(
            proposed[
                [
                    "architecture",
                    "validation",
                    "c_index",
                    "rank_c_index",
                    "auc_mean",
                    "rank_auc",
                    "ibs",
                    "rank_ibs",
                ]
            ]
        )

        folds = pd.read_csv(metrics_dir / "fold_metrics.csv")
        folds = folds.loc[folds["model"].eq(MODEL) & folds["strategy"].eq("combined_cv")]
        fold_rows.append(
            {
                "architecture": architecture,
                "c_index_mean": folds["c_index"].mean(),
                "c_index_sd": folds["c_index"].std(ddof=1),
                "auc_mean": folds["auc_mean"].mean(),
                "auc_sd": folds["auc_mean"].std(ddof=1),
                "ibs_mean": folds["ibs"].mean(),
                "ibs_sd": folds["ibs"].std(ddof=1),
            }
        )

    output_dir = ROOT / "results" / "sensitivity" / "architecture_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.concat(summary_rows, ignore_index=True).sort_values(["validation", "architecture"])
    fold_summary = pd.DataFrame(fold_rows)
    summary.to_csv(output_dir / "performance_summary.csv", index=False)
    fold_summary.to_csv(output_dir / "internal_fold_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\nInternal fold mean and standard deviation")
    print(fold_summary.to_string(index=False))


if __name__ == "__main__":
    main()
