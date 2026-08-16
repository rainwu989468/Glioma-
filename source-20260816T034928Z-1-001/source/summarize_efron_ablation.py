"""Create compact, report-ready summaries for the Efron sensitivity analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL = "cox_residual_kg_attention_nohazard"
SENSITIVITY = ROOT / "results" / "sensitivity" / "efron_ablation"


def main() -> None:
    primary = pd.read_csv(ROOT / "results" / "metrics" / "performance_summary.csv")
    ablation = pd.read_csv(SENSITIVITY / "metrics" / "performance_summary.csv")
    columns = ["validation", "c_index", "rank_c_index", "auc_mean", "rank_auc", "ibs", "rank_ibs"]
    primary = primary[primary["model"] == MODEL][columns].set_index("validation")
    ablation = ablation[ablation["model"] == MODEL][columns].set_index("validation")
    comparison = primary.add_prefix("primary_").join(ablation.add_prefix("ablation_")).reset_index()
    for metric in ("c_index", "auc_mean", "ibs"):
        comparison[f"delta_{metric}"] = comparison[f"ablation_{metric}"] - comparison[f"primary_{metric}"]
    comparison.to_csv(SENSITIVITY / "metrics" / "performance_delta.csv", index=False)

    manifest = json.loads((SENSITIVITY / "metrics" / "training_manifest.json").read_text())
    scores = []
    selections = []
    strengths = []
    for job in manifest:
        details = job["model_details"]
        selected = details["selected_config"]
        selections.append({"job_id": job["job_id"], "selected_config": selected})
        for row in details["selection_scores"]:
            scores.append({"job_id": job["job_id"], "selected_config": selected, **row})
        for member in details["members"]:
            strengths.append({
                "job_id": job["job_id"],
                "selected_config": selected,
                "learned_prior_strength": member["learned_prior_strength"],
                "learned_other_strength": member["learned_other_strength"],
            })
    pd.DataFrame(scores).to_csv(SENSITIVITY / "metrics" / "selection_scores.csv", index=False)
    selection_frame = pd.DataFrame(selections)
    selection_frame.to_csv(SENSITIVITY / "metrics" / "selected_configurations.csv", index=False)
    selection_frame.value_counts("selected_config").rename("jobs_selected").reset_index().to_csv(
        SENSITIVITY / "metrics" / "selection_frequency.csv", index=False
    )
    pd.DataFrame(strengths).to_csv(SENSITIVITY / "metrics" / "learned_bias_strengths.csv", index=False)


if __name__ == "__main__":
    main()
