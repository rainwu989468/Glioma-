"""Clean the corrected CGGA and TCGA workbooks and create validation splits.

Raw workbooks are read-only inputs. Predictor missingness is retained; only
records without a usable patient identifier, survival time, or event status are
excluded. All transformations are recorded in ``cleaning_audit.csv``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import EVENT_COL, FEATURES, ID_COL, PROCESSED_DIR, RAW_DIR, SEED, SPLIT_DIR, TIME_COL

MISSING_TEXT = {"", "na", "n/a", "nan", "none", "null", "missing", "?", "unknown"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_missing(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    for column in frame.columns:
        if frame[column].dtype == object or pd.api.types.is_string_dtype(frame[column]):
            values = frame[column].astype("string").str.strip()
            frame[column] = values.mask(values.str.lower().isin(MISSING_TEXT), pd.NA)
    return frame


def map_binary(values: pd.Series, positive: set[str], negative: set[str]) -> pd.Series:
    text = values.astype("string").str.strip().str.lower()
    result = pd.Series(np.nan, index=values.index, dtype=float)
    result.loc[text.isin(positive)] = 1.0
    result.loc[text.isin(negative)] = 0.0
    numeric = pd.to_numeric(values, errors="coerce")
    result.loc[numeric.isin([0, 1])] = numeric.loc[numeric.isin([0, 1])]
    return result


def derive_subtype(idh: pd.Series, codel: pd.Series, supplied: pd.Series | None = None) -> pd.Series:
    output = supplied.astype("string").copy() if supplied is not None else pd.Series(pd.NA, index=idh.index, dtype="string")
    normalized = output.str.strip().str.lower()
    canonical = pd.Series(pd.NA, index=idh.index, dtype="string")
    canonical.loc[normalized.str.contains("idhwt", na=False) | normalized.eq("wt")] = "IDHwt"
    canonical.loc[normalized.str.contains("mut-codel", na=False)] = "IDHmut-codel"
    canonical.loc[normalized.str.contains("mut-non-codel", na=False)] = "IDHmut-non-codel"
    canonical.loc[canonical.isna() & idh.eq(0)] = "IDHwt"
    canonical.loc[canonical.isna() & idh.eq(1) & codel.eq(1)] = "IDHmut-codel"
    canonical.loc[canonical.isna() & idh.eq(1) & codel.eq(0)] = "IDHmut-non-codel"
    return canonical


def add_missingness(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("age", "grade", "idh_mutant", "codel_1p19q", "idh_codel_subtype", "mgmt_methylated"):
        frame[f"{column}_missing"] = frame[column].isna().astype(int)
    return frame


def clean_cgga(path: Path) -> tuple[pd.DataFrame, list[dict]]:
    raw = normalize_missing(pd.read_excel(path))
    time_days = pd.to_numeric(raw["OS"], errors="coerce")
    event = pd.to_numeric(raw["Censor (alive=0; dead=1)"], errors="coerce")
    patient = raw["CGGA_ID"].astype("string")
    eligible = patient.notna() & time_days.notna() & event.isin([0, 1]) & (time_days > 0)

    idh = map_binary(raw["IDH_mutation_status"], {"mutant", "mutated"}, {"wildtype", "wild type", "wt"})
    codel = map_binary(raw["1p19q_codeletion_status"], {"codel", "codeleted", "co-deleted"}, {"non-codel", "noncodel"})
    mgmt = map_binary(raw["MGMTp_methylation_status"], {"methylated"}, {"un-methylated", "unmethylated"})
    grade = raw["Grade"].astype("string").str.extract(r"(IV|III|II|4|3|2)", expand=False).map(
        {"II": 2.0, "III": 3.0, "IV": 4.0, "2": 2.0, "3": 3.0, "4": 4.0}
    )
    frame = pd.DataFrame({
        ID_COL: patient,
        "dataset": "CGGA",
        "age": pd.to_numeric(raw["Age"], errors="coerce"),
        "sex": raw["Gender"].astype("string").str.title(),
        "grade": grade.astype(float),
        "idh_mutant": idh,
        "codel_1p19q": codel,
        "idh_codel_subtype": derive_subtype(idh, codel),
        "mgmt_methylated": mgmt,
        TIME_COL: time_days / 30.44,
        EVENT_COL: event,
        "prs_type": raw["PRS_type"].astype("string"),
        "histology_cgga": raw["Histology"].astype("string"),
    }).loc[eligible].reset_index(drop=True)
    frame[EVENT_COL] = frame[EVENT_COL].astype(int)
    frame = add_missingness(frame)
    audit = [
        {"dataset": "CGGA", "stage": "Raw records", "n": len(raw)},
        {"dataset": "CGGA", "stage": "Excluded: missing/nonpositive OS or missing event", "n": int((~eligible).sum())},
        {"dataset": "CGGA", "stage": "Outcome-eligible records", "n": len(frame)},
    ]
    return frame, audit


def clean_tcga(path: Path) -> tuple[pd.DataFrame, list[dict]]:
    raw = normalize_missing(pd.read_excel(path))
    time = pd.to_numeric(raw["Overall Survival (Months)"], errors="coerce")
    status = raw["Overall Survival Status"].astype("string").str.lower()
    event = pd.Series(np.nan, index=raw.index, dtype=float)
    event.loc[status.str.contains("deceased", na=False)] = 1.0
    event.loc[status.str.contains("living", na=False)] = 0.0
    patient = raw["Patient ID"].astype("string")
    eligible = patient.notna() & time.notna() & event.isin([0, 1]) & (time > 0)

    idh = map_binary(raw["IDH status"], {"mutant", "mutated"}, {"wt", "wildtype", "wild type"})
    supplied = raw["IDH/codel subtype"].astype("string")
    fallback = raw["IDH-1P10Q Subtype"].astype("string")
    subtype_text = supplied.fillna(fallback).str.lower()
    codel = pd.Series(np.nan, index=raw.index, dtype=float)
    codel.loc[subtype_text.str.contains("codel", na=False) & ~subtype_text.str.contains("non", na=False)] = 1.0
    codel.loc[subtype_text.str.contains("non", na=False) | subtype_text.str.contains("idhwt|wt", na=False)] = 0.0
    mgmt = map_binary(raw["MGMT promoter status"], {"methylated"}, {"unmethylated", "un-methylated"})
    grade = pd.to_numeric(raw["Neoplasm Histologic Grade"].astype("string").str.extract(r"([234])", expand=False), errors="coerce")

    frame = pd.DataFrame({
        ID_COL: patient,
        "dataset": "TCGA",
        "age": pd.to_numeric(raw["Diagnosis Age"], errors="coerce"),
        "sex": raw["Sex"].astype("string").str.title(),
        "grade": grade,
        "idh_mutant": idh,
        "codel_1p19q": codel,
        "idh_codel_subtype": derive_subtype(idh, codel, supplied),
        "mgmt_methylated": mgmt,
        TIME_COL: time,
        EVENT_COL: event,
        "tcga_atrx_status": raw["ATRX status"].astype("string"),
        "tcga_tert_promoter_status": raw["TERT promoter status"].astype("string"),
        "tcga_mutation_count": pd.to_numeric(raw["Mutation Count"], errors="coerce"),
        "tcga_tmb_nonsynonymous": pd.to_numeric(raw["TMB (nonsynonymous)"], errors="coerce"),
        "tcga_percent_aneuploidy": pd.to_numeric(raw["Percent aneuploidy"], errors="coerce"),
        "tcga_absolute_purity": pd.to_numeric(raw["Absolute Purity"], errors="coerce"),
        "tcga_transcriptome_subtype": raw["Transcriptome Subtype"].astype("string"),
        "tcga_pan-glioma_dna_methylation_cluster": raw["Pan-Glioma DNA Methylation Cluster"].astype("string"),
        "tcga_pan-glioma_rna_expression_cluster": raw["Pan-Glioma RNA Expression Cluster"].astype("string"),
    }).loc[eligible].reset_index(drop=True)
    frame[EVENT_COL] = frame[EVENT_COL].astype(int)
    frame = add_missingness(frame)
    if frame[ID_COL].duplicated().any():
        raise ValueError("TCGA contains duplicate eligible patient identifiers")
    audit = [
        {"dataset": "TCGA", "stage": "Raw records", "n": len(raw)},
        {"dataset": "TCGA", "stage": "Excluded: missing/nonpositive OS or missing event", "n": int((~eligible).sum())},
        {"dataset": "TCGA", "stage": "Outcome-eligible records", "n": len(frame)},
    ]
    return frame, audit


def summarize(frame: pd.DataFrame) -> dict:
    male = int(frame["sex"].eq("Male").sum())
    female = int(frame["sex"].eq("Female").sum())
    return {
        "dataset": frame["dataset"].iloc[0], "n": len(frame), "events": int(frame[EVENT_COL].sum()),
        "censored": int((1 - frame[EVENT_COL]).sum()), "event_rate": float(frame[EVENT_COL].mean()),
        "male": male, "female": female, "age_mean": float(frame["age"].mean()),
        "age_sd": float(frame["age"].std(ddof=1)), "os_median_months": float(frame[TIME_COL].median()),
        "grade_2": int(frame["grade"].eq(2).sum()), "grade_3": int(frame["grade"].eq(3).sum()),
        "grade_4": int(frame["grade"].eq(4).sum()), "idh_mutant": int(frame["idh_mutant"].sum(skipna=True)),
        "idh_observed": int(frame["idh_mutant"].notna().sum()),
        "codel_1p19q": int(frame["codel_1p19q"].sum(skipna=True)),
        "codel_observed": int(frame["codel_1p19q"].notna().sum()),
        "mgmt_methylated": int(frame["mgmt_methylated"].sum(skipna=True)),
        "mgmt_observed": int(frame["mgmt_methylated"].notna().sum()),
    }


def write_splits(harmonized: pd.DataFrame, split_dir: Path = SPLIT_DIR) -> None:
    """Write deterministic internal and external splits below ``split_dir``."""

    combined = split_dir / "combined_cv"
    combined.mkdir(parents=True, exist_ok=True)
    strata = (
        harmonized["dataset"].astype(str) + "_e" + harmonized[EVENT_COL].astype(str)
        + "_g" + harmonized["grade"].fillna(-1).astype(int).astype(str)
    )
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    assignments = pd.Series(index=harmonized.index, dtype=int)
    for fold, (_, test_index) in enumerate(splitter.split(harmonized, strata), start=1):
        assignments.iloc[test_index] = fold
        folder = combined / f"fold_{fold}"
        folder.mkdir(exist_ok=True)
        harmonized.loc[assignments.index.difference(test_index)].to_csv(folder / "train.csv", index=False)
        harmonized.iloc[test_index].to_csv(folder / "test.csv", index=False)
    assignment_frame = harmonized[[ID_COL, "dataset", EVENT_COL, "grade"]].copy()
    assignment_frame["fold"] = assignments.astype(int)
    assignment_frame.to_csv(combined / "fold_assignments.csv", index=False)

    external = split_dir / "external_validation"
    for train_name, test_name in (("CGGA", "TCGA"), ("TCGA", "CGGA")):
        folder = external / f"train_{train_name}_test_{test_name}"
        folder.mkdir(parents=True, exist_ok=True)
        harmonized[harmonized["dataset"].eq(train_name)].to_csv(folder / "train.csv", index=False)
        harmonized[harmonized["dataset"].eq(test_name)].to_csv(folder / "test.csv", index=False)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    cgga_path = RAW_DIR / "CGGA_raw.xlsx"
    tcga_path = RAW_DIR / "TCGA_raw.xlsx"
    cgga, cgga_audit = clean_cgga(cgga_path)
    tcga, tcga_audit = clean_tcga(tcga_path)
    harmonized_columns = [ID_COL, "dataset", *FEATURES, TIME_COL, EVENT_COL]
    harmonized = pd.concat([cgga[harmonized_columns], tcga[harmonized_columns]], ignore_index=True)

    cgga.to_csv(PROCESSED_DIR / "cgga_clean_enriched.csv", index=False)
    tcga.to_csv(PROCESSED_DIR / "tcga_clean_enriched.csv", index=False)
    harmonized.to_csv(PROCESSED_DIR / "glioma_harmonized.csv", index=False)
    pd.DataFrame([summarize(cgga), summarize(tcga)]).to_csv(PROCESSED_DIR / "dataset_summary.csv", index=False)
    missing = []
    for dataset, frame in (("CGGA", cgga), ("TCGA", tcga)):
        for variable in FEATURES:
            missing.append({"dataset": dataset, "variable": variable, "missing_n": int(frame[variable].isna().sum()), "missing_pct": float(frame[variable].isna().mean() * 100)})
    pd.DataFrame(missing).to_csv(PROCESSED_DIR / "missingness_summary.csv", index=False)
    pd.DataFrame(cgga_audit + tcga_audit).to_csv(PROCESSED_DIR / "cleaning_audit.csv", index=False)
    provenance = {
        "cleaning_policy": "Outcome eligibility only; predictor missingness retained with explicit indicators.",
        "cgga_os_conversion": "days / 30.44",
        "raw_files": {str(cgga_path): sha256(cgga_path), str(tcga_path): sha256(tcga_path)},
        "cleaned_rows": {"CGGA": len(cgga), "TCGA": len(tcga)},
        "seed": SEED,
        "fold_stratification": "dataset x event x WHO grade",
    }
    (PROCESSED_DIR / "cleaning_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    write_splits(harmonized)
    print(f"CGGA: {len(cgga)} patients, {int(cgga[EVENT_COL].sum())} events")
    print(f"TCGA: {len(tcga)} patients, {int(tcga[EVENT_COL].sum())} events")


if __name__ == "__main__":
    main()
