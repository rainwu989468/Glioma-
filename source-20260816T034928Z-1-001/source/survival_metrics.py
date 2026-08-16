"""Comparable-pair C-index, cumulative/dynamic AUC, and integrated Brier score."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score
from sksurv.util import Surv

from config import EVENT_COL, HORIZONS, TIME_COL


def survival_y(df: pd.DataFrame) -> np.ndarray:
    return Surv.from_arrays(
        event=df[EVENT_COL].astype(bool).to_numpy(),
        time=df[TIME_COL].astype(float).to_numpy(),
    )


def concordance_index(time, event, risk) -> float:
    """Reference-paper comparable-pair concordance index.

    A pair is comparable when the earlier observed time is an event. Tied risk
    receives half credit. Higher risk denotes shorter survival.
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    r = np.asarray(risk, dtype=float)
    good = np.isfinite(t) & np.isfinite(r) & (t > 0)
    t, e, r = t[good], e[good], r[good]
    permissible = 0
    concordant = 0.0
    for i in range(len(t)):
        if e[i] != 1:
            continue
        later = t > t[i]
        permissible += int(later.sum())
        concordant += float((r[i] > r[later]).sum())
        concordant += 0.5 * float((r[i] == r[later]).sum())
    return float(concordant / permissible) if permissible else float("nan")


def valid_times(train_df: pd.DataFrame, test_df: pd.DataFrame, requested) -> np.ndarray:
    lower = max(float(train_df[TIME_COL].min()), float(test_df[TIME_COL].min())) + 1e-6
    upper = min(float(train_df[TIME_COL].max()), float(test_df[TIME_COL].max())) - 1e-6
    return np.asarray([float(t) for t in requested if lower < float(t) < upper], dtype=float)


def evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    risk: np.ndarray,
    survival: np.ndarray,
    survival_times: np.ndarray,
) -> dict:
    risk = np.asarray(risk, dtype=float)
    cindex = concordance_index(test_df[TIME_COL], test_df[EVENT_COL], risk)
    horizon_times = valid_times(train_df, test_df, HORIZONS)
    auc_values = np.full(len(HORIZONS), np.nan)
    if len(horizon_times):
        computed, _ = cumulative_dynamic_auc(survival_y(train_df), survival_y(test_df), risk, horizon_times)
        for time, value in zip(horizon_times, np.asarray(computed, dtype=float)):
            auc_values[list(HORIZONS).index(float(time))] = value

    grid = valid_times(train_df, test_df, survival_times)
    if len(grid) >= 2:
        indices = [int(np.argmin(np.abs(survival_times - t))) for t in grid]
        ibs = float(integrated_brier_score(
            survival_y(train_df),
            survival_y(test_df),
            np.clip(survival[:, indices], 1e-5, 1 - 1e-5),
            grid,
        ))
    else:
        ibs = float("nan")
    out = {
        "n": int(len(test_df)),
        "events": int(test_df[EVENT_COL].sum()),
        "c_index": cindex,
        "auc_mean": float(np.nanmean(auc_values)) if np.isfinite(auc_values).any() else float("nan"),
        "ibs": ibs,
    }
    for horizon, value in zip(HORIZONS, auc_values):
        out[f"auc_{int(horizon)}"] = float(value)
    return out


def bootstrap_ci(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    risk: np.ndarray,
    survival: np.ndarray,
    survival_times: np.ndarray,
    repeats: int,
    seed: int,
) -> dict:
    if repeats <= 0:
        return {}
    rng = np.random.default_rng(seed)
    values = {"c_index": [], "auc_mean": [], "ibs": []}
    for _ in range(repeats):
        idx = rng.integers(0, len(test_df), len(test_df))
        sampled = test_df.iloc[idx].reset_index(drop=True)
        try:
            row = evaluate(train_df, sampled, risk[idx], survival[idx], survival_times)
        except (ValueError, ZeroDivisionError):
            continue
        for key in values:
            if np.isfinite(row[key]):
                values[key].append(row[key])
    out = {}
    for key, array in values.items():
        if array:
            out[f"{key}_ci_low"] = float(np.percentile(array, 2.5))
            out[f"{key}_ci_high"] = float(np.percentile(array, 97.5))
    return out
