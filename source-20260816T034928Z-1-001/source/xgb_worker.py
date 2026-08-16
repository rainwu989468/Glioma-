"""Isolated XGBoost AFT worker.

XGBoost and scikit-learn load conflicting OpenMP runtimes in this macOS
environment. Running XGBoost in a clean subprocess prevents a native crash.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.sparse import csr_matrix
import xgboost as xgb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    data = np.load(args.input)
    dtrain = xgb.DMatrix(csr_matrix(data["x_train"]))
    dtest = xgb.DMatrix(csr_matrix(data["x_test"]))
    dtrain.set_float_info("label_lower_bound", data["lower"])
    dtrain.set_float_info("label_upper_bound", data["upper"])
    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.2,
        "tree_method": "hist",
        "grow_policy": "lossguide",
        "max_depth": 0,
        "max_leaves": 15,
        "eta": 0.04,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "min_child_weight": 1,
        "lambda": 0.01,
        "seed": args.seed,
        "nthread": 4,
    }
    model = xgb.train(params, dtrain, num_boost_round=args.rounds, verbose_eval=False)
    np.save(args.output, np.asarray(model.predict(dtest), dtype=float))


if __name__ == "__main__":
    main()
