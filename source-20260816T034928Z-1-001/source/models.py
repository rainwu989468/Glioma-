"""Survival baselines and the Cox-residual knowledge-guided attention model."""

from __future__ import annotations

import copy
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM

from config import EVENT_COL, FEATURES, SEED, TIME_COL
from survival_metrics import concordance_index, survival_y


def preprocessor(*, scale_numeric: bool = True) -> ColumnTransformer:
    numeric = [x for x in FEATURES if x not in {"sex", "idh_codel_subtype"}]
    categorical = ["sex", "idh_codel_subtype"]
    return ColumnTransformer(
        [
            ("num", Pipeline(
                [("impute", SimpleImputer(strategy="median"))]
                + ([("scale", StandardScaler())] if scale_numeric else [])
            ), numeric),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical),
        ],
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )


def nonlinear_cox_preprocessor() -> ColumnTransformer:
    """Training-fitted Cox design with nonlinear age and categorical grade effects."""
    ordinary_numeric = [
        feature for feature in FEATURES
        if feature not in {"age", "grade", "sex", "idh_codel_subtype"}
    ]
    return ColumnTransformer(
        [
            (
                "age_spline",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    (
                        "spline",
                        SplineTransformer(
                            n_knots=4,
                            degree=3,
                            include_bias=False,
                            extrapolation="linear",
                        ),
                    ),
                    ("scale", StandardScaler()),
                ]),
                ["age"],
            ),
            (
                "grade_levels",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                ["grade"],
            ),
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                ordinary_numeric,
            ),
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                ["sex", "idh_codel_subtype"],
            ),
        ],
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )


def interpolate_functions(functions, times: np.ndarray) -> np.ndarray:
    return np.vstack([np.asarray(fn(times), dtype=float) for fn in functions])


def breslow_survival(train_df: pd.DataFrame, train_risk: np.ndarray, test_risk: np.ndarray, times: np.ndarray) -> np.ndarray:
    time = train_df[TIME_COL].to_numpy(dtype=float)
    event = train_df[EVENT_COL].to_numpy(dtype=int)
    center = float(np.mean(train_risk))
    train_exp = np.exp(np.clip(train_risk - center, -12, 12))
    test_exp = np.exp(np.clip(test_risk - center, -12, 12))
    event_times = np.sort(np.unique(time[event == 1]))
    increments = []
    for value in event_times:
        denom = train_exp[time >= value].sum()
        if denom > 0:
            increments.append((value, float(((time == value) & (event == 1)).sum() / denom)))
    base = []
    cumulative = 0.0
    pos = 0
    for horizon in times:
        while pos < len(increments) and increments[pos][0] <= horizon:
            cumulative += increments[pos][1]
            pos += 1
        base.append(cumulative)
    return np.exp(-np.asarray(base)[None, :] * test_exp[:, None])


@dataclass
class Prediction:
    risk: np.ndarray
    survival: np.ndarray
    times: np.ndarray
    details: dict


@dataclass(frozen=True)
class CoxResConfig:
    """Prespecified CoxRes-KGA setting eligible for training-only selection."""

    name: str
    cox_alpha: float
    residual_scale_init: float
    ranking_weight: float
    residual_penalty: float
    learnable_knowledge_bias: bool
    knowledge_anchor_penalty: float
    tie_method: str
    orthogonality_penalty: float = 0.0


PRIMARY_COXRES_CONFIG = CoxResConfig(
    "primary_fixed_bias", 0.10, float(torch.nn.functional.softplus(torch.tensor(-1.2))),
    0.16, 0.010, False, 0.0, "legacy",
)
COXRES_CONFIGS = (
    CoxResConfig("efron_fixed", 0.10, 0.25, 0.16, 0.010, False, 0.0, "efron"),
    CoxResConfig("efron_learnable", 0.10, 0.25, 0.16, 0.010, True, 0.05, "efron"),
    CoxResConfig("efron_shrunk", 0.30, 0.10, 0.12, 0.020, True, 0.05, "efron"),
    CoxResConfig("efron_flexible", 0.03, 0.50, 0.16, 0.005, True, 0.05, "efron"),
)

# Fixed-architecture candidates for the discrimination-focused sensitivity run.
# The grid changes only the Cox offset penalty and neural loss balance.  It keeps
# the publication architecture, knowledge bias, residual initialization, and tie
# handling unchanged so that the experiment isolates the proposed tuning axes.
RANK_TUNING_CONFIGS = (
    CoxResConfig("rank_a001_r016_p010", 0.01, PRIMARY_COXRES_CONFIG.residual_scale_init, 0.16, 0.010, False, 0.0, "legacy"),
    CoxResConfig("rank_a005_r016_p010", 0.05, PRIMARY_COXRES_CONFIG.residual_scale_init, 0.16, 0.010, False, 0.0, "legacy"),
    PRIMARY_COXRES_CONFIG,
    CoxResConfig("rank_a030_r016_p010", 0.30, PRIMARY_COXRES_CONFIG.residual_scale_init, 0.16, 0.010, False, 0.0, "legacy"),
    CoxResConfig("rank_a100_r016_p010", 1.00, PRIMARY_COXRES_CONFIG.residual_scale_init, 0.16, 0.010, False, 0.0, "legacy"),
    CoxResConfig("rank_a005_r010_p020", 0.05, PRIMARY_COXRES_CONFIG.residual_scale_init, 0.10, 0.020, False, 0.0, "legacy"),
    CoxResConfig("rank_a005_r025_p010", 0.05, PRIMARY_COXRES_CONFIG.residual_scale_init, 0.25, 0.010, False, 0.0, "legacy"),
    CoxResConfig("rank_a010_r025_p010", 0.10, PRIMARY_COXRES_CONFIG.residual_scale_init, 0.25, 0.010, False, 0.0, "legacy"),
    CoxResConfig("rank_a010_r040_p020", 0.10, PRIMARY_COXRES_CONFIG.residual_scale_init, 0.40, 0.020, False, 0.0, "legacy"),
    CoxResConfig("rank_a030_r025_p020", 0.30, PRIMARY_COXRES_CONFIG.residual_scale_init, 0.25, 0.020, False, 0.0, "legacy"),
)


def fit_cox_offset(
    train: pd.DataFrame,
    test: pd.DataFrame,
    times: np.ndarray,
    *,
    alpha: float = 0.1,
    nonlinear_main_effects: bool = False,
) -> Prediction:
    """Fit the internal Cox component used only by the residual KG model."""
    prep = nonlinear_cox_preprocessor() if nonlinear_main_effects else preprocessor()
    x_train = np.asarray(prep.fit_transform(train[list(FEATURES)]), dtype=float)
    x_test = np.asarray(prep.transform(test[list(FEATURES)]), dtype=float)
    model = CoxPHSurvivalAnalysis(alpha=alpha, n_iter=200)
    model.fit(x_train, survival_y(train))
    risk = np.asarray(model.predict(x_test), dtype=float)
    train_risk = np.asarray(model.predict(x_train), dtype=float)
    survival = breslow_survival(train, train_risk, risk, times)
    return Prediction(
        risk,
        survival,
        times,
        {
            "role": "internal Cox offset; not a comparator",
            "alpha": alpha,
            "nonlinear_main_effects": nonlinear_main_effects,
        },
    )


def fit_rsf(train: pd.DataFrame, test: pd.DataFrame, times: np.ndarray, seed: int = SEED, smoke: bool = False) -> Prediction:
    prep = preprocessor(scale_numeric=False)
    x_train = np.asarray(prep.fit_transform(train[list(FEATURES)]), dtype=np.float32)
    x_test = np.asarray(prep.transform(test[list(FEATURES)]), dtype=np.float32)
    model = RandomSurvivalForest(
        n_estimators=40 if smoke else 500,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(x_train, survival_y(train))
    risk = np.asarray(model.predict(x_test), dtype=float)
    survival = interpolate_functions(model.predict_survival_function(x_test), times)
    return Prediction(risk, survival, times, {"trees": model.n_estimators, "min_samples_leaf": 5})


def fit_survival_svm(train: pd.DataFrame, test: pd.DataFrame, times: np.ndarray, seed: int = SEED, smoke: bool = False) -> Prediction:
    prep = preprocessor()
    x_train = np.asarray(prep.fit_transform(train[list(FEATURES)]), dtype=np.float32)
    x_test = np.asarray(prep.transform(test[list(FEATURES)]), dtype=np.float32)
    model = FastSurvivalSVM(alpha=0.5, rank_ratio=0.7, max_iter=30 if smoke else 1000, random_state=seed)
    model.fit(x_train, survival_y(train))
    train_risk = np.asarray(model.predict(x_train), dtype=float)
    risk = np.asarray(model.predict(x_test), dtype=float)
    if concordance_index(train[TIME_COL], train[EVENT_COL], -train_risk) > concordance_index(
        train[TIME_COL], train[EVENT_COL], train_risk
    ):
        train_risk = -train_risk
        risk = -risk
    survival = breslow_survival(train, train_risk, risk, times)
    return Prediction(risk, survival, times, {"alpha": 0.5, "rank_ratio": 0.7})


def fit_linear_regression(train: pd.DataFrame, test: pd.DataFrame, times: np.ndarray, seed: int = SEED) -> Prediction:
    """Requested censoring-unaware comparator; fits observed deaths only."""
    prep = preprocessor()
    x_train = np.asarray(prep.fit_transform(train[list(FEATURES)]), dtype=float)
    x_test = np.asarray(prep.transform(test[list(FEATURES)]), dtype=float)
    events = train[EVENT_COL].to_numpy(dtype=int) == 1
    model = Ridge(alpha=1.0).fit(x_train[events], np.log(train.loc[events, TIME_COL].to_numpy(dtype=float)))
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    sigma = max(float(np.std(np.log(train.loc[events, TIME_COL]) - pred_train[events], ddof=1)), 0.1)
    survival = 1.0 - norm.cdf((np.log(times)[None, :] - pred_test[:, None]) / sigma)
    return Prediction(-pred_test, survival, times, {"warning": "Censoring-unaware event-only log-time ridge regression", "sigma": sigma})


def fit_xgboost_aft(train: pd.DataFrame, test: pd.DataFrame, times: np.ndarray, seed: int = SEED, smoke: bool = False) -> Prediction:
    prep = preprocessor()
    x_train = np.asarray(prep.fit_transform(train[list(FEATURES)]), dtype=np.float32)
    x_test = np.asarray(prep.transform(test[list(FEATURES)]), dtype=np.float32)
    lower = train[TIME_COL].to_numpy(dtype=float)
    upper = np.where(train[EVENT_COL].to_numpy(dtype=int) == 1, lower, np.inf)
    scale = 1.2
    rounds = 30 if smoke else 350
    with tempfile.TemporaryDirectory(prefix="rain_adv_xgb_") as temp_dir:
        input_path = Path(temp_dir) / "input.npz"
        output_path = Path(temp_dir) / "prediction.npy"
        np.savez(input_path, x_train=x_train, x_test=x_test, lower=lower, upper=upper)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("xgb_worker.py")),
                "--input", str(input_path),
                "--output", str(output_path),
                "--rounds", str(rounds),
                "--seed", str(seed),
            ],
            check=True,
        )
        predicted_time = np.load(output_path)
        # XGBoost AFT returns predictions on the time scale (exp(location)).
        # Convert back to the log-time location used by the log-normal survival
        # distribution and by the risk score.
        location = np.log(np.clip(predicted_time, 1e-8, None))
    survival = 1.0 - norm.cdf((np.log(times)[None, :] - location[:, None]) / scale)
    return Prediction(-location, survival, times, {"rounds": rounds, "distribution": "normal", "scale": scale})


def cox_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Legacy comparator loss; retained unchanged for the regular neural network."""
    order = torch.argsort(time, descending=True)
    r = risk[order]
    e = event[order]
    log_cumsum = torch.logcumsumexp(r, dim=0)
    return -torch.sum((r - log_cumsum) * e) / torch.clamp(e.sum(), min=1.0)


def efron_cox_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Negative Cox partial log-likelihood with Efron's correction for tied deaths."""
    event_mask = event > 0
    event_count = event_mask.sum()
    if event_count == 0:
        return risk.sum() * 0.0
    shift = risk.max()
    exp_shifted = torch.exp(risk - shift)
    log_likelihood = risk.new_zeros(())
    for event_time in torch.unique(time[event_mask]):
        deaths = event_mask & (time == event_time)
        death_count = int(deaths.sum().item())
        risk_sum = exp_shifted[time >= event_time].sum()
        death_risk_sum = exp_shifted[deaths].sum()
        log_likelihood = log_likelihood + risk[deaths].sum()
        fractions = torch.arange(death_count, device=risk.device, dtype=risk.dtype) / death_count
        denominators = torch.clamp(risk_sum - fractions * death_risk_sum, min=torch.finfo(risk.dtype).tiny)
        log_likelihood = log_likelihood - torch.sum(shift + torch.log(denominators))
    return -log_likelihood / event_count.to(risk.dtype)


class ScalarTokenizer(torch.nn.Module):
    def __init__(self, n_features: int, dim: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(n_features, dim) * 0.05)
        self.bias = torch.nn.Parameter(torch.zeros(n_features, dim))
        self.cls = torch.nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        body = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        return torch.cat([self.cls.expand(x.shape[0], -1, -1), body], dim=1)


class KnowledgeBlock(torch.nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.attn = torch.nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 3), torch.nn.GELU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(dim * 3, dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        z = self.norm1(x)
        attended, _ = self.attn(z, z, z, attn_mask=mask, need_weights=False)
        x = x + attended
        return x + self.ff(self.norm2(x))


class CoxResidualKGAttentionNoHazard(torch.nn.Module):
    """Knowledge-guided residual risk model with no discrete hazard head."""

    def __init__(
        self,
        n_features: int,
        knowledge_mask: np.ndarray,
        dim: int = 16,
        heads: int = 1,
        layers: int = 1,
        dropout: float = 0.12,
        config: CoxResConfig | None = None,
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("CoxRes-KGA requires at least one attention layer")
        if dim < 1 or heads < 1 or dim % heads:
            raise ValueError("Token dimension must be positive and divisible by the attention-head count")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Dropout must be in the interval [0, 1)")
        self.tokenizer = ScalarTokenizer(n_features, dim)
        self.blocks = torch.nn.ModuleList(
            [KnowledgeBlock(dim, heads, dropout) for _ in range(layers)]
        )
        config = config or PRIMARY_COXRES_CONFIG
        prior = np.zeros((n_features + 1, n_features + 1), dtype=np.float32)
        other = np.zeros_like(prior)
        prior[1:, 1:][knowledge_mask > 0] = 1.0
        other[1:, 1:][knowledge_mask <= 0] = 1.0
        np.fill_diagonal(prior, 0.0)
        np.fill_diagonal(other, 0.0)
        self.register_buffer("knowledge_prior_mask", torch.as_tensor(prior))
        self.register_buffer("knowledge_other_mask", torch.as_tensor(other))
        prior_strength = torch.tensor(0.45, dtype=torch.float32)
        other_strength = torch.tensor(-0.35, dtype=torch.float32)
        if config.learnable_knowledge_bias:
            self.knowledge_prior_strength = torch.nn.Parameter(prior_strength)
            self.knowledge_other_strength = torch.nn.Parameter(other_strength)
        else:
            self.register_buffer("knowledge_prior_strength", prior_strength)
            self.register_buffer("knowledge_other_strength", other_strength)
        self.learnable_knowledge_bias = config.learnable_knowledge_bias
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim, 1),
        )
        raw_scale = math.log(math.expm1(config.residual_scale_init))
        self.scale_raw = torch.nn.Parameter(torch.tensor(raw_scale, dtype=torch.float32))

    def attention_bias(self) -> torch.Tensor:
        return (
            self.knowledge_prior_strength * self.knowledge_prior_mask
            + self.knowledge_other_strength * self.knowledge_other_mask
        )

    def knowledge_anchor_loss(self) -> torch.Tensor:
        if not self.learnable_knowledge_bias:
            return self.scale_raw.new_zeros(())
        return (self.knowledge_prior_strength - 0.45).square() + (self.knowledge_other_strength + 0.35).square()

    def forward(self, x: torch.Tensor, cox_offset: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(x)
        bias = self.attention_bias()
        for block in self.blocks:
            tokens = block(tokens, bias)
        residual = self.head(tokens[:, 0]).squeeze(1)
        scale = torch.nn.functional.softplus(self.scale_raw)
        return cox_offset + scale * residual, residual


class DeepSurv(torch.nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 96), torch.nn.BatchNorm1d(96), torch.nn.ReLU(), torch.nn.Dropout(0.15),
            torch.nn.Linear(96, 48), torch.nn.BatchNorm1d(48), torch.nn.ReLU(), torch.nn.Dropout(0.15),
            torch.nn.Linear(48, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def knowledge_matrix(feature_names: list[str]) -> np.ndarray:
    matrix = np.eye(len(feature_names), dtype=float)
    pairs = [
        ("age", "grade"), ("grade", "idh_mutant"), ("grade", "mgmt_methylated"),
        ("idh_mutant", "codel_1p19q"), ("idh_mutant", "mgmt_methylated"),
    ]
    for left, right in pairs:
        left_idx = [i for i, name in enumerate(feature_names) if name == left or name.startswith(left + "_")]
        right_idx = [i for i, name in enumerate(feature_names) if name == right or name.startswith(right + "_")]
        for i in left_idx:
            for j in right_idx:
                matrix[i, j] = matrix[j, i] = 1.0
    return matrix


def pairwise_rank_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Logistic ranking loss over comparable event-before-later pairs."""
    comparable = (event[:, None] > 0) & (time[:, None] < time[None, :])
    if not torch.any(comparable):
        return risk.sum() * 0.0
    differences = risk[:, None] - risk[None, :]
    return torch.nn.functional.softplus(-differences[comparable]).mean()


def _stratification_labels(frame: pd.DataFrame) -> pd.Series:
    return frame[EVENT_COL].astype(str) + "_" + frame["grade"].fillna(-1).astype(int).astype(str)


def _coxres_arrays(
    train: pd.DataFrame,
    test: pd.DataFrame,
    times: np.ndarray,
    *,
    cox_alpha: float,
    nonlinear_cox: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Fit all preprocessing and the Cox offset on ``train`` only."""
    prep = preprocessor()
    x_train = np.asarray(prep.fit_transform(train[list(FEATURES)]), dtype=np.float32)
    x_test = np.asarray(prep.transform(test[list(FEATURES)]), dtype=np.float32)
    feature_names = list(prep.get_feature_names_out())
    combined = pd.concat([train, test], ignore_index=True)
    offset = np.asarray(
        fit_cox_offset(
            train,
            combined,
            times,
            alpha=cox_alpha,
            nonlinear_main_effects=nonlinear_cox,
        ).risk,
        dtype=np.float32,
    )
    offset_mean = float(offset[: len(train)].mean())
    offset_std = max(float(offset[: len(train)].std()), 1e-6)
    offset = (offset - offset_mean) / offset_std
    return x_train, x_test, offset[: len(train)], offset[len(train) :], feature_names


def _coxres_objective(
    model: CoxResidualKGAttentionNoHazard,
    features: torch.Tensor,
    offset: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    config: CoxResConfig,
) -> torch.Tensor:
    risk, residual = model(features, offset)
    centered_residual = residual - residual.mean()
    centered_offset = offset - offset.mean()
    correlation = (centered_residual * centered_offset).mean() / (
        centered_residual.square().mean().sqrt()
        * centered_offset.square().mean().sqrt()
        + 1e-6
    )
    return (
        (efron_cox_loss(risk, time, event) if config.tie_method == "efron" else cox_loss(risk, time, event))
        + config.ranking_weight * pairwise_rank_loss(risk, time, event)
        + config.residual_penalty * residual.square().mean()
        + config.orthogonality_penalty * correlation.square()
        + config.knowledge_anchor_penalty * model.knowledge_anchor_loss()
    )


def coxres_optimizer(
    model: torch.nn.Module,
    optimizer_name: str,
    *,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-3,
) -> torch.optim.Optimizer:
    """Build the requested CoxRes-KGA optimizer with explicit regularization."""
    optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
    try:
        optimizer_class = optimizers[optimizer_name.lower()]
    except KeyError as error:
        raise ValueError("optimizer_name must be 'adam' or 'adamw'") from error
    return optimizer_class(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def select_coxres_config(
    train: pd.DataFrame,
    times: np.ndarray,
    *,
    seed: int,
    smoke: bool,
    candidates: tuple[CoxResConfig, ...] = COXRES_CONFIGS,
    token_dimension: int = 16,
    attention_heads: int = 1,
    dropout: float = 0.12,
    optimizer_name: str = "adamw",
    nonlinear_cox: bool = False,
) -> tuple[CoxResConfig, list[dict]]:
    """Select a prespecified configuration using only the current outer training set."""
    labels = _stratification_labels(train)
    folds = 2 if smoke else 3
    epochs = 5 if smoke else 60
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + 700)
    rows: list[dict] = []
    if not candidates:
        raise ValueError("CoxRes-KGA selection requires at least one candidate")
    if len({config.name for config in candidates}) != len(candidates):
        raise ValueError("CoxRes-KGA candidate names must be unique")
    if token_dimension < 1 or attention_heads < 1 or token_dimension % attention_heads:
        raise ValueError("Token dimension must be positive and divisible by the attention-head count")
    for config_index, config in enumerate(candidates):
        scores = []
        for fold_index, (fit_idx, val_idx) in enumerate(splitter.split(np.zeros(len(train)), labels), start=1):
            fit_frame = train.iloc[fit_idx].reset_index(drop=True)
            val_frame = train.iloc[val_idx].reset_index(drop=True)
            fit_x, val_x, fit_offset, val_offset, feature_names = _coxres_arrays(
                fit_frame,
                val_frame,
                times,
                cox_alpha=config.cox_alpha,
                nonlinear_cox=nonlinear_cox,
            )
            member_seed = seed + 8000 + 100 * config_index + fold_index
            torch.manual_seed(member_seed)
            np.random.seed(member_seed)
            model = CoxResidualKGAttentionNoHazard(
                fit_x.shape[1],
                knowledge_matrix(feature_names),
                dim=token_dimension,
                heads=attention_heads,
                dropout=dropout,
                config=config,
            )
            optimizer = coxres_optimizer(model, optimizer_name)
            x_fit = torch.as_tensor(fit_x, dtype=torch.float32)
            x_val = torch.as_tensor(val_x, dtype=torch.float32)
            cox_fit = torch.as_tensor(fit_offset, dtype=torch.float32)
            cox_val = torch.as_tensor(val_offset, dtype=torch.float32)
            time_fit = torch.as_tensor(fit_frame[TIME_COL].to_numpy(np.float32))
            event_fit = torch.as_tensor(fit_frame[EVENT_COL].to_numpy(np.float32))
            for _ in range(epochs):
                model.train()
                optimizer.zero_grad()
                loss = _coxres_objective(model, x_fit, cox_fit, time_fit, event_fit, config)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_risk, _ = model(x_val, cox_val)
            scores.append(float(concordance_index(val_frame[TIME_COL], val_frame[EVENT_COL], val_risk.numpy())))
        rows.append({
            "config": config.name,
            "mean_inner_c_index": float(np.mean(scores)),
            "sd_inner_c_index": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "fold_c_indices": scores,
        })
    selected_row = max(rows, key=lambda row: (row["mean_inner_c_index"], -row["sd_inner_c_index"]))
    selected = next(config for config in candidates if config.name == selected_row["config"])
    return selected, rows


def train_coxres(
    train: pd.DataFrame,
    test: pd.DataFrame,
    times: np.ndarray,
    *,
    seed: int,
    smoke: bool,
    config: CoxResConfig,
    token_dimension: int = 16,
    attention_heads: int = 1,
    dropout: float = 0.12,
    optimizer_name: str = "adamw",
    nonlinear_cox: bool = False,
    max_epochs: int = 170,
) -> Prediction:
    """Select an epoch on training data, then refit one CoxRes-KGA member."""
    labels = _stratification_labels(train)
    fit_idx, val_idx = train_test_split(
        np.arange(len(train)), test_size=0.18, random_state=seed, stratify=labels
    )
    fit_frame = train.iloc[fit_idx].reset_index(drop=True)
    validation_frame = train.iloc[val_idx].reset_index(drop=True)
    fit_x, val_x, fit_offset, val_offset, fit_names = _coxres_arrays(
        fit_frame,
        validation_frame,
        times,
        cox_alpha=config.cox_alpha,
        nonlinear_cox=nonlinear_cox,
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CoxResidualKGAttentionNoHazard(
        fit_x.shape[1],
        knowledge_matrix(fit_names),
        dim=token_dimension,
        heads=attention_heads,
        dropout=dropout,
        config=config,
    )
    optimizer = coxres_optimizer(model, optimizer_name)
    x_fit = torch.as_tensor(fit_x, dtype=torch.float32)
    x_val = torch.as_tensor(val_x, dtype=torch.float32)
    cox_fit = torch.as_tensor(fit_offset, dtype=torch.float32)
    cox_val = torch.as_tensor(val_offset, dtype=torch.float32)
    time_fit = torch.as_tensor(fit_frame[TIME_COL].to_numpy(np.float32))
    event_fit = torch.as_tensor(fit_frame[EVENT_COL].to_numpy(np.float32))
    best_score = -np.inf
    best_epoch = 1
    stale = 0
    training_max_epochs = 12 if smoke else max_epochs
    if not smoke and training_max_epochs < 70:
        raise ValueError("max_epochs must be at least 70 for CoxRes-KGA training")
    for epoch in range(1, training_max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = _coxres_objective(model, x_fit, cox_fit, time_fit, event_fit, config)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_risk, _ = model(x_val, cox_val)
        score = concordance_index(
            validation_frame[TIME_COL], validation_frame[EVENT_COL], validation_risk.numpy()
        )
        if np.isfinite(score) and score > best_score + 1e-4:
            best_score = float(score)
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch >= 70 and stale >= 22:
            break
    if not smoke:
        best_epoch = max(70, best_epoch)

    x_train, x_test, train_offset, test_offset, feature_names = _coxres_arrays(
        train,
        test,
        times,
        cox_alpha=config.cox_alpha,
        nonlinear_cox=nonlinear_cox,
    )
    torch.manual_seed(seed + 1000)
    final = CoxResidualKGAttentionNoHazard(
        x_train.shape[1],
        knowledge_matrix(feature_names),
        dim=token_dimension,
        heads=attention_heads,
        dropout=dropout,
        config=config,
    )
    optimizer = coxres_optimizer(final, optimizer_name)
    train_features = torch.as_tensor(x_train, dtype=torch.float32)
    test_features = torch.as_tensor(x_test, dtype=torch.float32)
    train_cox = torch.as_tensor(train_offset, dtype=torch.float32)
    test_cox = torch.as_tensor(test_offset, dtype=torch.float32)
    train_time = torch.as_tensor(train[TIME_COL].to_numpy(np.float32))
    train_event = torch.as_tensor(train[EVENT_COL].to_numpy(np.float32))
    for _ in range(best_epoch):
        final.train()
        optimizer.zero_grad()
        loss = _coxres_objective(final, train_features, train_cox, train_time, train_event, config)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(final.parameters(), 5.0)
        optimizer.step()
    final.eval()
    with torch.no_grad():
        train_risk, _ = final(train_features, train_cox)
        risk, _ = final(test_features, test_cox)
    train_risk_np = train_risk.numpy()
    risk_np = risk.numpy()
    survival = breslow_survival(train, train_risk_np, risk_np, times)
    return Prediction(
        risk_np,
        survival,
        times,
        {
            "token_dimension": token_dimension,
            "attention_heads": attention_heads,
            "attention_layers": 1,
            "dropout": dropout,
            "optimizer": optimizer_name.lower(),
            "learning_rate": 3e-4,
            "weight_decay": 1e-3,
            "architecture": "fixed",
            "selected_config": config.name,
            "cox_ties": config.tie_method,
            "cox_alpha": config.cox_alpha,
            "residual_scale_init": config.residual_scale_init,
            "ranking_weight": config.ranking_weight,
            "residual_penalty": config.residual_penalty,
            "orthogonality_penalty": config.orthogonality_penalty,
            "nonlinear_cox": nonlinear_cox,
            "learnable_knowledge_bias": config.learnable_knowledge_bias,
            "knowledge_anchor_penalty": config.knowledge_anchor_penalty,
            "learned_prior_strength": float(final.knowledge_prior_strength.detach()),
            "learned_other_strength": float(final.knowledge_other_strength.detach()),
            "selected_epoch": int(best_epoch),
            "max_epochs": int(training_max_epochs),
            "inner_validation_c_index": float(best_score),
            "feature_names": feature_names,
        },
    )


def train_neural(
    train: pd.DataFrame,
    test: pd.DataFrame,
    times: np.ndarray,
    *,
    kind: str,
    seed: int,
    smoke: bool,
) -> Prediction:
    if kind != "regular_neural_network":
        raise ValueError("train_neural is reserved for the regular neural-network comparator")
    torch.manual_seed(seed)
    np.random.seed(seed)
    prep = preprocessor()
    x_train = np.asarray(prep.fit_transform(train[list(FEATURES)]), dtype=np.float32)
    x_test = np.asarray(prep.transform(test[list(FEATURES)]), dtype=np.float32)
    feature_names = list(prep.get_feature_names_out())
    n = len(train)
    strata = train[EVENT_COL].astype(str) + "_" + train["grade"].fillna(-1).astype(int).astype(str)
    fit_idx, val_idx = train_test_split(np.arange(n), test_size=0.18, random_state=seed, stratify=strata)
    device = torch.device("cpu")
    xt = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    tt = torch.as_tensor(train[TIME_COL].to_numpy(np.float32), device=device)
    et = torch.as_tensor(train[EVENT_COL].to_numpy(np.float32), device=device)

    model = DeepSurv(x_train.shape[1]).to(device)
    learning_rate = 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    best_state = copy.deepcopy(model.state_dict())
    best_score = -np.inf
    best_epoch = 1
    stale = 0
    max_epochs = 12 if smoke else (120 if kind == "regular_neural_network" else 170)
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        risk_fit = model(xt[fit_idx])
        loss = cox_loss(risk_fit, tt[fit_idx], et[fit_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            risk_val = model(xt[val_idx]).cpu().numpy()
        score = concordance_index(train.iloc[val_idx][TIME_COL], train.iloc[val_idx][EVENT_COL], risk_val)
        if np.isfinite(score) and score > best_score + 1e-4:
            best_state = copy.deepcopy(model.state_dict())
            best_score = score
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
    if not smoke:
        best_epoch = 120

    # Retrain from scratch for exactly the selected epoch count on the full outer training set.
    torch.manual_seed(seed + 1000)
    final = DeepSurv(x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(final.parameters(), lr=learning_rate, weight_decay=1e-3)
    for _ in range(best_epoch):
        final.train()
        optimizer.zero_grad()
        risk_all = final(xt)
        loss = cox_loss(risk_all, tt, et)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(final.parameters(), 5.0)
        optimizer.step()

    x_test_t = torch.as_tensor(x_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        train_risk = final(xt).cpu().numpy()
        risk = final(x_test_t).cpu().numpy()
    survival = breslow_survival(train, train_risk, risk, times)
    return Prediction(
        np.asarray(risk),
        survival,
        times,
        {"selected_epoch": int(best_epoch), "inner_validation_c_index": float(best_score), "feature_names": feature_names},
    )


MODEL_ORDER = [
    "linear_regression",
    "random_survival_forest",
    "survival_svm",
    "xgboost_aft",
    "regular_neural_network",
    "cox_residual_kg_attention_nohazard",
]


def fit_coxres_ensemble(
    train: pd.DataFrame,
    test: pd.DataFrame,
    times: np.ndarray,
    seed: int,
    smoke: bool,
    *,
    run_ablation: bool,
    candidate_configs: tuple[CoxResConfig, ...] | None = None,
    selection_label: str = "prespecified candidates; selected within outer training data",
    token_dimension: int = 16,
    attention_heads: int = 1,
    dropout: float = 0.12,
    optimizer_name: str = "adamw",
    nonlinear_cox: bool = False,
    ensemble_mode: str = "seed",
    max_epochs: int = 170,
) -> Prediction:
    if ensemble_mode not in {"seed", "crossfit"}:
        raise ValueError("ensemble_mode must be 'seed' or 'crossfit'")
    if run_ablation:
        selected_config, selection_scores = select_coxres_config(
            train,
            times,
            seed=seed,
            smoke=smoke,
            candidates=candidate_configs or COXRES_CONFIGS,
            token_dimension=token_dimension,
            attention_heads=attention_heads,
            dropout=dropout,
            optimizer_name=optimizer_name,
            nonlinear_cox=nonlinear_cox,
        )
        selection = selection_label
    else:
        selected_config = PRIMARY_COXRES_CONFIG
        selection_scores = []
        selection = "fixed primary publication configuration"
    seeds = [seed] if smoke else [seed + offset for offset in range(5)]
    if ensemble_mode == "seed" or smoke:
        member_frames = [train] * len(seeds)
    else:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 1700)
        labels = _stratification_labels(train)
        member_frames = [
            train.iloc[fit_idx].reset_index(drop=True)
            for fit_idx, _ in splitter.split(np.zeros(len(train)), labels)
        ]
    members = [
        train_coxres(
            member_frame,
            test,
            times,
            seed=value,
            smoke=smoke,
            config=selected_config,
            token_dimension=token_dimension,
            attention_heads=attention_heads,
            dropout=dropout,
            optimizer_name=optimizer_name,
            nonlinear_cox=nonlinear_cox,
            max_epochs=max_epochs,
        )
        for value, member_frame in zip(seeds, member_frames)
    ]
    risk = np.mean([member.risk for member in members], axis=0)
    survival = np.mean([member.survival for member in members], axis=0)
    survival = np.minimum.accumulate(np.clip(survival, 1e-5, 1 - 1e-5), axis=1)
    return Prediction(
        risk,
        survival,
        times,
        {
            "seeds": seeds,
            "architecture": {
                "token_dimension": token_dimension,
                "attention_heads": attention_heads,
                "attention_layers": 1,
                "dropout": dropout,
                "optimizer": optimizer_name.lower(),
                "learning_rate": 3e-4,
                "weight_decay": 1e-3,
                "selection": selection,
                "nonlinear_cox": nonlinear_cox,
                "ensemble_mode": ensemble_mode,
                "max_epochs": max_epochs,
            },
            "selected_config": selected_config.name,
            "config": selected_config.__dict__,
            "selection_scores": selection_scores,
            "member_training_sizes": [len(frame) for frame in member_frames],
            "members": [member.details for member in members],
        },
    )


def fit_model(name: str, train: pd.DataFrame, test: pd.DataFrame, times: np.ndarray, seed: int, smoke: bool) -> Prediction:
    if name == "linear_regression":
        return fit_linear_regression(train, test, times, seed)
    if name == "random_survival_forest":
        return fit_rsf(train, test, times, seed, smoke)
    if name == "survival_svm":
        return fit_survival_svm(train, test, times, seed, smoke)
    if name == "xgboost_aft":
        return fit_xgboost_aft(train, test, times, seed, smoke)
    if name == "regular_neural_network":
        return train_neural(train, test, times, kind="regular_neural_network", seed=seed, smoke=smoke)
    if name == "cox_residual_kg_attention_nohazard":
        return fit_coxres_ensemble(
            train,
            test,
            times,
            seed,
            smoke,
            run_ablation=True,
            candidate_configs=RANK_TUNING_CONFIGS,
            selection_label="Cox penalty and loss balance selected within outer training data",
        )
    raise ValueError(f"Unknown model: {name}")
