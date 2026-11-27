"""Models behind the adaptive conformal intervals: difficulty and quantiles.

Everything works on the log1p(kcal) scale of the baseline and reads only
``train`` and ``val``; ``calib`` and ``test`` never influence these models.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .baseline import BaselineData, RidgeFit, group_labels
from .conformal import (
    conformal_intervals,
    cqr_interval,
    normalized_interval,
    resampled_coverage_of,
)

DIFFICULTY_ALPHAS = (1, 10, 100, 1000, 10000, 100000)
QUANTILE_L1 = (0.001, 0.003, 0.01)
SIGMA_FLOOR_PERCENTILE = 5
_EPS = 1e-3  # keeps log|residual| finite for exact predictions
VARIANTS = ("additive", "multiplicative", "normalized", "cqr")


# --------------------------------------------------------------------------
# Honest training residuals: grouped cross-fitting
# --------------------------------------------------------------------------


def group_kfold(groups: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    """Fold index (0..k-1) per item, keeping each group in a single fold.

    Groups are shuffled and assigned one by one to the currently smallest
    fold, so folds end up with similar sizes.
    """
    groups = np.asarray(groups)
    labels, inverse = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inverse)
    fold_of_group = np.empty(len(labels), dtype=int)
    load = np.zeros(k, dtype=int)
    for g in np.random.default_rng(seed).permutation(len(labels)):
        f = int(np.argmin(load))
        fold_of_group[g] = f
        load[f] += sizes[g]
    return fold_of_group[inverse]


def cross_fitted_residuals(X: np.ndarray, kcal: np.ndarray, folds: np.ndarray, ridge_alpha: float) -> np.ndarray:
    """Out-of-fold absolute residuals of the baseline ridge on log1p(kcal)."""
    y = np.log1p(kcal)
    residuals = np.empty(len(y))
    for f in np.unique(folds):
        held = folds == f
        model = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha)).fit(X[~held], y[~held])
        residuals[held] = np.abs(y[held] - model.predict(X[held]))
    return residuals


# --------------------------------------------------------------------------
# Difficulty model for normalized conformal
# --------------------------------------------------------------------------


@dataclass
class DifficultyFit:
    model: Pipeline  # predicts log(|residual| + eps)
    alpha: float
    floor: float  # lower bound applied to sigma
    search: list[dict]  # [{"alpha": ..., "val_mse": ...}]

    def sigma(self, X: np.ndarray) -> np.ndarray:
        """Estimated difficulty (typical absolute log residual), at least ``floor``."""
        return np.maximum(np.exp(self.model.predict(X)), self.floor)


def fit_difficulty(
    data: BaselineData,
    ridge: RidgeFit,
    k: int = 5,
    seed: int = 0,
    alphas: Sequence[float] = DIFFICULTY_ALPHAS,
) -> DifficultyFit:
    """Ridge on the embeddings predicting log|residual| of the baseline.

    Trained on cross-fitted residuals of ``train`` (grouped folds), so the
    targets are as honest as residuals on new dishes. Regularization is
    chosen by squared error on ``val``, whose residuals come from the final
    baseline ridge, which never saw those dishes.
    """
    X, kcal = data.X, data.kcal
    folds = group_kfold(group_labels(data.parts["train"], data.dishes), k=k, seed=seed)
    target = np.log(cross_fitted_residuals(X["train"], kcal["train"], folds, ridge.alpha) + _EPS)
    val_target = np.log(np.abs(np.log1p(kcal["val"]) - ridge.model.predict(X["val"])) + _EPS)

    def fit(a):
        return make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(X["train"], target)

    search = [{"alpha": a, "val_mse": float(np.mean((fit(a).predict(X["val"]) - val_target) ** 2))} for a in alphas]
    best = min(search, key=lambda r: r["val_mse"])["alpha"]
    model = fit(best)
    floor = float(np.percentile(np.exp(model.predict(X["train"])), SIGMA_FLOOR_PERCENTILE))
    return DifficultyFit(model=model, alpha=best, floor=floor, search=search)


# --------------------------------------------------------------------------
# Quantile models for CQR
# --------------------------------------------------------------------------


def pinball_loss(y: np.ndarray, pred: np.ndarray, quantile: float) -> float:
    """Mean pinball (quantile) loss: the loss minimized by the ``quantile``-th quantile."""
    r = np.asarray(y) - np.asarray(pred)
    return float(np.mean(np.maximum(quantile * r, (quantile - 1) * r)))


@dataclass
class QuantileFit:
    alpha: float  # miscoverage level the quantiles target
    lower: Pipeline  # predicts the alpha/2 quantile of log1p(kcal)
    upper: Pipeline  # predicts the 1 - alpha/2 quantile
    l1: dict[str, float]  # chosen penalty per side
    search: list[dict]  # [{"side": ..., "l1": ..., "val_pinball": ...}]

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """Lower and upper log quantiles, swapped where they cross, and the number of crossings."""
        lo, hi = self.lower.predict(X), self.upper.predict(X)
        crossed = int(np.sum(lo > hi))
        return np.minimum(lo, hi), np.maximum(lo, hi), crossed


def fit_quantiles(
    data: BaselineData, alpha: float, l1_values: Sequence[float] = QUANTILE_L1
) -> QuantileFit:
    """Linear quantile regression for the alpha/2 and 1 - alpha/2 quantiles of log1p(kcal).

    The L1 penalty of each side is chosen by pinball loss on ``val``.
    """
    X, y = data.X, {k: np.log1p(v) for k, v in data.kcal.items()}
    chosen, models, search = {}, {}, []
    for side, q in (("lower", alpha / 2), ("upper", 1 - alpha / 2)):
        best = None
        for l1 in l1_values:
            model = make_pipeline(StandardScaler(), QuantileRegressor(quantile=q, alpha=l1, solver="highs"))
            model.fit(X["train"], y["train"])
            loss = pinball_loss(y["val"], model.predict(X["val"]), q)
            search.append({"side": side, "l1": l1, "val_pinball": loss})
            if best is None or loss < best[0]:
                best = (loss, l1, model)
        chosen[side], models[side] = best[1], best[2]
    return QuantileFit(alpha=alpha, lower=models["lower"], upper=models["upper"], l1=chosen, search=search)


# --------------------------------------------------------------------------
# All four variants
# --------------------------------------------------------------------------


def adaptive_intervals(
    data: BaselineData,
    ridge: RidgeFit,
    difficulty: DifficultyFit,
    quantiles: QuantileFit,
    alpha: float,
    split: str = "test",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The four conformal variants for ``split``, calibrated on ``calib``, in kcal.

    ``quantiles`` must have been fitted for this ``alpha``.
    """
    if not np.isclose(quantiles.alpha, alpha):
        raise ValueError(f"quantile models target alpha={quantiles.alpha}, not {alpha}")
    X, y_cal = data.X, data.kcal["calib"]
    pred_cal, pred = ridge.model.predict(X["calib"]), ridge.model.predict(X[split])
    out = conformal_intervals(y_cal, pred_cal, pred, alpha)
    out["normalized"] = normalized_interval(
        y_cal, pred_cal, difficulty.sigma(X["calib"]), pred, difficulty.sigma(X[split]), alpha
    )
    lo_cal, hi_cal, _ = quantiles.predict(X["calib"])
    lo, hi, _ = quantiles.predict(X[split])
    out["cqr"] = cqr_interval(y_cal, lo_cal, hi_cal, lo, hi, alpha)
    return {v: out[v] for v in VARIANTS}


def resampled_adaptive_coverage(
    data: BaselineData,
    ridge: RidgeFit,
    difficulty: DifficultyFit,
    quantiles: QuantileFit,
    n_reps: int = 1000,
    respect_groups: bool = True,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Coverage of the four variants over random re-splits of ``calib`` + ``test``.

    All models stay fixed; only which dishes calibrate and which are tested
    changes. Calibration keeps the size of ``calib``.
    """
    pool = data.parts["calib"] + data.parts["test"]
    X = np.concatenate([data.X["calib"], data.X["test"]])
    y = np.concatenate([data.kcal["calib"], data.kcal["test"]])
    pred, sigma = ridge.model.predict(X), difficulty.sigma(X)
    lo, hi, _ = quantiles.predict(X)
    alpha = quantiles.alpha

    def intervals(cal):
        out = conformal_intervals(y[cal], pred[cal], pred[~cal], alpha)
        out["normalized"] = normalized_interval(y[cal], pred[cal], sigma[cal], pred[~cal], sigma[~cal], alpha)
        out["cqr"] = cqr_interval(y[cal], lo[cal], hi[cal], lo[~cal], hi[~cal], alpha)
        return {v: out[v] for v in VARIANTS}

    groups = group_labels(pool, data.dishes) if respect_groups else None
    return resampled_coverage_of(y, intervals, len(data.parts["calib"]), n_reps, groups, seed)
