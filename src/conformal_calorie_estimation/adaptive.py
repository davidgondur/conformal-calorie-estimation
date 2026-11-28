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
    coverage,
    cqr_interval,
    interval_score,
    normalized_interval,
    resampled_coverage_of,
)
from .data import Dish

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


# --------------------------------------------------------------------------
# What any point model must provide
# --------------------------------------------------------------------------


@dataclass
class PointModelOutputs:
    """Everything the four conformal variants need from a point model on log1p(kcal).

    ``pred_log`` and ``emb`` come from the final model for every partition.
    ``honest_residual`` and ``honest_emb`` describe ``train`` dishes as if they
    were new (out-of-fold), and are what the difficulty and quantile models
    are trained on.
    """

    dishes: dict[str, Dish]
    parts: dict[str, list[str]]
    kcal: dict[str, np.ndarray]
    pred_log: dict[str, np.ndarray]
    emb: dict[str, np.ndarray]
    honest_residual: np.ndarray | None  # |log1p(kcal) - prediction| on train, out of fold
    honest_emb: np.ndarray | None  # embeddings of train matching those residuals


def baseline_outputs(data: BaselineData, ridge: RidgeFit, k: int = 5, seed: int = 0) -> PointModelOutputs:
    """Outputs of the frozen-backbone ridge baseline.

    The frozen embeddings never saw a label, so they are honest as they are;
    only the ridge residuals of ``train`` need grouped cross-fitting.
    """
    folds = group_kfold(group_labels(data.parts["train"], data.dishes), k=k, seed=seed)
    return PointModelOutputs(
        dishes=data.dishes,
        parts=data.parts,
        kcal=data.kcal,
        pred_log={p: ridge.model.predict(x) for p, x in data.X.items()},
        emb=data.X,
        honest_residual=cross_fitted_residuals(data.X["train"], data.kcal["train"], folds, ridge.alpha),
        honest_emb=data.X["train"],
    )


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


def val_residual(out: PointModelOutputs) -> np.ndarray:
    """Absolute log residual of the final model on ``val`` (it never saw those dishes)."""
    return np.abs(np.log1p(out.kcal["val"]) - out.pred_log["val"])


def fit_difficulty(out: PointModelOutputs, alphas: Sequence[float] = DIFFICULTY_ALPHAS) -> DifficultyFit:
    """Ridge on the embeddings predicting log|residual| of the point model.

    Trained on the honest (out-of-fold) residuals of ``train``. Regularization
    is chosen by squared error on ``val``.
    """
    target = np.log(out.honest_residual + _EPS)
    val_target = np.log(val_residual(out) + _EPS)

    def fit(a):
        return make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(out.honest_emb, target)

    search = [{"alpha": a, "val_mse": float(np.mean((fit(a).predict(out.emb["val"]) - val_target) ** 2))} for a in alphas]
    best = min(search, key=lambda r: r["val_mse"])["alpha"]
    model = fit(best)
    floor = float(np.percentile(np.exp(model.predict(out.honest_emb)), SIGMA_FLOOR_PERCENTILE))
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


def fit_quantiles(out: PointModelOutputs, alpha: float, l1_values: Sequence[float] = QUANTILE_L1) -> QuantileFit:
    """Linear quantile regression for the alpha/2 and 1 - alpha/2 quantiles of log1p(kcal).

    Trained on the honest embeddings of ``train``; the L1 penalty of each side
    is chosen by pinball loss on ``val``.
    """
    y_train, y_val = np.log1p(out.kcal["train"]), np.log1p(out.kcal["val"])
    chosen, models, search = {}, {}, []
    for side, q in (("lower", alpha / 2), ("upper", 1 - alpha / 2)):
        best = None
        for l1 in l1_values:
            model = make_pipeline(StandardScaler(), QuantileRegressor(quantile=q, alpha=l1, solver="highs"))
            model.fit(out.honest_emb, y_train)
            loss = pinball_loss(y_val, model.predict(out.emb["val"]), q)
            search.append({"side": side, "l1": l1, "val_pinball": loss})
            if best is None or loss < best[0]:
                best = (loss, l1, model)
        chosen[side], models[side] = best[1], best[2]
    return QuantileFit(alpha=alpha, lower=models["lower"], upper=models["upper"], l1=chosen, search=search)


# --------------------------------------------------------------------------
# All four variants
# --------------------------------------------------------------------------


def _four_variants(y_cal, pred_cal, pred, sigma_cal, sigma, q_cal, q, alpha):
    out = conformal_intervals(y_cal, pred_cal, pred, alpha)
    out["normalized"] = normalized_interval(y_cal, pred_cal, sigma_cal, pred, sigma, alpha)
    out["cqr"] = cqr_interval(y_cal, *q_cal, *q, alpha)
    return {v: out[v] for v in VARIANTS}


def adaptive_intervals(
    out: PointModelOutputs,
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
    e = out.emb
    return _four_variants(
        out.kcal["calib"], out.pred_log["calib"], out.pred_log[split],
        difficulty.sigma(e["calib"]), difficulty.sigma(e[split]),
        quantiles.predict(e["calib"])[:2], quantiles.predict(e[split])[:2], alpha,
    )


def resampled_adaptive_coverage(
    out: PointModelOutputs,
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
    pool = out.parts["calib"] + out.parts["test"]
    E = np.concatenate([out.emb["calib"], out.emb["test"]])
    y = np.concatenate([out.kcal["calib"], out.kcal["test"]])
    pred = np.concatenate([out.pred_log["calib"], out.pred_log["test"]])
    sigma = difficulty.sigma(E)
    lo, hi, _ = quantiles.predict(E)
    alpha = quantiles.alpha

    def intervals(cal):
        return _four_variants(
            y[cal], pred[cal], pred[~cal], sigma[cal], sigma[~cal],
            (lo[cal], hi[cal]), (lo[~cal], hi[~cal]), alpha,
        )

    groups = group_labels(pool, out.dishes) if respect_groups else None
    return resampled_coverage_of(y, intervals, len(out.parts["calib"]), n_reps, groups, seed)


def variant_metrics(intervals: dict[str, tuple[np.ndarray, np.ndarray]], y: np.ndarray, alpha: float) -> dict[str, dict]:
    """Coverage, mean and median width, and interval score of each variant."""
    return {
        v: {
            "coverage": coverage(y, lo, hi),
            "width_mean": float(np.mean(hi - lo)),
            "width_median": float(np.median(hi - lo)),
            "interval_score": interval_score(y, lo, hi, alpha),
        }
        for v, (lo, hi) in intervals.items()
    }
