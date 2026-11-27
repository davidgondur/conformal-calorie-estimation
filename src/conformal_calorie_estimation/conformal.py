"""Split conformal prediction for regression.

Kept small and explicit on purpose: the formulas are the point of the project.
"""

import math
from itertools import pairwise

import numpy as np
from scipy.stats import betabinom


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected (1 - alpha) quantile of calibration scores.

    Returns the ceil((n + 1)(1 - alpha))-th smallest score. If calibration and
    test points are exchangeable, a new score is <= this value with
    probability >= 1 - alpha. Returns +inf when n is too small for alpha.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    scores = np.sort(np.asarray(scores, dtype=float).ravel())
    k = math.ceil((len(scores) + 1) * (1 - alpha))
    return math.inf if k > len(scores) else float(scores[k - 1])


def absolute_residual_interval(
    y_cal: np.ndarray, pred_cal: np.ndarray, pred: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Basic split conformal: ``pred ± q``, with q from |y - pred| on calibration."""
    q = conformal_quantile(np.abs(np.asarray(y_cal) - np.asarray(pred_cal)), alpha)
    pred = np.asarray(pred, dtype=float)
    return pred - q, pred + q


def coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of ``y`` inside ``[lower, upper]``."""
    y = np.asarray(y)
    return float(np.mean((y >= lower) & (y <= upper)))


# --------------------------------------------------------------------------
# Interval variants used in the project
# --------------------------------------------------------------------------

VARIANTS = ("multiplicative", "additive")


def kcal_interval(
    y_cal: np.ndarray, pred_cal: np.ndarray, pred: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Additive interval in kcal: ``pred ± q``, with the lower end clipped at 0."""
    lo, hi = absolute_residual_interval(y_cal, pred_cal, pred, alpha)
    return lo.clip(0), hi


def log_scale_interval(
    y_cal: np.ndarray, pred_log_cal: np.ndarray, pred_log: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Multiplicative interval in kcal.

    Residuals are measured on the log1p(kcal) scale, where predictions live,
    and the interval is transformed back, so its width is proportional to
    the predicted size of the dish.
    """
    lo, hi = absolute_residual_interval(np.log1p(y_cal), pred_log_cal, pred_log, alpha)
    return np.expm1(lo).clip(0), np.expm1(hi)


def conformal_intervals(
    y_cal: np.ndarray, pred_log_cal: np.ndarray, pred_log: np.ndarray, alpha: float
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Both variants for a model that predicts log1p(kcal): ``{variant: (lower, upper)}``."""
    pred_cal = np.expm1(pred_log_cal).clip(0)
    pred = np.expm1(pred_log).clip(0)
    return {
        "multiplicative": log_scale_interval(y_cal, pred_log_cal, pred_log, alpha),
        "additive": kcal_interval(y_cal, pred_cal, pred, alpha),
    }


def normalized_interval(
    y_cal: np.ndarray,
    pred_log_cal: np.ndarray,
    sigma_cal: np.ndarray,
    pred_log: np.ndarray,
    sigma: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalized conformal on the log1p(kcal) scale, returned in kcal.

    Scores are residuals divided by the estimated difficulty ``sigma`` of each
    point, so the interval ``pred ± q * sigma`` is wider where the model
    expects to be wrong. ``sigma`` must be positive.
    """
    scores = np.abs(np.log1p(y_cal) - pred_log_cal) / sigma_cal
    q = conformal_quantile(scores, alpha)
    half = q * np.asarray(sigma, dtype=float)
    return np.expm1(pred_log - half).clip(0), np.expm1(pred_log + half)


def cqr_interval(
    y_cal: np.ndarray,
    lo_log_cal: np.ndarray,
    hi_log_cal: np.ndarray,
    lo_log: np.ndarray,
    hi_log: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Conformalized quantile regression (Romano et al., 2019) on log1p(kcal), in kcal.

    ``lo_log`` / ``hi_log`` are predicted lower and upper quantiles. The score
    is how far the truth falls outside them (negative when inside); one
    correction ``q`` widens (or narrows, if negative) both ends.
    """
    y_log = np.log1p(y_cal)
    scores = np.maximum(lo_log_cal - y_log, y_log - hi_log_cal)
    q = conformal_quantile(scores, alpha)
    return np.expm1(np.asarray(lo_log) - q).clip(0), np.expm1(np.asarray(hi_log) + q)


def interval_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> float:
    """Mean interval (Winkler) score: width plus 2/alpha times the miss distance.

    Lower is better. It rewards narrow intervals but penalizes every unit by
    which the truth falls outside, so it balances width against misses.
    """
    y, lower, upper = (np.asarray(a, dtype=float) for a in (y, lower, upper))
    below = np.clip(lower - y, 0, None)
    above = np.clip(y - upper, 0, None)
    return float(np.mean((upper - lower) + 2 / alpha * (below + above)))


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def quantile_groups(values: np.ndarray, n: int = 4) -> tuple[np.ndarray, list[str]]:
    """Group index (0..n-1) of each value by its quantile, and a label per group."""
    values = np.asarray(values, dtype=float)
    edges = np.quantile(values, np.linspace(0, 1, n + 1))
    idx = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, n - 1)
    labels = [f"{a:.0f}–{b:.0f}" for a, b in pairwise(edges)]
    return idx, labels


def coverage_by_group(
    y: np.ndarray, lower: np.ndarray, upper: np.ndarray, groups: np.ndarray
) -> dict:
    """Per group: number of points, coverage and mean interval width."""
    y, lower, upper, groups = map(np.asarray, (y, lower, upper, groups))
    out = {}
    for g in np.unique(groups):
        m = groups == g
        out[g.item()] = {
            "n": int(m.sum()),
            "coverage": coverage(y[m], lower[m], upper[m]),
            "mean_width": float(np.mean(upper[m] - lower[m])),
        }
    return out


# --------------------------------------------------------------------------
# What the guarantee promises: coverage over random calibration sets
# --------------------------------------------------------------------------


def resample_splits(
    n_items: int,
    n_cal: int,
    n_reps: int = 1000,
    groups: np.ndarray | None = None,
    seed: int = 0,
):
    """Yield ``n_reps`` boolean calibration masks over a pool of ``n_items``.

    Each mask selects about ``n_cal`` items (whole ``groups`` at a time); the
    complement is the test set.
    """
    labels = np.arange(n_items) if groups is None else np.unique(groups, return_inverse=True)[1]
    sizes = np.bincount(labels)
    rng = np.random.default_rng(seed)
    for _ in range(n_reps):
        order = rng.permutation(len(sizes))
        k = int(np.searchsorted(np.cumsum(sizes[order]), n_cal)) + 1
        yield np.isin(labels, order[:k])


def resampled_coverage_of(
    y: np.ndarray,
    interval_fn,
    n_cal: int,
    n_reps: int = 1000,
    groups: np.ndarray | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Test coverage of any set of variants over random calibration/test splits.

    ``interval_fn(cal_mask)`` must return ``{variant: (lower, upper)}`` for the
    points outside ``cal_mask``, calibrated on the points inside it, with a
    model that is fixed and has not seen the pool.
    """
    y = np.asarray(y, dtype=float)
    out: dict[str, list[float]] = {}
    for cal in resample_splits(len(y), n_cal, n_reps, groups, seed):
        for v, (lo, hi) in interval_fn(cal).items():
            out.setdefault(v, []).append(coverage(y[~cal], lo, hi))
    return {v: np.array(c) for v, c in out.items()}


def resampled_coverage(
    y: np.ndarray,
    pred_log: np.ndarray,
    alpha: float,
    n_cal: int,
    n_reps: int = 1000,
    groups: np.ndarray | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Coverage of the additive and multiplicative variants over random re-splits.

    The model is fixed (``pred_log`` are its predictions for a pool of points
    it has not seen). Returns ``{variant: array of n_reps coverages}``.
    """
    y, pred_log = np.asarray(y, dtype=float), np.asarray(pred_log, dtype=float)
    return resampled_coverage_of(
        y,
        lambda cal: conformal_intervals(y[cal], pred_log[cal], pred_log[~cal], alpha),
        n_cal, n_reps, groups, seed,
    )


def coverage_distribution(n_cal: int, n_test: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact distribution of the coverage measured on ``n_test`` points.

    Given a calibration set of size ``n_cal``, the true coverage of split
    conformal follows Beta(k, n_cal + 1 - k) with k = ceil((n_cal + 1)(1 - alpha)).
    Measuring it on a finite test set adds binomial noise, so the observed
    fraction is Beta-Binomial. Returns the possible coverages and their
    probabilities.
    """
    k = math.ceil((n_cal + 1) * (1 - alpha))
    if k > n_cal:
        raise ValueError("n_cal too small for alpha: the interval is infinite")
    x = np.arange(n_test + 1)
    return x / n_test, betabinom.pmf(x, n_test, k, n_cal + 1 - k)
