"""Split conformal prediction for regression.

Kept small and explicit on purpose: the formulas are the point of the project.
"""

import math

import numpy as np


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
