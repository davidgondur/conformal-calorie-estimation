import math

import numpy as np
import pytest

from conformal_calorie_estimation.conformal import (
    absolute_residual_interval,
    conformal_quantile,
    coverage,
)


def test_quantile_uses_finite_sample_correction():
    scores = np.arange(1, 10)  # n = 9
    # ceil(10 * 0.9) = 9 -> 9th smallest, not the plain 90 % quantile.
    assert conformal_quantile(scores, 0.1) == 9
    # ceil(10 * 0.5) = 5 -> 5th smallest.
    assert conformal_quantile(scores, 0.5) == 5


def test_quantile_infinite_when_too_few_points():
    assert conformal_quantile(np.arange(5), 0.1) == math.inf


def test_quantile_rejects_bad_alpha():
    with pytest.raises(ValueError):
        conformal_quantile(np.arange(5), 1.0)


def test_coverage_holds_on_exchangeable_data():
    rng = np.random.default_rng(0)
    alpha, covs = 0.1, []
    for _ in range(200):
        y = rng.normal(size=600)
        pred = np.zeros_like(y)
        lo, hi = absolute_residual_interval(y[:300], pred[:300], pred[300:], alpha)
        covs.append(coverage(y[300:], lo, hi))
    assert np.mean(covs) >= 1 - alpha - 0.005
