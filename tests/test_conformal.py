import math

import numpy as np
import pytest

from conformal_calorie_estimation.conformal import (
    VARIANTS,
    absolute_residual_interval,
    conformal_intervals,
    conformal_quantile,
    coverage,
    coverage_by_group,
    coverage_distribution,
    log_scale_interval,
    quantile_groups,
    resampled_coverage,
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


def test_log_scale_interval_is_positive_and_contains_prediction():
    rng = np.random.default_rng(1)
    y_cal = rng.uniform(0, 800, 200)
    pred_log_cal = np.log1p(y_cal) + rng.normal(0, 0.3, 200)
    pred_log = np.log1p(np.array([0.0, 5.0, 300.0, 900.0]))
    lo, hi = log_scale_interval(y_cal, pred_log_cal, pred_log, 0.1)
    pred = np.expm1(pred_log)
    assert np.all(lo >= 0) and np.all(lo <= pred) and np.all(pred <= hi)
    # Multiplicative: the upper/prediction ratio is the same for large dishes.
    assert (hi[3] + 1) / (pred[3] + 1) == pytest.approx((hi[2] + 1) / (pred[2] + 1))


def test_conformal_intervals_has_both_variants():
    y = np.array([10.0, 100.0, 1000.0] * 20)
    ints = conformal_intervals(y, np.log1p(y), np.log1p(np.array([50.0])), 0.2)
    assert set(ints) == set(VARIANTS)
    assert ints["additive"][0][0] >= 0


def test_coverage_by_group():
    y = np.array([1, 2, 3, 4])
    lo, hi = np.zeros(4), np.array([5, 5, 0, 5])
    out = coverage_by_group(y, lo, hi, np.array([0, 0, 1, 1]))
    assert out[0] == {"n": 2, "coverage": 1.0, "mean_width": 5.0}
    assert out[1]["coverage"] == 0.5


def test_quantile_groups_splits_evenly():
    idx, labels = quantile_groups(np.arange(100), n=4)
    assert np.bincount(idx).tolist() == [25, 25, 25, 25] and len(labels) == 4


def test_resampled_coverage_matches_theory():
    rng = np.random.default_rng(0)
    n, n_cal, alpha = 1000, 500, 0.1
    y = rng.lognormal(5, 0.8, n)
    pred_log = np.log1p(y) + rng.normal(0, 0.4, n)  # a fixed, imperfect model
    covs = resampled_coverage(y, pred_log, alpha, n_cal, n_reps=400, seed=0)
    grid, pmf = coverage_distribution(n_cal, n - n_cal, alpha)
    mean, sd = float(grid @ pmf), float(np.sqrt(((grid - grid @ pmf) ** 2) @ pmf))
    for c in covs.values():
        assert abs(c.mean() - mean) < 0.005  # theory mean is about 1 - alpha
        assert 0.7 * sd < c.std() < 1.3 * sd
    assert mean >= 1 - alpha


def test_resampled_coverage_keeps_groups_together():
    y = np.arange(1, 101, dtype=float)
    groups = np.repeat(np.arange(20), 5)  # 20 groups of 5
    # Calibration of ~23 points must take whole groups: 25 points, so 75 in test.
    covs = resampled_coverage(y, np.log1p(y), 0.1, n_cal=23, n_reps=5, groups=groups)
    assert all(np.isfinite(c).all() for c in covs.values())
