import numpy as np

from conformal_calorie_estimation.adaptive import (
    DifficultyFit,
    QuantileFit,
    cross_fitted_residuals,
    group_kfold,
    pinball_loss,
)


def test_group_kfold_keeps_groups_and_balances():
    groups = np.repeat(np.arange(40), np.arange(40) % 4 + 1)  # sizes 1..4
    folds = group_kfold(groups, k=5, seed=0)
    for g in np.unique(groups):
        assert len(set(folds[groups == g])) == 1
    sizes = np.bincount(folds)
    assert sizes.max() - sizes.min() <= 4


def test_cross_fitted_residuals_are_out_of_fold():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    kcal = np.expm1(5 + X @ np.array([1.0, -1.0, 0.5]))
    folds = np.repeat(np.arange(3), 20)
    # A memorizing fit would give ~0 residuals; out-of-fold ridge on a
    # noiseless linear target is accurate but not exact.
    r = cross_fitted_residuals(X, kcal, folds, ridge_alpha=1.0)
    assert r.shape == (60,) and np.all(r > 0) and r.mean() < 0.5


class _Const:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float) if np.ndim(self.value) == 0 else self.value


def test_difficulty_sigma_has_a_floor():
    fit = DifficultyFit(model=_Const(np.log(np.array([1e-6, 0.5]))), alpha=1, floor=0.1, search=[])
    assert np.allclose(fit.sigma(np.zeros((2, 1))), [0.1, 0.5])


def test_crossed_quantiles_are_swapped_and_counted():
    fit = QuantileFit(
        alpha=0.1, lower=_Const(np.array([1.0, 3.0])), upper=_Const(np.array([2.0, 2.5])), l1={}, search=[]
    )
    lo, hi, crossed = fit.predict(np.zeros((2, 1)))
    assert crossed == 1 and lo.tolist() == [1.0, 2.5] and hi.tolist() == [2.0, 3.0]


def test_pinball_loss_is_minimized_by_the_quantile():
    y = np.random.default_rng(0).normal(size=5000)
    q = 0.9
    losses = {c: pinball_loss(y, np.full_like(y, c), q) for c in (0.0, np.quantile(y, q), 2.5)}
    assert min(losses, key=losses.get) == np.quantile(y, q)
