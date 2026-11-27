import numpy as np

from conformal_calorie_estimation.baseline import fit_log_ridge


def test_fit_log_ridge_picks_best_val_alpha_and_ignores_calib_and_test():
    rng = np.random.default_rng(0)
    w = rng.normal(size=20)

    def make(n):
        X = rng.normal(size=(n, 20))
        return X, np.expm1(4 + 0.1 * X @ w + rng.normal(0, 0.1, n))

    X, kcal = {}, {}
    for part, n in [("train", 300), ("val", 100)]:
        X[part], kcal[part] = make(n)
    fit = fit_log_ridge(X, kcal, alphas=(0.1, 10, 1e6))
    assert fit.alpha == min(fit.search, key=lambda r: r["val_mae"])["alpha"]
    assert fit.alpha != 1e6  # heavy regularization predicts a constant

    # calib and test are not read: garbage there changes nothing.
    X["calib"], kcal["calib"] = np.full((5, 20), np.nan), np.full(5, np.nan)
    X["test"], kcal["test"] = np.full((5, 20), np.nan), np.full(5, np.nan)
    assert fit_log_ridge(X, kcal, alphas=(0.1, 10, 1e6)).alpha == fit.alpha
