"""Frozen-backbone baseline: data preparation and ridge on log1p(kcal).

Shared by notebooks/baseline.py and docs/make_figures.py so both always run
the same recipe.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import (
    Dish,
    clean_train,
    incremental_groups,
    official_splits,
    read_dishes,
    split_by_groups,
)
from .features import DEFAULT_MODEL, cached_features

RIDGE_ALPHAS = (1, 10, 30, 100, 300, 1000, 3000, 10000)
DEFAULT_FRACTIONS = {"val": 0.1, "calib": 0.2}


@dataclass
class BaselineData:
    dishes: dict[str, Dish]
    parts: dict[str, list[str]]  # train / val / calib / test dish ids
    removed: list[str]  # training dishes dropped by clean_train
    X: dict[str, np.ndarray]  # embeddings per part
    kcal: dict[str, np.ndarray]  # true calories per part


def make_parts(
    root: Path, seed: int = 0, fractions: dict[str, float] = DEFAULT_FRACTIONS
) -> tuple[dict[str, Dish], dict[str, list[str]], list[str]]:
    """Dishes, the project partitions and the training dishes removed by cleaning.

    ``test`` is the official test split. ``val`` and ``calib`` come from the
    official train split by incremental groups; only ``train`` is cleaned.
    """
    dishes = read_dishes(root)
    official = official_splits(root)
    raw = split_by_groups(official["train"], dishes, fractions, seed=seed)
    train = clean_train(raw["train"], dishes)
    parts = {"train": train, **{k: raw[k] for k in fractions}, "test": official["test"]}
    return dishes, parts, sorted(set(raw["train"]) - set(train))


def prepare_baseline(
    root: Path,
    features_dir: Path,
    seed: int = 0,
    fractions: dict[str, float] = DEFAULT_FRACTIONS,
    model_name: str = DEFAULT_MODEL,
    img_size: int = 224,
) -> BaselineData:
    """The project partitions (see :func:`make_parts`) with frozen-backbone embeddings."""
    dishes, parts, removed = make_parts(root, seed, fractions)
    official = official_splits(root)
    feats = cached_features(
        root, official["train"] + official["test"], features_dir,
        model_name=model_name, img_size=img_size,
    )
    return BaselineData(
        dishes=dishes,
        parts=parts,
        removed=removed,
        X={k: np.stack([feats[i] for i in ids]) for k, ids in parts.items()},
        kcal={k: np.array([dishes[i].calories for i in ids]) for k, ids in parts.items()},
    )


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


@dataclass
class RidgeFit:
    model: Pipeline  # predicts log1p(kcal)
    alpha: float
    search: list[dict]  # [{"alpha": ..., "val_mae": ...}, ...]


def fit_log_ridge(
    X: dict[str, np.ndarray], kcal: dict[str, np.ndarray], alphas: Sequence[float] = RIDGE_ALPHAS
) -> RidgeFit:
    """Standardized ridge on log1p(kcal), regularization chosen by val MAE in kcal.

    Only ``train`` and ``val`` are read; calibration and test never influence
    the model.
    """

    def fit(a):
        return make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(X["train"], np.log1p(kcal["train"]))

    search = [
        {"alpha": a, "val_mae": mae(kcal["val"], np.expm1(fit(a).predict(X["val"])).clip(0))}
        for a in alphas
    ]
    best = min(search, key=lambda r: r["val_mae"])["alpha"]
    return RidgeFit(model=fit(best), alpha=best, search=search)


def group_labels(dish_ids: Sequence[str], dishes: dict[str, Dish]) -> np.ndarray:
    """Incremental-group index of each dish, aligned with ``dish_ids``."""
    label = {}
    for g, group in enumerate(incremental_groups(dishes[i] for i in dish_ids)):
        label.update(dict.fromkeys(group, g))
    return np.array([label[i] for i in dish_ids])
