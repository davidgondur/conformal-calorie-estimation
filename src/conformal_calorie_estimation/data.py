"""Nutrition5k overhead-RGB dataset: metadata parsing, splits, and a PyTorch Dataset.

Expects the layout produced by ``conformal_calorie_estimation.download``.

Splits: only the official train/test splits are stored on disk
(:func:`official_splits`). Validation, calibration or any other partition is
carved out of ``train`` inside each script or notebook with
:func:`split_by_groups`, with an explicit seed, so the choice of fractions
lives next to the experiment that uses it. ``test`` is never subdivided.

Metadata format (verified against the files; the official README lists a
``num_ingrs`` column that does not exist): no header, 6 dish fields
``dish_id, total_calories, total_mass, total_fat, total_carb, total_protein``
followed by 7 fields per ingredient
``id, name, grams, calories, fat, carb, protein``.
"""

import csv
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

TARGETS = ("calories", "mass", "fat", "carb", "protein")
_DISH_FIELDS = 6
_INGR_FIELDS = 7


@dataclass(frozen=True)
class Ingredient:
    id: str
    name: str
    grams: float
    calories: float
    fat: float
    carb: float
    protein: float


@dataclass(frozen=True)
class Dish:
    id: str
    calories: float
    mass: float
    fat: float
    carb: float
    protein: float
    ingredients: tuple[Ingredient, ...]

    @property
    def timestamp(self) -> int:
        return int(self.id.removeprefix("dish_"))


def parse_dish_row(row: Sequence[str]) -> Dish:
    extra = len(row) - _DISH_FIELDS
    if extra < 0 or extra % _INGR_FIELDS:
        raise ValueError(f"malformed row for {row[0] if row else '?'}: {len(row)} fields")
    ingredients = tuple(
        Ingredient(row[i], row[i + 1], *map(float, row[i + 2 : i + _INGR_FIELDS]))
        for i in range(_DISH_FIELDS, len(row), _INGR_FIELDS)
    )
    return Dish(row[0], *map(float, row[1:_DISH_FIELDS]), ingredients=ingredients)


def read_dishes(root: Path) -> dict[str, Dish]:
    """All dishes from both cafeterias, keyed by dish id."""
    dishes = {}
    for path in sorted((Path(root) / "metadata").glob("dish_metadata_cafe*.csv")):
        with path.open(newline="") as f:
            for row in csv.reader(f):
                if row:
                    dish = parse_dish_row(row)
                    dishes[dish.id] = dish
    return dishes


def read_ids(path: Path) -> list[str]:
    return Path(path).read_text().split()


def image_path(root: Path, dish_id: str) -> Path:
    return Path(root) / "imagery" / "realsense_overhead" / dish_id / "rgb.png"


def incremental_groups(dishes: Iterable[Dish], max_gap_s: int = 300) -> list[list[str]]:
    """Group dishes that look like steps of one incremental scan.

    Nutrition5k plates are built one ingredient at a time and each step is its
    own dish, so consecutive dishes share most of their content. Two dishes are
    chained when they are less than ``max_gap_s`` apart and the earlier one's
    ingredients are a subset of the later one's. This is a heuristic: the
    official grouping is unpublished and does not match it exactly.
    """
    ordered = sorted(dishes, key=lambda d: d.timestamp)
    groups: list[list[str]] = []
    prev = None
    for dish in ordered:
        ingr = {i.id for i in dish.ingredients}
        if (
            prev is not None
            and dish.timestamp - prev.timestamp < max_gap_s
            and {i.id for i in prev.ingredients} <= ingr
        ):
            groups[-1].append(dish.id)
        else:
            groups.append([dish.id])
        prev = dish
    return groups


def official_splits(root: Path) -> dict[str, list[str]]:
    """The official ``depth_train`` / ``depth_test`` splits, as stored on disk.

    These are the only splits persisted locally. Dishes without metadata or
    an overhead image are dropped (3 train dishes have no ``rgb.png``).
    Any further division of ``train`` (validation, calibration, ...) is done
    in each script or notebook with :func:`split_by_groups`.
    """
    root = Path(root)
    dishes = read_dishes(root)
    splits_dir = root / "dish_ids" / "splits"

    def usable(name: str) -> list[str]:
        ids = read_ids(splits_dir / f"depth_{name}_ids.txt")
        return sorted(i for i in ids if i in dishes and image_path(root, i).exists())

    return {"train": usable("train"), "test": usable("test")}


def split_by_groups(
    dish_ids: Sequence[str],
    dishes: dict[str, Dish],
    fractions: dict[str, float],
    seed: int = 0,
) -> dict[str, list[str]]:
    """Divide ``dish_ids`` into parts of roughly the given fractions.

    Whole incremental groups (see :func:`incremental_groups`) are assigned to
    one part, so near-duplicate steps of the same plate never end up on both
    sides; otherwise validation or conformal calibration scores would be
    optimistic. The remaining dishes go to ``"train"``. Deterministic for a
    given ``seed``.

    Example: ``split_by_groups(ids, dishes, {"val": 0.1, "calib": 0.2})``.
    """
    if "train" in fractions:
        raise ValueError('"train" receives the remainder; do not list it in fractions')
    if sum(fractions.values()) >= 1:
        raise ValueError("fractions must sum to less than 1")
    groups = incremental_groups(dishes[i] for i in dish_ids)
    random.Random(seed).shuffle(groups)
    quotas = {name: round(f * len(dish_ids)) for name, f in fractions.items()}
    parts: dict[str, list[str]] = {name: [] for name in [*fractions, "train"]}
    for group in groups:
        name = next((n for n, q in quotas.items() if len(parts[n]) < q), "train")
        parts[name].extend(group)
    return {name: sorted(ids) for name, ids in parts.items()}


MAX_PLAUSIBLE_MASS_G = 2000.0


def label_problem(dish: Dish) -> str | None:
    """Why ``dish`` has an obviously wrong label, or None if it looks fine.

    Rules (checked against the photos in notebooks/EDA.py):

    - an empty "plate only" label or 0 kcal: food is visible in the photo;
    - total mass above ``MAX_PLAUSIBLE_MASS_G``: the one case is a ~3 kg
      rice entry for a small portion. The next heaviest dish is ~1.1 kg.

    Very low calories are *not* a problem: they are real one-ingredient
    plates (a few grams of spinach, cucumber...).
    """
    if dish.calories <= 0 or any(i.name == "plate only" for i in dish.ingredients):
        return "empty plate label / 0 kcal"
    if dish.mass > MAX_PLAUSIBLE_MASS_G:
        return f"mass > {MAX_PLAUSIBLE_MASS_G:.0f} g"
    return None


def clean_train(dish_ids: Iterable[str], dishes: dict[str, Dish]) -> list[str]:
    """Drop dishes with obviously wrong labels (see :func:`label_problem`).

    Apply it only to the part used to fit model weights, after
    :func:`split_by_groups`. Validation, calibration and test stay as they
    are: calibration must look like test for the coverage guarantee to hold,
    and removing hard cases from test would make the reported coverage
    optimistic.
    """
    return [i for i in dish_ids if label_problem(dishes[i]) is None]


class Nutrition5kOverhead(Dataset):
    """Overhead RGB photo -> nutrition targets.

    Returns ``(image, target)`` where ``image`` is whatever ``transform``
    makes of the PIL image (the PIL image itself if ``transform`` is None)
    and ``target`` is a float32 tensor with one value per name in ``targets``.
    """

    def __init__(
        self,
        root: Path,
        dish_ids: Sequence[str],
        targets: Sequence[str] = ("calories",),
        transform: Callable[[Image.Image], object] | None = None,
        dishes: dict[str, Dish] | None = None,
    ):
        unknown = set(targets) - set(TARGETS)
        if unknown:
            raise ValueError(f"unknown targets {sorted(unknown)}; choose from {TARGETS}")
        self.root = Path(root)
        self.dish_ids = list(dish_ids)
        self.targets = tuple(targets)
        self.transform = transform
        self.dishes = dishes if dishes is not None else read_dishes(self.root)

    def __len__(self) -> int:
        return len(self.dish_ids)

    def __getitem__(self, index: int):
        dish = self.dishes[self.dish_ids[index]]
        with Image.open(image_path(self.root, dish.id)) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        target = torch.tensor([getattr(dish, t) for t in self.targets], dtype=torch.float32)
        return image, target
