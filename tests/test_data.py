from pathlib import Path

import pytest
import torch
from PIL import Image

from conformal_calorie_estimation.data import (
    Nutrition5kOverhead,
    incremental_groups,
    official_splits,
    parse_dish_row,
    read_dishes,
    split_by_groups,
)

# Real row from dish_metadata_cafe1.csv
ROW = (
    "dish_1562688426,137.569992,88.000000,8.256000,5.190000,10.297000,"
    "ingr_0000000433,roasted potatoes,17.000000,23.97,1.156,3.06,0.357,"
    "ingr_0000000510,chicken apple sausage,71.000000,113.6,7.1,2.13,9.94"
)


def _row(dish_id: str, ingr_ids: list[str]) -> str:
    fields = [dish_id, "100", "50", "1", "2", "3"]
    for i in ingr_ids:
        fields += [i, i, "10", "20", "1", "1", "1"]
    return ",".join(fields)


def test_parse_dish_row():
    dish = parse_dish_row(ROW.split(","))
    assert dish.id == "dish_1562688426"
    assert dish.calories == pytest.approx(137.569992)
    assert dish.mass == 88.0
    assert [i.name for i in dish.ingredients] == ["roasted potatoes", "chicken apple sausage"]
    assert dish.ingredients[1].grams == 71.0


def test_parse_rejects_truncated_row():
    with pytest.raises(ValueError):
        parse_dish_row(ROW.split(",")[:-1])


def test_incremental_groups_chain_supersets_within_gap():
    rows = [
        _row("dish_1000", ["a"]),
        _row("dish_1030", ["a", "b"]),  # adds an ingredient -> same group
        _row("dish_1060", ["c"]),  # new plate
        _row("dish_9000", ["c", "d"]),  # superset but too late
    ]
    dishes = [parse_dish_row(r.split(",")) for r in rows]
    assert incremental_groups(dishes) == [["dish_1000", "dish_1030"], ["dish_1060"], ["dish_9000"]]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "metadata").mkdir()
    splits = tmp_path / "dish_ids" / "splits"
    splits.mkdir(parents=True)
    # 10 two-step plates for train, 1 test dish, 1 train id without image.
    rows, train_ids = [], []
    for k in range(10):
        t = 10_000 * (k + 1)
        rows += [_row(f"dish_{t}", [f"x{k}"]), _row(f"dish_{t + 10}", [f"x{k}", "y"])]
        train_ids += [f"dish_{t}", f"dish_{t + 10}"]
    rows.append(_row("dish_999999", ["z"]))
    (tmp_path / "metadata" / "dish_metadata_cafe1.csv").write_text("\n".join(rows) + "\n")
    (splits / "depth_train_ids.txt").write_text("\n".join(train_ids + ["dish_123"]) + "\n")
    (splits / "depth_test_ids.txt").write_text("dish_999999\n")
    for dish_id in train_ids + ["dish_999999"]:
        d = tmp_path / "imagery" / "realsense_overhead" / dish_id
        d.mkdir(parents=True)
        Image.new("RGB", (8, 6), "red").save(d / "rgb.png")
    return tmp_path


def test_official_splits_drop_dishes_without_image(root: Path):
    splits = official_splits(root)
    assert splits["test"] == ["dish_999999"]
    assert len(splits["train"]) == 20
    assert "dish_123" not in splits["train"]


def test_split_by_groups_keeps_groups_together(root: Path):
    train = official_splits(root)["train"]
    parts = split_by_groups(train, read_dishes(root), {"val": 0.1, "calib": 0.2}, seed=1)
    assert [len(parts[k]) for k in ("val", "calib", "train")] == [2, 4, 14]
    assert sorted(sum(parts.values(), [])) == train
    plates = {k: {int(i[5:]) // 10_000 for i in v} for k, v in parts.items()}
    assert not (plates["val"] & plates["calib"] or plates["val"] & plates["train"]
                or plates["calib"] & plates["train"])
    assert parts == split_by_groups(train, read_dishes(root), {"val": 0.1, "calib": 0.2}, seed=1)


def test_split_by_groups_rejects_bad_fractions(root: Path):
    dishes = read_dishes(root)
    with pytest.raises(ValueError):
        split_by_groups([], dishes, {"train": 0.5})
    with pytest.raises(ValueError):
        split_by_groups([], dishes, {"val": 0.6, "calib": 0.4})


def test_dataset_returns_image_and_targets(root: Path):
    ds = Nutrition5kOverhead(
        root, ["dish_999999"], targets=("calories", "mass"), dishes=read_dishes(root)
    )
    image, target = ds[0]
    assert image.size == (8, 6)
    assert torch.equal(target, torch.tensor([100.0, 50.0]))


def test_dataset_rejects_unknown_target(root: Path):
    with pytest.raises(ValueError):
        Nutrition5kOverhead(root, [], targets=("sugar",))
