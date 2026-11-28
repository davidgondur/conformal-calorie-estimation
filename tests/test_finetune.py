"""Fine-tuning loop on CPU with a tiny randomly initialised ViT and fake data."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conformal_calorie_estimation.finetune import (
    FinetuneConfig,
    build_model,
    make_transforms,
    param_groups,
    train,
    warmup_cosine,
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """40 dishes in 20 two-step incremental groups: 36 official train, 4 test."""
    (tmp_path / "metadata").mkdir()
    splits = tmp_path / "dish_ids" / "splits"
    splits.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows, ids = [], []
    for k in range(20):
        t = 10_000 * (k + 1)
        for step, ingr in enumerate(([f"a{k}"], [f"a{k}", "b"])):
            dish_id = f"dish_{t + 10 * step}"
            kcal = float(rng.uniform(50, 500))
            fields = [dish_id, f"{kcal}", "100", "1", "2", "3"]
            for i in ingr:
                fields += [i, i, "50", "10", "1", "1", "1"]
            rows.append(",".join(fields))
            ids.append(dish_id)
            d = tmp_path / "imagery" / "realsense_overhead" / dish_id
            d.mkdir(parents=True)
            Image.fromarray(rng.integers(0, 255, (30, 40, 3), dtype=np.uint8)).save(d / "rgb.png")
    (tmp_path / "metadata" / "dish_metadata_cafe1.csv").write_text("\n".join(rows) + "\n")
    (splits / "depth_train_ids.txt").write_text("\n".join(ids[:36]) + "\n")
    (splits / "depth_test_ids.txt").write_text("\n".join(ids[36:]) + "\n")
    return tmp_path


def tiny(root: Path, tmp_path: Path, **kw) -> FinetuneConfig:
    defaults = {
        "model_name": "vit_tiny_patch16_224", "pretrained": False, "img_size": 32, "epochs": 3,
        "warmup_epochs": 1, "batch_size": 4, "num_workers": 0, "root": str(root), "out_dir": str(tmp_path / "models"),
    }
    return FinetuneConfig(**(defaults | kw))


def test_full_run_writes_outputs_and_keeps_the_best_epoch(root, tmp_path):
    out = train(tiny(root, tmp_path, name="full"))
    history = list(csv.DictReader((out / "history.csv").open()))
    assert len(history) == 3
    outputs = np.load(out / "outputs.npz")
    best = int(np.argmin([float(r["val_mae"]) for r in history]))
    assert int(outputs["best_epoch"]) == best
    for part in ("train", "val", "calib", "test"):
        n = len(outputs[f"{part}_ids"])
        assert outputs[f"{part}_pred"].shape == (n,) and outputs[f"{part}_emb"].shape == (n, 192)
    assert "oof_ids" not in outputs
    assert not (out / "model.pt").exists()
    assert json.loads((out / "config.json").read_text())["name"] == "full"


def test_fold_run_predicts_the_left_out_groups(root, tmp_path):
    full = np.load(train(tiny(root, tmp_path, name="full", epochs=1)) / "outputs.npz")
    fold = np.load(train(tiny(root, tmp_path, name="fold0", epochs=1, fold=0)) / "outputs.npz")
    train_ids, oof_ids = set(fold["train_ids"]), set(fold["oof_ids"])
    assert oof_ids and not train_ids & oof_ids
    assert train_ids | oof_ids == set(full["train_ids"])
    # Both steps of an incremental scan share a timestamp block of 10_000.
    plates = lambda ids: {int(i[5:]) // 10_000 for i in ids}
    assert not plates(train_ids) & plates(oof_ids)


def test_layer_decay_and_partial_fine_tuning(root, tmp_path):
    cfg = tiny(root, tmp_path, name="x", lr_backbone=1e-4, lr_head=1e-2, layer_decay=0.5)
    model = build_model(cfg)
    lrs = {lr for g in param_groups(model, cfg) for lr in [g["lr"]]}
    assert max(lrs) == 1e-2 and min(lrs) == pytest.approx(1e-4 * 0.5**12)

    partial = tiny(root, tmp_path, name="y", train_last_blocks=2)
    model = build_model(partial)
    trainable = {n.split(".")[0] + ("." + n.split(".")[1] if n.startswith("blocks.") else "")
                 for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"blocks.10", "blocks.11", "norm", "head"}


def test_transforms_keep_the_input_size():
    cfg = FinetuneConfig(name="t", model_name="vit_tiny_patch16_224", pretrained=False, img_size=32)
    eval_tf, train_tf = make_transforms(build_model(cfg), 32)
    img = Image.fromarray(np.zeros((30, 40, 3), dtype=np.uint8))
    assert eval_tf(img).shape == train_tf(img).shape == (3, 32, 32)


def test_warmup_cosine():
    assert warmup_cosine(0, 100, 10) == pytest.approx(0.1)
    assert warmup_cosine(10, 100, 10) == pytest.approx(1.0)
    assert warmup_cosine(100, 100, 10) == pytest.approx(0.0)
