"""Fine-tune a ViT backbone to predict log1p(kcal) from the overhead photo.

    uv run n5k-finetune --name dinov2_lr1e-5 --lr-backbone 1e-5

Each run writes to ``models/<name>/``: ``config.json``, ``history.csv`` (one
row per epoch), ``outputs.npz`` (predictions and embeddings of every
partition, from the epoch with the lowest val MAE) and, optionally,
``model.pt``. Model selection reads ``val`` only; ``calib`` and ``test`` are
predicted once, at the end, and never influence training.

With ``--fold k`` the run leaves fold ``k`` of ``train`` out (grouped
5-fold) and also predicts it (``oof_*``): those out-of-fold residuals and
embeddings train the difficulty and quantile models of the adaptive
intervals, as the ridge cross-fitting does for the frozen baseline.
"""

import argparse
import copy
import csv
import json
import math
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import timm
import torch
from timm.data import resolve_data_config
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from .adaptive import PointModelOutputs, group_kfold
from .baseline import DEFAULT_FRACTIONS, group_labels, make_parts
from .data import Nutrition5kOverhead
from .features import DEFAULT_MODEL

TARGETS_SINGLE = ("calories",)
TARGETS_MULTI = ("calories", "mass", "fat", "carb", "protein")


@dataclass
class FinetuneConfig:
    name: str
    model_name: str = DEFAULT_MODEL
    pretrained: bool = True
    img_size: int = 224
    seed: int = 0  # training randomness (head init, shuffling, augmentation)
    split_seed: int = 0  # project partitions; keep 0 to match the baseline
    lr_backbone: float = 1e-5
    lr_head: float = 1e-3
    layer_decay: float = 0.75
    train_last_blocks: int | None = None  # None: fine-tune every block
    multitask: bool = False
    epochs: int = 40
    warmup_epochs: int = 2
    weight_decay: float = 0.05
    batch_size: int = 64
    num_workers: int = 8
    fold: int | None = None  # leave this fold of train out (grouped k-fold)
    n_folds: int = 5
    save_weights: bool = False
    root: str = "data/nutrition5k"
    out_dir: str = "models"

    @property
    def targets(self) -> tuple[str, ...]:
        return TARGETS_MULTI if self.multitask else TARGETS_SINGLE


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


class RandomRot90(nn.Module):
    """Rotate by a random multiple of 90 degrees: an overhead photo has no up."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, int(torch.randint(4, ()).item()), dims=(-2, -1))


def make_transforms(model: nn.Module, img_size: int) -> tuple[v2.Compose, v2.Compose]:
    """Eval transform (the model's own) and train transform (eval + augmentation).

    Augmentations keep the apparent amount of food: rotations, flips and mild
    colour changes, but no random zoom crops, which would teach the model
    that the same plate can look larger or smaller.
    """
    cfg = resolve_data_config({"input_size": (3, img_size, img_size)}, model=model)
    base = [
        v2.PILToTensor(),
        v2.Resize(img_size, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        v2.CenterCrop(img_size),
    ]
    normalize = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(cfg["mean"], cfg["std"])]
    augment = [
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        RandomRot90(),
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    ]
    return v2.Compose(base + normalize), v2.Compose(base + augment + normalize)


# --------------------------------------------------------------------------
# Model and optimizer
# --------------------------------------------------------------------------


def build_model(cfg: FinetuneConfig) -> nn.Module:
    model = timm.create_model(
        cfg.model_name, pretrained=cfg.pretrained, num_classes=len(cfg.targets), img_size=cfg.img_size
    )
    if cfg.train_last_blocks is not None:
        n_frozen = len(model.blocks) - cfg.train_last_blocks
        for name, p in model.named_parameters():
            in_frozen_block = name.startswith("blocks.") and int(name.split(".")[1]) < n_frozen
            if in_frozen_block or not (name.startswith(("blocks.", "norm", "fc_norm", "head"))):
                p.requires_grad = False
    return model


def param_groups(model: nn.Module, cfg: FinetuneConfig) -> list[dict]:
    """AdamW groups with layer-wise learning-rate decay.

    The head gets ``lr_head``. Block ``i`` of ``L`` gets
    ``lr_backbone * layer_decay ** (L - 1 - i)``, so blocks near the input
    change least; the patch embedding gets the smallest rate. Biases, norms
    and embeddings get no weight decay.
    """
    n_blocks = len(model.blocks)
    groups: dict[tuple[float, float], list] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(("head", "fc_norm")):
            lr = cfg.lr_head
        elif name.startswith("blocks."):
            lr = cfg.lr_backbone * cfg.layer_decay ** (n_blocks - 1 - int(name.split(".")[1]))
        elif name.startswith("norm"):
            lr = cfg.lr_backbone
        else:  # patch_embed, cls_token, pos_embed, reg_token
            lr = cfg.lr_backbone * cfg.layer_decay**n_blocks
        no_decay = p.ndim <= 1 or name.endswith(("cls_token", "pos_embed", "reg_token"))
        groups.setdefault((lr, 0.0 if no_decay else cfg.weight_decay), []).append(p)
    return [{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items()]


def warmup_cosine(step: int, total: int, warmup: int) -> float:
    """Multiplier of the base learning rate: linear warm-up, then cosine to 0."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_parts(cfg: FinetuneConfig, dishes, parts) -> dict[str, list[str]]:
    """Partitions for this run: with ``fold`` set, train loses one grouped fold (``oof``)."""
    if cfg.fold is None:
        return dict(parts)
    folds = group_kfold(group_labels(parts["train"], dishes), k=cfg.n_folds, seed=cfg.split_seed)
    train = np.array(parts["train"])
    return {**parts, "train": train[folds != cfg.fold].tolist(), "oof": train[folds == cfg.fold].tolist()}


@torch.inference_mode()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Log-target predictions (n, n_targets) and pooled embeddings (n, dim)."""
    model.eval()
    preds, embs = [], []
    for x, _ in loader:
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            emb = model.forward_head(model.forward_features(x.to(device)), pre_logits=True)
            preds.append(model.head(emb).float().cpu())
            embs.append(emb.float().cpu())
    return torch.cat(preds).numpy(), torch.cat(embs).numpy()


def train(cfg: FinetuneConfig) -> Path:
    """Run one fine-tuning and write its outputs. Returns the run directory."""
    out = Path(cfg.out_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    seed_everything(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dishes, parts, _ = make_parts(Path(cfg.root), seed=cfg.split_seed, fractions=DEFAULT_FRACTIONS)
    parts = run_parts(cfg, dishes, parts)
    model = build_model(cfg).to(device)
    eval_tf, train_tf = make_transforms(model, cfg.img_size)

    def loader(ids, transform, shuffle=False):
        ds = Nutrition5kOverhead(cfg.root, ids, targets=cfg.targets, transform=transform, dishes=dishes)
        return DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=shuffle, drop_last=shuffle,
            num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0,
            generator=torch.Generator().manual_seed(cfg.seed),
        )

    train_loader = loader(parts["train"], train_tf, shuffle=True)
    val_loader = loader(parts["val"], eval_tf)
    val_kcal = np.array([dishes[i].calories for i in parts["val"]])

    opt = torch.optim.AdamW(param_groups(model, cfg))
    base_lrs = [g["lr"] for g in opt.param_groups]
    total = cfg.epochs * len(train_loader)
    warmup = cfg.warmup_epochs * len(train_loader)

    history, best, step = [], None, 0
    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for x, y in train_loader:
            for g, lr in zip(opt.param_groups, base_lrs, strict=True):
                g["lr"] = lr * warmup_cosine(step, total, warmup)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                pred = model(x.to(device)).float()
            loss = (pred - torch.log1p(y.to(device))).abs().mean()  # L1 on log1p targets
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            step += 1
        val_pred, _ = predict(model, val_loader, device)
        val_mae = float(np.mean(np.abs(np.expm1(val_pred[:, 0]).clip(0) - val_kcal)))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_mae": val_mae})
        if best is None or val_mae < best[0]:
            best = (val_mae, epoch, copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}))
        print(f"[{cfg.name}] epoch {epoch:2d}  loss {history[-1]['train_loss']:.4f}  val MAE {val_mae:6.1f}", flush=True)

    with (out / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    model.load_state_dict(best[2])
    if cfg.save_weights:
        torch.save(best[2], out / "model.pt")
    arrays = {"best_epoch": np.array(best[1]), "best_val_mae": np.array(best[0])}
    for part, ids in parts.items():
        pred, emb = predict(model, loader(ids, eval_tf), device)
        arrays |= {f"{part}_ids": np.array(ids), f"{part}_pred": pred[:, 0], f"{part}_emb": emb}
    np.savez(out / "outputs.npz", **arrays)
    return out


# --------------------------------------------------------------------------
# Reading finished runs
# --------------------------------------------------------------------------


def load_config(run_dir: Path) -> FinetuneConfig:
    return FinetuneConfig(**json.loads((Path(run_dir) / "config.json").read_text()))


def run_summary(run_dir: Path) -> dict:
    """Configuration, best epoch and best val MAE of a finished run, plus its history."""
    run_dir = Path(run_dir)
    o = np.load(run_dir / "outputs.npz")
    with (run_dir / "history.csv").open() as f:
        history = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    return {
        **asdict(load_config(run_dir)),
        "best_epoch": int(o["best_epoch"]),
        "best_val_mae": float(o["best_val_mae"]),
        "history": history,
    }


def finetuned_outputs(run_dir: Path, require_folds: bool = True) -> PointModelOutputs:
    """A finished run as :class:`PointModelOutputs`, ready for the conformal variants.

    Predictions and embeddings come from the final model. The honest train
    residuals and embeddings come from the grouped-fold runs
    ``<name>_fold<k>`` next to it: each predicted the fold it never saw.
    Without them (``require_folds=False``) the honest fields are left empty
    and only the additive and multiplicative variants can be computed.
    """
    run_dir = Path(run_dir)
    cfg = load_config(run_dir)
    dishes, parts, _ = make_parts(Path(cfg.root), seed=cfg.split_seed, fractions=DEFAULT_FRACTIONS)
    o = np.load(run_dir / "outputs.npz")
    for part, ids in parts.items():
        if o[f"{part}_ids"].tolist() != ids:
            raise ValueError(f"{run_dir}: saved {part} ids do not match the project partitions")
    kcal = {p: np.array([dishes[i].calories for i in ids]) for p, ids in parts.items()}

    fold_dirs = [run_dir.parent / f"{cfg.name}_fold{k}" for k in range(cfg.n_folds)]
    honest_residual = honest_emb = None
    if all((d / "outputs.npz").exists() for d in fold_dirs):
        pred, emb = {}, {}
        for d in fold_dirs:
            f = np.load(d / "outputs.npz")
            for i, dish_id in enumerate(f["oof_ids"].tolist()):
                pred[dish_id], emb[dish_id] = f["oof_pred"][i], f["oof_emb"][i]
        if set(pred) != set(parts["train"]):
            raise ValueError(f"{run_dir}: fold runs do not cover train exactly once")
        honest_emb = np.stack([emb[i] for i in parts["train"]])
        honest_residual = np.abs(np.log1p(kcal["train"]) - np.array([pred[i] for i in parts["train"]]))
    elif require_folds:
        raise FileNotFoundError(f"{run_dir}: missing fold runs {[d.name for d in fold_dirs]}")

    return PointModelOutputs(
        dishes=dishes,
        parts=parts,
        kcal=kcal,
        pred_log={p: o[f"{p}_pred"] for p in parts},
        emb={p: o[f"{p}_emb"] for p in parts},
        honest_residual=honest_residual,
        honest_emb=honest_emb,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for f in fields(FinetuneConfig):
        flag = "--" + f.name.replace("_", "-")
        if f.name == "name":
            parser.add_argument(flag, required=True)
        elif f.type is bool:
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        elif f.type in (int, float, str):
            parser.add_argument(flag, type=f.type, default=f.default)
        else:  # int | None
            parser.add_argument(flag, type=int, default=f.default)
    train(FinetuneConfig(**vars(parser.parse_args(argv))))


if __name__ == "__main__":
    main()
