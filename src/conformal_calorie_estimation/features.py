"""Frozen-backbone image embeddings, cached on disk.

Used by baselines that fit a light regressor on top of a pretrained ViT
without fine-tuning it.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import timm
import torch
from timm.data import create_transform, resolve_data_config
from torch.utils.data import DataLoader

from .data import Nutrition5kOverhead

DEFAULT_MODEL = "vit_small_patch14_dinov2.lvd142m"


@torch.inference_mode()
def extract_features(
    root: Path,
    dish_ids: Sequence[str],
    model_name: str = DEFAULT_MODEL,
    img_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 8,
    device: str | None = None,
) -> np.ndarray:
    """Pooled embedding of each dish's overhead photo, shape (n, dim).

    Images go through the model's own eval transform (resize, center crop,
    normalization) at ``img_size``.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=img_size)
    model = model.eval().to(device)
    # The pretrained config still says 518 px; override it with img_size.
    config = resolve_data_config({"input_size": (3, img_size, img_size)}, model=model)
    transform = create_transform(**config)
    loader = DataLoader(
        Nutrition5kOverhead(root, dish_ids, transform=transform),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    out = []
    for images, _ in loader:
        with torch.autocast(device_type=device, enabled=device == "cuda"):
            out.append(model(images.to(device)).float().cpu())
    return torch.cat(out).numpy()


def cached_features(
    root: Path,
    dish_ids: Sequence[str],
    cache_dir: Path,
    model_name: str = DEFAULT_MODEL,
    img_size: int = 224,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Embeddings for ``dish_ids`` as ``{dish_id: vector}``, computed once.

    The cache file is keyed by model and image size. If it lacks any of the
    requested ids, everything is recomputed for ``dish_ids``.
    """
    path = Path(cache_dir) / f"{model_name.replace('/', '_')}_{img_size}.npz"
    if path.exists():
        cached = np.load(path)
        feats = dict(zip(cached["ids"].tolist(), cached["features"]))
        if set(dish_ids) <= feats.keys():
            return feats
    ids = list(dish_ids)
    array = extract_features(root, ids, model_name, img_size, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.array(ids), features=array)
    return dict(zip(ids, array))
