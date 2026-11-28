"""Run the fine-tuning experiments, skipping runs that already finished.

    uv run python scripts/finetune_sweep.py search   # 12 runs, one seed
    uv run python scripts/finetune_sweep.py dinov2   # best recipe, 3 seeds x (full + 5 folds)
    uv run python scripts/finetune_sweep.py dinov3   # best recipe on DINOv3, 3 seeds
    uv run python scripts/finetune_sweep.py all

The recipe for the dinov2 and dinov3 phases is the search run with the
lowest val MAE. calib and test play no part in any choice.
"""

import argparse
import itertools
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from conformal_calorie_estimation.finetune import FinetuneConfig, train

OUT = Path("models")
SEEDS = (0, 1, 2)
N_FOLDS = 5
DINOV3 = "vit_small_patch16_dinov3.lvd1689m"


def search_configs() -> list[FinetuneConfig]:
    configs = []
    for lr, last, multi in itertools.product((5e-6, 1e-5, 3e-5), (None, 4), (False, True)):
        name = f"search_lr{lr:g}_{'all' if last is None else f'last{last}'}_{'multi' if multi else 'kcal'}"
        configs.append(FinetuneConfig(name=name, lr_backbone=lr, train_last_blocks=last, multitask=multi))
    return configs


def done(cfg: FinetuneConfig) -> bool:
    return (OUT / cfg.name / "outputs.npz").exists()


def best_recipe() -> FinetuneConfig:
    """The finished search run with the lowest val MAE."""
    scored = [(float(np.load(OUT / c.name / "outputs.npz")["best_val_mae"]), c) for c in search_configs() if done(c)]
    if len(scored) < len(search_configs()):
        raise SystemExit("run the search phase first")
    mae, cfg = min(scored, key=lambda t: t[0])
    print(f"best recipe: {cfg.name} (val MAE {mae:.1f})")
    return cfg


def phase_configs(phase: str) -> list[FinetuneConfig]:
    if phase == "search":
        return search_configs()
    recipe = best_recipe()
    if phase == "dinov2":
        configs = []
        for seed in SEEDS:
            base = replace(recipe, name=f"dinov2_s{seed}", seed=seed, save_weights=True)
            configs.append(base)
            configs += [replace(base, name=f"{base.name}_fold{k}", fold=k, save_weights=False) for k in range(N_FOLDS)]
        return configs
    if phase == "dinov3":
        return [replace(recipe, name=f"dinov3_s{seed}", seed=seed, model_name=DINOV3) for seed in SEEDS]
    raise ValueError(phase)


def run(phase: str) -> None:
    configs = phase_configs(phase)
    todo = [c for c in configs if not done(c)]
    print(f"{phase}: {len(configs) - len(todo)} done, {len(todo)} to run", flush=True)
    for i, cfg in enumerate(todo, 1):
        print(f"--- {phase} {i}/{len(todo)}: {cfg.name} {asdict(cfg)}", flush=True)
        train(replace(cfg, out_dir=str(OUT)))
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("phase", choices=["search", "dinov2", "dinov3", "all"])
    phase = parser.parse_args().phase
    for p in ("search", "dinov2", "dinov3") if phase == "all" else (phase,):
        run(p)
