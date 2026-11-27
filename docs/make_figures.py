"""Generate the figures and numbers used by the Typst documents in docs/.

    uv run python docs/make_figures.py

Writes SVG charts, photo thumbnails and ``metrics.json`` to ``docs/figures/``.
Reproduces notebooks/baseline.py (seed 0, alpha = 0.1) and adds the
evidence behind the decisions in decisions.typ.
"""

import json
from pathlib import Path

import altair as alt
import numpy as np
import polars as pl
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conformal_calorie_estimation.conformal import (
    absolute_residual_interval,
    conformal_quantile,
    coverage,
)
from conformal_calorie_estimation.data import (
    clean_train,
    image_path,
    incremental_groups,
    official_splits,
    read_dishes,
    read_ids,
    split_by_groups,
)
from conformal_calorie_estimation.features import DEFAULT_MODEL, cached_features

PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "data" / "nutrition5k"
FEATURES = PROJECT / "data" / "features"
OUT = PROJECT / "docs" / "figures"
SEED, ALPHA = 0, 0.1
RIDGE_ALPHAS = [1, 10, 30, 100, 300, 1000, 3000, 10000]
BACKBONES = [  # (label, timm name, input size)
    ("DINOv2-S/14 @ 224", DEFAULT_MODEL, 224),
    ("DINOv3-S/16 @ 224", "vit_small_patch16_dinov3.lvd1689m", 224),
    ("DINOv3-S/16 @ 256", "vit_small_patch16_dinov3.lvd1689m", 256),
]

# Validated with the dataviz palette checker (CVD, contrast, chroma): both pass.
BLUE, ORANGE = "#1B6FA8", "#D4521F"
INK, MUTED, GRID = "#111111", "#5B6270", "#E3E5E8"
MUL, ADD = "multiplicative (log)", "additive (kcal)"
VARIANTS = [MUL, ADD]


@alt.theme.register("docs", enable=True)
def theme():
    return alt.theme.ThemeConfig({
        "config": {
            "font": "Source Sans 3",
            "view": {"stroke": None},
            "background": "white",
            "axis": {
                "labelColor": MUTED, "titleColor": INK, "titleFontWeight": 500,
                "gridColor": GRID, "domainColor": GRID, "tickColor": GRID,
                "labelFontSize": 11, "titleFontSize": 12,
            },
            "legend": {"labelColor": INK, "titleColor": INK, "labelFontSize": 11, "orient": "top"},
            "header": {"labelColor": INK, "labelFontSize": 12, "titleFontSize": 12},
            "title": {"color": INK, "fontSize": 13, "fontWeight": 600, "anchor": "start"},
        }
    })


def save(chart, name: str) -> None:
    chart.save(OUT / f"{name}.svg")


def thumbnail(dish_id: str, name: str, width: int = 480) -> None:
    img = Image.open(image_path(ROOT, dish_id)).convert("RGB")
    img.thumbnail((width, width))
    img.save(OUT / f"{name}.jpg", quality=88)


def mae(t, p) -> float:
    return float(np.mean(np.abs(t - p)))


def fit_baseline(X, y):
    """Ridge on log1p(kcal), regularization chosen on val. Returns (model, alpha, val MAE)."""
    best = None
    for a in RIDGE_ALPHAS:
        model = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(X["train"], np.log1p(y["train"]))
        score = mae(y["val"], np.expm1(model.predict(X["val"])))
        if best is None or score < best[2]:
            best = (model, a, score)
    return best


def conformal_intervals(y, pred_log, pred, alpha):
    """Both split conformal variants on test, in kcal: {variant: (lower, upper)}."""
    lo, hi = absolute_residual_interval(np.log1p(y["calib"]), pred_log["calib"], pred_log["test"], alpha)
    alo, ahi = absolute_residual_interval(y["calib"], pred["calib"], pred["test"], alpha)
    return {MUL: (np.expm1(lo).clip(0), np.expm1(hi)), ADD: (alo.clip(0), ahi)}


def grouping_evidence(dishes) -> dict:
    """How well two grouping heuristics respect the official rgb train/test split."""
    splits = ROOT / "dish_ids" / "splits"
    train = set(read_ids(splits / "rgb_train_ids.txt"))
    test = set(read_ids(splits / "rgb_test_ids.txt"))
    ids = sorted((train | test) & dishes.keys(), key=lambda i: dishes[i].timestamp)

    def summary(groups):
        mixed = [g for g in groups if set(g) & train and set(g) & test]
        return {"groups": len(groups), "mixed": len(mixed), "largest": max(map(len, groups))}

    by_time = {}
    for gap in (60, 300, 1800):
        groups = [[ids[0]]]
        for a, b in zip(ids, ids[1:]):
            if dishes[b].timestamp - dishes[a].timestamp < gap:
                groups[-1].append(b)
            else:
                groups.append([b])
        by_time[str(gap)] = summary(groups)
    return {"subset_rule": summary(incremental_groups(dishes[i] for i in ids)), "time_only": by_time}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dishes = read_dishes(ROOT)
    official = official_splits(ROOT)
    raw = split_by_groups(official["train"], dishes, {"val": 0.1, "calib": 0.2}, seed=SEED)
    parts = {
        "train": clean_train(raw["train"], dishes),
        "val": raw["val"],
        "calib": raw["calib"],
        "test": official["test"],
    }
    removed = sorted(set(raw["train"]) - set(parts["train"]))

    # ------------------------------------------------------------------ EDA
    df = pl.DataFrame(
        [
            {"split": split, "calories": dishes[i].calories, "mass": dishes[i].mass,
             "n_ingredients": len(dishes[i].ingredients)}
            for split, ids in official.items()
            for i in ids
        ]
    )
    train_df = df.filter(pl.col("split") == "train")
    eda = {
        key[0]: {
            "median_kcal": float(g["calories"].median()),
            "p95_kcal": float(g["calories"].quantile(0.95)),
            "median_ingredients": float(g["n_ingredients"].median()),
        }
        for key, g in df.group_by("split", maintain_order=True)
    }
    eda["kcal_q25"] = float(df["calories"].quantile(0.25))
    eda["kcal_q75"] = float(df["calories"].quantile(0.75))
    eda["corr_mass_kcal_train"] = float(train_df.select(pl.corr("mass", "calories")).item())
    eda["single_ingredient_share"] = float((df["n_ingredients"] == 1).mean())

    split_colors = alt.Scale(domain=["train", "test"], range=[BLUE, ORANGE])
    edges = np.linspace(0, 3.6, 37)
    hist_rows = []
    for split in ("train", "test"):
        kcal = df.filter((pl.col("split") == split) & (pl.col("calories") > 0))["calories"].to_numpy()
        counts, _ = np.histogram(np.log10(kcal), bins=edges)
        hist_rows += [
            {"split": split, "lo": float(a), "hi": float(b), "share": c / len(kcal)}
            for a, b, c in zip(edges, edges[1:], counts)
        ]
    hist = (
        alt.Chart(pl.DataFrame(hist_rows))
        .mark_rect(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("lo:Q", title="kcal (log scale)", scale=alt.Scale(domain=[0, 3.6]),
                    axis=alt.Axis(values=[0, 1, 2, 3], labelExpr="pow(10, datum.value)")),
            x2="hi:Q",
            y=alt.Y("share:Q", title="Share of dishes", axis=alt.Axis(format="%")),
            y2=alt.datum(0),
            color=alt.Color("split:N", scale=split_colors, legend=None),
            row=alt.Row("split:N", sort=["train", "test"], title=None,
                        header=alt.Header(labelAngle=0, labelAlign="left")),
        )
        .properties(width=420, height=110)
    )
    save(hist, "calories_hist")

    scatter = (
        alt.Chart(df.filter((pl.col("calories") > 0) & (pl.col("mass") < 2000)))
        .mark_circle(size=12, opacity=0.35, color=BLUE)
        .encode(
            x=alt.X("mass:Q", scale=alt.Scale(type="log"), title="Mass (g, log scale)"),
            y=alt.Y("calories:Q", scale=alt.Scale(type="log"), title="kcal (log scale)"),
        )
        .properties(width=300, height=260)
    )
    save(scatter, "mass_vs_kcal")

    for dish_id, name in [("dish_1560974769", "error_rice"), ("dish_1556575700", "error_plate")]:
        thumbnail(dish_id, name)

    # ------------------------------------------------------------ baseline
    y = {k: np.array([dishes[i].calories for i in ids]) for k, ids in parts.items()}
    all_ids = official["train"] + official["test"]
    backbones = []
    for label, name, size in BACKBONES:
        feats = cached_features(ROOT, all_ids, FEATURES, model_name=name, img_size=size)
        X = {k: np.stack([feats[i] for i in ids]) for k, ids in parts.items()}
        model, best_alpha, val_mae = fit_baseline(X, y)
        pred_log = {k: model.predict(x) for k, x in X.items()}
        pred = {k: np.expm1(p).clip(0) for k, p in pred_log.items()}
        lo, hi = conformal_intervals(y, pred_log, pred, ALPHA)[MUL]
        backbones.append({
            "label": label, "model": name, "img_size": size, "dim": int(X["train"].shape[1]),
            "val_mae": val_mae, "test_mae": mae(y["test"], pred["test"]),
            "coverage": coverage(y["test"], lo, hi),
            "width_mean": float(np.mean(hi - lo)), "width_median": float(np.median(hi - lo)),
        })
        if name == DEFAULT_MODEL:
            main_run = (best_alpha, pred_log, pred)
            raw_target_val_mae = min(  # same model, fitted on kcal instead of log1p(kcal)
                mae(y["val"], make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(X["train"], y["train"]).predict(X["val"]))
                for a in RIDGE_ALPHAS
            )
    best_alpha, pred_log, pred = main_run

    ints = conformal_intervals(y, pred_log, pred, ALPHA)
    yt = y["test"]
    q_edges = np.quantile(yt, [0, 0.25, 0.5, 0.75, 1])
    labels = [f"{a:.0f}–{b:.0f}" for a, b in zip(q_edges, q_edges[1:])]
    bins = np.clip(np.searchsorted(q_edges, yt, side="right") - 1, 0, 3)
    by_size = pl.DataFrame(
        [
            {"variant": v, "quartile": labels[b], "order": b,
             "coverage": coverage(yt[bins == b], lo[bins == b], hi[bins == b]),
             "width": float(np.mean(hi[bins == b] - lo[bins == b]))}
            for v, (lo, hi) in ints.items()
            for b in range(4)
        ]
    )
    variant_colors = alt.Scale(domain=VARIANTS, range=[BLUE, ORANGE])
    x_q = alt.X("quartile:N", sort=labels, title="True kcal (test quartile)", axis=alt.Axis(labelAngle=0))
    bar = dict(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, width=alt.RelativeBandSize(0.8))
    cov_chart = (
        alt.Chart(by_size).mark_bar(**bar).encode(
            x=x_q, xOffset=alt.XOffset("variant:N", sort=VARIANTS),
            y=alt.Y("coverage:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title="Coverage"),
            color=alt.Color("variant:N", scale=variant_colors, title=None),
        )
        + alt.Chart(pl.DataFrame({"y": [1 - ALPHA]})).mark_rule(color=INK, strokeWidth=1).encode(y="y:Q")
    ).properties(width=260, height=200, title="Coverage")
    width_chart = (
        alt.Chart(by_size).mark_bar(**bar).encode(
            x=x_q, xOffset=alt.XOffset("variant:N", sort=VARIANTS),
            y=alt.Y("width:Q", title="kcal"),
            color=alt.Color("variant:N", scale=variant_colors, title=None),
        )
    ).properties(width=260, height=200, title="Mean interval width")
    save(alt.hconcat(cov_chart, width_chart, spacing=30), "coverage_by_size")

    curve = pl.DataFrame(
        [
            {"target": round(1 - a, 2), "variant": v, "coverage": coverage(yt, lo, hi)}
            for a in np.round(np.arange(0.05, 0.55, 0.05), 2)
            for v, (lo, hi) in conformal_intervals(y, pred_log, pred, float(a)).items()
        ]
    )
    diag = alt.Chart(pl.DataFrame({"x": [0.45, 0.95]})).mark_line(color=MUTED, strokeWidth=1).encode(x="x:Q", y="x:Q")
    lines = alt.Chart(curve).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=40, filled=True)).encode(
        x=alt.X("target:Q", scale=alt.Scale(domain=[0.45, 0.95]), axis=alt.Axis(format="%"), title="Target coverage (1 − α)"),
        y=alt.Y("coverage:Q", scale=alt.Scale(domain=[0.45, 1]), axis=alt.Axis(format="%"), title="Test coverage"),
        color=alt.Color("variant:N", scale=variant_colors, title=None),
    )
    save((diag + lines).properties(width=300, height=260), "coverage_vs_alpha")

    lo, hi = ints[MUL]
    examples = []
    for n, i in enumerate(np.random.default_rng(SEED).choice(len(yt), 4, replace=False)):
        name = f"example_{n}"
        thumbnail(parts["test"][i], name, width=360)
        examples.append({"file": f"figures/{name}.jpg", "real": float(yt[i]), "pred": float(pred["test"][i]),
                         "lo": float(lo[i]), "hi": float(hi[i])})

    metrics = {
        "n": {k: len(v) for k, v in parts.items()},
        "n_official": {k: len(v) for k, v in official.items()},
        "removed": removed,
        "eda": eda,
        "grouping": grouping_evidence(dishes),
        "best_alpha": best_alpha,
        "alpha": ALPHA,
        "mae": {k: mae(y[k], pred[k]) for k in ("val", "test")},
        "mae_pct": {k: mae(y[k], pred[k]) / float(y[k].mean()) for k in ("val", "test")},
        "val_mae_log_target": backbones[0]["val_mae"],
        "val_mae_raw_target": raw_target_val_mae,
        "mae_naive_test": mae(yt, np.full_like(yt, float(np.median(y["train"])))),
        "q_add": conformal_quantile(np.abs(y["calib"] - pred["calib"]), ALPHA),
        "q_log_factor": float(np.exp(conformal_quantile(np.abs(np.log1p(y["calib"]) - pred_log["calib"]), ALPHA))),
        "coverage": {v: coverage(yt, l, h) for v, (l, h) in ints.items()},
        "width_mean": {v: float(np.mean(h - l)) for v, (l, h) in ints.items()},
        "width_median": {v: float(np.median(h - l)) for v, (l, h) in ints.items()},
        "by_size": by_size.to_dicts(),
        "cov_sd": float(np.sqrt(ALPHA * (1 - ALPHA) / len(yt))),
        "cov_sd_calib": float(np.sqrt(ALPHA * (1 - ALPHA) / len(parts["calib"]))),
        "backbones": backbones,
        "examples": examples,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps({k: metrics[k] for k in ("n", "eda", "grouping", "backbones")}, indent=1))


if __name__ == "__main__":
    main()
