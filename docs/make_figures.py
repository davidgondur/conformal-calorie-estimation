"""Generate the figures and numbers used by the Typst documents in docs/.

    uv run python docs/make_figures.py

Writes SVG charts, photo thumbnails and ``metrics.json`` to ``docs/figures/``.
Reproduces notebooks/baseline.py (seed 0, alpha = 0.1) and adds the
evidence behind the decisions in decisions.typ.
"""

import json
from itertools import pairwise
from pathlib import Path

import altair as alt
import numpy as np
import polars as pl
from PIL import Image
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conformal_calorie_estimation.adaptive import (
    adaptive_intervals,
    fit_difficulty,
    fit_quantiles,
    pinball_loss,
    resampled_adaptive_coverage,
)
from conformal_calorie_estimation.baseline import (
    RIDGE_ALPHAS,
    fit_log_ridge,
    group_labels,
    mae,
    prepare_baseline,
)
from conformal_calorie_estimation.conformal import (
    conformal_intervals,
    conformal_quantile,
    coverage,
    coverage_by_group,
    coverage_distribution,
    interval_score,
    quantile_groups,
    resampled_coverage,
)
from conformal_calorie_estimation.data import (
    image_path,
    incremental_groups,
    official_splits,
    read_ids,
)
from conformal_calorie_estimation.features import DEFAULT_MODEL

PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "data" / "nutrition5k"
FEATURES = PROJECT / "data" / "features"
OUT = PROJECT / "docs" / "figures"
SEED, ALPHA = 0, 0.1
N_REPS = 1000
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
LABELS = {"multiplicative": MUL, "additive": ADD}
NORM, CQR = "normalized (log)", "CQR (log)"
ALL_LABELS = {"additive": ADD, "multiplicative": MUL, "normalized": NORM, "cqr": CQR}
# Four-colour palette validated with the dataviz checker (CVD, contrast, chroma).
ALL_COLORS = {MUL: BLUE, ADD: ORANGE, NORM: "#2E8B57", CQR: "#8A4FBF"}


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


def test_intervals(y, pred_log, alpha):
    """Both variants on test, keyed by their document labels."""
    ints = conformal_intervals(y["calib"], pred_log["calib"], pred_log["test"], alpha)
    return {LABELS[v]: lohi for v, lohi in ints.items()}


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
        for a, b in pairwise(ids):
            if dishes[b].timestamp - dishes[a].timestamp < gap:
                groups[-1].append(b)
            else:
                groups.append([b])
        by_time[str(gap)] = summary(groups)
    return {"subset_rule": summary(incremental_groups(dishes[i] for i in ids)), "time_only": by_time}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    official = official_splits(ROOT)
    data = prepare_baseline(ROOT, FEATURES, seed=SEED)  # DINOv2, the chosen backbone
    dishes, parts, removed, y = data.dishes, data.parts, data.removed, data.kcal

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
    backbones = []
    for label, name, size in BACKBONES:
        run = data if name == DEFAULT_MODEL else prepare_baseline(ROOT, FEATURES, seed=SEED, model_name=name, img_size=size)
        fit = fit_log_ridge(run.X, run.kcal)
        run_pred_log = {k: fit.model.predict(x) for k, x in run.X.items()}
        lo, hi = test_intervals(run.kcal, run_pred_log, ALPHA)[MUL]
        backbones.append({
            "label": label, "model": name, "img_size": size, "dim": int(run.X["train"].shape[1]),
            "val_mae": min(r["val_mae"] for r in fit.search),
            "test_mae": mae(run.kcal["test"], np.expm1(run_pred_log["test"]).clip(0)),
            "coverage": coverage(run.kcal["test"], lo, hi),
            "width_mean": float(np.mean(hi - lo)), "width_median": float(np.median(hi - lo)),
        })

    X = data.X
    fit = fit_log_ridge(X, y)
    best_alpha = fit.alpha
    pred_log = {k: fit.model.predict(x) for k, x in X.items()}
    pred = {k: np.expm1(p).clip(0) for k, p in pred_log.items()}
    raw_target_val_mae = min(  # same embeddings, ridge fitted on kcal instead of log1p(kcal)
        mae(y["val"], make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(X["train"], y["train"]).predict(X["val"]))
        for a in RIDGE_ALPHAS
    )

    ints = test_intervals(y, pred_log, ALPHA)
    yt = y["test"]
    bins, labels = quantile_groups(yt, n=4)
    by_size = pl.DataFrame(
        [
            {"variant": v, "quartile": labels[g], "order": g, "coverage": st["coverage"], "width": st["mean_width"]}
            for v, (lo, hi) in ints.items()
            for g, st in coverage_by_group(yt, lo, hi, bins).items()
        ]
    )
    variant_colors = alt.Scale(domain=VARIANTS, range=[BLUE, ORANGE])
    x_q = alt.X("quartile:N", sort=labels, title="True kcal (test quartile)", axis=alt.Axis(labelAngle=0))
    bar = {"cornerRadiusTopLeft": 3, "cornerRadiusTopRight": 3, "width": alt.RelativeBandSize(0.8)}
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
            for v, (lo, hi) in test_intervals(y, pred_log, float(a)).items()
        ]
    )
    diag = alt.Chart(pl.DataFrame({"x": [0.45, 0.95]})).mark_line(color=MUTED, strokeWidth=1).encode(x="x:Q", y="x:Q")
    lines = alt.Chart(curve).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=40, filled=True)).encode(
        x=alt.X("target:Q", scale=alt.Scale(domain=[0.45, 0.95]), axis=alt.Axis(format="%"), title="Target coverage (1 − α)"),
        y=alt.Y("coverage:Q", scale=alt.Scale(domain=[0.45, 1]), axis=alt.Axis(format="%"), title="Test coverage"),
        color=alt.Color("variant:N", scale=variant_colors, title=None),
    )
    save((diag + lines).properties(width=300, height=260), "coverage_vs_alpha")

    # What the guarantee promises: fixed model, random calibration/test re-splits.
    pool = parts["calib"] + parts["test"]
    pool_y = np.concatenate([y["calib"], y["test"]])
    pool_pred = np.concatenate([pred_log["calib"], pred_log["test"]])
    n_cal, n_pool_test = len(parts["calib"]), len(pool) - len(parts["calib"])
    grid, pmf = coverage_distribution(n_cal, n_pool_test, ALPHA)
    theory_mean = float(grid @ pmf)
    theory_sd = float(np.sqrt(((grid - theory_mean) ** 2) @ pmf))
    resampled = {
        mode: resampled_coverage(pool_y, pool_pred, ALPHA, n_cal, N_REPS, groups=groups, seed=SEED)
        for mode, groups in (("groups", group_labels(pool, dishes)), ("dishes", None))
    }
    bin_width = 0.005
    edges_cov = np.arange(0.80, 1.0 + bin_width / 2, bin_width)
    hist_rows = [
        {"variant": LABELS[v], "lo": float(a), "hi": float(b), "density": c / (N_REPS * bin_width)}
        for v, covs in resampled["groups"].items()
        for a, b, c in zip(edges_cov, edges_cov[1:], np.histogram(covs, bins=edges_cov)[0])
    ]
    theory = pl.DataFrame({"coverage": grid, "density": pmf * n_pool_test}).filter(
        (pl.col("coverage") >= 0.80) & (pl.col("coverage") <= 1.0)
    )
    panels = []
    for label, color in ((MUL, BLUE), (ADD, ORANGE)):
        panels.append(
            (
                alt.Chart(pl.DataFrame(hist_rows).filter(pl.col("variant") == label))
                .mark_rect(color=color, opacity=0.75)
                .encode(
                    x=alt.X("lo:Q", title="Coverage measured on test", axis=alt.Axis(format="%"),
                            scale=alt.Scale(domain=[0.82, 0.98])),
                    x2="hi:Q",
                    y=alt.Y("density:Q", title="Density"),
                    y2=alt.datum(0),
                )
                + alt.Chart(theory).mark_line(color=INK, strokeWidth=2).encode(x="coverage:Q", y="density:Q")
                + alt.Chart(pl.DataFrame({"x": [1 - ALPHA]})).mark_rule(color=MUTED, strokeWidth=1).encode(x="x:Q")
            ).properties(width=250, height=180, title=label)
        )
    save(alt.hconcat(*panels, spacing=30), "coverage_resampled")

    # ------------------------------------------------------ adaptive variants
    difficulty = fit_difficulty(data, fit, seed=SEED)
    qfit = fit_quantiles(data, ALPHA)
    adaptive = {ALL_LABELS[v]: lohi for v, lohi in adaptive_intervals(data, fit, difficulty, qfit, ALPHA).items()}
    val_residual = np.abs(np.log1p(y["val"]) - pred_log["val"])
    pred_groups, pred_labels = quantile_groups(pred["test"], n=4)
    n_ingr = np.array([len(dishes[i].ingredients) for i in parts["test"]])
    ingr_groups, ingr_labels = np.digitize(n_ingr, [2, 6, 13]), ["1", "2–5", "6–12", "13+"]
    adaptive_resampled = resampled_adaptive_coverage(data, fit, difficulty, qfit, n_reps=N_REPS, seed=SEED)

    def by(groups, names):
        return [
            {"variant": v, "group": names[g], "order": g, "n": st["n"], "coverage": st["coverage"], "width": st["mean_width"]}
            for v, (lo, hi) in adaptive.items()
            for g, st in coverage_by_group(yt, lo, hi, groups).items()
        ]

    by_pred, by_ingr = by(pred_groups, pred_labels), by(ingr_groups, ingr_labels)
    all_colors = alt.Scale(domain=list(ALL_COLORS), range=list(ALL_COLORS.values()))
    x_g = alt.X("group:N", sort=pred_labels, title="Predicted kcal (test quartile)", axis=alt.Axis(labelAngle=0))
    off = alt.XOffset("variant:N", sort=list(ALL_COLORS))
    color = alt.Color("variant:N", scale=all_colors, title=None)
    bar4 = {"cornerRadiusTopLeft": 2, "cornerRadiusTopRight": 2, "width": alt.RelativeBandSize(0.85)}
    cov4 = (
        alt.Chart(pl.DataFrame(by_pred)).mark_bar(**bar4).encode(
            x=x_g, xOffset=off, color=color,
            y=alt.Y("coverage:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title="Coverage"),
        )
        + alt.Chart(pl.DataFrame({"y": [1 - ALPHA]})).mark_rule(color=INK, strokeWidth=1).encode(y="y:Q")
    ).properties(width=260, height=200, title="Coverage")
    width4 = (
        alt.Chart(pl.DataFrame(by_pred)).mark_bar(**bar4).encode(
            x=x_g, xOffset=off, color=color, y=alt.Y("width:Q", title="kcal"),
        )
    ).properties(width=260, height=200, title="Mean interval width")
    save(alt.hconcat(cov4, width4, spacing=30), "adaptive_by_predicted")

    def spread(rows):
        out = {}
        for r in rows:
            lo_, hi_ = out.get(r["variant"], (1.0, 0.0))
            out[r["variant"]] = (min(lo_, r["coverage"]), max(hi_, r["coverage"]))
        return {v: {"worst": a, "best": b, "gap": b - a} for v, (a, b) in out.items()}

    adaptive_metrics = {
        "difficulty": {
            "alpha": difficulty.alpha,
            "floor": difficulty.floor,
            "spearman_val": float(spearmanr(difficulty.sigma(X["val"]), val_residual).statistic),
            "floored_share_test": float(np.mean(difficulty.sigma(X["test"]) <= difficulty.floor)),
        },
        "quantiles": {
            "l1": qfit.l1,
            "crossed_test": qfit.predict(X["test"])[2],
            "crossed_calib": qfit.predict(X["calib"])[2],
            # Why a linear model: val pinball loss of the upper quantile against gradient boosting.
            "val_pinball_upper_linear": min(r["val_pinball"] for r in qfit.search if r["side"] == "upper"),
            "val_pinball_upper_boosting": pinball_loss(
                np.log1p(y["val"]),
                HistGradientBoostingRegressor(loss="quantile", quantile=1 - ALPHA / 2, random_state=SEED)
                .fit(X["train"], np.log1p(y["train"]))
                .predict(X["val"]),
                1 - ALPHA / 2,
            ),
        },
        "variants": {
            v: {
                "coverage": coverage(yt, lo, hi),
                "width_mean": float(np.mean(hi - lo)),
                "width_median": float(np.median(hi - lo)),
                "interval_score": interval_score(yt, lo, hi, ALPHA),
            }
            for v, (lo, hi) in adaptive.items()
        },
        "by_predicted": by_pred,
        "by_ingredients": by_ingr,
        "spread_predicted": spread(by_pred),
        "spread_ingredients": spread(by_ingr),
        "resampled": {
            ALL_LABELS[v]: {"mean": float(c.mean()), "sd": float(c.std()),
                            "share_below_official": float(np.mean(c <= coverage(yt, *adaptive[ALL_LABELS[v]])))}
            for v, c in adaptive_resampled.items()
        },
    }

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
        "resampling": {
            "n_reps": N_REPS, "n_cal": n_cal, "n_test": n_pool_test,
            "theory_mean": theory_mean, "theory_sd": theory_sd,
            **{
                mode: {LABELS[v]: {"mean": float(c.mean()), "sd": float(c.std())} for v, c in covs.items()}
                for mode, covs in resampled.items()
            },
        },
        "examples": examples,
        "adaptive": adaptive_metrics,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps({k: metrics["adaptive"][k] for k in ("difficulty", "quantiles", "variants", "spread_predicted", "spread_ingredients", "resampled")}, indent=1))


if __name__ == "__main__":
    main()
