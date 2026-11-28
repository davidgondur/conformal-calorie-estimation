import marimo

__generated_with = "0.26.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import numpy as np
    import polars as pl
    from scipy.stats import spearmanr

    from conformal_calorie_estimation.adaptive import (
        adaptive_intervals,
        baseline_outputs,
        fit_difficulty,
        fit_quantiles,
        resampled_adaptive_coverage,
        val_residual,
        variant_metrics,
    )
    from conformal_calorie_estimation.baseline import fit_log_ridge, mae, prepare_baseline
    from conformal_calorie_estimation.conformal import (
        conformal_intervals,
        coverage_by_group,
        quantile_groups,
    )
    from conformal_calorie_estimation.finetune import finetuned_outputs, run_summary

    return (
        adaptive_intervals,
        alt,
        baseline_outputs,
        conformal_intervals,
        coverage_by_group,
        finetuned_outputs,
        fit_difficulty,
        fit_log_ridge,
        fit_quantiles,
        mae,
        mo,
        np,
        pl,
        prepare_baseline,
        quantile_groups,
        resampled_adaptive_coverage,
        run_summary,
        spearmanr,
        val_residual,
        variant_metrics,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fine-tuning del ViT

    Hasta ahora DINOv2 estaba **congelado**: sus embeddings se calculaban una vez y un
    ridge predecía las calorías. Aquí se ajusta la red entera (o sus últimos bloques) para
    predecir log(1 + kcal), y se evalúan las mismas cuatro variantes conformales.

    Los entrenamientos se lanzan con `scripts/finetune_sweep.py` y guardan sus
    predicciones y embeddings en `models/`; este notebook solo los lee. Ninguna decisión
    mira `calib` ni `test`: la receta y la mejor época se eligen con `val`.
    """)


@app.cell
def _(mo):
    ROOT = mo.notebook_dir().parent / "data" / "nutrition5k"
    FEATURES_DIR = mo.notebook_dir().parent / "data" / "features"
    MODELS = mo.notebook_dir().parent / "models"
    SEED = 0
    SEEDS = (0, 1, 2)
    ALPHAS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    VARIANT_LABELS = {
        "additive": "aditiva",
        "multiplicative": "multiplicativa",
        "normalized": "normalizada",
        "cqr": "CQR",
    }
    # Validated palette (CVD, contrast, chroma), as in adaptive.py.
    VARIANT_COLORS = {"multiplicativa": "#1B6FA8", "aditiva": "#D4521F", "normalizada": "#2E8B57", "CQR": "#8A4FBF"}
    return (
        ALPHAS,
        FEATURES_DIR,
        MODELS,
        ROOT,
        SEED,
        SEEDS,
        VARIANT_COLORS,
        VARIANT_LABELS,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Búsqueda de la receta

    Doce entrenamientos con una semilla: tasa de aprendizaje del backbone × ajustar todo
    o solo los últimos 4 bloques × predecir solo kcal o también masa y macros
    (multitarea). Se elige la combinación con menor MAE en `val`.
    """)


@app.cell
def _(MODELS, pl, run_summary):
    search_runs = [run_summary(_d) for _d in sorted(MODELS.glob("search_*")) if (_d / "outputs.npz").exists()]
    search = pl.DataFrame(
        [
            {
                "entrenamiento": _r["name"],
                "lr backbone": _r["lr_backbone"],
                "bloques": "todos" if _r["train_last_blocks"] is None else f"últimos {_r['train_last_blocks']}",
                "objetivo": "multitarea" if _r["multitask"] else "kcal",
                "MAE val": _r["best_val_mae"],
                "mejor época": _r["best_epoch"],
            }
            for _r in search_runs
        ]
    ).sort("MAE val")
    curves = pl.DataFrame(
        [
            {"entrenamiento": _r["name"], "época": int(_h["epoch"]), "MAE val": _h["val_mae"]}
            for _r in search_runs
            for _h in _r["history"]
        ]
    )
    best_recipe = search["entrenamiento"][0]
    return best_recipe, curves, search


@app.cell
def _(alt, best_recipe, curves, mo, search):
    _chart = (
        alt.Chart(curves)
        .mark_line(strokeWidth=1.5)
        .encode(
            x=alt.X("época:Q"),
            y=alt.Y("MAE val:Q", scale=alt.Scale(zero=False, domain=[45, 110], clamp=True), title="MAE en val (kcal)"),
            color=alt.condition(alt.datum.entrenamiento == best_recipe, alt.value("#1B6FA8"), alt.value("#C9CED6")),
            detail="entrenamiento:N",
            tooltip=["entrenamiento", "época", alt.Tooltip("MAE val:Q", format=".1f")],
        )
        .properties(width=420, height=260, title=f"MAE en val por época (en azul, la elegida: {best_recipe})")
    )
    mo.vstack([search.with_columns(search["MAE val"].round(1)), _chart])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Semillas y comparación de modelos

    La receta elegida se entrena con 3 semillas en DINOv2 (cada una con su validación
    cruzada de 5 partes, para las variantes adaptativas) y con 3 semillas en DINOv3. El
    modelo congelado del baseline es la referencia.
    """)


@app.cell
def _(
    FEATURES_DIR,
    MODELS,
    ROOT,
    SEED,
    SEEDS,
    baseline_outputs,
    finetuned_outputs,
    fit_log_ridge,
    mae,
    np,
    prepare_baseline,
):
    _data = prepare_baseline(ROOT, FEATURES_DIR, seed=SEED)
    models = {"DINOv2 congelado + ridge": baseline_outputs(_data, fit_log_ridge(_data.X, _data.kcal), seed=SEED)}
    models |= {f"DINOv2 ajustado (semilla {_s})": finetuned_outputs(MODELS / f"dinov2_s{_s}") for _s in SEEDS}
    models |= {
        f"DINOv3 ajustado (semilla {_s})": finetuned_outputs(MODELS / f"dinov3_s{_s}", require_folds=False)
        for _s in SEEDS
    }
    dinov2_names = [_n for _n in models if _n.startswith("DINOv2 ajustado")]
    # The seed to export is chosen on val only.
    export_name = min(
        dinov2_names, key=lambda _n: mae(models[_n].kcal["val"], np.expm1(models[_n].pred_log["val"]).clip(0))
    )
    return export_name, models


@app.cell
def _(mae, models, np, pl):
    def family(name):
        return name.split(" (")[0]

    point = pl.DataFrame(
        [
            {
                "modelo": family(_n),
                "entrenamiento": _n,
                "MAE val": mae(_o.kcal["val"], np.expm1(_o.pred_log["val"]).clip(0)),
                "MAE test": mae(_o.kcal["test"], np.expm1(_o.pred_log["test"]).clip(0)),
            }
            for _n, _o in models.items()
        ]
    )
    point_summary = point.group_by("modelo", maintain_order=True).agg(
        [pl.len().alias("semillas")]
        + [pl.col(_c).mean().alias(f"{_c} media") for _c in ("MAE val", "MAE test")]
        + [pl.col(_c).std().alias(f"{_c} desv.") for _c in ("MAE val", "MAE test")]
    )
    point_summary
    return family, point_summary


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Las cuatro variantes conformales

    Para cada modelo se calibran las cuatro variantes con `calib` y se miden en `test`.
    Las del DINOv2 ajustado se dan como media ± desviación entre sus 3 semillas. DINOv3
    no tiene validación cruzada, así que solo tiene las dos variantes del modelo base.
    """)


@app.cell
def _(ALPHAS, mo):
    alpha_slider = mo.ui.slider(steps=list(ALPHAS), value=0.10, label="α", show_value=True)
    alpha_slider
    return (alpha_slider,)


@app.cell
def _(fit_difficulty, fit_quantiles, mo, models):
    difficulties = {_n: fit_difficulty(_o) for _n, _o in models.items() if _o.honest_emb is not None}

    @mo.cache
    def quantiles_for(name, alpha):
        return fit_quantiles(models[name], alpha)

    return difficulties, quantiles_for


@app.cell
def _(
    VARIANT_LABELS,
    adaptive_intervals,
    alpha_slider,
    conformal_intervals,
    difficulties,
    mo,
    models,
    quantiles_for,
):
    def intervals_of(name, alpha):
        """Test intervals of every variant available for this model, keyed by Spanish label."""
        _o = models[name]
        if name in difficulties:
            _ints = adaptive_intervals(_o, difficulties[name], quantiles_for(name, alpha), alpha)
        else:
            _ints = conformal_intervals(_o.kcal["calib"], _o.pred_log["calib"], _o.pred_log["test"], alpha)
        return {VARIANT_LABELS[_v]: _lohi for _v, _lohi in _ints.items()}

    with mo.status.spinner(title="Ajustando los modelos de cuantiles para este α..."):
        all_intervals = {_n: intervals_of(_n, alpha_slider.value) for _n in models}
    return (all_intervals,)


@app.cell
def _(all_intervals, alpha_slider, family, models, pl, variant_metrics):
    per_run = pl.DataFrame(
        [
            {"modelo": family(_n), "entrenamiento": _n, "variante": _v, **_m}
            for _n, _ints in all_intervals.items()
            for _v, _m in variant_metrics(_ints, models[_n].kcal["test"], alpha_slider.value).items()
        ]
    )
    _metrics = {"coverage": "cobertura", "width_median": "ancho mediano", "interval_score": "interval score"}
    variants_summary = per_run.group_by("modelo", "variante", maintain_order=True).agg(
        [pl.col(_c).mean().alias(_es) for _c, _es in _metrics.items()]
        + [pl.col(_c).std().alias(f"{_es} desv.") for _c, _es in _metrics.items()]
    )
    variants_summary
    return (variants_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ¿Sigue funcionando el modelo de dificultad?

    Con el backbone ajustado, los errores honestos de `train` salen de 5 modelos parciales
    (validación cruzada), pero el modelo final es otro. Si sus embeddings difieren mucho,
    el modelo de dificultad entrenado con los parciales ordenaría peor los platos del
    modelo final. La correlación de rangos en `val` lo mide.
    """)


@app.cell
def _(difficulties, models, pl, spearmanr, val_residual):
    difficulty_table = pl.DataFrame(
        [
            {
                "entrenamiento": _n,
                "Spearman σ̂ vs error (val)": float(spearmanr(_d.sigma(models[_n].emb["val"]), val_residual(models[_n])).statistic),
                "regularización": _d.alpha,
            }
            for _n, _d in difficulties.items()
        ]
    )
    difficulty_table
    return (difficulty_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cobertura por kcal predichas (modelo que se exportará)

    El modelo que se exportará es la semilla de DINOv2 con menor MAE en `val`.
    """)


@app.cell
def _(
    VARIANT_COLORS,
    all_intervals,
    alpha_slider,
    alt,
    coverage_by_group,
    export_name,
    models,
    np,
    pl,
    quantile_groups,
):
    _o = models[export_name]
    _groups, _labels = quantile_groups(np.expm1(_o.pred_log["test"]).clip(0), n=4)
    by_predicted = pl.DataFrame(
        [
            {"variante": _v, "grupo": _labels[_g], "orden": _g, "cobertura": _s["coverage"], "ancho medio": _s["mean_width"]}
            for _v, (_lo, _hi) in all_intervals[export_name].items()
            for _g, _s in coverage_by_group(_o.kcal["test"], _lo, _hi, _groups).items()
        ]
    )
    _color = alt.Color("variante:N", scale=alt.Scale(domain=list(VARIANT_COLORS), range=list(VARIANT_COLORS.values())), title=None)
    _x = alt.X("grupo:N", sort=_labels, title="kcal predichas (cuartil)", axis=alt.Axis(labelAngle=0))
    _off = alt.XOffset("variante:N", sort=list(VARIANT_COLORS))
    _cov = (
        alt.Chart(by_predicted).mark_bar().encode(
            x=_x, xOffset=_off, color=_color,
            y=alt.Y("cobertura:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title="Cobertura"),
        )
        + alt.Chart(pl.DataFrame({"y": [1 - alpha_slider.value]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
    ).properties(width=300, height=230, title=f"Cobertura · {export_name}")
    _width = (
        alt.Chart(by_predicted).mark_bar().encode(x=_x, xOffset=_off, color=_color, y=alt.Y("ancho medio:Q", title="kcal"))
    ).properties(width=300, height=230, title="Ancho medio del intervalo")
    # Static chart: mo.ui.altair_chart would duplicate the legend selection signal.
    _cov | _width


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## La garantía con el modelo ajustado

    Con los modelos fijos, `calib` + `test` se reparten al azar 1.000 veces (por grupos de
    montaje). La cobertura media debe seguir en torno a 1 − α.
    """)


@app.cell
def _(
    SEED,
    VARIANT_LABELS,
    alpha_slider,
    difficulties,
    export_name,
    models,
    pl,
    quantiles_for,
    resampled_adaptive_coverage,
):
    _r = resampled_adaptive_coverage(
        models[export_name], difficulties[export_name], quantiles_for(export_name, alpha_slider.value), n_reps=1000, seed=SEED
    )
    resampled_table = pl.DataFrame(
        [{"variante": VARIANT_LABELS[_v], "media": float(_c.mean()), "desv. típica": float(_c.std())} for _v, _c in _r.items()]
    )
    resampled_table
    return (resampled_table,)


@app.cell
def _(
    alpha_slider,
    best_recipe,
    difficulty_table,
    mo,
    point_summary,
    resampled_table,
    variants_summary,
):
    _p = {_r["modelo"]: _r for _r in point_summary.iter_rows(named=True)}
    _base, _ft, _v3 = (_p.get(_k) for _k in ("DINOv2 congelado + ridge", "DINOv2 ajustado", "DINOv3 ajustado"))
    _vs = variants_summary.filter(variants_summary["modelo"] == "DINOv2 ajustado").sort("interval score")
    _best_v = _vs.row(0, named=True)
    _base_v = variants_summary.filter(
        (variants_summary["modelo"] == "DINOv2 congelado + ridge") & (variants_summary["variante"] == _best_v["variante"])
    ).row(0, named=True)
    _rho = dict(zip(difficulty_table["entrenamiento"], difficulty_table["Spearman σ̂ vs error (val)"]))
    _rho_base = _rho.pop("DINOv2 congelado + ridge")
    _ok = all(abs(_m - (1 - alpha_slider.value)) < 0.01 for _m in resampled_table["media"])

    mo.md(
        f"""
    ## Conclusiones (α = {alpha_slider.value:.2f})

    - Receta elegida en `val`: **{best_recipe}**.
    - El fine-tuning baja el MAE en test de {_base["MAE test media"]:.0f} kcal (congelado) a
      **{_ft["MAE test media"]:.0f} ± {_ft["MAE test desv."]:.0f} kcal** (3 semillas).
      DINOv3 con la misma receta: {_v3["MAE test media"]:.0f} ± {_v3["MAE test desv."]:.0f} kcal en test
      y {_v3["MAE val media"]:.0f} kcal en `val`, frente a {_ft["MAE val media"]:.0f} de DINOv2.
    - La mejor variante del modelo ajustado es la **{_best_v["variante"]}**: *interval score*
      {_best_v["interval score"]:.0f} y ancho mediano {_best_v["ancho mediano"]:.0f} kcal, frente a
      {_base_v["interval score"]:.0f} y {_base_v["ancho mediano"]:.0f} kcal con el modelo congelado.
    - El modelo de dificultad ordena los platos con Spearman entre {min(_rho.values()):.2f} y
      {max(_rho.values()):.2f} según la semilla, frente a {_rho_base:.2f} con el modelo congelado.
    - Sobre 1.000 repartos, {"todas las variantes mantienen" if _ok else "no todas las variantes mantienen"}
      la cobertura media en torno al objetivo.
    """
    )


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
