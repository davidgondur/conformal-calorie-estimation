import marimo

__generated_with = "0.26.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import numpy as np
    import polars as pl
    from PIL import Image

    from conformal_calorie_estimation.baseline import (
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
        quantile_groups,
        resampled_coverage,
    )
    from conformal_calorie_estimation.data import image_path, label_problem

    return (
        Image,
        alt,
        conformal_intervals,
        conformal_quantile,
        coverage,
        coverage_by_group,
        coverage_distribution,
        fit_log_ridge,
        group_labels,
        image_path,
        label_problem,
        mae,
        mo,
        np,
        pl,
        prepare_baseline,
        quantile_groups,
        resampled_coverage,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Baseline: DINOv2 congelado + ridge + split conformal

    Primer pipeline completo, sin fine-tuning:

    1. Embedding de cada foto con **DINOv2-small congelado** (384 dimensiones, en caché).
    2. **Ridge** sobre los embeddings para predecir **log(1 + kcal)**.
    3. **Split conformal** con la partición de calibración, en dos variantes:
       margen fijo en **kcal** (intervalo aditivo) o en **log-kcal** (intervalo multiplicativo).
    4. Cobertura y ancho medidos en **test**.

    Sirve de referencia para el fine-tuning y para probar la parte conformal.
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Particiones

    Las particiones oficiales se dividen aquí (ver README): `val` y `calib` salen de
    train por grupos de montaje incremental. **Solo se limpia `train`**. `val`,
    `calib` y `test` quedan como están, para que calibración se parezca a test.
    """)


@app.cell
def _(mo, prepare_baseline):
    ROOT = mo.notebook_dir().parent / "data" / "nutrition5k"
    FEATURES_DIR = mo.notebook_dir().parent / "data" / "features"
    SEED = 0

    # Splits, cleaning of the training part and cached embeddings (see README).
    data = prepare_baseline(ROOT, FEATURES_DIR, seed=SEED, fractions={"val": 0.1, "calib": 0.2})
    dishes, parts, removed, X, kcal = data.dishes, data.parts, data.removed, data.X, data.kcal
    return ROOT, SEED, X, dishes, kcal, parts, removed


@app.cell
def _(dishes, label_problem, mo, parts, pl, removed):
    mo.vstack(
        [
            mo.md(
                "Platos por partición: "
                + ", ".join(f"**{k}** {len(v)}" for k, v in parts.items())
            ),
            mo.md("Quitados de train por etiqueta errónea:"),
            pl.DataFrame(
                {
                    "dish_id": removed,
                    "kcal": [dishes[i].calories for i in removed],
                    "masa_g": [dishes[i].mass for i in removed],
                    "motivo": [label_problem(dishes[i]) for i in removed],
                }
            ),
        ]
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Embeddings y regresión

    Los embeddings de DINOv2 se calculan una vez y se guardan en `data/features/`
    (en GPU, unos segundos). El objetivo es `log1p(kcal)`: la distribución de calorías
    es muy asimétrica, y en escala logarítmica los errores son relativos (±30 % en lugar
    de ±150 kcal). `log1p` en vez de `log` porque test tiene un plato con 0 kcal.
    La regularización se elige solo con `val`: `calib` y `test` no influyen en el modelo.
    """)


@app.cell
def _(X, fit_log_ridge, kcal, np, pl):
    ridge = fit_log_ridge(X, kcal)
    ridge_search = pl.DataFrame(ridge.search)
    best_alpha = ridge.alpha
    pred_log = {k: ridge.model.predict(x) for k, x in X.items()}
    pred_kcal = {k: np.expm1(p).clip(0) for k, p in pred_log.items()}
    return best_alpha, pred_kcal, pred_log, ridge_search


@app.cell
def _(alt, best_alpha, kcal, mae, mo, np, pred_kcal, ridge_search):
    _chart = (
        alt.Chart(ridge_search)
        .mark_line(point=True)
        .encode(
            x=alt.X("alpha:Q", scale=alt.Scale(type="log"), title="alpha de ridge"),
            y=alt.Y("val_mae:Q", scale=alt.Scale(zero=False), title="MAE en val (kcal)"),
        )
        .properties(width=320, height=220)
    )
    _rows = []
    for _k in ("val", "test"):
        _naive = np.full_like(kcal[_k], np.median(kcal["train"]))
        _rows.append(
            f"| {_k} | {mae(kcal[_k], pred_kcal[_k]):.1f} "
            f"({mae(kcal[_k], pred_kcal[_k]) / kcal[_k].mean():.1%}) "
            f"| {mae(kcal[_k], _naive):.1f} ({mae(kcal[_k], _naive) / kcal[_k].mean():.1%}) |"
        )
    mo.hstack(
        [
            _chart,
            mo.md(
                f"""
    **alpha elegido: {best_alpha}**

    | Partición | MAE ridge (kcal, % de la media) | MAE prediciendo la mediana |
    |---|---|---|
    {chr(10).join(_rows)}

    Referencia del artículo (otro split; InceptionV2 con fine-tuning, solo RGB):
    70,6 kcal (26,1 %).
    """
            ),
        ],
        align="center",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Split conformal

    Con las predicciones en `calib` se calcula el cuantil corregido
    $\hat q = $ el $\lceil (n+1)(1-\alpha) \rceil$-ésimo menor residuo absoluto.
    El intervalo para un plato nuevo es $\hat y \pm \hat q$:

    - **Aditivo**: residuos en kcal → el mismo ancho para todos los platos.
    - **Multiplicativo**: residuos en log1p(kcal) → al deshacer el log, el intervalo
      es proporcional al tamaño estimado del plato.

    Las dos variantes garantizan cobertura ≥ 1 − α *en promedio*. La diferencia está en
    cómo se reparte entre platos pequeños y grandes.
    """)


@app.cell
def _(mo):
    alpha_slider = mo.ui.slider(0.05, 0.30, step=0.05, value=0.10, label="α", show_value=True)
    alpha_slider
    return (alpha_slider,)


@app.cell
def _(alpha_slider, conformal_intervals, kcal, pred_log):
    VARIANT_LABELS = {"multiplicative": "multiplicativo (log)", "additive": "aditivo (kcal)"}


    def intervals(alpha, split="test"):
        """Both conformal variants for ``split``, in kcal: {label: (lower, upper)}."""
        _ints = conformal_intervals(kcal["calib"], pred_log["calib"], pred_log[split], alpha)
        return {VARIANT_LABELS[v]: lohi for v, lohi in _ints.items()}


    test_intervals = intervals(alpha_slider.value)
    return VARIANT_LABELS, intervals, test_intervals


@app.cell
def _(
    alpha_slider,
    conformal_quantile,
    coverage,
    kcal,
    mo,
    np,
    pred_kcal,
    pred_log,
    test_intervals,
):
    _q_log = conformal_quantile(
        np.abs(np.log1p(kcal["calib"]) - pred_log["calib"]), alpha_slider.value
    )
    _q_add = conformal_quantile(np.abs(kcal["calib"] - pred_kcal["calib"]), alpha_slider.value)
    _rows = [
        f"| {name} | {coverage(kcal['test'], lo, hi):.1%} | {np.mean(hi - lo):.0f} | {np.median(hi - lo):.0f} |"
        for name, (lo, hi) in test_intervals.items()
    ]
    mo.md(
        f"""
    **Objetivo: cobertura ≥ {1 - alpha_slider.value:.0%}** · $\\hat q$ aditivo = ±{_q_add:.0f} kcal ·
    $\\hat q$ log = ×/÷{np.exp(_q_log):.2f} (aprox.)

    | Variante | Cobertura en test | Ancho medio (kcal) | Ancho mediano (kcal) |
    |---|---|---|---|
    {chr(10).join(_rows)}

    Con {len(kcal["test"])} platos de test, la cobertura medida fluctúa unos
    ±{np.sqrt(alpha_slider.value * (1 - alpha_slider.value) / len(kcal["test"])):.1%} (1 σ)
    alrededor de la real.
    """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cobertura por tamaño de plato

    Aquí se ve la diferencia: con margen fijo, los platos pequeños reciben intervalos
    mucho más anchos de lo necesario y los grandes se quedan cortos. El intervalo
    multiplicativo reparte la cobertura de forma más uniforme.
    """)


@app.cell
def _(
    alpha_slider,
    alt,
    coverage_by_group,
    kcal,
    pl,
    quantile_groups,
    test_intervals,
):
    _groups, _labels = quantile_groups(kcal["test"], n=4)
    by_size = pl.DataFrame(
        [
            {
                "variante": _name,
                "kcal reales (cuartil)": _labels[_g],
                "orden": _g,
                "cobertura": _s["coverage"],
                "ancho medio": _s["mean_width"],
            }
            for _name, (_lo, _hi) in test_intervals.items()
            for _g, _s in coverage_by_group(kcal["test"], _lo, _hi, _groups).items()
        ]
    )

    _x = alt.X("kcal reales (cuartil):N", sort=alt.SortField("orden"), title="kcal reales (cuartil)")
    _color = alt.Color("variante:N", title=None)
    _cov = (
        alt.Chart(by_size)
        .mark_bar()
        .encode(x=_x, xOffset="variante:N", y=alt.Y("cobertura:Q", scale=alt.Scale(domain=[0, 1])), color=_color)
        + alt.Chart(pl.DataFrame({"y": [1 - alpha_slider.value]}))
        .mark_rule(strokeDash=[4, 4])
        .encode(y="y:Q")
    ).properties(width=300, height=250, title="Cobertura")
    _width = (
        alt.Chart(by_size)
        .mark_bar()
        .encode(x=_x, xOffset="variante:N", y=alt.Y("ancho medio:Q", title="kcal"), color=_color)
        .properties(width=300, height=250, title="Ancho medio del intervalo")
    )
    # Static chart: mo.ui.altair_chart would add the same legend
    # selection to both panels, which Vega rejects as a duplicate signal.
    _cov | _width
    return (by_size,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cobertura frente a α

    Si la calibración funciona, la cobertura en test sigue la diagonal 1 − α.
    """)


@app.cell
def _(alt, coverage, intervals, kcal, np, pl):
    _rows = [
        {"1 - α": 1 - _a, "variante": _name, "cobertura": coverage(kcal["test"], _lo, _hi)}
        for _a in np.round(np.arange(0.05, 0.55, 0.05), 2)
        for _name, (_lo, _hi) in intervals(float(_a)).items()
    ]
    _diag = alt.Chart(pl.DataFrame({"x": [0.45, 0.95]})).mark_line(color="gray", strokeDash=[4, 4]).encode(
        x="x:Q", y="x:Q"
    )
    _lines = (
        alt.Chart(pl.DataFrame(_rows))
        .mark_line(point=True)
        .encode(
            x=alt.X("1 - α:Q", scale=alt.Scale(domain=[0.45, 0.95]), title="Cobertura objetivo (1 − α)"),
            y=alt.Y("cobertura:Q", scale=alt.Scale(domain=[0.45, 1]), title="Cobertura en test"),
            color=alt.Color("variante:N", title=None),
        )
    )
    (_diag + _lines).properties(width=360, height=300)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ¿Qué promete exactamente la garantía?

    La cobertura ≥ 1 − α es un **promedio sobre posibles conjuntos de calibración**. Con
    otra calibración, la cobertura medida habría sido otra. Para verlo, se deja el modelo
    fijo, se juntan `calib` y `test` y se reparten al azar muchas veces en una calibración
    del mismo tamaño y un test con el resto.

    La teoría da la distribución exacta de la cobertura medida: con $n$ puntos de
    calibración y $k = \lceil (n+1)(1-\alpha) \rceil$, la cobertura real sigue una
    $\mathrm{Beta}(k,\, n+1-k)$, y medirla en un test finito añade ruido binomial, así
    que lo observado sigue una **Beta-Binomial**. La anchura viene de esas dos fuentes:
    la calibración y el test finito. Es la misma para los dos intervalos, porque no
    depende ni del modelo ni de la escala de los residuos.

    La teoría supone platos intercambiables uno a uno. Si se reparten **grupos** de
    montaje enteros (como en el resto del proyecto), los platos de un grupo se parecen y
    hay menos información independiente de la que parece: la cobertura varía algo más que
    la curva teórica.
    """)


@app.cell
def _(mo):
    respect_groups = mo.ui.checkbox(value=True, label="Repartir por grupos de montaje")
    respect_groups
    return (respect_groups,)


@app.cell
def _(
    SEED,
    alpha_slider,
    coverage_distribution,
    dishes,
    group_labels,
    kcal,
    np,
    parts,
    pred_log,
    resampled_coverage,
    respect_groups,
):
    _pool = parts["calib"] + parts["test"]
    n_cal = len(parts["calib"])
    n_pool_test = len(_pool) - n_cal
    resampled = resampled_coverage(
        np.concatenate([kcal["calib"], kcal["test"]]),
        np.concatenate([pred_log["calib"], pred_log["test"]]),
        alpha_slider.value,
        n_cal,
        n_reps=1000,
        groups=group_labels(_pool, dishes) if respect_groups.value else None,
        seed=SEED,
    )
    theory_grid, theory_pmf = coverage_distribution(n_cal, n_pool_test, alpha_slider.value)
    return n_cal, n_pool_test, resampled, theory_grid, theory_pmf


@app.cell
def _(
    VARIANT_LABELS,
    alpha_slider,
    alt,
    mo,
    n_cal,
    n_pool_test,
    np,
    pl,
    resampled,
    theory_grid,
    theory_pmf,
):
    _bin_width = 0.005
    _edges = np.arange(0.70, 1.0 + _bin_width, _bin_width)
    _hist = pl.DataFrame(
        [
            {"variante": VARIANT_LABELS[_v], "lo": float(_a), "hi": float(_b), "densidad": _c / (len(_covs) * _bin_width)}
            for _v, _covs in resampled.items()
            for _a, _b, _c in zip(_edges, _edges[1:], np.histogram(_covs, bins=_edges)[0])
        ]
    )
    # Beta-Binomial pmf over coverage values spaced 1/n_test apart -> density.
    _theory = pl.DataFrame(
        {"cobertura": theory_grid, "densidad": theory_pmf * n_pool_test}
    ).filter((pl.col("cobertura") >= 0.70) & (pl.col("cobertura") <= 1.0))
    _panels = [
        (
            alt.Chart(_hist.filter(pl.col("variante") == _label))
            .mark_rect(opacity=0.6)
            .encode(
                x=alt.X("lo:Q", title="Cobertura medida en test", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0.75, 1])),
                x2="hi:Q",
                y=alt.Y("densidad:Q", title="Densidad"),
                y2=alt.datum(0),
            )
            + alt.Chart(_theory).mark_line(color="black").encode(x="cobertura:Q", y="densidad:Q")
            + alt.Chart(pl.DataFrame({"x": [1 - alpha_slider.value]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
        ).properties(width=300, height=220, title=_label)
        for _label in VARIANT_LABELS.values()
    ]
    _m = float(theory_grid @ theory_pmf)
    _sd = float(np.sqrt(((theory_grid - _m) ** 2) @ theory_pmf))
    mo.vstack(
        [
            alt.hconcat(*_panels),
            mo.md(
                f"Teoría: {_m:.1%} ± {_sd:.1%} · "
                + " · ".join(
                    f"{VARIANT_LABELS[_v]}: {_c.mean():.1%} ± {_c.std():.1%}" for _v, _c in resampled.items()
                )
                + f" (1.000 repeticiones, calibración de {n_cal} y test de unos {n_pool_test} platos)."
            ),
        ]
    )


@app.cell
def _(mo):
    mo.md("""
    ## Ejemplos de test
    """)


@app.cell
def _(mo):
    example_seed = mo.ui.number(start=0, stop=10_000, value=0, label="Semilla de la muestra")
    example_seed
    return (example_seed,)


@app.cell
def _(
    Image,
    ROOT,
    alpha_slider,
    example_seed,
    image_path,
    kcal,
    mo,
    np,
    parts,
    pred_kcal,
    test_intervals,
):
    _lo, _hi = test_intervals["multiplicativo (log)"]
    _idx = np.random.default_rng(example_seed.value).choice(len(parts["test"]), 8, replace=False)
    _cards = []
    for _i in _idx:
        _inside = _lo[_i] <= kcal["test"][_i] <= _hi[_i]
        _cards.append(
            mo.vstack(
                [
                    mo.image(Image.open(image_path(ROOT, parts["test"][_i])), width=180),
                    mo.md(
                        f"real **{kcal['test'][_i]:.0f}** · pred {pred_kcal['test'][_i]:.0f}  \n"
                        f"[{_lo[_i]:.0f}, {_hi[_i]:.0f}] {'✅' if _inside else '❌'}"
                    ),
                ],
                align="center",
            )
        )
    mo.vstack(
        [
            mo.md(f"Intervalo multiplicativo al {1 - alpha_slider.value:.0%}, en kcal."),
            mo.hstack(_cards, wrap=True),
        ]
    )


@app.cell(hide_code=True)
def _(
    SEED,
    VARIANT_LABELS,
    alpha_slider,
    by_size,
    coverage,
    kcal,
    mae,
    mo,
    np,
    pred_kcal,
    test_intervals,
):
    _mul, _add = VARIANT_LABELS["multiplicative"], VARIANT_LABELS["additive"]
    _target = 1 - alpha_slider.value
    _cov = {_name: coverage(kcal["test"], _lo, _hi) for _name, (_lo, _hi) in test_intervals.items()}
    _q = by_size.pivot(on="variante", index="orden", values="cobertura").sort("orden")
    _spread = {_name: float(_q[_name].max() - _q[_name].min()) for _name in (_mul, _add)}
    _balanced, _other = (_mul, _add) if _spread[_mul] <= _spread[_add] else (_add, _mul)
    _worst = {_name: (float(_q[_name].min()), int(_q["orden"][int(_q[_name].arg_min())]) + 1) for _name in (_mul, _add)}
    _naive = mae(kcal["test"], np.full_like(kcal["test"], np.median(kcal["train"])))
    _sd = np.sqrt(alpha_slider.value * (1 - alpha_slider.value) / len(kcal["test"]))
    _close = all(abs(_c - _target) <= 2 * _sd for _c in _cov.values())

    mo.md(
        f"""
    ## Conclusiones (α = {alpha_slider.value:.2f}, semilla {SEED})

    - Sin fine-tuning, el MAE en test es de **{mae(kcal["test"], pred_kcal["test"]):.0f} kcal**
      ({mae(kcal["test"], pred_kcal["test"]) / kcal["test"].mean():.0%} de la media), frente a
      {_naive:.0f} kcal prediciendo siempre la mediana. El modelo del artículo, con fine-tuning
      y otra partición, llega a 70,6 kcal.
    - Cobertura en test: **{_cov[_mul]:.1%}** (multiplicativo) y **{_cov[_add]:.1%}** (aditivo),
      para un objetivo de {_target:.0%}. {"Las dos están" if _close else "No todas están"} dentro
      de ±2σ del objetivo (σ ≈ {_sd:.1%} con {len(kcal["test"])} platos).
    - El intervalo **{_balanced}** reparte mejor la cobertura entre cuartiles de kcal
      (diferencia entre el mejor y el peor cuartil de {_spread[_balanced]:.0%}, frente a
      {_spread[_other]:.0%}). El peor cuartil del {_add} se queda en {_worst[_add][0]:.0%}
      (cuartil {_worst[_add][1]}) y el del {_mul} en {_worst[_mul][0]:.0%} (cuartil {_worst[_mul][1]}).
    - Siguientes pasos: conformal normalizado y CQR para adaptar el ancho a la dificultad de
      cada plato, no solo a su tamaño; y fine-tuning del ViT.
    """
    )


if __name__ == "__main__":
    app.run()
