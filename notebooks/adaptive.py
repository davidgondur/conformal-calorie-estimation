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
    from scipy.stats import spearmanr

    from conformal_calorie_estimation.adaptive import (
        adaptive_intervals,
        baseline_outputs,
        fit_difficulty,
        fit_quantiles,
        resampled_adaptive_coverage,
    )
    from conformal_calorie_estimation.baseline import fit_log_ridge, prepare_baseline
    from conformal_calorie_estimation.conformal import (
        coverage,
        coverage_by_group,
        interval_score,
        quantile_groups,
    )
    from conformal_calorie_estimation.data import image_path

    return (
        Image,
        adaptive_intervals,
        baseline_outputs,
        alt,
        coverage,
        coverage_by_group,
        fit_difficulty,
        fit_log_ridge,
        fit_quantiles,
        image_path,
        interval_score,
        mo,
        np,
        pl,
        prepare_baseline,
        quantile_groups,
        resampled_adaptive_coverage,
        spearmanr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Intervalos adaptativos: conformal normalizado y CQR

    El modelo base (`baseline.py`) cumple la cobertura en promedio, pero su ancho solo
    depende del **tamaño** estimado del plato. Aquí se añaden dos variantes cuyo ancho
    depende de la **dificultad** de cada plato, sobre los mismos embeddings de DINOv2:

    | Variante | Intervalo | Escala |
    |---|---|---|
    | aditiva | $\hat y \pm \hat q$ | kcal |
    | multiplicativa | $\hat y \pm \hat q$ | log(1 + kcal) |
    | **normalizada** | $\hat y \pm \hat q \cdot \hat\sigma(x)$, con $\hat\sigma$ la dificultad estimada | log(1 + kcal) |
    | **CQR** | $[\hat q_{\alpha/2}(x) - \hat q,\ \hat q_{1-\alpha/2}(x) + \hat q]$ | log(1 + kcal) |

    Todas se calibran con `calib`. Los modelos nuevos solo leen `train` y `val`.
    """)


@app.cell
def _(baseline_outputs, fit_log_ridge, mo, np, prepare_baseline):
    ROOT = mo.notebook_dir().parent / "data" / "nutrition5k"
    FEATURES_DIR = mo.notebook_dir().parent / "data" / "features"
    SEED = 0
    ALPHAS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    VARIANT_LABELS = {
        "additive": "aditiva",
        "multiplicative": "multiplicativa",
        "normalized": "normalizada",
        "cqr": "CQR",
    }
    # Validated palette (CVD, contrast, chroma); blue/orange as in baseline.py.
    VARIANT_COLORS = {"multiplicativa": "#1B6FA8", "aditiva": "#D4521F", "normalizada": "#2E8B57", "CQR": "#8A4FBF"}

    data = prepare_baseline(ROOT, FEATURES_DIR, seed=SEED)
    ridge = fit_log_ridge(data.X, data.kcal)
    # Predictions, embeddings and honest (cross-fitted) train residuals.
    outputs = baseline_outputs(data, ridge, seed=SEED)
    pred_kcal_test = np.expm1(outputs.pred_log["test"]).clip(0)
    return (
        ALPHAS,
        ROOT,
        SEED,
        VARIANT_COLORS,
        VARIANT_LABELS,
        data,
        outputs,
        pred_kcal_test,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Modelo de dificultad (normalizado)

    Un ridge sobre los embeddings predice $\log|r|$, el logaritmo del error del modelo
    base. Para que esos errores sean honestos, se obtienen con **validación cruzada de 5
    partes en `train`, por grupos de montaje**: cada plato se predice con un ridge que no
    lo ha visto. La regularización se elige con `val`. $\hat\sigma$ se recorta por abajo al
    percentil 5 de `train`, para que ningún plato reciba un intervalo casi vacío.
    """)


@app.cell
def _(fit_difficulty, np, outputs, spearmanr):
    difficulty = fit_difficulty(outputs)
    _val_residual = np.abs(np.log1p(outputs.kcal["val"]) - outputs.pred_log["val"])
    _sigma_val = difficulty.sigma(outputs.emb["val"])
    difficulty_rho = float(spearmanr(_sigma_val, _val_residual).statistic)
    floored_share = float(np.mean(difficulty.sigma(outputs.emb["test"]) <= difficulty.floor))
    return difficulty, difficulty_rho, floored_share


@app.cell
def _(difficulty, difficulty_rho, floored_share, mo, pl):
    mo.hstack(
        [
            pl.DataFrame(difficulty.search),
            mo.md(
                f"""
    - Regularización elegida: **{difficulty.alpha}**
    - Correlación de rangos (Spearman) entre $\\hat\\sigma$ y el error real en `val`:
      **{difficulty_rho:.2f}**. 0 sería no predecir nada; 1, ordenar perfectamente.
    - Mínimo de $\\hat\\sigma$: {difficulty.floor:.3f} · platos de test en el mínimo: {floored_share:.0%}
    """
            ),
        ],
        align="center",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Modelos de cuantiles (CQR)

    Dos regresiones cuantílicas lineales (penalización L1) predicen los cuantiles
    $\alpha/2$ y $1-\alpha/2$ de log(1 + kcal). Cada penalización se elige con la pérdida
    *pinball* en `val`. Como dependen de α, se ajustan una vez para cada α del slider
    (unos segundos por α la primera vez).
    """)


@app.cell
def _(ALPHAS, fit_quantiles, mo, outputs):
    with mo.status.spinner(title="Ajustando los modelos de cuantiles para cada α..."):
        quantile_fits = {round(_a, 2): fit_quantiles(outputs, _a) for _a in ALPHAS}
    return (quantile_fits,)


@app.cell
def _(ALPHAS, mo):
    alpha_slider = mo.ui.slider(
        steps=list(ALPHAS), value=0.10, label="α", show_value=True
    )
    alpha_slider
    return (alpha_slider,)


@app.cell
def _(alpha_slider, mo, outputs, quantile_fits):
    qfit = quantile_fits[round(alpha_slider.value, 2)]
    _crossed = {s: qfit.predict(outputs.emb[s])[2] for s in ("calib", "test")}
    mo.md(
        f"Cuantiles {alpha_slider.value / 2:.3f} y {1 - alpha_slider.value / 2:.3f} · "
        f"L1 elegida: inferior {qfit.l1['lower']}, superior {qfit.l1['upper']} · "
        f"cuantiles cruzados (se intercambian): {_crossed['calib']} en `calib`, {_crossed['test']} en `test`"
    )
    return (qfit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparación en test

    El ***interval score*** (Winkler) resume ancho y fallos en un número: el ancho del
    intervalo más $2/\alpha$ por cada kcal que el valor real queda fuera. **Más bajo es
    mejor.**
    """)


@app.cell
def _(
    VARIANT_LABELS,
    adaptive_intervals,
    alpha_slider,
    difficulty,
    outputs,
    qfit,
):
    test_intervals = {
        VARIANT_LABELS[_v]: _lohi
        for _v, _lohi in adaptive_intervals(outputs, difficulty, qfit, alpha_slider.value).items()
    }
    return (test_intervals,)


@app.cell
def _(
    alpha_slider,
    coverage,
    data,
    interval_score,
    mo,
    np,
    pl,
    test_intervals,
):
    _y = data.kcal["test"]
    summary = pl.DataFrame(
        [
            {
                "variante": _name,
                "cobertura": coverage(_y, _lo, _hi),
                "ancho medio": float(np.mean(_hi - _lo)),
                "ancho mediano": float(np.median(_hi - _lo)),
                "interval score": interval_score(_y, _lo, _hi, alpha_slider.value),
            }
            for _name, (_lo, _hi) in test_intervals.items()
        ]
    )
    _sd = np.sqrt(alpha_slider.value * (1 - alpha_slider.value) / len(_y))
    mo.vstack(
        [
            summary.with_columns(
                pl.col("cobertura").round(3), pl.col("^ancho.*$").round(0), pl.col("interval score").round(0)
            ),
            mo.md(
                f"Objetivo: cobertura ≥ {1 - alpha_slider.value:.0%}. Con {len(_y)} platos de test, la "
                f"cobertura medida fluctúa unos ±{_sd:.1%} (1σ) solo por el tamaño del test, y algo más "
                "porque los platos de un mismo grupo se parecen (ver la última sección)."
            ),
        ]
    )
    return (summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ¿Dónde se reparte la cobertura?

    Una buena variante adaptativa debería acercarse al objetivo en **todos** los grupos,
    no solo en promedio. Las agrupaciones:

    - **kcal predichas** (cuartiles): lo único que se conoce al predecir.
    - **número de ingredientes**: complejidad del plato. Es una etiqueta, no se conoce
      al predecir, pero describe qué platos son difíciles.
    - **kcal reales** (cuartiles): la agrupación del modelo base, para comparar.
    """)


@app.cell
def _(mo):
    grouping = mo.ui.dropdown(
        options=["kcal predichas", "número de ingredientes", "kcal reales"],
        value="kcal predichas",
        label="Agrupar por",
    )
    grouping
    return (grouping,)


@app.cell
def _(
    coverage_by_group,
    data,
    grouping,
    np,
    pl,
    pred_kcal_test,
    quantile_groups,
    test_intervals,
):
    def group_test_dishes(kind):
        """Group index per test dish and a label per group."""
        if kind == "kcal predichas":
            return quantile_groups(pred_kcal_test, n=4)
        if kind == "kcal reales":
            return quantile_groups(data.kcal["test"], n=4)
        _n = np.array([len(data.dishes[_i].ingredients) for _i in data.parts["test"]])
        return np.digitize(_n, [2, 6, 13]), ["1", "2–5", "6–12", "13+"]

    _groups, group_names = group_test_dishes(grouping.value)
    by_group = pl.DataFrame(
        [
            {
                "variante": _name,
                "grupo": group_names[_g],
                "orden": _g,
                "platos": _s["n"],
                "cobertura": _s["coverage"],
                "ancho medio": _s["mean_width"],
            }
            for _name, (_lo, _hi) in test_intervals.items()
            for _g, _s in coverage_by_group(data.kcal["test"], _lo, _hi, _groups).items()
        ]
    )
    return by_group, group_names


@app.cell
def _(
    VARIANT_COLORS,
    alpha_slider,
    alt,
    by_group,
    group_names,
    grouping,
    mo,
    pl,
):
    _color = alt.Color(
        "variante:N",
        scale=alt.Scale(domain=list(VARIANT_COLORS), range=list(VARIANT_COLORS.values())),
        title=None,
    )
    _x = alt.X("grupo:N", sort=group_names, title=grouping.value, axis=alt.Axis(labelAngle=0))
    _off = alt.XOffset("variante:N", sort=list(VARIANT_COLORS))
    _tooltip = ["variante", "grupo", "platos", alt.Tooltip("cobertura:Q", format=".1%"), alt.Tooltip("ancho medio:Q", format=".0f")]
    _cov = (
        alt.Chart(by_group).mark_bar().encode(
            x=_x, xOffset=_off, color=_color, tooltip=_tooltip,
            y=alt.Y("cobertura:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title="Cobertura"),
        )
        + alt.Chart(pl.DataFrame({"y": [1 - alpha_slider.value]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
    ).properties(width=320, height=240, title="Cobertura")
    _width = (
        alt.Chart(by_group).mark_bar().encode(
            x=_x, xOffset=_off, color=_color, tooltip=_tooltip,
            y=alt.Y("ancho medio:Q", title="kcal"),
        )
    ).properties(width=320, height=240, title="Ancho medio del intervalo")
    _spread = (
        by_group.group_by("variante", maintain_order=True)
        .agg(
            peor=pl.col("cobertura").min(),
            mejor=pl.col("cobertura").max(),
        )
        .with_columns(diferencia=pl.col("mejor") - pl.col("peor"))
        .sort("diferencia")
    )
    # Static chart: mo.ui.altair_chart would add the same legend selection to
    # both panels, which Vega rejects as a duplicate signal.
    mo.vstack([_cov | _width, mo.md("Diferencia de cobertura entre el mejor y el peor grupo (menor es mejor):"), _spread])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Qué platos reciben intervalos estrechos y anchos

    Ancho **relativo** (ancho / kcal predichas): con él se ve qué platos considera fáciles
    o difíciles cada variante, sin que domine el tamaño.
    """)


@app.cell
def _(VARIANT_LABELS, mo):
    example_variant = mo.ui.dropdown(
        options=[VARIANT_LABELS[_v] for _v in ("normalized", "cqr", "multiplicative")],
        value=VARIANT_LABELS["normalized"],
        label="Variante",
    )
    example_variant
    return (example_variant,)


@app.cell
def _(
    Image,
    ROOT,
    data,
    example_variant,
    image_path,
    mo,
    np,
    pred_kcal_test,
    test_intervals,
):
    _lo, _hi = test_intervals[example_variant.value]
    _rel = (_hi - _lo) / np.maximum(pred_kcal_test, 1)
    _order = np.argsort(_rel)

    def _card(_i):
        _y = data.kcal["test"][_i]
        _ok = "dentro" if _lo[_i] <= _y <= _hi[_i] else "**fuera**"
        return mo.vstack(
            [
                mo.image(Image.open(image_path(ROOT, data.parts["test"][_i])), width=160),
                mo.md(
                    f"real **{_y:.0f}** · pred {pred_kcal_test[_i]:.0f}  \n"
                    f"[{_lo[_i]:.0f}, {_hi[_i]:.0f}] · {_ok}  \n"
                    f"ancho ×{_rel[_i]:.1f} la predicción"
                ),
            ],
            align="center",
        )

    mo.vstack(
        [
            mo.md("**Intervalos más estrechos (relativos)**"),
            mo.hstack([_card(_i) for _i in _order[:4]], wrap=True),
            mo.md("**Intervalos más anchos (relativos)**"),
            mo.hstack([_card(_i) for _i in _order[-4:][::-1]], wrap=True),
        ]
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## La garantía, variante por variante

    Como en el modelo base: con todos los modelos fijos, se juntan `calib` y `test` y se
    reparten al azar 1.000 veces. La garantía promete una cobertura media ≥ 1 − α para
    **todas** las variantes; la cobertura de la partición oficial es solo una de esas
    repeticiones.
    """)


@app.cell
def _(mo):
    respect_groups = mo.ui.checkbox(value=True, label="Repartir por grupos de montaje")
    respect_groups
    return (respect_groups,)


@app.cell
def _(
    SEED,
    VARIANT_LABELS,
    alpha_slider,
    difficulty,
    mo,
    np,
    outputs,
    pl,
    qfit,
    resampled_adaptive_coverage,
    respect_groups,
    summary,
):
    resampled = resampled_adaptive_coverage(
        outputs, difficulty, qfit, n_reps=1000, respect_groups=respect_groups.value, seed=SEED
    )
    _official = dict(zip(summary["variante"], summary["cobertura"]))
    resampled_table = pl.DataFrame(
        [
            {
                "variante": VARIANT_LABELS[_v],
                "media": float(_c.mean()),
                "desv. típica": float(_c.std()),
                "partición oficial": _official[VARIANT_LABELS[_v]],
                "repartos con menos cobertura que la oficial": float(np.mean(_c <= _official[VARIANT_LABELS[_v]])),
            }
            for _v, _c in resampled.items()
        ]
    )
    mo.vstack(
        [
            resampled_table.with_columns(pl.exclude("variante").round(3)),
            mo.md(
                f"Objetivo ≥ {1 - alpha_slider.value:.0%}. La última columna dice cuán inusual es la "
                "partición oficial: un valor pequeño significa que tocó un test más difícil que la calibración."
            ),
        ]
    )
    return (resampled_table,)


@app.cell
def _(
    SEED,
    alpha_slider,
    by_group,
    difficulty_rho,
    grouping,
    mo,
    resampled_table,
    summary,
):
    _best = summary.sort("interval score").row(0, named=True)
    _base = summary.filter(summary["variante"] == "multiplicativa").row(0, named=True)
    _spread = {
        _name: float(_g["cobertura"].max() - _g["cobertura"].min())
        for (_name,), _g in by_group.group_by("variante")
    }
    _most_even = min(_spread, key=_spread.get)
    _means = dict(zip(resampled_table["variante"], resampled_table["media"]))
    _all_ok = all(_m >= 1 - alpha_slider.value - 0.005 for _m in _means.values())

    mo.md(
        f"""
    ## Conclusiones (α = {alpha_slider.value:.2f}, semilla {SEED})

    - La variante con mejor *interval score* es la **{_best["variante"]}**
      ({_best["interval score"]:.0f}, frente a {_base["interval score"]:.0f} de la multiplicativa
      del modelo base), con un ancho mediano de {_best["ancho mediano"]:.0f} kcal.
    - Agrupando por {grouping.value}, la cobertura más uniforme es la de la **{_most_even}**
      (diferencia de {_spread[_most_even]:.0%} entre su mejor y su peor grupo).
    - El modelo de dificultad ordena los platos por su error de forma
      {"débil" if difficulty_rho < 0.3 else "moderada" if difficulty_rho < 0.6 else "fuerte"}
      (Spearman {difficulty_rho:.2f} en `val`){": hay margen para un mejor $\\hat\\sigma$" if difficulty_rho < 0.6 else ""}.
    - Sobre 1.000 repartos, {"todas las variantes" if _all_ok else "no todas las variantes"} tienen
      cobertura media en torno al objetivo; las diferencias en la partición oficial son en parte azar.
    """
    )


if __name__ == "__main__":
    app.run()
