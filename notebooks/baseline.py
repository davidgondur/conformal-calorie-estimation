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
        label_problem,
        official_splits,
        read_dishes,
        split_by_groups,
    )
    from conformal_calorie_estimation.features import DEFAULT_MODEL, cached_features

    return (
        DEFAULT_MODEL,
        Image,
        Ridge,
        StandardScaler,
        absolute_residual_interval,
        alt,
        cached_features,
        clean_train,
        conformal_quantile,
        coverage,
        image_path,
        label_problem,
        make_pipeline,
        mo,
        np,
        official_splits,
        pl,
        read_dishes,
        split_by_groups,
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
def _(clean_train, mo, official_splits, read_dishes, split_by_groups):
    ROOT = mo.notebook_dir().parent / "data" / "nutrition5k"
    FEATURES_DIR = mo.notebook_dir().parent / "data" / "features"
    SEED = 0

    dishes = read_dishes(ROOT)
    official = official_splits(ROOT)
    _parts = split_by_groups(official["train"], dishes, {"val": 0.1, "calib": 0.2}, seed=SEED)
    removed = sorted(set(_parts["train"]) - set(clean_train(_parts["train"], dishes)))
    parts = {
        "train": clean_train(_parts["train"], dishes),
        "val": _parts["val"],
        "calib": _parts["calib"],
        "test": official["test"],
    }
    return FEATURES_DIR, ROOT, dishes, official, parts, removed


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

    Los embeddings se calculan una vez y se guardan en `data/features/` (en GPU, unos
    segundos). El objetivo es `log1p(kcal)`: la distribución de calorías es muy
    asimétrica, y en escala logarítmica los errores son relativos (±30 % en lugar
    de ±150 kcal). `log1p` en vez de `log` porque test tiene un plato con 0 kcal.
    """)


@app.cell
def _(
    DEFAULT_MODEL,
    FEATURES_DIR,
    ROOT,
    cached_features,
    dishes,
    np,
    official,
    parts,
):
    _features = cached_features(
        ROOT, official["train"] + official["test"], FEATURES_DIR, model_name=DEFAULT_MODEL
    )
    X = {k: np.stack([_features[i] for i in ids]) for k, ids in parts.items()}
    kcal = {k: np.array([dishes[i].calories for i in ids]) for k, ids in parts.items()}
    return X, kcal


@app.cell
def _(Ridge, StandardScaler, X, kcal, make_pipeline, np, pl):
    def fit_ridge(alpha):
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(
            X["train"], np.log1p(kcal["train"])
        )

    def mae(y, pred):
        return float(np.mean(np.abs(y - pred)))

    # Regularization chosen on val only; calib and test are not looked at.
    _alphas = [1, 10, 30, 100, 300, 1000, 3000, 10000]
    ridge_search = pl.DataFrame(
        {
            "alpha": _alphas,
            "val_MAE_kcal": [
                mae(kcal["val"], np.expm1(fit_ridge(a).predict(X["val"]))) for a in _alphas
            ],
        }
    )
    best_alpha = ridge_search.sort("val_MAE_kcal")["alpha"][0]
    model = fit_ridge(best_alpha)
    pred_log = {k: model.predict(x) for k, x in X.items()}
    pred_kcal = {k: np.expm1(p).clip(0) for k, p in pred_log.items()}
    return best_alpha, mae, pred_kcal, pred_log, ridge_search


@app.cell
def _(alt, best_alpha, kcal, mae, mo, np, pred_kcal, ridge_search):
    _chart = (
        alt.Chart(ridge_search)
        .mark_line(point=True)
        .encode(
            x=alt.X("alpha:Q", scale=alt.Scale(type="log"), title="alpha de ridge"),
            y=alt.Y("val_MAE_kcal:Q", scale=alt.Scale(zero=False), title="MAE en val (kcal)"),
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
def _(absolute_residual_interval, alpha_slider, kcal, np, pred_kcal, pred_log):
    def intervals(alpha, split="test"):
        """Both conformal variants for ``split``, in kcal: {name: (lower, upper)}."""
        _lo, _hi = absolute_residual_interval(
            np.log1p(kcal["calib"]), pred_log["calib"], pred_log[split], alpha
        )
        _alo, _ahi = absolute_residual_interval(
            kcal["calib"], pred_kcal["calib"], pred_kcal[split], alpha
        )
        return {
            "multiplicativo (log)": (np.expm1(_lo).clip(0), np.expm1(_hi)),
            "aditivo (kcal)": (_alo.clip(0), _ahi),
        }

    test_intervals = intervals(alpha_slider.value)
    return intervals, test_intervals


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
def _(alpha_slider, alt, coverage, kcal, np, pl, test_intervals):
    _y = kcal["test"]
    _edges = np.quantile(_y, [0, 0.25, 0.5, 0.75, 1])
    _labels = [f"{a:.0f}–{b:.0f}" for a, b in zip(_edges, _edges[1:])]
    _bin = np.clip(np.searchsorted(_edges, _y, side="right") - 1, 0, 3)
    _rows = []
    for _name, (_lo, _hi) in test_intervals.items():
        for _b, _label in enumerate(_labels):
            _m = _bin == _b
            _rows.append(
                {
                    "variante": _name,
                    "kcal reales (cuartil)": _label,
                    "orden": _b,
                    "cobertura": coverage(_y[_m], _lo[_m], _hi[_m]),
                    "ancho medio": float(np.mean(_hi[_m] - _lo[_m])),
                }
            )
    by_size = pl.DataFrame(_rows)

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
def _(mo):
    mo.md(r"""
    ## Conclusiones (α = 0,1, semilla 0)

    - Sin fine-tuning, el MAE en test es de unas **76 kcal (~30 % de la media)**, cerca de
      las 70,6 kcal del modelo ajustado del artículo (con otro split y otras imágenes).
    - Las dos variantes conformales cumplen la cobertura **en promedio** (~89 %, dentro
      del ruido con 507 platos).
    - El margen **aditivo** se reparte mal: cubre el 100 % de los platos pequeños con
      intervalos de ~200 kcal y solo ~64 % de los grandes. El **multiplicativo** está
      mucho más equilibrado (81–94 % por cuartil), aunque sigue cubriendo de menos los
      platos muy pequeños y los muy grandes.
    - Siguientes pasos: conformal normalizado y CQR para adaptar el ancho a la
      dificultad de cada plato, no solo a su tamaño; y fine-tuning del ViT.
    """)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
