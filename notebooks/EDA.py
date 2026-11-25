import marimo

__generated_with = "0.26.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import polars as pl
    from PIL import Image

    from conformal_calorie_estimation.data import image_path, official_splits, read_dishes

    return Image, alt, image_path, mo, official_splits, pl, read_dishes


@app.cell
def _(mo):
    mo.md(r"""
    # EDA: Nutrition5k (fotos cenitales)

    Análisis exploratorio de las particiones oficiales **train** y **test**
    (`depth_{train,test}_ids.txt`). Solo se usan las fotos cenitales RGB.
    """)


@app.cell
def load(mo, official_splits, pl, read_dishes):
    ROOT = mo.notebook_dir().parent / "data" / "nutrition5k"
    dishes = read_dishes(ROOT)
    splits = official_splits(ROOT)

    df = pl.DataFrame(
        [
            {
                "dish_id": d.id,
                "split": split,
                "calories": d.calories,
                "mass": d.mass,
                "fat": d.fat,
                "carb": d.carb,
                "protein": d.protein,
                "n_ingredients": len(d.ingredients),
            }
            for split, ids in splits.items()
            for d in (dishes[i] for i in ids)
        ]
    ).with_columns(kcal_per_g=pl.col("calories") / pl.col("mass"))
    df
    return ROOT, df, dishes, splits


@app.cell
def md_summary(mo):
    mo.md("""
    ## Resumen por partición
    """)


@app.cell
def summary(df, pl):
    summary = (
        df.group_by("split")
        .agg(
            n=pl.len(),
            kcal_mean=pl.col("calories").mean(),
            kcal_median=pl.col("calories").median(),
            kcal_p95=pl.col("calories").quantile(0.95),
            kcal_max=pl.col("calories").max(),
            mass_median=pl.col("mass").median(),
            ingr_median=pl.col("n_ingredients").median(),
        )
        .sort("split", descending=True)
    )
    summary


@app.cell
def md_calories(mo):
    mo.md(r"""
    ## Distribución de calorías

    Muy asimétrica a la derecha. En escala logarítmica (derecha) se ve mejor la forma
    y la cola de platos casi vacíos. Train y test tienen distribuciones parecidas.
    """)


@app.cell
def chart_calories(alt, df, mo):
    _base = alt.Chart(df).encode(color=alt.Color("split:N", title="Partición"))
    _lin = _base.mark_bar(opacity=0.6).encode(
        x=alt.X("calories:Q", bin=alt.Bin(maxbins=60), title="kcal"),
        y=alt.Y("count():Q", stack=None, title="Platos"),
    )
    _log = (
        _base.transform_filter("datum.calories > 0")
        .transform_calculate(log_kcal="log(datum.calories) / LN10")
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X("log_kcal:Q", bin=alt.Bin(maxbins=50), title="log10(kcal)"),
            y=alt.Y("count():Q", stack=None, title="Platos"),
        )
    )
    mo.ui.altair_chart((_lin.properties(width=320) | _log.properties(width=320)))


@app.cell
def md_mass(mo):
    mo.md(r"""
    ## Masa frente a calorías

    La masa explica gran parte de las calorías, pero la densidad energética (kcal/g)
    varía mucho entre platos. Esa es la parte que la foto tiene que aportar: dos
    platos del mismo tamaño pueden diferir mucho en calorías.
    """)


@app.cell
def chart_mass(alt, df, mo):
    _scatter = (
        alt.Chart(df)
        .mark_circle(size=14, opacity=0.4)
        .encode(
            x=alt.X("mass:Q", scale=alt.Scale(type="symlog"), title="Masa (g)"),
            y=alt.Y("calories:Q", scale=alt.Scale(type="symlog"), title="kcal"),
            color=alt.Color("split:N", title="Partición"),
            tooltip=["dish_id", "calories", "mass", "n_ingredients"],
        )
        .properties(width=340, height=300)
    )
    _density = (
        alt.Chart(df)
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X("kcal_per_g:Q", bin=alt.Bin(maxbins=50), title="kcal/g"),
            y=alt.Y("count():Q", stack=None, title="Platos"),
            color="split:N",
        )
        .properties(width=300, height=300)
    )
    mo.ui.altair_chart(_scatter | _density)


@app.cell
def corr(df, mo, pl):
    mo.md(
        f"Correlación de Pearson (train): masa–kcal = "
        f"**{df.filter(pl.col('split') == 'train').select(pl.corr('mass', 'calories')).item():.2f}**, "
        f"nº ingredientes–kcal = "
        f"**{df.filter(pl.col('split') == 'train').select(pl.corr('n_ingredients', 'calories')).item():.2f}**"
    )


@app.cell
def md_ingr(mo):
    mo.md(r"""
    ## Ingredientes

    El 28 % de los platos tiene un único ingrediente. Son los primeros pasos de los
    montajes incrementales o platos sencillos. Los platos complejos de cafetería
    llegan a más de 30 ingredientes.
    """)


@app.cell
def ingredients(alt, df, dishes, mo, pl, splits):
    ingredients = pl.DataFrame(
        [
            {"split": split, "dish_id": i, "ingredient": ing.name, "grams": ing.grams}
            for split, ids in splits.items()
            for i in ids
            for ing in dishes[i].ingredients
        ]
    )
    _n_ingr = (
        alt.Chart(df)
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X("n_ingredients:O", title="Nº de ingredientes"),
            y=alt.Y("count():Q", stack=None, title="Platos"),
            color="split:N",
        )
        .properties(width=360, height=260)
    )
    _top = (
        ingredients.filter(pl.col("split") == "train")
        .group_by("ingredient")
        .agg(platos=pl.len())
        .sort("platos", descending=True)
        .head(20)
    )
    _top_chart = (
        alt.Chart(_top)
        .mark_bar()
        .encode(
            x=alt.X("platos:Q", title="Platos de train que lo contienen"),
            y=alt.Y("ingredient:N", sort="-x", title=None),
        )
        .properties(width=260, height=260)
    )
    mo.ui.altair_chart(_n_ingr | _top_chart)


@app.cell
def md_outliers(mo):
    mo.md(r"""
    ## Etiquetas sospechosas

    Hay errores de etiqueta claros, comprobados mirando la foto:

    - `dish_1560974769`: **3.943 kcal y 3.051 g**, por 2.991 g de arroz blanco. En la
      foto se ve una ración pequeña de arroz con espinacas.
    - `dish_1556575700`: etiquetado como *"plate only"* (86 g, 0 kcal), pero en la
      foto hay tomates cherry.

    Abajo están los platos con kcal < 5 o con más de 1.500 g. Habrá que decidir si se
    filtran antes de entrenar. En calibración y test no se pueden quitar sin más:
    quitar platos difíciles del test haría que la cobertura medida fuera optimista.
    """)


@app.cell
def suspects(df, pl):
    suspects = df.filter((pl.col("calories") < 5) | (pl.col("mass") > 1500)).sort("calories")
    suspects


@app.cell
def md_gallery(mo):
    mo.md("""
    ## Ejemplos de platos (train)
    """)


@app.cell
def gallery_n(mo):
    gallery_seed = mo.ui.number(start=0, stop=10_000, value=0, label="Semilla de la muestra")
    gallery_seed
    return (gallery_seed,)


@app.cell
def gallery(Image, ROOT, df, gallery_seed, image_path, mo, pl):
    _sample = df.filter(pl.col("split") == "train").sample(8, seed=gallery_seed.value)
    mo.hstack(
        [
            mo.vstack(
                [
                    mo.image(Image.open(image_path(ROOT, r["dish_id"])), width=180),
                    mo.md(f"**{r['calories']:.0f} kcal** · {r['mass']:.0f} g · {r['n_ingredients']} ingr."),
                ],
                align="center",
            )
            for r in _sample.iter_rows(named=True)
        ],
        wrap=True,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
