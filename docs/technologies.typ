#import "template.typ": *

#show: project.with(title: [Technologies used], subtitle: [A guided tour of the stack, from data to intervals])

= Introduction <sec:intro>

This document explains, one by one, the technologies that the project *actually uses so far*, and how each one fits in. It is meant to be read by someone who knows Python and basic machine learning but has not necessarily met these particular tools. Technologies that are planned but not yet used (fine-tuning, ONNX export, the web demo) are left out; they will be added when they enter the code.

The reasons for each choice are recorded in the companion document, _Decision log_. Here the focus is on *how things work*.

The project turns a photo of a dish into a calorie estimate with an interval that comes with a coverage guarantee:

#pipeline((
  [Overhead photo \ 640 × 480],
  [DINOv2-small \ (frozen, 384-d)],
  [Ridge regression \ on log(1 + kcal)],
  [Split conformal \ on `calib`],
  [Interval \ in kcal],
))

#figure(
  table(
    columns: (auto, 1fr, auto),
    align: (left, left, left),
    table.header[Layer][Technologies][Section],
    [Project tooling], [`uv`, pytest], [@sec:tooling],
    [Data], [Nutrition5k, Google Cloud Storage over HTTPS], [@sec:data],
    [Deep learning], [PyTorch, `timm`], [@sec:dl],
    [Image representation], [Vision Transformers, DINOv2], [@sec:vit],
    [Point prediction], [scikit-learn: standardization, ridge and quantile regression], [@sec:regression],
    [Uncertainty], [Split conformal prediction, normalized conformal, CQR], [@sec:conformal],
    [Analysis], [marimo, Polars, Altair / Vega-Lite], [@sec:analysis],
    [Documentation], [Typst], [@sec:typst],
  ),
  caption: [The stack at a glance.],
)

= Project tooling <sec:tooling>

== uv

#link("https://docs.astral.sh/uv/")[`uv`] is a Python package and project manager written in Rust. It replaces the usual combination of `pip`, `venv`, `pip-tools` and `pyenv` with one fast command. Three ideas cover almost all of its use in this project:

- *The project file.* `pyproject.toml` declares the package name, the supported Python versions and the dependencies. Dependencies are split into the package's own (`torch`, `timm`, ...) and a `dev` *dependency group* for tools that are only needed while developing (`marimo`, `pytest`, ...).
- *The lockfile.* `uv.lock` records the exact version of every package, including indirect ones. Anyone running `uv sync` gets the same environment, which makes results reproducible.
- *Running things.* `uv run <command>` makes sure the virtual environment in `.venv/` matches the lockfile and then runs the command inside it. There is no environment to activate by hand.

The project is an installable *package* with a `src/` layout: the code lives in `src/conformal_calorie_estimation/` and is installed into the environment, so notebooks, scripts and tests all import it the same way. `pyproject.toml` also declares an *entry point*, which is how `uv run n5k-download` becomes a command that calls `download.main()`.

```bash
uv sync                              # create or update .venv from uv.lock
uv add timm                          # add a dependency and update the lockfile
uv add --dev pytest                  # add a development-only dependency
uv run pytest -q                     # run a command inside the environment
```

== pytest

#link("https://docs.pytest.org/")[pytest] runs every function whose name starts with `test_` in the `tests/` folder and reports which assertions fail. Two features are used:

- *Fixtures*: a function decorated with `@pytest.fixture` builds something a test needs, and a test receives it just by naming it as a parameter. Here a fixture builds a tiny fake Nutrition5k folder (a few CSV rows and 8 × 6 pixel images) in a temporary directory, so the tests never need the network or the real data.
- *`pytest.raises`*: checks that bad input raises an error, for example a CSV row with the wrong number of fields.

One test is statistical rather than exact: it simulates 200 calibration/test experiments and checks that the average coverage of the conformal intervals reaches the promised level (@sec:conformal).

= Data <sec:data>

== Nutrition5k

Nutrition5k @thames2021nutrition5k is a dataset of about 5,000 real dishes served in Google cafeterias. Its labels are unusually accurate because they were *measured*, not estimated: each plate was assembled one ingredient at a time on a scale, and each ingredient's weight was multiplied by its nutritional content per gram from the USDA food database. Each dish therefore has total calories, total mass, fat, carbohydrate and protein, plus the same values per ingredient.

Each dish was recorded by two kinds of camera:

- four *side-angle* cameras that rotate around the plate and record short videos;
- one *overhead* camera (an Intel RealSense) that takes a colour photo and a depth map from above.

Only the overhead colour photo is used in this project. @fig:hist shows how calories are distributed: very skewed, with the middle half of the dishes between #num(m.eda.kcal_q25, d: 0) and #num(m.eda.kcal_q75, d: 0) kcal and a long tail.

#figure(
  image("figures/calories_hist.svg", width: 58%),
  caption: [Calories per dish in the official splits, on a log scale. Each panel is normalized by its number of dishes.],
) <fig:hist>

Mass explains a large part of the calories (correlation #num(m.eda.corr_mass_kcal_train, d: 2) in train), but energy density varies a lot (@fig:mass): two dishes of the same size can differ widely in calories. That remaining variation is what the photo has to explain.

#figure(
  image("figures/mass_vs_kcal.svg", width: 38%),
  caption: [Mass against calories, both on log scales.],
) <fig:mass>

#note(title: [Incremental scans])[
  Because plates were built one ingredient at a time and each step was photographed and weighed, consecutive "dishes" are often the same plate with one more ingredient. This is a property of the dataset to keep in mind whenever data is split: near-duplicates should not end up on both sides of a split.
]

== Google Cloud Storage over HTTPS

The dataset lives in a Google Cloud Storage *bucket*. A bucket is a flat collection of *objects* (files) whose names can contain slashes, which makes them look like folders. A public bucket can be used without any Google tool, through two plain HTTPS endpoints:

- *Listing*: `https://storage.googleapis.com/storage/v1/b/<bucket>/o?prefix=<path>` returns JSON with up to 1,000 object names and a `nextPageToken` to request the next page.
- *Download*: `https://storage.googleapis.com/<bucket>/<object>` returns the file itself.

`download.py` lists the bucket, keeps only the files it needs, and downloads them with 16 threads in parallel. Each file is written to a temporary `.part` name and renamed when complete, so an interrupted run never leaves a half-written file behind, and a re-run only downloads what is missing.

= Deep learning stack <sec:dl>

== PyTorch

#link("https://pytorch.org/")[PyTorch] is the deep learning library underneath everything. Three of its pieces appear in the code:

- *Tensors*: n-dimensional arrays, like NumPy arrays, that can live on the GPU. A batch of 64 RGB images of 224 × 224 pixels is a tensor of shape $(64, 3, 224, 224)$.
- *`Dataset` and `DataLoader`*: a `Dataset` knows how many items there are and how to load item $i$. `Nutrition5kOverhead` returns the photo of dish $i$ and its calories. A `DataLoader` wraps it, groups items into batches and loads them in parallel worker processes while the GPU is busy.
- *Inference mode and mixed precision*: `torch.inference_mode()` tells PyTorch that no gradients are needed, which saves memory and time. `torch.autocast` runs the matrix multiplications in 16-bit floating point on the GPU, which is faster and accurate enough for extracting features.

== timm

#link("https://github.com/huggingface/pytorch-image-models")[`timm`] (PyTorch Image Models) is a library of hundreds of pretrained vision models behind one interface. A single call downloads the weights from the Hugging Face Hub and builds the network:

```python
model = timm.create_model(
    "vit_small_patch14_dinov2.lvd142m",  # architecture . pretrained weights
    pretrained=True,
    num_classes=0,                       # no classification head: return the embedding
    img_size=224,                        # input size to use
)
```

Each pretrained model carries a *configuration* describing how its images were preprocessed during training. `timm` turns it into the matching transform, which here is:

+ resize the shorter side to 224 pixels with bicubic interpolation (a 640 × 480 photo becomes 299 × 224);
+ crop the central 224 × 224 square, which discards about a quarter of the photo's width (an eighth on each side);
+ convert to a tensor and *normalize* each colour channel with the ImageNet mean and standard deviation, so that inputs have the statistics the network was trained on.

Using the model's own preprocessing matters: a network fed differently normalized images produces worse features without any error message.

= Vision Transformers and DINOv2 <sec:vit>

== Vision Transformers

A Vision Transformer (ViT) @dosovitskiy2021image applies the Transformer architecture, created for text, to images. The idea is to treat an image as a sentence whose words are small square patches:

+ *Patches.* The 224 × 224 image is cut into patches of 14 × 14 pixels: a 16 × 16 grid, so 256 patches.
+ *Tokens.* Each patch is flattened and projected linearly into a vector of 384 numbers, its *token*. A learned *position embedding* is added so the model knows where each patch came from. One extra learned token, the *class token*, is prepended.
+ *Transformer blocks.* Twelve blocks follow. In each one, *self-attention* lets every token look at every other token and mix in information from the ones it finds relevant. Each block has 6 attention *heads*, each of which can attend to different relations (colour, texture, shape, position). A small feed-forward network then processes each token.
+ *Output.* After the last block, the class token has gathered information from the whole image. Its 384 numbers are the *embedding* of the image.

Unlike a convolutional network, which only combines neighbouring pixels in early layers, attention can relate distant parts of the image from the very first block, for example the rice and the sauce on opposite sides of a plate.

== Self-supervised learning: DINO and DINOv2

Training a ViT normally needs millions of labelled images. *Self-supervised learning* removes the labels: the model learns from the images alone, by solving a task whose answer is contained in the data.

DINO @caron2021emerging uses *self-distillation*. Two copies of the network see different random crops of the same image: a *student* and a *teacher*. The student is trained to produce the same output as the teacher, and the teacher is a slowly moving average of the student. To agree on two different views of the same image, the network has to learn what the image *contains*, not what the pixels look like.

DINOv2 @oquab2024dinov2 scales this recipe up: it adds a masked-image objective (predict the features of hidden patches), trains on LVD-142M, a curated collection of 142 million images, and distils the result from a very large ViT into smaller ones. The model used here, ViT-S/14, has 21.6 million parameters.

The result is a general-purpose image representation: images with similar content get similar embeddings, even for categories the model was never told about. That is why it works on food without ever seeing a calorie label.

== Using a backbone frozen

A pretrained network used without changing its weights is called a *frozen backbone*. Its embeddings are computed once for every photo (about 12 seconds for 3,262 photos on the GPU) and stored in `data/features/`. Any number of cheap models can then be fitted on top of them in seconds. Fitting a linear model on frozen features is known as a *linear probe*, and it is the standard way to measure how much useful information a representation already contains.

The alternative, *fine-tuning*, updates the backbone's weights on the target task. It usually performs better but costs a full training run, so it comes later in the project.

= Point prediction with ridge regression <sec:regression>

== Standardization

The 384 embedding dimensions have different scales. `StandardScaler` subtracts each dimension's mean and divides by its standard deviation, both computed on the training set only, so that every dimension has mean 0 and variance 1. Without it, the regularization below would penalize dimensions unevenly.

== Ridge regression

Ridge regression @hoerl1970ridge is linear regression with a penalty on the size of the weights. With embeddings $bold(x)_i$ and targets $y_i$, it finds

$ hat(bold(w)) = arg min_(bold(w)) sum_(i=1)^n (y_i - bold(w)^top bold(x)_i)^2 + lambda ||bold(w)||^2 . $

The first term fits the data; the second keeps the weights small. With 384 correlated dimensions and fewer than 2,000 training dishes, plain least squares would overfit: it would find large weights that exploit noise. The penalty $lambda$ trades a little bias for much less variance. It is chosen by trying several values and keeping the one with the lowest error on `val` (here $lambda = #m.best_alpha$).

In scikit-learn, both steps are chained into a *pipeline*, so the same scaling is applied when fitting and when predicting:

```python
model = make_pipeline(StandardScaler(), Ridge(alpha=lam))
model.fit(X_train, np.log1p(kcal_train))
kcal_pred = np.expm1(model.predict(X_test))
```

== Predicting on a log scale

The model does not predict kcal but $log(1 + "kcal")$, and transforms back with $exp(dot) - 1$ (`np.log1p` and `np.expm1`). On a log scale, an error of $0.1$ means "about 10% too high" whatever the size of the dish, so the model treats errors as *relative*. The $+1$ keeps the transform defined for a dish with 0 kcal.

With this setup the test MAE is #num(m.mae.test) kcal, #pct(m.mae_pct.test) of the mean, against #num(m.mae_naive_test) kcal for always predicting the median.

= Conformal prediction <sec:conformal>

== The problem: a number is not enough

A point prediction says "300 kcal" but not how far off it could be. Conformal prediction @vovk2005algorithmic @angelopoulos2023gentle turns any point predictor into an *interval* predictor with a guarantee: the true value falls inside the interval with probability at least $1 - alpha$, for a level $alpha$ chosen by the user (for example $alpha = 0.1$ for 90%).

The guarantee is remarkable for what it does *not* require: it holds whatever the model and whatever the distribution of the data. The only assumption is *exchangeability*: the calibration examples and the new example are like random draws from the same population, in any order. A random split of the same dataset satisfies it.

== Split conformal, step by step

Split conformal @lei2018distribution needs a model that is already fitted and a *calibration set* that the model has never seen:

+ *Scores.* For each of the $n$ calibration dishes, compute the absolute residual $s_i = |y_i - hat(y)_i|$: how wrong the model was.
+ *Quantile.* Sort the scores and take the $k$-th smallest, with $k = ceil((n + 1)(1 - alpha))$. Call it $hat(q)$.
+ *Interval.* For a new dish with prediction $hat(y)$, return $[hat(y) - hat(q), hat(y) + hat(q)]$.

#note(title: [Why it works])[
  Add the new dish to the $n$ calibration dishes. By exchangeability, its score is equally likely to occupy any of the $n + 1$ positions in the sorted order. So the probability that it is among the $k$ smallest is $k \/ (n + 1) >= 1 - alpha$. Being among the $k$ smallest means $s_"new" <= hat(q)$, which is exactly "the true value is inside the interval". The $n + 1$ (instead of $n$) is the *finite-sample correction* that makes the guarantee exact rather than approximate.
]

The whole method fits in a few lines (`conformal.py`):

```python
def conformal_quantile(scores, alpha):
    scores = np.sort(scores)
    k = math.ceil((len(scores) + 1) * (1 - alpha))
    return math.inf if k > len(scores) else scores[k - 1]
```

If the calibration set is too small for the requested $alpha$, $k$ exceeds $n$ and the only honest interval is infinite.

== What the guarantee covers, and what it does not

The guarantee is *marginal*: it holds on average over dishes. It says nothing about any particular kind of dish. In the experiments, both variants below reach about 90% overall, and the coverage measured on #m.n.test test dishes tracks the target closely across all levels (@fig:alpha). With that many test dishes the measured coverage naturally fluctuates by about ±#pct(m.cov_sd) around the true value.

#figure(
  image("figures/coverage_vs_alpha.svg", width: 46%),
  caption: [Coverage on test against the target $1 - alpha$. The grey line is perfect calibration.],
) <fig:alpha>

== Coverage varies with the calibration set

The guarantee is also an average *over calibration sets*: a different calibration set would have given a slightly different $hat(q)$, and so a different coverage. This can be measured. The model is kept fixed, the #m.resampling.n_cal calibration dishes and the #m.resampling.n_test test dishes are pooled, and the pool is split again at random #thousands(m.resampling.n_reps) times into a calibration set of the same size and a test set with the rest. Each split gives one coverage value (@fig:resampled).

Theory predicts the exact distribution of those values. For a fixed model and $n$ calibration points, with $k = ceil((n + 1)(1 - alpha))$, the coverage on infinitely many new points follows a $"Beta"(k, n + 1 - k)$ distribution. Measuring it on a finite test set adds binomial noise, so the observed fraction follows a *Beta-Binomial* distribution, available as `scipy.stats.betabinom`. Its spread has two sources, the calibration draw and the finite test set, and here it is #pct(m.resampling.theory_mean) ± #pct(m.resampling.theory_sd).

#let rs = m.resampling
#figure(
  table(
    columns: (1fr, auto, auto),
    align: (left, right, right),
    table.header[Re-split by][Multiplicative (log)][Additive (kcal)],
    [Theory (Beta-Binomial)], [#pct(rs.theory_mean) ± #pct(rs.theory_sd)], [#pct(rs.theory_mean) ± #pct(rs.theory_sd)],
    [Single dishes], [#pct(rs.dishes.at(MUL).mean) ± #pct(rs.dishes.at(MUL).sd)], [#pct(rs.dishes.at(ADD).mean) ± #pct(rs.dishes.at(ADD).sd)],
    [Incremental groups], [#pct(rs.groups.at(MUL).mean) ± #pct(rs.groups.at(MUL).sd)], [#pct(rs.groups.at(ADD).mean) ± #pct(rs.groups.at(ADD).sd)],
  ),
  caption: [Mean ± standard deviation of test coverage over #thousands(rs.n_reps) random calibration/test re-splits ($alpha = #num(m.alpha)$).],
) <tab:resampled>

Two things stand out (@tab:resampled):

- *The theory does not depend on the model.* The additive and the multiplicative intervals, which spread coverage so differently across dishes (@sec:scale), have the same distribution of overall coverage, and both match the theoretical curve.
- *Exchangeability is about the unit that is shuffled.* When single dishes are shuffled, the spread matches the theory. When whole incremental groups are kept together, as everywhere else in the project, the spread is larger: dishes in a group are near-duplicates, so the pool holds less independent information than its size suggests. The average is unchanged; only the fluctuation around it grows.

#figure(
  image("figures/coverage_resampled.svg", width: 100%),
  caption: [Test coverage over #thousands(rs.n_reps) re-splits that keep incremental groups together. The black curve is the Beta-Binomial predicted by theory; the grey line is the target $1 - alpha$.],
) <fig:resampled>

== The scale of the scores <sec:scale>

The residuals can be measured on any scale, and the choice changes the shape of the intervals:

- *Additive*: residuals in kcal. Every dish gets the same margin, here ±#num(m.q_add, d: 0) kcal.
- *Multiplicative*: residuals on the $log(1 + "kcal")$ scale. Transforming back turns $plus.minus hat(q)$ into "multiply or divide by about #num(m.q_log_factor, d: 2)", so large dishes get wider intervals.

Both keep the marginal guarantee, but they spread coverage very differently (@fig:size). The fixed margin over-covers small dishes with intervals wider than the dish itself, and under-covers large ones (#pct(size-row(ADD, 3).coverage, d: 0) in the top quartile). The multiplicative version is much more even. Methods that adapt the width to each dish's difficulty, such as conformalized quantile regression @romano2019conformalized, push this further; they are the next step in the project.

#figure(
  image("figures/coverage_by_size.svg", width: 100%),
  caption: [Coverage and mean width per quartile of true calories in test ($alpha = #num(m.alpha)$).],
) <fig:size>

#figure(
  grid(
    columns: 4,
    column-gutter: 8pt,
    row-gutter: 4pt,
    ..m.examples.map(e => image(e.file, width: 100%)),
    ..m.examples.map(e => align(center, text(size: 8.5pt)[
      true *#num(e.real, d: 0)* · predicted #num(e.pred, d: 0) \
      [#num(e.lo, d: 0), #num(e.hi, d: 0)] kcal · #if e.lo <= e.real and e.real <= e.hi [inside] else [*outside*]
    ])),
  ),
  caption: [Test dishes with their prediction and their 90% multiplicative interval.],
)

== Adaptive intervals <sec:adaptive>

#let ad = m.adaptive
#let av(v) = ad.variants.at(v)
#let NORM = "normalized (log)"
#let CQRL = "CQR (log)"

The variants so far use the same margin for every dish on their scale: their width depends only on the predicted *size*. An *adaptive* interval should also depend on how *hard* the dish is: narrow for a plain salad, wide for an opaque stew. Two standard methods achieve this, both built here on the $log(1 + "kcal")$ scale (notebook `notebooks/adaptive.py`, code in `adaptive.py`).

=== Normalized conformal

The idea is to divide each residual by an estimate of how large it should be:

$ s_i = (|y_i - hat(y)_i|) / (hat(sigma)(bold(x)_i)), quad "interval" = hat(y) plus.minus hat(q) dot hat(sigma)(bold(x)) . $

The quantile $hat(q)$ is computed exactly as before, so the guarantee is unchanged: dividing by $hat(sigma)$ only changes the *shape* of the score. If $hat(sigma)$ is good, hard dishes get wide intervals and easy dishes narrow ones; if it is useless, the method falls back to roughly the plain interval.

$hat(sigma)$ is a second model, the *difficulty model*: a ridge on the embeddings that predicts the logarithm of the baseline's absolute residual. Training it needs residuals that look like residuals on new dishes, and residuals on the training set do not: the ridge has already fitted those dishes, so they are too small. *Cross-fitting* solves this: `train` is split into 5 folds (keeping incremental groups together), each fold is predicted by a ridge trained on the other four, and every dish ends up with an honest, out-of-fold residual.

=== Quantile regression and the pinball loss

Ordinary regression predicts the mean. *Quantile regression* predicts a quantile instead: for example the value below which 95% of dishes like this one fall. It is trained by minimizing the *pinball loss*, which penalizes under- and over-predictions asymmetrically:

$ L_tau (y, hat(y)) = cases(tau (y - hat(y)) & "if" y >= hat(y), (1 - tau)(hat(y) - y) & "otherwise") $

For $tau = 0.95$, predicting too low costs 19 times more than predicting too high, so the minimizer sits where only 5% of the values are above it. scikit-learn's `QuantileRegressor` fits a linear model with this loss and an L1 penalty.

=== Conformalized quantile regression (CQR)

Quantile regression alone has no guarantee: on new data, the predicted 5% and 95% quantiles may cover 80% or 95% of dishes. CQR @romano2019conformalized calibrates them. With predicted quantiles $hat(q)_"lo"(bold(x))$ and $hat(q)_"hi"(bold(x))$, the score measures how far the truth falls outside them, negative when it is inside:

$ s_i = max(hat(q)_"lo"(bold(x)_i) - y_i, quad y_i - hat(q)_"hi"(bold(x)_i)), quad "interval" = [hat(q)_"lo"(bold(x)) - hat(q), quad hat(q)_"hi"(bold(x)) + hat(q)] . $

If the quantile models are too narrow, $hat(q)$ is positive and widens them; if they are too wide, it is negative and narrows them. The width of each interval comes from the quantile models, so it adapts to the dish, and the calibration step restores the guarantee.

=== Comparing intervals: the interval score

Coverage alone rewards absurdly wide intervals, and width alone rewards intervals that miss. The *interval score* (or Winkler score) combines them:

$ "IS"_alpha = (u - l) + 2 / alpha (l - y) bb(1)[y < l] + 2 / alpha (y - u) bb(1)[y > u] , $

the width plus a penalty proportional to how far the truth falls outside. Lower is better.

#figure(
  table(
    columns: (1fr, auto, auto, auto),
    align: (left, right, right, right),
    table.header[Variant][Coverage][Median width (kcal)][Interval score],
    ..(ADD, MUL, NORM, CQRL).map(v => (v, pct(av(v).coverage), num(av(v).width_median, d: 0), num(av(v).interval_score, d: 0))).flatten(),
  ),
  caption: [The four variants on test, $alpha = #num(m.alpha)$ (target 90%).],
) <tab:adaptive>

Both adaptive variants improve on the baseline intervals (@tab:adaptive, @fig:adaptive). Normalized conformal has the best interval score even though its difficulty model is weak (Spearman correlation #num(ad.difficulty.spearman_val, d: 2) between predicted and actual error). CQR gives the smallest median width and the most even coverage across predicted sizes. Its coverage on the official test split is low (#pct(av(CQRL).coverage)), but over #thousands(m.resampling.n_reps) re-splits its mean is #pct(ad.resampled.at(CQRL).mean): the guarantee holds, and the official split happens to be an unlucky draw for it.

#figure(
  image("figures/adaptive_by_predicted.svg", width: 100%),
  caption: [Coverage and mean width of the four variants per quartile of predicted kcal in test ($alpha = #num(m.alpha)$).],
) <fig:adaptive>

= Analysis tools <sec:analysis>

== marimo

#link("https://marimo.io/")[marimo] is a notebook in which cells form a *dataflow graph*. marimo reads which variables each cell defines and which ones it uses; when a cell changes, every cell that depends on it re-runs automatically. Three consequences:

- *No hidden state.* What is on screen always matches the code, unlike a notebook where cells were run out of order.
- *Plain Python files.* A notebook is a `.py` file: it diffs cleanly in git and runs as a script (`uv run python notebooks/baseline.py`).
- *Interactive widgets.* A slider for $alpha$ (`mo.ui.slider`) is just another variable: moving it re-runs the cells that compute the intervals and redraws the charts.

The price is a rule: each variable can be defined in only one cell. Names that start with an underscore are private to their cell and can be reused freely.

== Polars

#link("https://pola.rs/")[Polars] is a DataFrame library written in Rust. Its *expression* API describes a computation on columns, which Polars then runs in parallel:

```python
df.group_by("split").agg(
    n=pl.len(),
    kcal_median=pl.col("calories").median(),
)
```

== Altair and Vega-Lite

#link("https://altair-viz.github.io/")[Altair] is a Python interface to Vega-Lite @satyanarayan2017vegalite, a *grammar of graphics*. Instead of drawing a chart step by step, one declares a *mark* (bar, point, line) and *encodings* that map columns to visual channels (x, y, colour). Altair produces a JSON specification that Vega renders.

```python
alt.Chart(df).mark_circle().encode(
    x=alt.X("mass:Q", scale=alt.Scale(type="log")),
    y=alt.Y("calories:Q", scale=alt.Scale(type="log")),
)
```

Charts combine with operators: `a | b` places them side by side and `a + b` layers them. In marimo they render interactively in the browser; for these documents, the `vl-convert` package renders them to SVG without a browser.

= Documentation with Typst <sec:typst>

#link("https://typst.app/")[Typst] is a typesetting system, an alternative to LaTeX with a simpler syntax and near-instant compilation. Text uses lightweight markup (`= Heading`, `*bold*`, `$x^2$` for maths), and a real scripting language is available behind `#`.

These documents use that scripting to stay in sync with the code. `docs/make_figures.py` re-runs the baseline and writes the charts and a `metrics.json` file; the documents read it with `json(...)` and insert every number from there:

```typ
#let m = json("figures/metrics.json")
The test MAE is #num(m.mae.test) kcal.
```

Shared styling (fonts, headings, the cover, note boxes) lives in `template.typ`, which each document imports and applies with a `show` rule.

```bash
uv run python docs/make_figures.py   # refresh figures and numbers
typst compile docs/technologies.typ  # build the PDF
```

#pagebreak()
#bibliography("bibliography.bib", title: "References")
