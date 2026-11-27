#import "template.typ": *

#show: project.with(title: [Decision log], subtitle: [What was decided, and why])

// One decision: what was chosen, what else was considered, why, and the evidence.
#let decision(id, title, chosen: none, alternatives: none, why: none, evidence: none, consequences: none) = {
  heading(level: 2)[#id #h(4pt) #title]
  let row(label, body) = if body != none { (text(weight: 600, fill: color-palette.secondary, label), body) } else { () }
  grid(
    columns: (2.6cm, 1fr),
    column-gutter: 10pt,
    row-gutter: 10pt,
    ..row([Decision], chosen),
    ..row([Alternatives], alternatives),
    ..row([Why], why),
    ..row([Evidence], evidence),
    ..row([Consequences], consequences),
  )
}

= Overview <sec:overview>

This document records every decision taken so far in the project, together with the alternatives that were considered and the reasons behind the choice. Each entry follows the same structure: the *decision*, the *alternatives*, *why* it was chosen, the *evidence* when there is any, and its *consequences* for later work.

All numbers come from `docs/figures/metrics.json`, which `uv run python docs/make_figures.py` regenerates from the data (seed 0, $alpha = #num(m.alpha)$). The technologies mentioned here are explained in the companion document, _Technologies used_.

#figure(
  table(
    columns: (auto, 1fr, auto),
    align: (left, left, left),
    table.header[ID][Decision][Section],
    [D1], [`uv` package with a `src/` layout, Python 3.13], [@sec:setup],
    [D2], [Overhead RGB photos only], [@sec:data],
    [D3], [Strict parsing of the dish metadata], [@sec:data],
    [D4], [Only the official splits are stored on disk], [@sec:splits],
    [D5], [Separate `val` and `calib` partitions], [@sec:splits],
    [D6], [Split by incremental-scan groups], [@sec:splits],
    [D7], [Clean wrong labels, in the training part only], [@sec:splits],
    [D8], [Predict $log(1 + "kcal")$], [@sec:model],
    [D9], [Frozen-backbone baseline before any fine-tuning], [@sec:model],
    [D10], [DINOv2-small rather than DINOv3-small], [@sec:model],
    [D11], [Own implementation of split conformal], [@sec:conformal],
    [D12], [Multiplicative (log-scale) conformal intervals], [@sec:conformal],
    [D13], [marimo, Polars and Altair for analysis], [@sec:tooling],
    [D14], [Typst documents with generated numbers], [@sec:tooling],
  ),
  caption: [Summary of decisions.],
)

= Project setup <sec:setup>

#decision("D1", [`uv` package with a `src/` layout, Python 3.13],
  chosen: [Manage the project with `uv` as an installable package (`src/conformal_calorie_estimation/`), with Python pinned to 3.13 and development tools in a separate dependency group.],
  alternatives: [A flat script folder; Python 3.15, which `uv` proposed by default.],
  why: [A package lets notebooks, scripts and tests import the same code instead of copying it, and gives the download script a command (`n5k-download`). Python 3.15 was too new: PyTorch and ONNX Runtime do not publish wheels for it yet, so the core dependencies would not install.],
  consequences: [Notebook and analysis tools (marimo, Polars, Altair, pytest) live in the `dev` group and are not shipped with the package.],
)

= Data <sec:data>

#decision("D2", [Overhead RGB photos only],
  chosen: [Use only the overhead RGB image of each dish (#thousands(m.n_official.train + m.n_official.test) dishes with an image, 1.4 GB, 640 × 480). Download it over HTTPS from the public bucket.],
  alternatives: [The side-angle videos, as the Nutrition5k paper does @thames2021nutrition5k; adding the depth map as a fourth channel or as a volume estimate.],
  why: [
    - The final demo receives *one photo from a phone*, which looks much more like the overhead image than like a frame of a rotating video.
    - The videos make up almost all of the 181 GB archive.
    - A phone has no depth sensor, so a model that needs depth could not run in the demo. Depth is still downloadable (`--depth`) for analysis, for example to check whether interval width grows with food volume.
    - The bucket is public over HTTPS, so `gsutil` and a Google Cloud account are unnecessary.
  ],
  consequences: [Only about 3,500 of the 5,000 dishes are usable. Results are not directly comparable with the paper, which trains on video frames.],
)

#decision("D3", [Strict parsing of the dish metadata],
  chosen: [Parse each CSV row as 6 dish fields followed by blocks of 7 fields per ingredient, and raise an error when the field count does not fit.],
  alternatives: [Follow the official README, which describes a `num_ingrs` column.],
  why: [The `num_ingrs` column *does not exist* in the files. A parser written from the README would shift every column by one without raising any error, silently corrupting all labels.],
  evidence: [Checked against the real `dish_metadata_cafe1.csv`; a test uses a real row.],
)

= Splits and labels <sec:splits>

#note(title: [Why splits matter so much here])[
  The conformal guarantee holds only if the calibration dishes and the test dishes are *exchangeable* and the model has not been influenced by either. Every decision in this section protects that condition.
]

#decision("D4", [Only the official splits are stored on disk],
  chosen: [Keep the official `depth_train` / `depth_test` lists (#thousands(m.n_official.train) and #m.n_official.test dishes after dropping 3 without a photo) as the only stored splits. Every other partition is derived from `train` inside the notebook or script that uses it, with `split_by_groups` and an explicit seed.],
  alternatives: [Saving derived partitions (`val`, `calib`) as files; making a single `make_splits` function decide them for every experiment.],
  why: [The proportions belong to the experiment: a fine-tuning run and a calibration study may want different `val` and `calib` sizes. Defining them next to the code that uses them, with a seed, keeps each experiment readable and reproducible without a growing set of split files. The official lists stay untouched, so results remain comparable with other work.],
  consequences: [`test` is never subdivided and is only used for final reporting.],
)

#decision("D5", [Separate `val` and `calib` partitions],
  chosen: [Carve both a validation set (10%) and a calibration set (20%) out of `train`.],
  alternatives: [No validation set, with fixed hyperparameters; cross-validation inside `train`; using `calib` for model selection as well.],
  why: [
    - The guarantee needs `calib` to have no influence on the model, *including* early stopping and hyperparameter choices. Otherwise its errors look smaller than they really are and the intervals come out too narrow.
    - Without `val` there would be no honest way to stop training or compare backbones.
    - Cross-validation multiplies training cost by $k$ and complicates the code for little gain in a portfolio project.
    - `calib` does not need to be large: the realised coverage fluctuates by about $sqrt(alpha(1 - alpha) \/ n)$, which is ±#pct(m.cov_sd_calib) for $n = #m.n.calib$.
  ],
  evidence: [Resulting sizes (seed 0): `train` #thousands(m.n.train), `val` #m.n.val, `calib` #m.n.calib, `test` #m.n.test.],
)

#decision("D6", [Split by incremental-scan groups],
  chosen: [Assign whole groups of related dishes to a single partition. Two consecutive dishes belong to the same group when they are less than 5 minutes apart and the ingredients of the first are a subset of those of the second.],
  alternatives: [A random split by dish; grouping by time alone.],
  why: [Nutrition5k plates are built one ingredient at a time and each step is recorded as a separate dish. With a random split, a dish and the same dish plus one ingredient can land in `train` and `calib`. The model then looks better on calibration than on truly new dishes, and the intervals come out too narrow. The official grouping is not published, so a heuristic is needed.],
  evidence: [
    Neither heuristic reproduces the official grouping. @tab:grouping counts how many groups contain dishes from both official train and official test (the official split is assumed to respect the true groups).

    #figure(
      table(
        columns: (1fr, auto, auto, auto),
        align: (left, right, right, right),
        table.header[Rule][Groups][Mixed train/test][Largest group],
        [Subset of ingredients, < 5 min (chosen)], [#thousands(m.grouping.subset_rule.groups)], [#m.grouping.subset_rule.mixed], [#m.grouping.subset_rule.largest],
        ..("60", "300", "1800").map(g => {
          let r = m.grouping.time_only.at(g)
          ([Time only, < #g s], thousands(r.groups), str(r.mixed), str(r.largest))
        }).flatten(),
      ),
      caption: [Grouping heuristics compared against the official rgb train/test split.],
    ) <tab:grouping>

    A 60-second window mixes slightly fewer groups than the chosen rule, but it cannot link two steps of the same plate taken more than a minute apart. Wider windows merge unrelated plates into groups of over 100 dishes. The subset rule follows the actual structure of an incremental scan and keeps groups small, so it was chosen even though the evidence does not favour it clearly.
  ],
  consequences: [Some leakage between partitions probably remains. Revisit if a better grouping signal appears.],
)

#decision("D7", [Clean wrong labels, in the training part only],
  chosen: [`clean_train` removes dishes labelled _"plate only"_ or with 0 kcal, and dishes heavier than 2 kg. It is applied to the training part only, after splitting.],
  alternatives: [Keeping every dish; cleaning every partition; removing very low-calorie dishes as well.],
  why: [
    - Both rules target labels that are *demonstrably wrong* when looking at the photo (@fig:errors). Fitting on them only adds noise.
    - `calib` has to look like `test`. If calibration were cleaner than test, the intervals would be slightly too narrow.
    - Removing hard or wrong cases from `test` would make the reported coverage optimistic.
    - Very low calories are real: a few grams of spinach or cucumber. They stay.
  ],
  evidence: [With seed 0 a single training dish is removed (#raw(m.removed.join(", "))). The _"plate only"_ dish of the official train set falls into `calib` and stays there, matching the one present in `test`.],
)

#figure(
  block(width: 60%, grid(
    columns: 2,
    column-gutter: 12pt,
    image("figures/error_rice.jpg", width: 100%),
    image("figures/error_plate.jpg", width: 100%),
    align(center, text(size: 9pt)[`dish_1560974769`: 3,943 kcal, 3,051 g \ (2,991 g of rice)]),
    align(center, text(size: 9pt)[`dish_1556575700`: _"plate only"_, 0 kcal]),
  )),
  caption: [Two wrong labels. Left: a small portion of rice recorded as almost 3 kg. Right: a plate labelled as empty that holds cherry tomatoes.],
) <fig:errors>

= Modelling <sec:model>

#decision("D8", [Predict $log(1 + "kcal")$],
  chosen: [Fit the regressor on $log(1 + "kcal")$ and transform predictions back with $exp(dot) - 1$.],
  alternatives: [Predict kcal directly; use $log("kcal")$.],
  why: [
    - Calories are very skewed: the middle half of the dishes lies between #num(m.eda.kcal_q25, d: 0) and #num(m.eda.kcal_q75, d: 0) kcal, with a long tail towards 1,000 kcal. On a log scale the distribution is much more regular.
    - Errors on a log scale are *relative*, which matches how the problem is perceived: 50 kcal is a big mistake for a salad and a small one for a stew.
    - `test` contains a 0 kcal dish, so $log$ is undefined; $log(1 + dot)$ is not.
  ],
  evidence: [With the same embeddings and the regularization tuned on `val` for each, the best ridge on the log target reaches a validation MAE of #num(m.val_mae_log_target) kcal, against #num(m.val_mae_raw_target) kcal when fitted on raw kcal.],
)

#decision("D9", [Frozen-backbone baseline before any fine-tuning],
  chosen: [Before fine-tuning, build the whole pipeline with frozen DINOv2 embeddings, a ridge regressor and split conformal.],
  alternatives: [Start directly with fine-tuning a Vision Transformer.],
  why: [
    - The embeddings are computed once (seconds on the GPU), so the whole pipeline runs in seconds and can be iterated on quickly.
    - It tests the part that matters most in this project, the conformal layer, without waiting for training runs.
    - It gives a reference: a fine-tuned model that does not beat it signals a bug.
  ],
  evidence: [
    The baseline reaches a test MAE of #num(m.mae.test) kcal (#pct(m.mae_pct.test) of the mean), against #num(m.mae_naive_test) kcal when always predicting the training median. The paper's RGB-only model, fine-tuned on video frames and on a different split, reports 70.6 kcal (26.1%).
  ],
)

#decision("D10", [DINOv2-small rather than DINOv3-small],
  chosen: [Use `vit_small_patch14_dinov2.lvd142m` @oquab2024dinov2 as the backbone, at 224 × 224.],
  alternatives: [DINOv3-small @simeoni2025dinov3 (`vit_small_patch16_dinov3.lvd1689m`), at 224 and at its native 256 pixels; DeiT-tiny.],
  why: [
    - DINOv2 was the original plan and DINOv3 does not improve on it for this baseline (@tab:backbones). On `val`, which is where the choice must be made, they are tied.
    - DINOv2 weights are released under Apache 2.0. DINOv3 has its own Meta license, which would have to be reviewed before publishing the model in a public web demo.
    - In DINOv3's favour: with 16-pixel patches a 224-pixel image is 196 tokens instead of 256, so browser inference would be somewhat faster.
  ],
  evidence: [
    #figure(
      table(
        columns: (1fr, auto, auto, auto, auto),
        align: (left, right, right, right, right),
        table.header[Backbone][Val MAE][Test MAE][Coverage][Mean (median) width],
        ..m.backbones.map(b => (
          b.label, num(b.val_mae), num(b.test_mae), pct(b.coverage),
          num(b.width_mean, d: 0) + " (" + num(b.width_median, d: 0) + ") kcal",
        )).flatten(),
      ),
      caption: [Frozen embeddings + ridge + multiplicative split conformal, $alpha = #num(m.alpha)$. Test columns are shown for information only; the choice is made on `val`.],
    ) <tab:backbones>
  ],
  consequences: [The comparison uses one seed and frozen features. It must be repeated after fine-tuning, where the ranking could change.],
)

= Uncertainty <sec:conformal>

#decision("D11", [Own implementation of split conformal],
  chosen: [Implement the conformal quantile, the interval and the coverage in `conformal.py` (about 40 lines, with tests).],
  alternatives: [MAPIE, which is already a dependency.],
  why: [The formulas are the point of the project. Writing them out keeps the finite-sample correction and the scale of the residuals explicit and easy to explain. A simulation test checks that coverage holds on exchangeable data. MAPIE stays available to cross-check results and for the methods that come next.],
)

#decision("D12", [Multiplicative (log-scale) conformal intervals],
  chosen: [Compute the conformal residuals on the $log(1 + "kcal")$ scale. After transforming back, the interval is proportional to the size of the predicted dish.],
  alternatives: [Residuals in kcal, which give the same additive margin to every dish.],
  why: [Both variants meet the target coverage *on average*, but they distribute it very differently across dishes (@fig:size). With a fixed margin, small dishes get useless intervals and large dishes fall well short of the promised coverage. The multiplicative interval is much more balanced, at the cost of wider intervals for large dishes.],
  evidence: [
    #figure(
      table(
        columns: (auto, auto, auto, auto, auto),
        align: (left, right, right, right, right),
        table.header(
          table.cell(rowspan: 2)[True kcal],
          table.cell(colspan: 2, align: center)[Multiplicative (log)],
          table.cell(colspan: 2, align: center)[Additive (kcal)],
          [Coverage], [Width], [Coverage], [Width],
        ),
        ..range(4).map(i => {
          let a = size-row(MUL, i)
          let b = size-row(ADD, i)
          (a.quartile, pct(a.coverage, d: 0), num(a.width, d: 0), pct(b.coverage, d: 0), num(b.width, d: 0))
        }).flatten(),
        table.cell(colspan: 5)[],
        [All of test], pct(m.coverage.at(MUL)), num(m.width_mean.at(MUL), d: 0), pct(m.coverage.at(ADD)), num(m.width_mean.at(ADD), d: 0),
      ),
      caption: [Coverage and mean width (kcal) per quartile of true calories in test, $alpha = #num(m.alpha)$ (target 90%).],
    )
  ],
  consequences: [Very small and very large dishes are still below target (#pct(size-row(MUL, 0).coverage, d: 0) and #pct(size-row(MUL, 3).coverage, d: 0)). Normalized conformal and conformalized quantile regression @romano2019conformalized are the next step to adapt the width to the difficulty of each dish, not only to its size.],
)

#figure(
  image("figures/coverage_by_size.svg", width: 100%),
  caption: [Coverage and mean interval width per quartile of true calories in test. The horizontal line marks the 90% target.],
) <fig:size>

= Tooling and documentation <sec:tooling>

#decision("D13", [marimo, Polars and Altair for analysis],
  chosen: [Write notebooks with marimo, manipulate tables with Polars and plot with Altair. Composite charts are rendered as plain Altair charts, not wrapped in `mo.ui.altair_chart`.],
  alternatives: [Jupyter, pandas and matplotlib.],
  why: [
    - marimo notebooks are plain Python files, so they diff cleanly in git, run as scripts and cannot hold hidden state: changing a cell re-runs everything that depends on it.
    - Polars and Altair have first-class support in marimo.
    - `mo.ui.altair_chart` adds the same legend selection to every panel of a composite chart, and Vega rejects it as a "Duplicate signal name". None of these charts needs selection, so plain charts are used.
  ],
)

#decision("D14", [Typst documents with generated numbers],
  chosen: [Write the documentation in Typst. Every number and figure is produced by `docs/make_figures.py` and read from `metrics.json`.],
  alternatives: [LaTeX; Markdown; typing results by hand.],
  why: [Typst compiles in well under a second and its scripting can read JSON directly. Generating the numbers means the documents cannot drift from the code: after any change, one command refreshes every table and figure.],
  consequences: [Run `uv run python docs/make_figures.py` before compiling whenever the baseline changes.],
)

= Open questions <sec:open>

+ *Grouping.* Is there a better signal than ingredients and timestamps for the official incremental groups?
+ *Adaptive intervals.* Do normalized conformal and CQR fix the under-coverage of very small and very large dishes?
+ *Fine-tuning.* Does a fine-tuned ViT beat the frozen baseline, and does DINOv3 overtake DINOv2 once fine-tuned?
+ *Quantization.* Does exporting the model to int8 ONNX break coverage, or only widen the intervals?

#pagebreak()
#bibliography("bibliography.bib", title: "References")
