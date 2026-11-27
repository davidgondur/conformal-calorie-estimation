# Conformal Calorie Estimation

Calorie estimation from Nutrition5k overhead photos with conformal prediction intervals. Goal and stack: `README.md`. Why each choice was made: `docs/decisions.typ`. How each technology works: `docs/technologies.typ`.

## Language

Talk to the user, and write the README and notebook prose, in Spanish. Everything in `docs/`, code, docstrings, comments, tests and commit messages are in English.

## Commands

- `uv run pytest -q`: tests use synthetic fixtures, no network or real data needed.
- `uv run n5k-download`: fetches the overhead-RGB subset (~1.4 GB) into `data/nutrition5k/`.
- `uv run python docs/make_figures.py`, then `typst compile docs/decisions.typ` and `docs/technologies.typ`: the documents read every number from `docs/figures/metrics.json`. Regenerate it whenever the baseline changes; never type results into a `.typ` by hand.
- When a decision is taken or reversed, add or update its entry in `docs/decisions.typ`. When a technology enters the code, explain it in `docs/technologies.typ`; that document only covers what the code actually uses.

Python is pinned to 3.13 because torch and onnxruntime have no 3.15 wheels.

## Splits: the coverage guarantee depends on these

- Only the official splits (`depth_{train,test}_ids.txt`) live on disk. Every other split (`val`, `calib`, ...) is carved from `train` inside the notebook or script that uses it, with `split_by_groups` and an explicit seed.
- `split_by_groups` keeps incremental-scan groups whole. Nutrition5k plates are built one ingredient at a time and each step is its own dish, so a random split leaks near-duplicates across partitions.
- `calib` never influences the model: no early stopping, hyperparameter or model choice on it. That is what `val` is for.
- `test` is never subdivided and is only used for final reporting.
- `clean_train` applies only to the training part, after splitting. `val`, `calib` and `test` keep their wrong labels so calibration still looks like test.

## Data gotchas

- The official README lists a `num_ingrs` column that does not exist: rows are 6 dish fields + 7 per ingredient, no header. `parse_dish_row` validates this.
- Test contains a 0 kcal dish: model calories as `log1p`, not `log`.
- The GCS bucket is public over HTTPS; `gsutil` is not needed.

## Notebooks (marimo)

- The user usually has a notebook open in `marimo edit`. Check with `uv run marimo pair list`; if it is open, edit it through `marimo pair execute`, because direct file edits get overwritten by the live session.
- Render composed Altair charts (`a | b`, `a + b`) as plain charts. `mo.ui.altair_chart` adds the same legend selection to every panel and Vega fails with "Duplicate signal name".
- Polars and Altair are the dataframe and plotting libraries (no pandas, no matplotlib).
