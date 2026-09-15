# Entropy-Guided FIM for PlantUML Generation

Data, training, and evaluation pipeline for entropy-guided fill-in-the-middle (FIM)
code generation on PlantUML diagrams.

## Overview

Three stages, each independently runnable:

1. **Data pipeline** (`data_pipeline/`) - download PlantUML source from the-stack-v2,
   import a UML-in-the-wild corpus, join the two, and filter down to a strictly
   validated train/test split.
2. **Training** (`training/`) - fine-tune a causal LM with the FIM objective. The
   entropy-guided middle-span selection lives in `training/train.py`'s `FIMDataCollator`.
3. **Evaluation** (`evaluation/`) - generate completions on the test set, score them
   (BERTScore similarity) and render them (does the generated PlantUML actually
   render as a diagram?).

All three are orchestrated through `main.py`, driven by YAML configs in `config/`.

## Prerequisites

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- A JDK (any recent version). The PlantUML jar itself is bundled at
  `data_pipeline/java/plantuml-mit-1.2026.6.jar` - you only need to provide a JDK
  installation path.
- AWS credentials with read access to the Software Heritage S3 bucket, and a
  HuggingFace token (`bigcode/the-stack-v2` is a gated dataset) - both only needed
  for the data pipeline's Stack download step. The UML-in-the-wild corpus is fetched
  from a public archive and needs no credentials.
- A WANDB account, if you want training metrics logged (optional - see below).

## Setup

```bash
uv sync --extra train
cp .env.example .env
# fill in .env with your AWS/HF/WANDB credentials
```

`generate` and `score` (in the evaluation stage) need incompatible `transformers`
versions - `score` depends on `bert_score`, which pins an older one than the trained
checkpoint needs to load. They're kept as two mutually exclusive extras from one
lockfile: switch between them with `uv sync --extra train` / `uv sync --extra score`
as needed, rather than trying to install both at once.

## 1. Data pipeline

Before running, make sure to supply the path to the local jdk installation in config/data_pipeline.yml.

Run the data preparation process in one call:

```bash
uv run main.py --config config/data_pipeline.yml data-prepare
```

Fill in `config/data_pipeline.yml`'s `data_pipeline.java_home` first (required, no
default - point it at your JDK installation). The UML-in-the-wild corpus is
downloaded automatically from `puml_files_zip_url`/`uml_metadata_url` (also in that
config, defaulting to its published Zenodo archive - update those two fields if that
location ever changes).

Or run each stage on its own (useful for re-running just one stage, or for the exact
CLI flags each script takes - pass `--help` to any of them):

```bash
uv run python data_pipeline/download_stack.py --target-dir ./data/raw
uv run python data_pipeline/download_uml_in_the_wild.py
uv run python data_pipeline/import_puml_files_dataset.py --java-home /path/to/jdk
uv run python data_pipeline/join_datasets.py
uv run python data_pipeline/build_strict_dataset.py --java-home /path/to/jdk
```

(`download_uml_in_the_wild.py` also takes `--puml-files-zip-url`/`--metadata-url`, and
skips whichever half is already present on disk, so it's safe to re-run.)

This produces `./data/comparison_datasets/` with `test`, `train_strict`,
`test_labeled`, and `train_strict_labeled` splits (the `_labeled` variants add an
`our_diagram_type` column, used by evaluation's per-diagram-type breakdown and by
`diagram_type_oversampling` during training).

## 2. Training

Sanity-check the pipeline first (tiny model, tiny data slice, runs on CPU in ca. 10min):

```bash
uv run main.py --config config/train_smoke_test.yml train
```

`training/train.py` always calls `wandb.init()`, regardless of the `logging.log_wandb`
config flag - so without a `WANDB_API_KEY` set, either add one to `.env`, or set
`WANDB_MODE=disabled` in your environment before running.

Then fill in `config/train_main.yml` with your model/hyperparameter choices and run
the proper training:

```bash
uv run main.py --config config/train_main.yml train
```

`data.entropy: true` runs the entropy-guided method; `data.entropy: false` runs the
random-FIM baseline it's compared against - same config otherwise. Any field can be
overridden from the CLI, e.g. `--training.num_train_epochs 5`.

## 3. Evaluation

Fill in `config/evaluate.yml`'s `model_name` (path to a trained checkpoint) and
`java_home`, then:

```bash
# generate needs the train extra, to load the checkpoint
uv run main.py --config config/evaluate.yml generate

# score needs the score extra, for bert_score
uv sync --extra score
uv run main.py --config config/evaluate.yml score

# aggregate every model's results in outputs/ into one CSV
uv run python evaluation/summarize_scores.py --outputs-dir ./outputs/eval --csv-path ./outputs/eval/overview.csv
```

`score` runs two things: `similarity_scores.py` (BERTScore between generated and
reference code, over the full snippet and over just the FIM middle span) and
`render_scores.py` (does the generated PlantUML actually render, via the bundled
jar - a produced PNG isn't sufficient on its own, since PlantUML draws an error image
on a rendering failure too).

## Repository layout

```
config/            YAML configs for every stage
data_pipeline/      download -> import -> join -> filter, plus the shared
                     PlantUML classification/repair library
training/           FIMDataCollator (entropy-guided + random-baseline FIM), the
                     training loop, and optional diagram-type oversampling
evaluation/          generation, similarity scoring, rendering, and aggregation
scripts/             generic config-loading/CLI-override plumbing shared by main.py
main.py              CLI entry point: data-prepare | train | generate | score
```
