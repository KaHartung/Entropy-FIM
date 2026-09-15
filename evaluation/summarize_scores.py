"""
Summarize per-model evaluation results in outputs/ into a single overview CSV.

Scans outputs/<model_name>/ for the *_scored.json (written by similarity_scores.py)
and *_rendered.json (written by render_scores.py) produced for each model. Instances
that differ only by training seed and/or eval seed (see parse_model_name) are treated
as repeats of the same setting and collapsed into one row reporting the mean, standard
deviation, and instance count (n) for each score.

With --test-labeled pointing at a labeled dataset (e.g. data_pipeline's
build_strict_dataset.py output, test_labeled), also writes a second CSV breaking the
same scores down by diagram type, matching each model's per-instance scores to that
dataset's 'our_diagram_type' column by position (see load_diagram_types).
"""

import argparse
import csv
import json
import logging
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Optional


# Fields that identify a distinct experimental setting. Rows that agree on all of these
# are aggregated together; "seed" (training seed) and "eval_seed" (evaluate.seed in
# config/evaluate.yml) are deliberately excluded so repeats over either kind of seed
# get averaged.
GROUP_FIELDS = [
    "method", "dataset_version", "learning_rate", "fim_rate", "epochs", "filtered pct", "run_tag", "temperature",
]

SCORE_FIELDS = ["rendered", "code-score", "mid-code-score"]

FIELDNAMES = GROUP_FIELDS + ["n"] + [
    f"{field}_{stat}" for field in SCORE_FIELDS for stat in ("mean", "std")
] + ["model names"]

FIELDNAMES_BY_TYPE = GROUP_FIELDS + ["diagram_type", "n"] + [
    f"{field}_{stat}" for field in SCORE_FIELDS for stat in ("mean", "std")
] + ["model names"]

# training.seed's default before run_name started encoding it (see training/train.py) -
# used to fill the seed column for model names that predate that tag.
DEFAULT_SEED = "10"

# <method>_train_<dataset_version>[_pct<filtered_pct>]_lr<learning_rate>_fim<fim_rate>[_ep<epochs>][_s<seed>][_<run_tag>]_fim_plantuml[_temp<temperature>]_evs<eval_seed>
# temperature/eval_seed come from evaluate.do_sample+temperature / evaluate.seed in
# config/evaluate.yml (see evaluation/generate.py), appended by generate/similarity_scores/
# render_scores after the fixed "_fim_plantuml" suffix so that multiple eval variations
# (sampling settings, eval seeds, ...) of the same checkpoint don't collide on disk.
MODEL_NAME_RX = re.compile(
    r"^(?P<method>baseline|entropy)_train_(?P<dataset_version>.+?)"
    r"(?:_pct(?P<filtered_pct>[^_]+))?"
    r"_lr(?P<learning_rate>[^_]+)_fim(?P<fim_rate>[^_]+)"
    r"(?:_ep(?P<epochs>\d+))?(?:_s(?P<seed>\d+))?(?:_(?P<run_tag>.+?))?_fim_plantuml"
    r"(?:_temp(?P<temperature>[^_]+))?(?:_evs(?P<eval_seed>\d+))?$"
)


def parse_model_name(model_name: str) -> dict:
    """
    Extract method/dataset_version/learning_rate/fim_rate/epochs/seed/filtered_pct/run_tag/
    temperature/eval_seed from a model name following <method>_train_<dataset_version>
    [_pct<pct>]_lr<lr>_fim<fim_rate>[_ep<epochs>][_s<seed>][_<run_tag>]_fim_plantuml
    [_temp<temperature>]_evs<eval_seed>. Returns empty values for names that don't follow the
    pattern, except seed, which defaults to DEFAULT_SEED whenever it's not present (older runs
    predate run_name encoding the training seed at all), and temperature, which defaults to
    "greedy" whenever the name has no "_temp<...>" segment (do_sample was False for that run).
    """
    match = MODEL_NAME_RX.match(model_name)
    if not match:
        return {"method": None, "dataset_version": None, "learning_rate": None,
                "fim_rate": None, "epochs": None, "seed": DEFAULT_SEED, "filtered pct": None, "run_tag": None,
                "temperature": None, "eval_seed": None}
    groups = match.groupdict()
    return {
        "method": groups["method"],
        "dataset_version": groups["dataset_version"],
        "learning_rate": groups["learning_rate"],
        "fim_rate": groups["fim_rate"],
        "epochs": groups["epochs"],
        "seed": groups["seed"] or DEFAULT_SEED,
        "filtered pct": groups["filtered_pct"],
        "run_tag": groups["run_tag"],
        "temperature": groups["temperature"] or "greedy",
        "eval_seed": groups["eval_seed"],
    }


def setup_logger():
    """Setup logging."""
    logger = logging.getLogger("summarize_scores")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


# Per-instance fields collected for each model dir, before grouping/aggregation.
RAW_FIELDS = GROUP_FIELDS + ["seed", "eval_seed", "model name"] + SCORE_FIELDS


def mean(values: list) -> Optional[float]:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def stdev(values: list) -> Optional[float]:
    """Sample standard deviation; undefined (None) for fewer than two data points."""
    values = [v for v in values if v is not None]
    return statistics.stdev(values) if len(values) > 1 else None


def load_model_row(model_dir: Path, logger) -> Optional[dict]:
    model_name = model_dir.name
    scored_path = model_dir / f"{model_name}_scored.json"
    rendered_path = model_dir / f"{model_name}_rendered.json"

    if not scored_path.exists() and not rendered_path.exists():
        return None

    row = {field: None for field in RAW_FIELDS}
    row["model name"] = model_name
    row.update(parse_model_name(model_name))

    if rendered_path.exists():
        with open(rendered_path) as f:
            passed = json.load(f)
        row["rendered"] = mean(passed)
    else:
        logger.warning(f" > {model_name}: no rendered.json found")

    if scored_path.exists():
        with open(scored_path) as f:
            scored = json.load(f)
        row["code-score"] = mean(scored.get("code-scores"))
        row["mid-code-score"] = mean(scored.get("mid-code-scores"))
    else:
        logger.warning(f" > {model_name}: no scored.json found")

    for field in SCORE_FIELDS:
        if row[field] is not None:
            row[field] = round(row[field], 4)

    return row


def group_rows(rows: list, group_fields: list = GROUP_FIELDS) -> list:
    """Collapse instances that agree on every group_fields value into one row of
    mean/std/n per score, ignoring seed and eval_seed differences between them."""
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups[key].append(row)

    grouped_rows = []
    for key, group in groups.items():
        grouped = dict(zip(group_fields, key))
        grouped["n"] = len(group)
        grouped["model names"] = "; ".join(sorted(r["model name"] for r in group))
        for field in SCORE_FIELDS:
            values = [r[field] for r in group]
            field_mean = mean(values)
            field_std = stdev(values)
            grouped[f"{field}_mean"] = round(field_mean, 4) if field_mean is not None else None
            grouped[f"{field}_std"] = round(field_std, 4) if field_std is not None else None
        grouped_rows.append(grouped)

    grouped_rows.sort(key=lambda r: tuple((r[f] is None, r[f]) for f in group_fields))
    return grouped_rows


def load_diagram_types(test_labeled_path: str, logger) -> list:
    """Load the `our_diagram_type` column of a test_labeled dataset (see
    data_pipeline/build_strict_dataset.py), in row order. Model outputs are generated
    by iterating the same underlying test set in order (see evaluation/generate.py),
    so position i in a model's *_scored.json/*_rendered.json lines up with position i
    here."""
    try:
        from datasets import load_from_disk
    except ImportError:
        raise SystemExit("--test-labeled requires the 'datasets' package (pip install datasets)")
    dataset = load_from_disk(test_labeled_path)
    if "our_diagram_type" not in dataset.column_names:
        raise SystemExit(f"{test_labeled_path} has no 'our_diagram_type' column")
    logger.info(f"Loaded {len(dataset)} diagram type labels from {test_labeled_path}")
    return list(dataset["our_diagram_type"])


def load_model_type_rows(model_dir: Path, diagram_types: list, logger) -> list:
    """Per-(diagram_type) breakdown of one model dir's per-instance scores, matched
    to diagram_types by dataset row (see load_diagram_types). Returns one row per
    diagram type present among this model's instances, shaped like load_model_row's
    output so it can be grouped with group_rows()."""
    model_name = model_dir.name
    scored_path = model_dir / f"{model_name}_scored.json"
    rendered_path = model_dir / f"{model_name}_rendered.json"

    raw = {field: None for field in SCORE_FIELDS}
    indices = None
    if scored_path.exists():
        with open(scored_path) as f:
            scored = json.load(f)
        raw["code-score"] = scored.get("code-scores")
        raw["mid-code-score"] = scored.get("mid-code-scores")
        indices = scored.get("indices")
    if rendered_path.exists():
        with open(rendered_path) as f:
            raw["rendered"] = json.load(f)

    lengths = {field: len(values) for field, values in raw.items() if values is not None}
    if not lengths:
        return []

    n_instances = next(iter(lengths.values()))
    if any(length != n_instances for length in lengths.values()):
        logger.warning(f" > {model_name}: scored/rendered instance counts disagree {lengths} "
                        f"-- skipping diagram-type breakdown for this model")
        return []

    if indices is not None and len(indices) != n_instances:
        logger.warning(f" > {model_name}: 'indices' length ({len(indices)}) doesn't match "
                        f"instance count ({n_instances}) -- ignoring it")
        indices = None

    if indices is None:
        # evaluation/generate.py silently drops any dataset row whose FIM-transformed
        # length hit the tokenizer's max length, without recording which one -- so for
        # outputs missing the 'indices' field, position i only equals dataset row i if
        # no row was dropped before i. A row dropped anywhere but the very end shifts
        # every diagram type assigned after it, not just the tail.
        indices = list(range(n_instances))
        if n_instances != len(diagram_types):
            logger.warning(f" > {model_name}: {n_instances} scored instances vs {len(diagram_types)} "
                            f"diagram-type labels, and this output has no 'indices' field, so exact "
                            f"alignment can't be recovered -- assuming instances 0..{n_instances - 1} "
                            f"map to dataset rows 0..{n_instances - 1} in order. If generation actually "
                            f"skipped a row before the end (likely), diagram types after it will be wrong.")

    base_fields = parse_model_name(model_name)
    by_type = defaultdict(lambda: {field: [] for field in SCORE_FIELDS})
    for i, dataset_index in enumerate(indices):
        if dataset_index >= len(diagram_types):
            continue
        dtype = diagram_types[dataset_index]
        for field, values in raw.items():
            if values is not None:
                by_type[dtype][field].append(values[i])

    rows = []
    for dtype, field_values in by_type.items():
        row = {**base_fields, "diagram_type": dtype, "model name": model_name}
        for field in SCORE_FIELDS:
            field_mean = mean(field_values[field])
            row[field] = round(field_mean, 4) if field_mean is not None else None
        rows.append(row)
    return rows


def main(outputs_dir: str = "./outputs", csv_path: str = "./outputs/overview.csv",
         test_labeled: Optional[str] = None, csv_path_by_type: str = "./outputs/overview_by_type.csv") -> None:
    logger = setup_logger()
    outputs_base = Path(outputs_dir)

    diagram_types = load_diagram_types(test_labeled, logger) if test_labeled else None

    rows = []
    type_rows = []
    for model_dir in sorted(outputs_base.iterdir()):
        if not model_dir.is_dir():
            continue
        row = load_model_row(model_dir, logger)
        if row is not None:
            rows.append(row)
        if diagram_types is not None:
            type_rows.extend(load_model_type_rows(model_dir, diagram_types, logger))

    if not rows:
        logger.warning(f"No scored/rendered results found under {outputs_base}")
        return

    grouped_rows = group_rows(rows)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(grouped_rows)

    logger.info(f"Wrote overview for {len(grouped_rows)} groups ({len(rows)} instances) to {csv_path}")

    if diagram_types is not None:
        grouped_type_rows = group_rows(type_rows, GROUP_FIELDS + ["diagram_type"])

        with open(csv_path_by_type, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES_BY_TYPE)
            writer.writeheader()
            writer.writerows(grouped_type_rows)

        logger.info(f"Wrote per-diagram-type overview for {len(grouped_type_rows)} groups "
                    f"({len(type_rows)} instances) to {csv_path_by_type}")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize per-model eval scores into an overview CSV.")
    parser.add_argument("--outputs-dir", default="./outputs", help="Directory containing per-model output folders")
    parser.add_argument("--csv-path", default="./outputs/overview.csv", help="Where to write the overview CSV")
    parser.add_argument("--test-labeled", default=None,
                         help="Path to a test_labeled dataset (e.g. data/comparison_datasets/test_labeled) with "
                              "an 'our_diagram_type' column; when given, also writes a per-diagram-type breakdown. "
                              "Diagram types are matched to each model's scored instances by position, so this "
                              "only makes sense if all evaluated models were generated over that same test set.")
    parser.add_argument("--csv-path-by-type", default="./outputs/overview_by_type.csv",
                         help="Where to write the per-diagram-type overview CSV (only used with --test-labeled)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(outputs_dir=args.outputs_dir, csv_path=args.csv_path,
         test_labeled=args.test_labeled, csv_path_by_type=args.csv_path_by_type)
