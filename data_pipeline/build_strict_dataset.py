"""
Build the strictly-filtered test + train_strict dataset splits: cheap semantic filter
first, real jar classification only on survivors.

"Strict" means extract_insights_jar classifies the row into one of REAL_DIAGRAM_TYPES
(real PlantUML-engine ground truth) -- a bare "does this parse" check lets through a
lot that isn't really a UML diagram: entity-only ER diagrams and title/skinparam-only
stubs are syntactically valid CLASS-grammar content but have zero real declarations;
alternate diagram families (git/board/wire/hcl/ebnf/chart/bpm/...) parse fine but
aren't UML; a block with non-trivial content left dangling outside it, or more than
one @start.../@end... block, both parse fine block-by-block. Only extract_insights_jar's
full resolution logic catches all of this.

semantic_pass (the cheap pre-filter) is kept because it does more than
extract_insights_jar's own non-uml gate (also checks for Latin-only content) and is
cheap enough to run first and shrink what needs the (much slower) jar classification.

Also produces test_labeled/train_strict_labeled with an our_diagram_type column (a
plain copy of diagram_type, no extra jar work) -- some downstream evaluation tooling
keys off that column name for a per-diagram-type breakdown.
"""

import argparse
import functools
import logging
import os
import shutil
from collections import Counter
from pathlib import Path

from datasets import load_from_disk, DatasetDict

from filter_data import passes_semantic_filter
from validate_plantuml import (
    REAL_DIAGRAM_TYPES, dedup_by_code, ensure_diagram_type_detector_compiled,
    extract_insights_jar, repair_row, split_multi_block_rows,
)

DEFAULT_JAR_PATH = str(Path(__file__).resolve().parent / "java" / "plantuml-mit-1.2026.6.jar")


def setup_logger():
    """Setup logging."""
    logger = logging.getLogger("build_strict_dataset")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def add_semantic_flag(example):
    semantic_pass, semantic_reason = passes_semantic_filter(example["code"])
    return {"semantic_pass": semantic_pass, "semantic_reason": semantic_reason or ""}


def add_diagram_type(example, jar_path: str, java_home: str, jar_timeout: int):
    insights = extract_insights_jar(example["code"], jar_path, java_home=java_home, timeout=jar_timeout)
    return {"diagram_type": insights["diagram_type"]}


def main(java_home: str, target_dir: str, jar_path: str = DEFAULT_JAR_PATH,
         joined_dataset_dir: str = "./data/puml_joined",
         test_size: int = 500, seed: int = 42, num_proc: int = 4, jar_num_proc: int = 4,
         jar_timeout: int = 30) -> None:
    """
    Build the test + train_strict splits, skipping the jar-based classification for
    any row the cheap semantic check already rejects.

    Args:
        java_home: path to a JDK installation; its bin/java (and bin/javac) runs
            (and compiles) the jar
        target_dir: directory to save the resulting DatasetDict (test + train_strict)
        jar_path: path to a local plantuml.jar (used for classification)
        joined_dataset_dir: path to the joined dataset (join_datasets.py's output)
        test_size: number of samples to hold out for the test split
        seed: random seed for test-set sampling
        num_proc: worker count for cheap (regex-based) filtering steps
        jar_num_proc: worker count for the jar-based classification step (each worker
            keeps one persistent JVM alive for its whole share of the data)
        jar_timeout: seconds to allow the jar to run per sample
    """
    logger = setup_logger()
    os.makedirs(target_dir, exist_ok=True)

    # Compile once here, before ds.map spawns jar_num_proc worker processes, to avoid
    # concurrent javac invocations racing on the same output file.
    ensure_diagram_type_detector_compiled(jar_path, java_home=java_home)

    # ---------------------- LOAD + CLEAN ----------------------

    logger.info(f" > loading joined dataset from {joined_dataset_dir}")
    ds = load_from_disk(joined_dataset_dir)
    logger.info(f" > total rows: {len(ds)}")

    logger.info(" > splitting multi-block rows into one row per @start.../@end... block")
    before_split = len(ds)
    ds = ds.map(split_multi_block_rows, batched=True, num_proc=num_proc)
    logger.info(f" > {before_split} rows -> {len(ds)} rows after splitting "
                f"({len(ds) - before_split} additional rows from multi-block files)")

    logger.info(" > removing exact-duplicate rows (by code content)")
    ds = dedup_by_code(ds, logger)
    logger.info(f" > total rows after dedup: {len(ds)}")

    logger.info(" > stripping SketchIt attribution footers, repairing unbalanced tag counts, "
                "trimming outer whitespace")
    ds = ds.map(repair_row, num_proc=num_proc)
    logger.info(f" > stripped {sum(ds['footer_stripped'])} SketchIt footers, "
                f"repaired {sum(ds['tags_repaired'])} unbalanced-tag rows, "
                f"trimmed whitespace on {sum(ds['whitespace_stripped'])} rows")

    # ---------------------- CHEAP SEMANTIC FILTER FIRST ----------------------

    logger.info(" > applying semantic filter")
    ds = ds.map(add_semantic_flag, num_proc=num_proc)
    logger.info(f" > semantic pass: {sum(ds['semantic_pass'])}/{len(ds)}")

    survivors = ds.filter(lambda x: x["semantic_pass"], num_proc=num_proc)
    logger.info(f" > {len(survivors)}/{len(ds)} rows pass the semantic filter "
                f"(only these get the slow jar classification)")

    # ---------------------- JAR CLASSIFICATION (survivors only) ----------------------

    logger.info(" > classifying survivors with the real PlantUML engine (this is the slow step)")
    survivors = survivors.map(
        functools.partial(add_diagram_type, jar_path=jar_path, java_home=java_home, jar_timeout=jar_timeout),
        num_proc=jar_num_proc,
    )
    type_counts = Counter(survivors["diagram_type"])
    logger.info(f" > diagram_type counts among semantic-pass survivors: {dict(type_counts.most_common())}")

    strict = survivors.filter(lambda x: x["diagram_type"] in REAL_DIAGRAM_TYPES, num_proc=num_proc)
    logger.info(f" > {len(strict)} rows classified as a real diagram type -- this is the strict pool")

    # ---------------------- TEST / TRAIN_STRICT SPLIT ----------------------

    actual_test_size = min(test_size, len(strict))
    if actual_test_size < test_size:
        logger.warning(f" > only {len(strict)} rows are strict-eligible; "
                        f"using all of them as the test set instead of {test_size}")

    strict = strict.shuffle(seed=seed)
    test_ds = strict.select(range(actual_test_size))
    train_strict = strict.select(range(actual_test_size, len(strict)))

    logger.info(f" > test set: {len(test_ds)} rows")
    logger.info(f" > train_strict: {len(train_strict)} rows")

    # ---------------------- LABELED VARIANTS ----------------------

    test_labeled = test_ds.map(lambda x: {"our_diagram_type": x["diagram_type"]})
    train_strict_labeled = train_strict.map(lambda x: {"our_diagram_type": x["diagram_type"]})

    ds_to_save = DatasetDict({
        "test": test_ds,
        "train_strict": train_strict,
        "test_labeled": test_labeled,
        "train_strict_labeled": train_strict_labeled,
    })
    ds_to_save.save_to_disk(target_dir)
    logger.info(f"Saved test + train_strict (+ _labeled variants) to {target_dir}")
    logger.info("Done!")


def parse_args():
    parser = argparse.ArgumentParser(description="Build the strictly-filtered test + train_strict dataset splits.")
    parser.add_argument("--java-home", required=True,
                         help="JDK installation whose bin/java is used to run the jar")
    parser.add_argument("--jar-path", default=DEFAULT_JAR_PATH,
                         help="Path to a local plantuml.jar (defaults to the bundled jar)")
    parser.add_argument("--joined-dataset-dir", default="./data/puml_joined",
                         help="Path to the joined dataset (join_datasets.py's output)")
    parser.add_argument("--target-dir", default="./data/comparison_datasets",
                         help="Where to save the dataset versions (test + train_strict)")
    parser.add_argument("--test-size", type=int, default=700, help="Number of samples held out for the test set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for test-set sampling")
    parser.add_argument("--num-proc", type=int, default=4, help="Workers for cheap (regex) filtering steps")
    parser.add_argument("--jar-num-proc", type=int, default=4,
                         help="Workers for the jar-based classification step (each keeps one persistent JVM alive)")
    parser.add_argument("--jar-timeout", type=int, default=30, help="Seconds to allow the jar to run per sample")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Expecting Jar at:", args.jar_path)
    assert os.path.exists(args.jar_path), args.jar_path
    main(
        java_home=args.java_home,
        target_dir=args.target_dir,
        jar_path=args.jar_path,
        joined_dataset_dir=args.joined_dataset_dir,
        test_size=args.test_size,
        seed=args.seed,
        num_proc=args.num_proc,
        jar_num_proc=args.jar_num_proc,
        jar_timeout=args.jar_timeout,
    )
