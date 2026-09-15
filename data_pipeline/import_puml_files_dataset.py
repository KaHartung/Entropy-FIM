"""
Import the puml_files/ dataset (one PlantUML file per sample, with metadata in
uml_metadata_enriched.json's "classifications") into a HuggingFace-format dataset,
keeping only samples not already present in the downloaded Stack dataset -- matched
by exact code string OR by original file path, since a sample can also be a
near-duplicate (e.g. a different version/revision of the same file) that shares a
path but not the exact code content.
"""

import argparse
import functools
import json
import logging
import re
from collections import Counter
from pathlib import Path

from datasets import Dataset, load_from_disk

from validate_plantuml import ensure_diagram_type_detector_compiled, extract_insights_jar

DEFAULT_JAR_PATH = str(Path(__file__).resolve().parent / "java" / "plantuml-mit-1.2026.6.jar")

# The metadata's classifier uses "usecase"; ours uses "use_case" -- normalized so
# the confusion matrix's diagonal reflects real agreement/disagreement, not a naming
# mismatch. Missing metadata (file has no classifications entry) gets its own explicit
# label rather than being silently folded into "unknown".
_NO_METADATA_LABEL = "<no metadata>"
_THEIR_TYPE_ALIASES = {"usecase": "use_case"}

# puml_files/ samples are prefixed with a header of real PlantUML comment lines
# (leading `'`), e.g.:
#   ' Blob ID: be8022fa89fd9a2c9fd019c57e4373074179cc8b
#   ' Original Path: /src/main/java/ex44/diagram04.puml
#   ' Source: World of Code
# (blank line)
# The reference dataset's code strings never have this, so it was making every
# genuinely-duplicate sample look new (the code strings differed by this header alone).
# Matches any run of leading `'` comment lines rather than these exact 3, in case the
# header varies. Blob ID/Original Path duplicate what's already in the structured
# metadata (build_row already reads them from the classification entry), so nothing is
# lost by stripping it entirely rather than trying to preserve it in `code`.
RX_LEADING_HEADER = re.compile(r"^(?:'[^\n]*\n)+\n?")


def normalize_their_type(primary_type: str | None) -> str:
    if primary_type is None:
        return _NO_METADATA_LABEL
    return _THEIR_TYPE_ALIASES.get(primary_type, primary_type)


def strip_leading_header(code: str) -> str:
    return RX_LEADING_HEADER.sub("", code, count=1)


def normalize_path(path: str | None) -> str | None:
    """Strip whitespace and a leading slash, so e.g. "/src/x.puml" and "src/x.puml" match."""
    if not path:
        return None
    return path.strip().lstrip("/")


def setup_logger():
    """Setup logging."""
    logger = logging.getLogger("import_puml_files_dataset")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_existing_codes_and_paths(reference_dataset: str, logger) -> tuple[set[str], set[str]]:
    """
    Load the Stack dataset's `code` and `path` columns as sets, for O(1) dedup
    lookups. Both criteria are used: an exact code match catches byte-identical
    samples; a path match additionally catches different versions/revisions of the
    same file, which have different code but are still duplicates for training
    purposes (we only want one copy of any given file's lineage).

    reference_dataset is a local `Dataset.save_to_disk` directory (the output of
    download_stack.py), loaded with load_from_disk -- not a HuggingFace Hub dataset
    name, so this never depends on anything outside this pipeline's own output.
    """
    logger.info(f" > loading reference dataset: {reference_dataset}")
    ds = load_from_disk(reference_dataset)
    # Stripped so purely-cosmetic leading/trailing whitespace differences between
    # otherwise-identical samples don't defeat the match.
    codes = {code.strip() for code in ds["code"]}
    paths = {normalize_path(p) for p in ds["path"]}
    paths.discard(None)
    logger.info(f" > reference dataset: {len(ds)} rows, {len(codes)} distinct code strings, "
                f"{len(paths)} distinct paths")
    return codes, paths


def load_classifications(metadata_path: Path, logger) -> dict:
    logger.info(f" > loading metadata from {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    classifications = data["classifications"]
    logger.info(f" > metadata covers {len(classifications)} files "
                f"(dataset total_files={data['metadata'].get('total_files')})")
    return classifications


def build_row(puml_path: Path, code: str, classification: dict | None) -> dict:
    classification = classification or {}
    return {
        "filename": puml_path.name,
        "code": code,
        "diagram_type": classification.get("diagram_type"),
        "primary_type": classification.get("primary_type"),
        "secondary_types": classification.get("secondary_types") or [],
        "reasoning": classification.get("reasoning"),
        "content_lines": classification.get("content_lines"),
        # elements/connections are dicts with a different set of keys per row (e.g.
        # {"class": 1} vs {"class": 3, "package": 8}) -- JSON-serialized to sidestep
        # Arrow schema inference over a variable-shape nested type entirely.
        "elements_json": json.dumps(classification.get("elements") or {}),
        "elements_total": classification.get("elements_total"),
        "connections_json": json.dumps(classification.get("connections") or {}),
        "connections_total": classification.get("connections_total"),
        "extraction_error": classification.get("extraction_error"),
        "blob_id": classification.get("blob_id"),
        "original_path": classification.get("original_path"),
        "repository": classification.get("repository"),
    }


def add_dup_flags(example, existing_codes: set[str], existing_paths: set[str]):
    return {
        "dup_by_code": example["code"].strip() in existing_codes,
        "dup_by_path": normalize_path(example["original_path"]) in existing_paths,
    }


def add_our_classification(example, jar_path: str, java_home: str, jar_timeout: int):
    insights = extract_insights_jar(example["code"], jar_path, java_home=java_home, timeout=jar_timeout)
    return {"our_diagram_type": insights["diagram_type"]}


def build_confusion_matrix(ds) -> Counter:
    """(our_diagram_type, normalized primary_type) -> count, over every row in ds."""
    confusion = Counter()
    for our_type, their_primary_type in zip(ds["our_diagram_type"], ds["primary_type"]):
        confusion[(our_type, normalize_their_type(their_primary_type))] += 1
    return confusion


def render_confusion_matrix_md(confusion: Counter) -> str:
    our_types = sorted({k[0] for k in confusion})
    their_types = sorted({k[1] for k in confusion})

    header = "| ours \\ theirs | " + " | ".join(their_types) + " | total |\n"
    header += "|" + "---|" * (len(their_types) + 2) + "\n"

    lines = [header]
    col_totals = {t: 0 for t in their_types}
    grand_total = 0
    for our_type in our_types:
        row_total = 0
        cells = []
        for their_type in their_types:
            count = confusion.get((our_type, their_type), 0)
            cells.append(str(count))
            row_total += count
            col_totals[their_type] += count
        grand_total += row_total
        lines.append(f"| {our_type} | " + " | ".join(cells) + f" | {row_total} |\n")

    footer = "| total | " + " | ".join(str(col_totals[t]) for t in their_types) + f" | {grand_total} |\n"
    lines.append(footer)
    return "".join(lines)


def main(puml_dir: str, metadata_file: str, output_dir: str, java_home: str,
         jar_path: str = DEFAULT_JAR_PATH,
         reference_dataset: str = "./data/raw/the-stack-v2_PlantUML",
         confusion_matrix_path: str = "./data/puml_files_confusion_matrix.md",
         num_proc: int = 4, jar_timeout: int = 30) -> None:
    """
    Args:
        puml_dir: directory containing the individual .puml files
        metadata_file: path to uml_metadata_enriched.json
        output_dir: where to save_to_disk the resulting new-samples-only dataset
        java_home: path to a JDK installation used to run (and compile) the jar
        jar_path: path to a local plantuml.jar, used to classify every file with our
            own extract_insights_jar before deduplication
        reference_dataset: the local Stack dataset (download_stack.py's output) to
            dedup against (by code string and path match)
        confusion_matrix_path: where to write the our-vs-their classification
            confusion matrix (markdown table)
        num_proc: worker count for the classification step
        jar_timeout: seconds to allow the jar to run per sample
    """
    logger = setup_logger()

    ensure_diagram_type_detector_compiled(jar_path, java_home=java_home)

    existing_codes, existing_paths = load_existing_codes_and_paths(reference_dataset, logger)
    classifications = load_classifications(Path(metadata_file), logger)

    puml_files = sorted(Path(puml_dir).glob("*.puml"))
    logger.info(f" > found {len(puml_files)} .puml files in {puml_dir}")

    # First pass: read every file and attach its metadata, WITHOUT filtering out
    # duplicates yet -- the confusion matrix below is meant to reflect our classifier
    # against theirs over the whole incoming set, not just the new-vs-old subset.
    rows = []
    num_missing_metadata = 0
    num_read_errors = 0
    num_header_stripped = 0

    for i, puml_path in enumerate(puml_files, start=1):
        if i % 2000 == 0:
            logger.info(f" > read {i}/{len(puml_files)} files")

        try:
            code = puml_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as e:
            num_read_errors += 1
            logger.warning(f" > skipping {puml_path.name}: {type(e).__name__}: {e}")
            continue

        stripped_code = strip_leading_header(code)
        if stripped_code != code:
            num_header_stripped += 1
            code = stripped_code

        classification = classifications.get(puml_path.name)
        if classification is None:
            num_missing_metadata += 1
            logger.warning(f" > no metadata entry for {puml_path.name} (kept, with null metadata fields)")

        rows.append(build_row(puml_path, code, classification))

    logger.info(f" > {len(rows)}/{len(puml_files)} files read ({num_read_errors} unreadable, "
                f"{num_missing_metadata} missing metadata, {num_header_stripped} had a leading header stripped)")

    # ---------------------- CLASSIFY + CONFUSION MATRIX ----------------------

    logger.info(" > classifying every file with our extract_insights_jar (this is the slow step)")
    ds = Dataset.from_list(rows)
    ds = ds.map(
        functools.partial(add_our_classification, jar_path=jar_path, java_home=java_home, jar_timeout=jar_timeout),
        num_proc=num_proc,
    )

    confusion = build_confusion_matrix(ds)
    confusion_md = (
        f"# Our classification vs. puml_files metadata's primary_type\n\n"
        f"Rows: our `extract_insights_jar` diagram_type. Columns: their `primary_type` "
        f"(\"usecase\" normalized to \"use_case\"; `{_NO_METADATA_LABEL}` = no classifications entry "
        f"for that file). Computed over all {len(ds)} files, before deduplication.\n\n"
        + render_confusion_matrix_md(confusion)
    )
    with open(confusion_matrix_path, "w", encoding="utf-8") as f:
        f.write(confusion_md)
    logger.info(f" > confusion matrix saved to {confusion_matrix_path}")
    logger.info("\n" + confusion_md)

    # ---------------------- DEDUPLICATE + SAVE ----------------------

    ds = ds.map(
        functools.partial(add_dup_flags, existing_codes=existing_codes, existing_paths=existing_paths),
        num_proc=num_proc,
    )
    num_dup_code = sum(ds["dup_by_code"])
    num_dup_path = sum(ds["dup_by_path"])
    num_dup_either = sum(1 for c, p in zip(ds["dup_by_code"], ds["dup_by_path"]) if c or p)
    logger.info(f" > duplicates of {reference_dataset}: {num_dup_code} by code match, {num_dup_path} by path "
                f"match, {num_dup_either} total (union)")

    new_ds = ds.filter(lambda x: not x["dup_by_code"] and not x["dup_by_path"], num_proc=num_proc)
    logger.info(f" > {len(new_ds)} new samples to save")

    new_ds.save_to_disk(output_dir)
    logger.info(f"Saved {len(new_ds)} new samples to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Import new (not-already-present) samples from puml_files/ into a HF dataset."
    )
    parser.add_argument("--java-home", required=True,
                         help="JDK installation whose bin/java (and bin/javac) is used")
    parser.add_argument("--jar-path", default=DEFAULT_JAR_PATH,
                         help="Path to a local plantuml.jar (defaults to the bundled jar)")
    parser.add_argument("--puml-dir", default="./data/puml_files", help="Directory of individual .puml files")
    parser.add_argument("--metadata-file", default="./data/puml_files/uml_metadata_enriched.json",
                         help="Path to uml_metadata_enriched.json")
    parser.add_argument("--output-dir", default="./data/uml_in_the_wild",
                         help="Where to save_to_disk the resulting new-samples-only dataset")
    parser.add_argument("--reference-dataset", default="./data/raw/the-stack-v2_PlantUML",
                         help="Local Stack dataset (download_stack.py's output) to dedup against")
    parser.add_argument("--confusion-matrix-path", default="./data/puml_files_confusion_matrix.md",
                         help="Where to write the our-vs-their classification confusion matrix")
    parser.add_argument("--num-proc", type=int, default=4, help="Workers for the classification step")
    parser.add_argument("--jar-timeout", type=int, default=30, help="Seconds to allow the jar to run per sample")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        puml_dir=args.puml_dir,
        metadata_file=args.metadata_file,
        output_dir=args.output_dir,
        jar_path=args.jar_path,
        java_home=args.java_home,
        reference_dataset=args.reference_dataset,
        confusion_matrix_path=args.confusion_matrix_path,
        num_proc=args.num_proc,
        jar_timeout=args.jar_timeout,
    )
