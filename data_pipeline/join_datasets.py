"""
Join the downloaded Stack dataset with the UML-in-the-wild dataset into one dataset
for filtering.

The two datasets have very different schemas (the Stack dataset carries the-stack-v2's
raw columns -- blob_id, path, repo/license/star metadata, code, ...; the UML-in-the-wild
dataset carries import_puml_files_dataset.py's classification columns -- filename, code,
diagram_type, blob_id, original_path, ...). Nothing downstream of this script (the
filtering pipeline in build_strict_dataset.py) reads anything but `code`, so rather than
reconciling the full column union, this keeps only the columns present on both sides.
"""

import argparse

from datasets import concatenate_datasets, load_from_disk


def main(stack_dataset: str, uml_in_the_wild_dataset: str, output_dir: str) -> None:
    """
    Args:
        stack_dataset: path to the Stack dataset (download_stack.py's output)
        uml_in_the_wild_dataset: path to the UML-in-the-wild dataset
            (import_puml_files_dataset.py's output)
        output_dir: where to save_to_disk the concatenated dataset
    """
    stack_ds = load_from_disk(stack_dataset)
    wild_ds = load_from_disk(uml_in_the_wild_dataset)

    shared_columns = [c for c in stack_ds.column_names if c in wild_ds.column_names]
    if "code" not in shared_columns:
        raise ValueError(
            f"Expected both datasets to share a 'code' column; got shared columns: {shared_columns}"
        )

    stack_ds = stack_ds.select_columns(shared_columns)
    wild_ds = wild_ds.select_columns(shared_columns)

    joined = concatenate_datasets([stack_ds, wild_ds])
    joined.save_to_disk(output_dir)
    print(f"Joined {len(stack_ds)} Stack rows + {len(wild_ds)} UML-in-the-wild rows "
          f"= {len(joined)} rows (columns kept: {shared_columns}) -> {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Join the Stack dataset with the UML-in-the-wild dataset.")
    parser.add_argument("--stack-dataset", default="./data/raw/the-stack-v2_PlantUML",
                         help="Path to the Stack dataset (download_stack.py's output)")
    parser.add_argument("--uml-in-the-wild-dataset", default="./data/uml_in_the_wild",
                         help="Path to the UML-in-the-wild dataset (import_puml_files_dataset.py's output)")
    parser.add_argument("--output-dir", default="./data/puml_joined",
                         help="Where to save_to_disk the concatenated dataset")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        stack_dataset=args.stack_dataset,
        uml_in_the_wild_dataset=args.uml_in_the_wild_dataset,
        output_dir=args.output_dir,
    )
