#!/usr/bin/env python3
"""
FIM Pipeline CLI Entry Point
Orchestrates data preparation, training, and evaluation workflows.

Usage:
    uv run main.py --config config/data_pipeline.yml data-prepare
    uv run main.py --config config/train_main.yml train
    uv run main.py --config config/train_smoke_test.yml train

    # generate needs the training venv's transformers version to load the checkpoint;
    # score needs bert_score's older, incompatible transformers version -- run them
    # in separate `uv sync --extra train` / `uv sync --extra score` environments:
    uv run main.py --config config/evaluate.yml generate
    uv run main.py --config config/evaluate.yml score
"""

import argparse
import logging
import sys
from pathlib import Path

# data_pipeline/*.py import each other with bare sibling imports (e.g.
# `from validate_plantuml import ...`), which only resolve if data_pipeline/ itself is
# on sys.path -- true automatically when one of those scripts is run directly, but not
# when this module imports them as data_pipeline.xyz. Added unconditionally since it's
# harmless for the other commands.
sys.path.insert(0, str(Path(__file__).parent / "data_pipeline"))

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass

from scripts.config_loader import load_config, merge_env_overrides, validate_required_fields
from scripts.utils import setup_logging, parse_cli_overrides


logger = setup_logging()


def data_prepare(config: dict) -> None:
    """Execute the full data pipeline: download the Stack data, import the
    UML-in-the-wild data, join the two, and filter down to test/train_strict."""
    logger.info("Starting data preparation pipeline...")

    try:
        validate_required_fields(config, "data-prepare")
    except ValueError as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

    dp = config.get("data_pipeline", {})
    java_home = dp.get("java_home")
    jar_path = dp.get("jar_path") or str(Path(__file__).parent / "data_pipeline" / "java" / "plantuml-mit-1.2026.6.jar")
    stack_target_dir = dp.get("stack_target_dir", "./data/raw")
    stack_dataset_path = str(Path(stack_target_dir) / "the-stack-v2_PlantUML")

    try:
        import download_stack
        import download_uml_in_the_wild
        import import_puml_files_dataset
        import join_datasets
        import build_strict_dataset
    except Exception as e:
        logger.exception(f"Failed to import data_pipeline scripts: {e}")
        sys.exit(1)

    logger.info("Running download_stack...")
    try:
        download_stack.main(stack_target_dir, dp.get("debug_download", False))
    except Exception as e:
        logger.exception(f"download_stack failed: {e}")
        sys.exit(1)

    logger.info("Running download_uml_in_the_wild...")
    try:
        download_uml_in_the_wild.main(
            puml_dir=dp.get("puml_dir", "./data/puml_files"),
            metadata_file=dp.get("metadata_file", "./data/puml_files/uml_metadata_enriched.json"),
            puml_files_zip_url=dp.get("puml_files_zip_url", download_uml_in_the_wild.DEFAULT_PUML_FILES_ZIP_URL),
            metadata_url=dp.get("uml_metadata_url", download_uml_in_the_wild.DEFAULT_METADATA_URL),
        )
    except Exception as e:
        logger.exception(f"download_uml_in_the_wild failed: {e}")
        sys.exit(1)

    logger.info("Running import_puml_files_dataset...")
    try:
        import_puml_files_dataset.main(
            puml_dir=dp.get("puml_dir", "./data/puml_files"),
            metadata_file=dp.get("metadata_file", "./data/puml_files/uml_metadata_enriched.json"),
            output_dir=dp.get("uml_in_the_wild_dir", "./data/uml_in_the_wild"),
            java_home=java_home,
            jar_path=jar_path,
            reference_dataset=stack_dataset_path,
            confusion_matrix_path=dp.get("confusion_matrix_path", "./data/puml_files_confusion_matrix.md"),
            num_proc=dp.get("num_proc", 4),
            jar_timeout=dp.get("jar_timeout", 30),
        )
    except Exception as e:
        logger.exception(f"import_puml_files_dataset failed: {e}")
        sys.exit(1)

    logger.info("Running join_datasets...")
    try:
        join_datasets.main(
            stack_dataset=stack_dataset_path,
            uml_in_the_wild_dataset=dp.get("uml_in_the_wild_dir", "./data/uml_in_the_wild"),
            output_dir=dp.get("joined_dataset_dir", "./data/puml_joined"),
        )
    except Exception as e:
        logger.exception(f"join_datasets failed: {e}")
        sys.exit(1)

    logger.info("Running build_strict_dataset...")
    try:
        build_strict_dataset.main(
            java_home=java_home,
            target_dir=dp.get("strict_dataset_dir", "./data/comparison_datasets"),
            jar_path=jar_path,
            joined_dataset_dir=dp.get("joined_dataset_dir", "./data/puml_joined"),
            test_size=dp.get("test_size", 700),
            seed=dp.get("seed", 42),
            num_proc=dp.get("num_proc", 4),
            jar_num_proc=dp.get("jar_num_proc", 4),
            jar_timeout=dp.get("jar_timeout", 30),
        )
    except Exception as e:
        logger.exception(f"build_strict_dataset failed: {e}")
        sys.exit(1)

    logger.info("Data preparation completed successfully!")


def train(config: dict) -> None:
    """Execute training pipeline."""
    logger.info("Starting training pipeline...")

    try:
        validate_required_fields(config, "train")
    except ValueError as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

    try:
        from training.train import main as train_main
        train_main(config)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)

    logger.info("Training completed successfully!")


def generate(config: dict) -> None:
    """Execute prediction generation only.

    Kept separate from `score` since generation needs the model/training
    venv, which may not have `bert_score`'s compatible transformers version.
    """
    logger.info("Starting generation step...")

    try:
        validate_required_fields(config, "evaluate")
    except ValueError as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

    logger.info("Running generate...")
    try:
        from evaluation.generate import main as generate_main
        generate_main(config)
    except Exception as e:
        logger.exception(f"generate failed: {e}")
        sys.exit(1)

    logger.info("Generation completed successfully!")


def score(config: dict) -> None:
    """Execute similarity scoring + rendering, on top of already-generated predictions.

    Kept separate from `generate` so it can run in its own venv (e.g. one with
    `bert_score`'s compatible transformers version).
    """
    logger.info("Starting scoring pipeline...")

    try:
        validate_required_fields(config, "evaluate")
    except ValueError as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

    # Compute similarity scores
    logger.info("Running similarity_scores...")
    try:
        from evaluation.similarity_scores import main as similarity_main
        similarity_main(config)
    except Exception as e:
        logger.exception(f"similarity_scores failed: {e}")
        sys.exit(1)

    # Render scores
    logger.info("Running render_scores...")
    try:
        from evaluation.render_scores import main as render_main
        render_main(config)
    except Exception as e:
        logger.warning(f"render_scores failed: {e}", exc_info=True)
        logger.info("Continuing without diagram rendering...")

    logger.info("Scoring completed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="FIM Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        uv run main.py --config config/data_pipeline.yml data-prepare
        uv run main.py --config config/train_main.yml train
        uv run main.py --config config/train_smoke_test.yml train --training.num_train_epochs 1
        uv run main.py --config config/evaluate.yml generate
        uv run main.py --config config/evaluate.yml score
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )

    parser.add_argument(
        "command",
        choices=["data-prepare", "train", "generate", "score"],
        help="Pipeline command to execute"
    )

    # Parse known args to extract command, then collect remaining for overrides
    args, remaining = parser.parse_known_args()

    # Load configuration
    logger.info(f"Loading config from {args.config}")
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.exception(f"Failed to load config: {e}")
        sys.exit(1)

    # Apply CLI overrides, e.g. --training.num_train_epochs 5
    if remaining:
        overrides = parse_cli_overrides(remaining)
        config = merge_env_overrides(config, overrides)

    # Execute command
    if args.command == "data-prepare":
        data_prepare(config)
    elif args.command == "train":
        train(config)
    elif args.command == "generate":
        generate(config)
    elif args.command == "score":
        score(config)


if __name__ == "__main__":
    main()
