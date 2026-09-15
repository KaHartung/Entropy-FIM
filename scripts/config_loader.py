"""
Config loader for FIM pipeline.
Loads YAML configuration files, validates required fields, and merges with environment overrides.
"""

import os
import re
from typing import Any, Dict, Optional
import yaml


# Top-level sections that appear in the config YAMLs (see config/*.yml). Every
# CLI override must be scoped under one of these, so a typo'd or missing
# section prefix (e.g. --diagram_type_oversampling instead of
# --data.diagram_type_oversampling) is rejected instead of silently creating
# an unused top-level key.
KNOWN_CONFIG_SECTIONS = {"data_pipeline", "data", "model", "training", "evaluate"}

# A valid dotted-path segment: letters, digits, underscore only. Catches typos
# like a stray trailing ':' (--training.learning_rate: instead of
# --training.learning_rate) that would otherwise silently become part of the
# key name.
_VALID_KEY_SEGMENT_RX = re.compile(r"^[A-Za-z0-9_]+$")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary with configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    return config

def merge_env_overrides(config: Dict[str, Any], overrides: Dict[str, str]) -> Dict[str, Any]:
    """
    Merge CLI overrides into the config, in place.

    Keys use dotted paths to address nested sections, e.g. "training.num_train_epochs".
    Values are parsed with the YAML loader so numbers/booleans keep their type
    instead of staying strings (e.g. "5" -> 5, "true" -> True).

    Args:
        config: Configuration dictionary to update
        overrides: Dict of dotted-path keys to raw string values (from parse_cli_overrides)

    Returns:
        The same config dictionary, updated in place

    Raises:
        ValueError: If an override key isn't scoped under a known top-level section, or
            contains characters other than letters/digits/underscore (e.g. a typo'd stray
            character like a trailing ':'). A malformed key would otherwise silently create
            an unused entry instead of overriding the field that was actually meant.
    """
    for dotted_key, raw_value in overrides.items():
        parts = dotted_key.split(".")
        if len(parts) < 2:
            raise ValueError(
                f"Invalid config override '--{dotted_key}': overrides must be scoped to a section, "
                f"e.g. '--data.{dotted_key} ...' or '--training.{dotted_key} ...' "
                f"(known sections: {', '.join(sorted(KNOWN_CONFIG_SECTIONS))})"
            )
        if not all(_VALID_KEY_SEGMENT_RX.match(part) for part in parts):
            raise ValueError(
                f"Invalid config override key '--{dotted_key}': each dotted segment may only contain "
                "letters, digits, and underscores - check for a stray character (e.g. a trailing ':')"
            )
        if parts[0] not in KNOWN_CONFIG_SECTIONS:
            raise ValueError(
                f"Invalid config override '--{dotted_key}': unknown top-level section '{parts[0]}' "
                f"(known sections: {', '.join(sorted(KNOWN_CONFIG_SECTIONS))})"
            )

        *parents, leaf = parts
        target = config
        for part in parents:
            target = target.setdefault(part, {})
        target[leaf] = yaml.safe_load(raw_value)

    return config


def validate_required_fields(config: Dict[str, Any], command: str) -> None:
    """
    Validate that required fields are present for a given command.

    Args:
        config: Configuration dictionary
        command: Command being run (data-prepare, train, evaluate)

    Raises:
        ValueError: If required fields are missing
    """
    if command == "data-prepare":
        required_env = ["AWS_REGION", "AWS_KEY", "AWS_SECRET"]
        missing_env = [var for var in required_env if not os.getenv(var, "").strip()]
        if missing_env:
            raise ValueError(
                f"Missing required environment variables for data-prepare: {', '.join(missing_env)}"
            )

        if not config.get("data_pipeline", {}).get("java_home"):
            raise ValueError("Required config field missing: data_pipeline.java_home "
                              "(set it to your local JDK installation)")

    elif command == "train":
        required_model = config.get("model", {}).get("model_name_or_path")
        if not required_model:
            raise ValueError("Required config field missing: model.model_name_or_path")

        required_data = config.get("data", {}).get("dataset_name")
        if not required_data:
            raise ValueError("Required config field missing: data.dataset_name")

    elif command == "evaluate":
        required_model = config.get("evaluate", {}).get("model_name")
        if not required_model:
            raise ValueError("Required config field missing: evaluate.model_name")

        required_dataset = config.get("evaluate", {}).get("dataset_path")
        if not required_dataset:
            raise ValueError("Required config field missing: evaluate.dataset_path")
