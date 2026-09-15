"""
Optional square-root-scaled oversampling by diagram type.

Groups our_diagram_type values into DIAGRAM_TYPE_GROUPS and resamples the
train split so the groups are represented in proportion to
sqrt(natural_count) rather than natural_count -- i.e. small groups are
oversampled, large groups undersampled, while the combined size of the
scaled groups stays the same. Any row whose our_diagram_type isn't in
DIAGRAM_TYPE_GROUPS (mixed, unknown, missing, or anything else) is left
untouched at its natural count.

Must run on the raw (pre-tokenization, pre-group_texts) train split: once
group_texts concatenates and re-chunks tokenized examples in train.py, a
single training example can span multiple original rows, so diagram_type
stops being attributable per example.
"""

import logging
import math
import random
from collections import defaultdict
from typing import Optional


# PlantUML doesn't distinguish use_case/component/deployment diagrams
# internally -- the real engine reports all three as DESCRIPTION (see
# validate_plantuml.py's _JAR_TYPE_MAP), so they're grouped here the same
# way. class/object are grouped on our own side for the same reason.
DIAGRAM_TYPE_GROUPS = {
    "class": "class_object",
    "object": "class_object",
    "sequence": "sequence",
    "use_case": "use_case_component",
    "deployment": "use_case_component",
    "component": "use_case_component",
    "activity": "activity",
    "state": "state",
}


def compute_sqrt_oversample_factors(group_counts: dict) -> dict:
    """
    For each group, the factor to multiply its natural count by so the
    resulting counts are proportional to sqrt(natural_count) instead of
    natural_count, while keeping the combined total across all groups
    unchanged.
    """
    total = sum(group_counts.values())
    sqrt_counts = {group: math.sqrt(count) for group, count in group_counts.items()}
    sqrt_total = sum(sqrt_counts.values())
    return {
        group: (total * sqrt_counts[group] / sqrt_total) / count
        for group, count in group_counts.items()
    }


def apply_diagram_type_oversampling(dataset, seed: int = 42, logger: Optional[logging.Logger] = None):
    """
    Return a resampled copy of `dataset` with DIAGRAM_TYPE_GROUPS groups
    square-root-scaled (see module docstring). Rows are duplicated (via
    random sampling with replacement) for groups being oversampled, and
    randomly subsampled without replacement for groups being undersampled.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if "our_diagram_type" not in dataset.column_names:
        raise ValueError(
            "diagram_type_oversampling is enabled but 'our_diagram_type' is not a column of this "
            "dataset -- point data.train_split at a dataset with that column populated "
            "(see data_pipeline/build_strict_dataset.py's train_strict_labeled output)"
        )

    group_indices = defaultdict(list)
    unscaled_indices = []
    for i, diagram_type in enumerate(dataset["our_diagram_type"]):
        group = DIAGRAM_TYPE_GROUPS.get(diagram_type)
        if group is None:
            unscaled_indices.append(i)
        else:
            group_indices[group].append(i)

    if not group_indices:
        logger.warning(" > diagram_type_oversampling: no rows matched any known diagram type group, nothing to do")
        return dataset

    group_counts = {group: len(idxs) for group, idxs in group_indices.items()}
    factors = compute_sqrt_oversample_factors(group_counts)
    target_counts = {group: round(count * factors[group]) for group, count in group_counts.items()}

    natural_total = sum(group_counts.values())
    target_total = sum(target_counts.values())

    rng = random.Random(seed)
    new_indices = list(unscaled_indices)
    for group, idxs in group_indices.items():
        target_count = target_counts[group]
        if target_count <= len(idxs):
            new_indices.extend(rng.sample(idxs, target_count))
        else:
            new_indices.extend(idxs)
            new_indices.extend(rng.choices(idxs, k=target_count - len(idxs)))

        logger.info(
            f" > diagram_type_oversampling: {group}: {len(idxs)} ({len(idxs) / natural_total:.1%}) -> "
            f"{target_count} ({target_count / target_total:.1%}) rows, factor {factors[group]:.2f}x"
        )

    logger.info(f" > diagram_type_oversampling: {len(unscaled_indices)} rows left untouched (not in a scaled group)")
    logger.info(f" > diagram_type_oversampling: {len(dataset)} -> {len(new_indices)} total rows")

    rng.shuffle(new_indices)
    return dataset.select(new_indices)
