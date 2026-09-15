"""
Compute similarity scores for generated vs reference code.
"""
import os
import json
import logging
from typing import Dict, Any

from tqdm import tqdm

import numpy as np
from bert_score import BERTScorer


def setup_logger():
    """Setup logging."""
    logger = logging.getLogger("similarity_scores")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def main(config: Dict[str, Any]) -> None:
    """
    Compute similarity scores between generated and reference code.

    Args:
        config: Configuration dictionary with evaluate settings
    """
    logger = setup_logger()

    eval_config = config.get("evaluate", {})
    model_name = eval_config.get("model_name")
    if not model_name:
        raise ValueError("model_name required in evaluate config")

    # Generate file path - similar to generate.py
    model_to_save = model_name.split("/")[-1]
    training_to_save = "fim"
    data_to_save = "plantuml"
    output_dir = eval_config.get("output_dir", "./outputs")
    do_sample = eval_config.get("do_sample", False)
    temperature = eval_config.get("temperature", 1.0)
    seed = eval_config.get("seed", 42)
    model_name_str = f"{model_to_save}_{training_to_save}_{data_to_save}"
    if do_sample:
        model_name_str = f"{model_name_str}_temp{temperature}"
    model_name_str = f"{model_name_str}_evs{seed}"
    base_path = f"{output_dir}/{model_name_str}/"
    file_path = f"{base_path}{model_name_str}.json"
    save_path = file_path.replace(".json", "_scored.json")
    if os.path.exists(save_path):
        logger.info(f" > Similarity scores already found at {save_path}. Skipping evaluation!")
        return

    logger.info(f"Evaluating generations at: {file_path}")

    def evaluate_similarity(generation, reference, bertscore):
        predictions = [generation]
        references = [reference]
        results = bertscore.score(predictions, references)
        return results

    ### ---------- DEFINE HELPER ----------
    score_model_id = "distilbert-base-uncased"
    bertscore = BERTScorer(model_type=score_model_id)

    ### ---------- LOAD DATA ----------
    with open(file_path, 'r') as jfile:
        generations = json.load(jfile)

    code_scores = []
    mid_code_scores = []

    ### ---------- RUN EVALUATION ----------
    for i in tqdm(range(len(generations["generations"]))):

        generation = generations["generations"][i]
        reference = generations["references"][i]
        gen_middle = generations["gen-middles"][i]
        ref_middle = generations["ref-middles"][i]

        _, _, code_score = evaluate_similarity(generation, reference, bertscore)
        _, _, mid_code_score = evaluate_similarity(gen_middle, ref_middle, bertscore)

        code_scores.append(code_score.item())
        mid_code_scores.append(mid_code_score.item())


    code_score = np.mean(code_scores)
    mid_code_score = np.mean(mid_code_scores)
    logger.info(
        f" >> {file_path.split('/')[-1]}"
        f"\n - code-score: {code_score}"
        f"\n - code-score-middles: {mid_code_score}"
    )

    generations["code-scores"] = code_scores
    generations["mid-code-scores"] = mid_code_scores
    logger.info(f"Saving scores to: {save_path}")
    with open(save_path, "w") as jfile:
        json.dump(generations, jfile)


if __name__ == "__main__":
    pass
