"""
Generate predictions using a trained FIM model.
"""

import os
import re
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, Any

from tqdm import tqdm

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, set_seed
import transformers
from datasets import load_from_disk

# Suppress transformers warnings
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*pad_token_id.*")


def setup_logger():
    """Setup logging."""
    logger = logging.getLogger("generate")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def main(config: Dict[str, Any]) -> None:
    """
    Generate predictions on validation set.

    Args:
        config: Configuration dictionary with evaluate settings
    """
    logger = setup_logger()

    eval_config = config.get("evaluate", {})
    model_name = eval_config.get("model_name")
    dataset = eval_config.get("dataset_path")
    output_dir = eval_config.get("output_dir", "./outputs")
    seed = eval_config.get("seed", 42)
    subset_size = eval_config.get("subset_size")
    do_sample = eval_config.get("do_sample", False)
    temperature = eval_config.get("temperature", 1.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not model_name or not dataset:
        raise ValueError("model_name and dataset_path required in evaluate config")

    np_rng = np.random.RandomState(seed=seed)
    set_seed(seed)

    # Construct save path - update as needed for your naming convention
    model_to_save = model_name.split("/")[-1]
    training_to_save = "fim"
    data_to_save = "plantuml"
    model_name_str = f"{model_to_save}_{training_to_save}_{data_to_save}"
    if do_sample:
        model_name_str = f"{model_name_str}_temp{temperature}"
    model_name_str = f"{model_name_str}_evs{seed}"
    base_path = f"{output_dir}/{model_name_str}/"
    base_path_obj = Path(base_path)
    base_path_obj.mkdir(parents=True, exist_ok=True)
    save_path = f"{base_path}{model_name_str}.json"
    print(f" > Generations will be saved to: {save_path}")
    if os.path.exists(save_path):
        logger.info(f" > generated predictions already found at {save_path}. Skipping generation!")
        return

    def load_model(model_name: str, device: str) -> PreTrainedModel:
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device)

    ### ---------- LOAD DATA ----------
    data = load_from_disk(os.path.join(dataset))
    valset = data if subset_size is None else data.select(range(min(subset_size, len(data))))

    ### ---------- LOAD MODEL ----------

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.convert_tokens_to_ids("<fim_prefix>") is not None:
        prefix_tok_id = [tokenizer.convert_tokens_to_ids("<fim_prefix>")]
        middle_tok_id = [tokenizer.convert_tokens_to_ids("<fim_middle>")]
        suffix_tok_id = [tokenizer.convert_tokens_to_ids("<fim_suffix>")]
    else:
        prefix_tok_id = tokenizer.encode("<fim_prefix>")
        middle_tok_id = tokenizer.encode("<fim_middle>")
        suffix_tok_id = tokenizer.encode("<fim_suffix>")
    tokenizer.pad_token_id = tokenizer.eos_token_id


    model = load_model(model_name, device)
    model.config.pad_token_id = tokenizer.eos_token_id


    ### ---------- SETUP FIM GENERATE ON TESTSET ----------

    def fim_transform(example: str):
            """
            This function performs random FIM transformation on a single example (list of tokens)
            """
            input_ids = tokenizer(example)["input_ids"]
            boundary_low = np_rng.randint(low=0, high=len(input_ids))
            boundary_span = np_rng.randint(low=1, high=len(input_ids)-boundary_low+1)
            boundary_span = min(boundary_span, int(len(input_ids)/4))
            boundary_high = boundary_low + boundary_span

            prefix = input_ids[: boundary_low]
            middle = input_ids[boundary_low : boundary_high]
            suffix = input_ids[boundary_high :]

            if np_rng.binomial(1, 0.5):
                # Apply Suffix-Prefix-Middle (SPM) transformation
                transformed_example = [*prefix_tok_id, *suffix_tok_id] + suffix + middle_tok_id + prefix
            else:
                # Apply Prefix-Suffix-Middle (PSM) transformation
                transformed_example = prefix_tok_id + prefix + suffix_tok_id + suffix + middle_tok_id

            return transformed_example, middle

    def apply_fim(examples: dict) -> dict:
            """
            Apply FIM transformation to a batch of examples
            """
            fim_transform_ids = []
            fim_middles = []
            for ids in examples["code"]:
                transform, middle = fim_transform(ids)
                fim_transform_ids.append(transform)
                fim_middles.append(tokenizer.decode(middle))

            examples = {
                "input_ids": fim_transform_ids,
                "labels": fim_transform_ids,
                "attention_mask": [[1] * len(mask) for mask in fim_transform_ids],
                "middle": fim_middles
            }
            return examples

    def sort_fim(sample: str) -> str:
        """
        Create full code sample from FIM sample in PSM or SPM order
        """
        sorted_gen = ""

        if sample.startswith("<fim_prefix><fim_suffix>"):
            # SPM
            prefix_pattern = r"<fim_prefix><fim_suffix>(.*)<fim_middle>(.*)"
            match = re.fullmatch(prefix_pattern, sample, re.DOTALL)
            if match:
                prefix_middles = match.group(2)
                suffix = match.group(1)
                sorted_gen = prefix_middles + suffix
        else:
            # PSM
            prefix_pattern = r"<fim_prefix>(.*)<fim_suffix>(.*)<fim_middle>(.*)"
            match = re.fullmatch(prefix_pattern, sample, re.DOTALL)
            if match:
                prefix = match.group(1)
                middle = match.group(3)
                suffix = match.group(2)
                sorted_gen = prefix + middle + suffix

        if sorted_gen == "":
            raise ValueError(f"unprocessed sample:\n{sample}")

        return sorted_gen

    valset_clean = valset.remove_columns(
        [col for col in valset.column_names if col not in ['code']]
    )

    fim_testset = valset_clean.map(
        apply_fim,
        batched=True,
        num_proc=None,
        load_from_cache_file=True,
        desc="Performing FIM transformation",
    )

    fim_testset = fim_testset.remove_columns(['code'])

    generations = []
    gen_middles = []
    references = []
    ref_middles = []
    indices = []

    ### ---------- RUN GENERATION ON TESTSET ----------

    for dataset_index, (fim_sample, sample) in enumerate(zip(tqdm(fim_testset), valset)):

        if len(fim_sample["input_ids"]) >= tokenizer.model_max_length:
            continue

        fim_sample["input_ids"] = torch.tensor([fim_sample["input_ids"]], device=device)
        fim_sample["attention_mask"] = torch.tensor([fim_sample["attention_mask"]], device=device)
        fim_sample["labels"] = torch.tensor([fim_sample["labels"]], device=device)
        fim_middle = fim_sample["middle"]
        del fim_sample["middle"]

        seq_length = fim_sample['input_ids'].shape[1]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=fim_sample['input_ids'].device)
        position_ids = position_ids.unsqueeze(0)
        generate_kwargs = dict(
            num_return_sequences=1,
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=do_sample,
        )
        if do_sample:
            generate_kwargs["temperature"] = temperature
        output_ids = model.generate(**fim_sample, **generate_kwargs)

        middle_ids = output_ids[:, fim_sample["input_ids"].shape[-1]:]
        generated_puml = tokenizer.batch_decode(output_ids,
                                                skip_special_tokens=False)[0]
        puml_middle = tokenizer.batch_decode(middle_ids,
                                            skip_special_tokens=False)[0]

        # generate() may have stopped early on EOS; skip_special_tokens=False keeps
        # it (and the <fim_*> markers sort_fim needs) in the decoded text, so strip
        # it out here rather than let it leak into the saved generations/scoring.
        generated_puml = generated_puml.replace(tokenizer.eos_token, "")
        puml_middle = puml_middle.replace(tokenizer.eos_token, "")

        puml = sort_fim(generated_puml)

        generations.append(puml)
        gen_middles.append(puml_middle)
        references.append(sample["code"])
        ref_middles.append(fim_middle)
        indices.append(dataset_index)

    if not os.path.isfile(save_path):
        logger.info(f"Saving generations to {save_path}")
        with open(save_path, "w") as jfile:
            json.dump({
                "generations": generations,
                "gen-middles": gen_middles,
                "references": references,
                "ref-middles": ref_middles,
                # Row index into `dataset` (evaluate.dataset_path) each entry above came
                # from -- rows above whose FIM-transformed length hit tokenizer.model_max_length
                # are skipped (see the `continue` above), so position i here does NOT always
                # equal dataset row i. Downstream tools that need to line generations back up
                # with per-row dataset metadata (e.g. evaluation/summarize_scores.py's
                # diagram-type breakdown) should use this instead of assuming positional alignment.
                "indices": indices,
                 }, jfile)


if __name__ == "__main__":
    pass
