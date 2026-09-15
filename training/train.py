"""
Fine-tuning the library models for causal language modeling using
Fill-in-the middle (FIM) objective on a text file or a dataset.
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_LOGS"] = "recompiles"  # log if torch.compile does a recompile of the model

# torch.compile/Triton default to caching compiled kernels under /tmp/torchinductor_<user>;
# on nodes where /tmp is mounted noexec, loading those .so files fails with
# "failed to map segment from shared object". Redirect the cache somewhere executable.
_TORCHINDUCTOR_CACHE_DIR = os.path.abspath("./cache/torchinductor")
os.makedirs(_TORCHINDUCTOR_CACHE_DIR, exist_ok=True)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", _TORCHINDUCTOR_CACHE_DIR)
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_TORCHINDUCTOR_CACHE_DIR, "triton"))

import sys
import logging
import math
from itertools import chain
from typing import Tuple

import datasets
import evaluate
import torch
import wandb
import numpy as np
from datasets import load_from_disk, DatasetDict
from dataclasses import asdict

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version
from huggingface_hub.utils import send_telemetry


# Suppress transformers warnings
transformers.logging.set_verbosity_error()

from training.arguments import ModelArguments, DataTrainingArguments
from training.oversampling import apply_diagram_type_oversampling


def load_training_config(config: dict) -> Tuple[ModelArguments, DataTrainingArguments, TrainingArguments]:
    """Load training configuration from config dictionary."""

    use_cuda = torch.cuda.is_available()
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    training_config = config.get("training", {})

    model_torch_dtype = model_config.get("torch_dtype", "float32")
    if not use_cuda and model_torch_dtype in ("bfloat16", "float16"):
        model_torch_dtype = "float32"

    training_bf16 = training_config.get("bf16", False) if use_cuda else False

    model_args = ModelArguments(
        model_name_or_path=model_config.get("model_name_or_path", "gpt2"),
        tokenizer_name=model_config.get("tokenizer_name"),
        config_name=model_config.get("config_name"),
        cache_dir=model_config.get("cache_dir", "./cache"),
        torch_dtype=model_torch_dtype,
        low_cpu_mem_usage=model_config.get("low_cpu_mem_usage", True),
        trust_remote_code=model_config.get("trust_remote_code", False),
    )

    data_args = DataTrainingArguments(
        dataset_name=data_config.get("dataset_name", "wikitext"),
        train_split=data_config.get("train_split", "train"),
        validation_split_percentage=data_config.get("validation_split_percentage", 5),
        fim_rate=data_config.get("fim_rate", 0.9),
        fim_spm_rate=data_config.get("fim_spm_rate", 0.5),
        block_size=data_config.get("block_size", 1024),
        max_train_samples=data_config.get("max_train_samples"),
        max_eval_samples=data_config.get("max_eval_samples"),
        entropy=data_config.get("entropy", True),
        window_size=data_config.get("window_size"),
        window_size_min=data_config.get("window_size_min"),
        window_size_random=data_config.get("window_size_random", False),
        diagram_type_oversampling=data_config.get("diagram_type_oversampling", False),
    )

    # Handle special tokens from config
    special_tokens = data_config.get("special_tokens", {})
    data_args.fim_prefix_token = special_tokens.get("prefix", "<fim_prefix>")
    data_args.fim_middle_token = special_tokens.get("middle", "<fim_middle>")
    data_args.fim_suffix_token = special_tokens.get("suffix", "<fim_suffix>")
    data_args.pad_token = special_tokens.get("pad", "<|pad|>")
    data_args.truncate_or_pad = True

    logging_config = training_config.get("logging", {})
    report_to = logging_config.get("report_to")
    log_wandb = logging_config.get("log_wandb", False)

    training_args = TrainingArguments(
        output_dir=training_config.get("output_dir", "./outputs"),
        do_eval=True,
        do_predict=False,
        do_train=True,
        num_train_epochs=training_config.get("num_train_epochs", 3),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 8),
        bf16=training_bf16,
        learning_rate=training_config.get("learning_rate", 5e-5),
        seed=training_config.get("seed", 42),
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=training_config.get("eval_steps", 500),
        save_steps=training_config.get("save_steps", 500),
        save_total_limit=training_config.get("save_total_limit", 3),
        # overwrite_output_dir=True,
        load_best_model_at_end=training_config.get("load_best_model_at_end", True),
        # [] rather than None when disabled: transformers' TrainingArguments only
        # special-cases the *unset* default (also None) into "all issues" -- an
        # explicitly-passed None isn't distinguishable from that at this point, so it
        # instead falls through to being wrapped as the single-item list [None],
        # which then fails validation ("None is not supported, only ... are
        # supported"). An empty list means "no integrations" unambiguously.
        report_to=report_to if log_wandb else [],
        push_to_hub=False,
        use_cpu=not torch.cuda.is_available(), # disables CUDA
        # use_mps_device=False                   # disables MPS
        dataloader_num_workers=training_config.get("dataloader_num_workers", 4),
        torch_compile=training_config.get("torch_compile", True),
        resume_from_checkpoint=training_config.get("resume_from_checkpoint"),
    )

    return model_args, data_args, training_args


def setup_logging(model_args, data_args, training_args):

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    return logger


def load_data(data_args, seed: int = 42, train_dataset_override=None) -> DatasetDict:

    # Loading a locally stored dataset, with one or more training variants
    # saved as splits (see `train_split`). If it has no "validation" split,
    # one is carved out of the selected train split using validation_split_percentage.
    #
    # `train_dataset_override`, if given, is used as the train split directly
    # instead of looking `train_split` up on disk (e.g. a dataset assembled
    # on the fly by train_percentage.py).
    if train_dataset_override is not None:
        splits = {}
        train_dataset = train_dataset_override
    else:
        dataset = load_from_disk(data_args.dataset_name)

        if data_args.train_split not in dataset:
            raise ValueError(
                f"train_split '{data_args.train_split}' not found in dataset splits: {list(dataset.keys())}"
            )

        splits = dict(dataset)
        train_dataset = splits.pop(data_args.train_split)

    if "validation" not in splits:
        split = train_dataset.train_test_split(
            test_size=data_args.validation_split_percentage / 100,
            seed=seed,
        )
        train_dataset = split["train"]
        splits["validation"] = split["test"]

    splits["train"] = train_dataset
    return DatasetDict(splits)


def load_tokenizer(model_args: ModelArguments,
                   data_args: DataTrainingArguments) -> PreTrainedTokenizerFast:

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }

    if model_args.tokenizer_name:
        # load from hub
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    else:
        # load from local
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)

    return tokenizer


def load_model(model_args: ModelArguments) -> Tuple[PretrainedConfig, PreTrainedModel]:

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }

    if model_args.config_name:
        # load from hub
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    else:
        # load from local
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        attn_implementation=model_args.attn_implementation,
    )

    return config, model


class FIMDataCollator:
    def __init__(
        self,
        tokenizer,
        block_size,
        fim_rate,
        fim_spm_rate,
        prefix_tok_id,
        middle_tok_id,
        suffix_tok_id,
        pad_token_id,
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        np_rng=None,
        entropy=True,
        truncate_or_pad=True,
        window_size=None,
        window_size_min=None,
        window_size_random=False,
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.fim_rate = fim_rate
        self.fim_spm_rate = fim_spm_rate
        self.pad_token_id = pad_token_id
        self.device = device
        self.np_rng = np_rng if np_rng is not None else np.random.RandomState()
        self.entropy = entropy
        self.truncate_or_pad = truncate_or_pad

        self.prefix_tok_id = prefix_tok_id
        self.suffix_tok_id = suffix_tok_id
        self.middle_tok_id = middle_tok_id

        self.reference_model = None

        # Window size (in tokens) for the FIM middle span, used both by the
        # entropy-guided search (_find_best_middle_boundary) and the random
        # placement (_fim_transform_random). `window_size` is the fixed/target
        # size; when `window_size_min` is also given, `current_window_size`
        # starts there and is grown towards `window_size` over training by
        # WindowSizeCurriculumCallback. `window_size_random` bypasses both of
        # these and instead draws a fresh random size per example (still
        # confined to the middle_start/middle_end frame).
        self.window_size = window_size if window_size is not None else block_size // 2
        self.window_size_min = window_size_min
        self.window_size_random = window_size_random
        self.current_window_size = window_size_min if window_size_min is not None else self.window_size

    def set_model(self, model):
        self.reference_model = model.eval().to(self.device)

    def set_window_size(self, window_size: int):
        self.current_window_size = window_size

    def _middle_frame(self, seq_len):
        """The [middle_start, middle_end) range the FIM middle span is confined to."""
        middle_start = int(seq_len * 0.1)
        middle_end = seq_len - middle_start
        return middle_start, middle_end

    def _pick_window_size(self, middle_start, middle_end):
        """
        Size (in tokens) of the FIM middle span for one example: a fresh
        uniform-random size within the middle frame when window_size_random
        is set, otherwise the fixed/curriculum current_window_size (clamped
        to what fits in the frame).
        """
        if self.window_size_random:
            return self.np_rng.randint(low=1, high=middle_end - middle_start + 1)
        return max(1, min(self.current_window_size, middle_end - middle_start))

    def _fim_transform_random(self, input_ids):
        """
        Applies random FIM transformation: a window is placed at a random
        start position within the middle frame. Used when entropy=False;
        this is the random-placement analogue of _find_best_middle_boundary's
        entropy-guided window search (same window size/curriculum, no
        entropy signal to pick where it goes).
        """
        example = input_ids.copy()
        seq_len = len(example)
        middle_start, middle_end = self._middle_frame(seq_len)

        window_size = self._pick_window_size(middle_start, middle_end)
        start = self.np_rng.randint(low=middle_start, high=middle_end - window_size + 1)
        end = start + window_size

        prefix = example[:start]
        middle = example[start:end]
        suffix = example[end:]

        if self.truncate_or_pad:
            total_length = len(prefix) + len(middle) + len(suffix) + 3
            diff = total_length - self.block_size

            if diff > 0:
                suffix = suffix[: max(0, len(suffix) - diff)]
            else:
                suffix+= [self.pad_token_id] * (self.block_size - total_length)

        if self.np_rng.binomial(1, self.fim_spm_rate):
            # Apply Suffix-Prefix-Middle (SPM) transformation
            transformed = [self.prefix_tok_id, self.suffix_tok_id] + suffix + [self.middle_tok_id] + prefix + middle
        else:
            # Apply Prefix-Suffix-Middle (PSM) transformation
            transformed = [self.prefix_tok_id] + prefix + [self.suffix_tok_id] + suffix + [self.middle_tok_id] + middle

        if len(transformed) > self.block_size:
            transformed = transformed[:self.block_size]

        return transformed


    def _find_best_middle_boundary(self, entropy):
        seq_len = entropy.size(0)
        middle_start, middle_end = self._middle_frame(seq_len)

        window_size = self._pick_window_size(middle_start, middle_end)
        cumsum = torch.zeros(seq_len + 1, device=entropy.device)
        cumsum[1:] = torch.cumsum(entropy, dim=0)

        starts = torch.arange(middle_start, middle_end-window_size+1, device=entropy.device)
        ends = starts + window_size

        window_sums = cumsum[ends] - cumsum[starts]
        best_idx = torch.argmax(window_sums)

        best_start = starts[best_idx].item()
        best_end = ends[best_idx].item()

        return best_start, best_end

    def _apply_fim_boundaries(self, input_ids, best_start, best_end):
        """Helper to apply the actual FIM token injection given boundaries."""
        prefix = input_ids[:best_start]
        middle = input_ids[best_start:best_end]
        suffix = input_ids[best_end:]

        # Pad or truncate to block size
        total_length = len(prefix) + len(middle) + len(suffix) + 3
        diff = total_length - self.block_size

        if diff > 0:
            suffix = suffix[: max(0, len(suffix) - diff)]
        else:
            suffix += [self.pad_token_id] * (self.block_size - total_length)

        if self.np_rng.binomial(1, self.fim_spm_rate):
            transformed = [self.prefix_tok_id, self.suffix_tok_id] + suffix + [self.middle_tok_id] + prefix + middle
        else:
            transformed = [self.prefix_tok_id] + prefix + [self.suffix_tok_id] + suffix + [self.middle_tok_id] + middle

        if len(transformed) > self.block_size:
            transformed = transformed[:self.block_size]

        return transformed

    def __call__(self, examples):
        # 1. Extract raw token lists
        input_ids_list = [ex["input_ids"] for ex in examples]
        batch_size = len(input_ids_list)
        transformed_ids = []

        # 2. Vectorized Batch Forward Pass for Entropy
        if self.entropy:
            # Stack into a single batch tensor: [batch_size, seq_len]
            batch_tensor = torch.tensor(input_ids_list, dtype=torch.long, device=self.device)
            with torch.no_grad():
                outputs = self.reference_model(batch_tensor)

            if isinstance(outputs, dict):
                logits = outputs["logits"]
            elif hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs[0]

            # Calculate entropy for the ENTIRE BATCH at once
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)
            entropy_batch = -(log_probs * probs).sum(dim=-1) # Shape: [batch_size, seq_len]
        else:
            entropy_batch = [None] * batch_size

        # 3. Apply transformations using the pre-computed batched entropy
        for i, input_ids in enumerate(input_ids_list):
            # Skip FIM transformation based on fim_rate
            if not self.np_rng.binomial(1, self.fim_rate):
                example = input_ids.copy()
                if len(example) > self.block_size:
                    example = example[:self.block_size]
                transformed_ids.append(example)
                continue

            if self.entropy:
                # Use the pre-computed row for this specific sequence
                best_start, best_end = self._find_best_middle_boundary(entropy_batch[i])
                trans = self._apply_fim_boundaries(input_ids, best_start, best_end)
            else:
                trans = self._fim_transform_random(input_ids)

            transformed_ids.append(trans)

        # 4. Construct final dictionary
        input_ids_batch = torch.tensor(transformed_ids, dtype=torch.long)
        # Padding introduced by truncate_or_pad isn't a real prediction target,
        # so exclude it from the loss (-100 is CrossEntropyLoss's default ignore_index).
        labels_batch = input_ids_batch.clone()
        if self.truncate_or_pad and self.pad_token_id is not None:
            labels_batch[input_ids_batch == self.pad_token_id] = -100

        return {
            "input_ids": input_ids_batch,
            "labels": labels_batch,
            "attention_mask": torch.tensor([[1] * self.block_size for _ in transformed_ids], dtype=torch.long),
        }


class WindowSizeCurriculumCallback(TrainerCallback):
    """Linearly grows a FIMDataCollator's entropy-search window size from
    window_size_min to window_size_max over the full course of training."""

    def __init__(self, data_collator: FIMDataCollator, window_size_min: int, window_size_max: int):
        self.data_collator = data_collator
        self.window_size_min = window_size_min
        self.window_size_max = window_size_max

    def on_step_begin(self, args, state, control, **kwargs):
        progress = min(1.0, state.global_step / state.max_steps) if state.max_steps else 0.0
        window_size = round(self.window_size_min + progress * (self.window_size_max - self.window_size_min))
        self.data_collator.set_window_size(window_size)


def args_to_dict(args):
    return {
        k: v for k, v in asdict(args).items()
        if isinstance(v, (str, int, float, bool, type(None)))
    }


def main(config: dict, train_dataset_override=None) -> None:
    """Main training function.

    Args:
        config: Configuration dictionary loaded from YAML
        train_dataset_override: Optional pre-built train Dataset to use instead of
            looking up data.train_split on disk (see load_data)
    """

    from accelerate import Accelerator
    a = Accelerator()
    print("accelerate:", a.num_processes, a.device)
    print("device count:", torch.cuda.device_count())

    if not torch.cuda.is_available():
        print("CUDA not available; running in CPU test mode.")

    model_args, data_args, training_args = load_training_config(config)

    # `overwrite_output_dir` no longer exists on TrainingArguments (removed in
    # transformers 5.x), so we check it ourselves before looking for a checkpoint
    # to resume from. Defaults to True: fresh run every time unless set to false.
    overwrite_output_dir = config.get("training", {}).get("overwrite_output_dir", True)

    # ========== LOGGING ==========
    train_type = "entropy" if data_args.entropy else "baseline"
    run_name = f"{train_type}_{data_args.train_split}_lr{training_args.learning_rate}_fim{data_args.fim_rate}_ep{training_args.num_train_epochs}_s{training_args.seed}"

    # Optional free-form tag to tell comparison runs apart that would otherwise
    # share this exact run_name (e.g. a dataloader_num_workers or window_size
    # curriculum A/B test), appended to the model name used for saving/logging.
    run_tag = config.get("training", {}).get("run_tag")
    if run_tag:
        run_name = f"{run_name}_{run_tag}"

    os.environ["WANDB_PROJECT"] = "entropy-fim"
    os.environ["WANDB_RUN_NAME"] = run_name

    run = wandb.init(project="entropy-fim",
                     name=run_name,
                     config=config,
                     )

    # Give each run its own output subdirectory so parallel/sequential runs
    # with different configs never write checkpoints into the same place.
    training_args.output_dir = os.path.join(training_args.output_dir, run_name)
    send_telemetry(topic="run_baseline",
                   library_name="transformers",
                   library_version=transformers.__version__,
                   user_agent={
                       **args_to_dict(model_args),
                       **args_to_dict(data_args)})

    logger = setup_logging(model_args, data_args, training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)
    # Set a numpy random state for FIM transformations
    np_rng = np.random.RandomState(seed=training_args.seed)

    text_column_name = "code"

    tokenizer = load_tokenizer(model_args, data_args)
    config_model, model = load_model(model_args)

    ##  ---------- INIT NEW TOKEN EMBEDDINGS ----------

    # Add the new FIM tokens to the tokenizer and resize model's vocab embeddings
    special_tokens = [data_args.fim_prefix_token, data_args.fim_middle_token, data_args.fim_suffix_token]
    if data_args.truncate_or_pad:
        special_tokens.append(data_args.pad_token)

    pad_factor = 1
    if torch.cuda.is_available():
        pad_factor = 8

    # Add the new tokens to the tokenizer
    tokenizer.add_tokens(special_tokens)

    original_embeddings = model.get_input_embeddings()

    if is_deepspeed_zero3_enabled():
        import deepspeed

        with deepspeed.zero.GatheredParameters(original_embeddings.weight, modifier_rank=0):
            # Get the pre-expansion embeddings of the model and resize the embedding layer
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=pad_factor)
            embeddings = model.get_input_embeddings()

            # Sample the embeddings for the new tokens from a multivariate normal distribution
            # We do this so that the new embeddings are close to the original embeddings and not necessarily zero
            # More on this: https://nlp.stanford.edu/~johnhew/vocab-expansion.html
            mean = original_embeddings.mean(dim=0)
            n = original_embeddings.size()[0]
            sigma = ((original_embeddings - mean).T @ (original_embeddings - mean)) / n
            dist = torch.distributions.multivariate_normal.MultivariateNormal(
                mean,
                covariance_matrix=1e-5 * sigma,
            )
            new_token_embeddings = torch.stack(
                tuple((dist.sample() for _ in range(len(special_tokens)))),
                dim=0,
            )
    else:
        original_embeddings = model.get_input_embeddings().weight
        # Get the pre-expansion embeddings of the model and resize the embedding layer
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=pad_factor)
        embeddings = model.get_input_embeddings()

        # Sample the embeddings for the new tokens from a multivariate normal distribution
        # We do this so that the new embeddings are close to the original embeddings and not necessarily zero
        # More on this: https://nlp.stanford.edu/~johnhew/vocab-expansion.html

        # Cholesky (used internally by MultivariateNormal) has no CPU/bf16 kernel,
        # so do the covariance computation and sampling in float32, then cast back.
        original_embeddings_fp32 = original_embeddings.float()
        mean = original_embeddings_fp32.mean(dim=0)
        n = original_embeddings_fp32.size()[0]

        sigma = ((original_embeddings_fp32 - mean).T @ (original_embeddings_fp32 - mean)) / n
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            mean,
            covariance_matrix=1e-5 * sigma,
        )
        new_token_embeddings = torch.stack(
            tuple((dist.sample() for _ in range(len(special_tokens)))),
            dim=0,
        ).to(original_embeddings.dtype)

    if is_deepspeed_zero3_enabled():
        import deepspeed

        with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=0):
            # Set the new tokens' embeddings to the newly sampled embeddings
            embeddings.weight.data[-len(special_tokens) :] = new_token_embeddings
    else:
        # Set the new tokens' embeddings to the newly sampled embeddings
        embeddings.weight.data[-len(special_tokens) :] = new_token_embeddings

    # Update the model's embeddings with the new embeddings
    model.set_input_embeddings(embeddings)
    logger.info("Added special tokens to the tokenizer and resized model's embedding layer")

    ##  ---------- PREPARE DATA ----------

    raw_dataset = load_data(data_args, seed=training_args.seed, train_dataset_override=train_dataset_override)
    logger.info(f"Training on split {data_args.train_split}")

    if data_args.diagram_type_oversampling:
        # Must run before tokenization/group_texts: once texts are concatenated and
        # re-chunked into block_size pieces, a single example can span multiple
        # original rows, so our_diagram_type stops being attributable per example.
        raw_dataset["train"] = apply_diagram_type_oversampling(
            raw_dataset["train"], seed=training_args.seed, logger=logger
        )

    print("Dataset:", raw_dataset)

    # First we tokenize all the texts.

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    assert tokenizer.eos_token_id is not None, "tokenizer must define an eos_token to mark document boundaries"

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm-fim input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        # group_texts concatenates every example's input_ids back to back with no
        # separator of its own, so mark document boundaries here with EOS.
        output["input_ids"] = [ids + [tokenizer.eos_token_id] for ids in output["input_ids"]]
        output["attention_mask"] = [mask + [1] for mask in output["attention_mask"]]
        return output

    if training_args.do_train:
        column_names = list(raw_dataset["train"].features)
    else:
        column_names = list(raw_dataset["validation"].features)

    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_dataset.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > config_model.max_position_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, config_model.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            block_size = min(1024, config_model.max_position_embeddings)
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Get the FIM-specific token ids
    prefix_tok_id = tokenizer.convert_tokens_to_ids(data_args.fim_prefix_token)
    middle_tok_id = tokenizer.convert_tokens_to_ids(data_args.fim_middle_token)
    suffix_tok_id = tokenizer.convert_tokens_to_ids(data_args.fim_suffix_token)
    pad_tok_id = None

    # If truncate_or_pad is on, also get pad token id
    if data_args.truncate_or_pad:
        pad_tok_id = tokenizer.convert_tokens_to_ids(data_args.pad_token)

    # Data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with training_args.main_process_first(desc="processing texts together"):
        if not data_args.streaming:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            shift_labels = eval_preds.label_ids[:, 1:].reshape(-1)
            shift_preds = eval_preds.predictions[:, :-1].reshape(-1)
            accuracy = metric.compute(predictions=shift_preds, references=shift_labels)
            return {"causal_accuracy": accuracy["accuracy"]}


    ##  ---------- SET UP TRAINING ----------

    data_collator = FIMDataCollator(
        tokenizer=tokenizer,
        block_size=block_size,
        fim_rate=data_args.fim_rate,
        fim_spm_rate=data_args.fim_spm_rate,
        prefix_tok_id = prefix_tok_id,
        middle_tok_id = middle_tok_id,
        suffix_tok_id = suffix_tok_id,
        pad_token_id=pad_tok_id,
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        np_rng=np_rng,
        entropy=data_args.entropy,
        truncate_or_pad=data_args.truncate_or_pad,
        window_size=data_args.window_size,
        window_size_min=data_args.window_size_min,
        window_size_random=data_args.window_size_random,
    )

    # Set the reference model for FIM transformation (entropy-based middle boundary selection)
    data_collator.set_model(model)

    if data_args.window_size_random and data_args.window_size_min is not None:
        raise ValueError("data.window_size_random and data.window_size_min are mutually exclusive")

    callbacks = []
    if data_args.window_size_min is not None:
        if data_args.window_size_min > data_collator.window_size:
            raise ValueError(
                f"data.window_size_min ({data_args.window_size_min}) must be <= data.window_size "
                f"({data_collator.window_size})"
            )
        callbacks.append(WindowSizeCurriculumCallback(
            data_collator=data_collator,
            window_size_min=data_args.window_size_min,
            window_size_max=data_collator.window_size,
        ))

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval  else None,
        preprocess_logits_for_metrics=(
            preprocess_logits_for_metrics if training_args.do_eval
             else None
        ),
        callbacks=callbacks or None,
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Set training.overwrite_output_dir: true in the config to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `output_dir` or set training.overwrite_output_dir: true in the config to train from scratch."
            )

    ##  ---------- TRAIN ----------

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    pass
