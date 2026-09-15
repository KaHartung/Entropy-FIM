from typing import Optional
from dataclasses import dataclass, field

from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING
)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str | None = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    pad_to_multiple_of: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad the embedding layer to a multiple depending on the device. ",
                "For NVIDIA GPUs, this will be a multiple of 8, for TPUs a multiple of 128.",
            )
        },
    )
    attn_implementation: Optional[str] = field(
        default="sdpa", metadata={"help": ("The attention implementation to use. ")}
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_split: str = field(
        default="train",
        metadata={
            "help": (
                "Name of the split within the on-disk dataset to use as the training set. "
                "Lets you keep several training variants as separate splits of the same "
                "saved dataset and pick between them via config."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    fim_rate: Optional[float] = field(
        default=0.5,
        metadata={
            "help": (
                "Optional probability with which the FIM transformation is applied to the example. "
                "Default is 0.5. A rate of 1.0 means every example will undergo FIM transformation, "
                "while a rate of 0.0 means no example will."
            )
        },
    )
    fim_spm_rate: Optional[float] = field(
        default=0.5,
        metadata={
            "help": (
                "Within the examples undergoing FIM transformation, this rate determines the probability "
                "of applying the Sentence Permutation Mode (SPM). "
                "Default is 0.5. A rate of 1.0 means all FIM transformations will use SPM, "
                "while a rate of 0.0 means none will."
            )
        },
    )
    truncate_or_pad: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "Indicates whether the transformed example should be truncated or padded to maintain "
                "the same length as the original example. "
                "Default is True. If False, the function will not truncate or pad the examples."
            )
        },
    )
    fim_prefix_token: Optional[str] = field(
        default="<fim_prefix>",
        metadata={"help": ("Fill-in-Middle Prefix token. Defaults to '<fim_prefix>'.")},
    )
    fim_middle_token: Optional[str] = field(
        default="<fim_middle>",
        metadata={"help": ("Fill-in-Middle Middle token. Defaults to '<fim_middle>'.")},
    )
    fim_suffix_token: Optional[str] = field(
        default="<fim_suffix>",
        metadata={"help": ("Fill-in-Middle Suffix token. Defaults to '<fim_suffix>'.")},
    )
    pad_token: Optional[str] = field(
        default="<fim_pad>",
        metadata={
            "help": (
                "Fill-in-Middle Pad token. Used only when 'truncate_or_pad' is set to True. Defaults to '<fim_pad>'."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    entropy: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to use entropy-based middle boundary selection (True) or random boundaries (False). "
                "Default is True. Set to False to use random FIM transformation."
            )
        },
    )
    window_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Size (in tokens) of the FIM middle span: the window searched for the "
                "highest-entropy span when entropy=True, or the window placed at a random "
                "start position when entropy=False. Defaults to block_size // 2 if unset. If "
                "window_size_min is also set, this is the size the window grows to by the end "
                "of training instead of a fixed value."
            )
        },
    )
    window_size_min: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Starting window size (in tokens) for a curriculum that linearly grows the FIM "
                "window from this value up to window_size over the course of training (applies "
                "in both entropy=True and entropy=False modes). "
                "Leave unset to use window_size as a fixed value throughout training. Mutually "
                "exclusive with window_size_random."
            )
        },
    )
    window_size_random: bool = field(
        default=False,
        metadata={
            "help": (
                "If set, draws a fresh uniform-random window size per example instead of using "
                "window_size/window_size_min - the size varies, but is still confined to the "
                "same [0.1, 0.9] * seq_len middle frame the fixed-size search uses. Applies in "
                "both entropy=True and entropy=False modes. Mutually exclusive with "
                "window_size_min (which drives a curriculum over a fixed size)."
            )
        },
    )
    diagram_type_oversampling: bool = field(
        default=False,
        metadata={
            "help": (
                "If set, resample the train split (before tokenization) so diagram-type groups "
                "(class+object, sequence, use_case+deployment+component, activity, state) are "
                "represented in proportion to sqrt(natural_count) instead of natural_count - "
                "small groups oversampled, large groups undersampled, combined size unchanged. "
                "Rows whose our_diagram_type isn't one of those (mixed, unknown, or missing) are "
                "left at their natural count. Requires the dataset to have an our_diagram_type "
                "column populated (see data_pipeline/build_strict_dataset.py's train_strict_labeled "
                "output); see training/oversampling.py."
            )
        },
    )
