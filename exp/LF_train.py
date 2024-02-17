from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import os
import torch
from transformers import PreTrainedModel, set_seed, Seq2SeqTrainingArguments, HfArgumentParser
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers import DataCollatorForLanguageModeling, Trainer, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import sys
from transformers.trainer_utils import get_last_checkpoint

sys.path.insert(0, "../src")


from llmtuner.extras.logging import get_logger
from llmtuner.extras.callbacks import LogCallback
from llmtuner.hparams import get_infer_args, get_train_args
from llmtuner.hparams.parser import _verify_model_args, _set_transformers_logging, _parse_train_args, _parse_args
from llmtuner.data import get_dataset, split_dataset
from llmtuner.model import load_model_and_tokenizer

from llmtuner.hparams.data_args import DataArguments
from llmtuner.hparams.finetuning_args import FinetuningArguments
from llmtuner.hparams.generating_args import GeneratingArguments
from llmtuner.hparams.model_args import ModelArguments


logger = get_logger(__name__)

_TRAIN_ARGS = [ModelArguments, DataArguments, Seq2SeqTrainingArguments, FinetuningArguments, GeneratingArguments]

def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: Optional[List["TrainerCallback"]] = None):
    parser = HfArgumentParser(_TRAIN_ARGS)

    namespace, remaining_args = parser.parse_known_args(sys.argv[1:])

    for dtype in parser.dataclass_types:
        keys = {f.name for f in parser.dataclasses.fields(dtype) if f.init}

    model, tokenizer = load_model_and_tokenizer()

    model_args, data_args, training_args, finetuning_args, generating_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)


    _verify_model_args(model_args, finetuning_args)

    # Setup logging
    if training_args.should_log:
        _set_transformers_logging()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    callbacks = [LogCallback()] if callbacks is None else callbacks

    run_pt(model_args, data_args, training_args, finetuning_args, model, tokenizer, callbacks)


def load_model_and_tokenizer(
) -> Tuple["PreTrainedModel", "PreTrainedTokenizer"]:
    r"""
    Loads pretrained model and tokenizer.

    Support both training and inference.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/phi-2",
        use_fast=False,
        split_special_tokens=False,
        padding_side="right",
    )

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )

    model.train()

    return model, tokenizer


def run_pt(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    model,
    tokenizer,
    callbacks: Optional[List["TrainerCallback"]] = None,

):
    dataset = get_dataset(tokenizer, model_args, data_args, training_args, stage="pt")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        **split_dataset(dataset, data_args, training_args),
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)


if __name__ == "__main__":
    run_exp()
