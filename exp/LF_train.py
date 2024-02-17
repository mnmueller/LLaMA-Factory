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
from llmtuner.hparams.parser import _verify_model_args, _set_transformers_logging, _parse_train_args
from llmtuner.data import get_dataset, split_dataset
from llmtuner.model import load_model_and_tokenizer

from llmtuner.hparams.data_args import DataArguments
from llmtuner.hparams.finetuning_args import FinetuningArguments
from llmtuner.hparams.generating_args import GeneratingArguments
from llmtuner.hparams.model_args import ModelArguments


logger = get_logger(__name__)

_TRAIN_ARGS = [ModelArguments, DataArguments, Seq2SeqTrainingArguments, FinetuningArguments, GeneratingArguments]

def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: Optional[List["TrainerCallback"]] = None):
    model_args, data_args, training_args, finetuning_args, generating_args = _parse_train_args(args)

    # Setup logging
    if training_args.should_log:
        _set_transformers_logging()

    _verify_model_args(model_args, finetuning_args)

    # postprocess training_args
    if (
        training_args.local_rank != -1
        and training_args.ddp_find_unused_parameters is None
        and finetuning_args.finetuning_type == "lora"
    ):
        print("Ping 1")
        logger.warning("`ddp_find_unused_parameters` needs to be set as False for LoRA in DDP training.")
        training_args_dict = training_args.to_dict()
        training_args_dict.update(dict(ddp_find_unused_parameters=False))
        training_args = Seq2SeqTrainingArguments(**training_args_dict)

    can_resume_from_checkpoint = True

    if (
        training_args.resume_from_checkpoint is None
        and training_args.do_train
        and os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
        and can_resume_from_checkpoint
    ):
        print("Ping 2")
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError("Output directory already exists and is not empty. Please set `overwrite_output_dir`.")

        if last_checkpoint is not None:
            training_args_dict = training_args.to_dict()
            training_args_dict.update(dict(resume_from_checkpoint=last_checkpoint))
            training_args = Seq2SeqTrainingArguments(**training_args_dict)
            logger.info(
                "Resuming training from {}. Change `output_dir` or use `overwrite_output_dir` to avoid.".format(
                    training_args.resume_from_checkpoint
                )
            )

    # postprocess model_args
    model_args.compute_dtype = (
        torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)
    )
    model_args.model_max_length = data_args.cutoff_len

    # Log on each process the small summary:
    logger.info(
        "Process rank: {}, device: {}, n_gpu: {}\n  distributed training: {}, compute dtype: {}".format(
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            bool(training_args.local_rank != -1),
            str(model_args.compute_dtype),
        )
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    model, tokenizer = load_model_and_tokenizer()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

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
