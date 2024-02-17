from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedModel
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers import DataCollatorForLanguageModeling, Trainer, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import sys

sys.path.insert(0, "../src")


from llmtuner.extras.callbacks import LogCallback
from llmtuner.hparams import get_infer_args, get_train_args
from llmtuner.data import get_dataset, split_dataset
from llmtuner.model import load_model_and_tokenizer


def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: Optional[List["TrainerCallback"]] = None):
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    model, tokenizer = load_model_and_tokenizer()

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

    # config = AutoConfig.from_pretrained(model_name_or_path)

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        # config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
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
