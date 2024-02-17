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
    model, tokenizer = load_model_and_tokenizer()

    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    callbacks = [LogCallback()] if callbacks is None else callbacks

    run_pt(model_args, data_args, training_args, finetuning_args, model, tokenizer, callbacks)


def load_model_and_tokenizer(
    model_name_or_path="microsoft/phi-2",
    compute_dtype=torch.bfloat16,
) -> Tuple["PreTrainedModel", "PreTrainedTokenizer"]:
    r"""
    Loads pretrained model and tokenizer.

    Support both training and inference.
    """

    # try_download_model_from_ms(model_args)

    config_kwargs = {}
    #     "trust_remote_code": True,
    #     "cache_dir": model_args.cache_dir,
    #     "revision": model_args.model_revision,
    #     "token": model_args.hf_hub_token,
    # }

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
        split_special_tokens=False,
        padding_side="right",
        **config_kwargs,
    )
    # patch_tokenizer(tokenizer)

    config = AutoConfig.from_pretrained(model_name_or_path, **config_kwargs)
    # patch_config(config, tokenizer, model_args, config_kwargs, is_trainable)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
        **config_kwargs,
    )

    # patch_model(model, tokenizer, model_args, is_trainable)
    # register_autoclass(config, model, tokenizer)
    #
    # model = init_adapter(model, model_args, finetuning_args, is_trainable)

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
