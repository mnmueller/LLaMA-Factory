from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from transformers import PreTrainedModel
from transformers import DataCollatorForLanguageModeling, Trainer
import sys

sys.path.insert(0, "../src")


from llmtuner.extras.callbacks import LogCallback
from llmtuner.hparams import get_infer_args, get_train_args
from llmtuner.data import get_dataset, split_dataset
from llmtuner.model import load_model_and_tokenizer


def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: Optional[List["TrainerCallback"]] = None):
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    callbacks = [LogCallback()] if callbacks is None else callbacks

    run_pt(model_args, data_args, training_args, finetuning_args, callbacks)

def run_pt(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train)
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
