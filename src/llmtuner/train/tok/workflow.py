# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/language-modeling/run_clm.py

from typing import TYPE_CHECKING, List, Optional

from transformers import DataCollatorForLanguageModeling, Trainer

from ...data import get_dataset, split_dataset
from ...model.loader import load_tokenizer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments

    from ...hparams import DataArguments, ModelArguments

def run_tok(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: "str",
    ):
    model, tokenizer = load_tokenizer(model_args)
    get_dataset(tokenizer, model_args, data_args, training_args, stage=stage)
    DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
