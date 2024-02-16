import os
import sys
import torch
import tqdm
import numpy as np
import pathlib
from pathlib import Path
import argparse
import joblib

from transformers.integrations import is_deepspeed_zero3_enabled

from torch.utils.data import Dataset
from transformers import (
    LlamaForCausalLM,
    MistralForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    AutoModelForCausalLM,
    AutoConfig
)

from datasets import DatasetDict

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

print("Hello world")
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())

os.environ["WANDB_PROJECT"] = "bg-llama"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="/home/markmueller",
        help="Root directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/phi-2",
        help="HuggingFace model to load",
    )
    parser.add_argument(
        "--context_length", type=int, default=2048, help="Context length"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Global batch size.",
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=2,
        help="Batch size per GPU. Try to maximize this.",
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=1,
        help="Log training stats every n steps.",
    )
    parser.add_argument(
        "--save_steps",
        type=float,
        default=500,
        help="Save the checkpoint every n steps.",
    )
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default="zero_stage2_config.json",
        help="DeepSpeed config file.",
    )
    parser.add_argument(
        "--flash_attn",
        action="store_true"
    )
    parser.add_argument(
        "--bf16",
        action="store_true"
    )
    parser.add_argument(
        "--fp16",
        action="store_true"
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_torch",  # not wasting resources on adamW
        help="Adam-like optimizer",
    )
    args = parser.parse_args()

    assert not (args.bf16 and args.fp16), "Can only enable either bf16 or fp16 not both."

    return args


def memmap_iterator(data: np.memmap, full_length: int, context_length: int):
    end_index = int(full_length / context_length)
    for idx in range(end_index):
        yield {"input_ids": data[idx * context_length: (idx + 1) * context_length]}


def sequence_packing(
        data_folder: str,
        context_length: int,
):
    ds_dict = DatasetDict()
    num_proc = max(joblib.cpu_count() - 4, 1)

    for split_file_path in [x for x in os.listdir(data_folder) if x.endswith("bin")]:
        split_file_name = split_file_path.split("/")[-1]
        split_name = split_file_name.split("_")[0]

        if int(split_file_name.split(".")[0][-2:]) == 16:
            dtype = np.uint16
        elif int(split_file_name.split(".")[0][-2:]) == 32:
            dtype = np.uint32
        else:
            assert False, "could not read dataset"

        data_split = np.memmap(os.path.join(data_folder, split_file_path), dtype=dtype, mode="r")
        arr_len = len(data_split)

        ds = Dataset.from_generator(
            memmap_iterator,
            gen_kwargs={
                "data": data_split,
                "full_length": arr_len,
                "context_length": context_length,
            },
            cache_dir= os.path.join(data_folder, ".cache"),
            num_proc=num_proc,
        )
        ds_dict[split_name] = ds

    return ds_dict

def load_data(dataset_path: Path, context_length):
    ds_dict = sequence_packing(dataset_path, context_length)

    return ds_dict["train"], ds_dict["val"]

def train(
    root_dir: str,
    model_name: str,
    context_length: int,
    deepspeed_config: str,
    optim: str,
    num_gpus: int,
    batch_size: int = 512,
    per_gpu_batch_size: int = 4,
    log_steps: int = 1,
    save_steps: int = 500,
    compute_dtype: torch.dtype = torch.bfloat16,
    use_flash_attention_2: bool = True
):
    SCRATCH_PATH = pathlib.Path(root_dir)
    DATA_ROOT = SCRATCH_PATH / "bg_data"
    MODELS_PATH = SCRATCH_PATH / "models"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.unk_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    train_dataset, val_dataset = load_data(DATA_ROOT, context_length)

    print(f"Zero3 enabled: {is_deepspeed_zero3_enabled()}")
    config = AutoConfig.from_pretrained(model_name)

    print(config)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=compute_dtype,
        use_flash_attention_2 = use_flash_attention_2,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
    )

    model_dir = MODELS_PATH / (model_name + "-debug")
    args = TrainingArguments(
        output_dir=str(model_dir),
        per_device_train_batch_size=per_gpu_batch_size,
        per_device_eval_batch_size=per_gpu_batch_size,
        evaluation_strategy="steps",
        logging_steps=log_steps,
        gradient_accumulation_steps=int(batch_size / (per_gpu_batch_size * num_gpus)),
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.05,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        optim=optim,
        num_train_epochs=1,
        save_steps=save_steps,
        bf16=compute_dtype==torch.bfloat16,
        fp16=compute_dtype==torch.float16,
        deepspeed=deepspeed_config,
        report_to="wandb",
        run_name=model_name + "-test",  # name of the W&B run (optional)
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    args = parse_arguments()
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    train(
        root_dir=args.root,
        model_name=args.model,
        context_length=args.context_length,
        deepspeed_config=args.deepspeed_config,
        optim=args.optim,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        per_gpu_batch_size=args.per_gpu_batch_size,
        log_steps=args.log_steps,
        save_steps=args.save_steps,
        compute_dtype=dtype,
        use_flash_attention_2=args.flash_attn
    )
