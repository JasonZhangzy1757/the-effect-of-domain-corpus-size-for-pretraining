import datetime
import json
import math
import pathlib
import random
import string
from typing import Dict
import re
import deepspeed
import fire
import loguru
import numpy as np
import pytz
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import BertModel, BertForMaskedLM, BertForPreTraining
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
    set_seed
)

logger = loguru.logger


def get_unique_identifier(length: int = 8) -> str:
    """Create a unique identifier by choosing `length`
    random characters from list of ascii characters and numbers
    """
    alphabet = string.ascii_lowercase + string.digits
    uuid = "".join(alphabet[ix] for ix in np.random.choice(len(alphabet), length))
    return uuid


def create_experiment_dir(
        checkpoint_dir: pathlib.Path, all_arguments: Dict
) -> pathlib.Path:
    """ Create an experiment directory and save all arguments in it."""
    current_time = datetime.datetime.now(pytz.timezone("US/Pacific"))
    expname = f"bert_pretrain.{current_time.year}.{current_time.month}.{current_time.day}.{current_time.hour}.{current_time.minute}.{current_time.second}.{get_unique_identifier()}"
    exp_dir = checkpoint_dir / expname
    exp_dir.mkdir(exist_ok=False)
    hparams_file = exp_dir / "hparams.json"
    with hparams_file.open("w") as handle:
        json.dump(obj=all_arguments, fp=handle, indent=2)

    # Create the Tensorboard Dir
    tb_dir = exp_dir / "tb_dir"
    tb_dir.mkdir()
    return exp_dir


def main(
    checkpoint_dir: str = None,
    load_checkpoint_dir: str = None,
    # Dataset Params
    train_file: str = None,
    validation_file: str = None,
    mask_prob: float = 0.15,
    # Model Params
    model_name_or_path: str = None,
    # Training Params
    epoch: int = 3,
    batch_size: int = 8,
    checkpoint_every: int = 1000,
    weight_decay: float = 0.0,
    learning_rate: float = 5e-5,
    gradient_accumulation_steps: int = 1,
    lr_scheduler_type: SchedulerType = "linear",
    num_warmup_steps: int = 0,
    log_every: int = 10,
    seed=666,
    local_rank: int = -1
):
    """ Train a BERT style MLM model """
    # local_rank will be automatically decided, don't need us to manually specify
    device = (
        torch.device("cuda", local_rank)
        if (local_rank > -1) and torch.cuda.is_available()
        else torch.device("cpu")
    )
    logger.info(f"local_rank: {local_rank}")

    if checkpoint_dir is None and load_checkpoint_dir is None:
        logger.error("Need to specify one of checkpoint_dir or load_checkpoint_dir")
        return
    if checkpoint_dir is not None and load_checkpoint_dir is not None:
        logger.error("Cannot specify both checkpoint_dir and load_checkpoint_dir")
        return

    if checkpoint_dir:
        logger.info("Creating Experiment Directory")
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        all_arguments = {
            # Dataset Params
            "train_file": train_file,
            "validation_file": validation_file,
            "mask_prob": mask_prob,
            # Training Params
            "batch_size": batch_size,
            "checkpoint_every": checkpoint_every,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "lr_scheduler_type": lr_scheduler_type,
            "num_warmup_steps": num_warmup_steps,
            "seed": seed
        }
        exp_dir = create_experiment_dir(checkpoint_dir, all_arguments)
        logger.info(f"Experiment Directory created at {exp_dir}")
    else:
        logger.info("Loading from Experiment Directory")
        load_checkpoint_dir = pathlib.Path(load_checkpoint_dir)
        assert load_checkpoint_dir.exists()
        with (load_checkpoint_dir / "hparams.json").open("r") as handle:
            hparams = json.load(handle)
        # Set the hparams
        # Dataset Params
        train_file = hparams.get("train_file", train_file)
        validation_file = hparams.get("validation_file", validation_file)
        mask_prob = hparams.get("mask_prob", mask_prob)
        # Training Params
        epoch = hparams.get("epoch", batch_size)
        batch_size = hparams.get("batch_size", batch_size)
        checkpoint_every = hparams.get("checkpoint_every", checkpoint_every)
        learning_rate = hparams.get("learning_rate", learning_rate)
        weight_decay = hparams.get("weight_decay", weight_decay)
        gradient_accumulation_steps = hparams.get("gradient_accumulation_steps", gradient_accumulation_steps)
        lr_scheduler_type = hparams.get("lr_scheduler_type", lr_scheduler_type)
        num_warmup_steps = hparams.get("num_warmup_steps", num_warmup_steps)
        seed = hparams.get("seed", seed)
        exp_dir = load_checkpoint_dir

    # Tensorboard writer
    tb_dir = exp_dir / "tb_dir"
    assert tb_dir.exists()
    summary_writer = SummaryWriter(log_dir=tb_dir)

    set_seed(seed)

    ######### Create Dataset #########
    data_files = {}
    if train_file is None or validation_file is None:
        logger.error("Need to specify both train_file and val_file")
        return
    if (not train_file.endswith(".json")) or (not validation_file.endswith(".json")):
        logger.error("train_file and val_file should all be json files")

    data_files["train"] = train_file
    data_files["validation"] = validation_file

    raw_datasets = load_dataset("json", data_files=data_files)

    if model_name_or_path is None:
        logger.error("You need to specify model_name_or_path")
        return

    model = BertForMaskedLM.from_pretrained(
        model_name_or_path,
        from_tf=False,
        config=AutoConfig.from_pretrained(model_name_or_path)
    )
    # tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    tokenizer = BertTokenizer.from_pretrained("wp-vocab-30500-vocab.txt")
    max_seq_length = tokenizer.model_max_length

    def tokenize_function(examples):
        # Remove empty lines
        examples["text"] = [
            line for line in examples["text"] if len(line) > 0 and not line.isspace()
        ]
        return tokenizer(
            examples["text"],
            padding=False,  # We use dynamic padding
            truncation=True,
            max_length=max_seq_length,
            # We use this option because DataCollatorForLanguageModeling (see below)
            # is more efficient when it receives the `special_tokens_mask`.
            return_special_tokens_mask=True
        )

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        remove_columns=["text"],
        load_from_cache_file=True,
        desc="Running tokenizer on dataset line by line"
    )

    train_dataset = tokenized_datasets["train"]

    # Log a few random samples from the training set
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}")

    # Data collator
    # This one will take care of randomly masking the tokens
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=mask_prob
    )

    # DataLoaders creation
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    # Prepare everything using DeepSpeed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = epoch * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    ####### DeepSpeed Engine ########
    logger.info("Creating DeepSpeed Engine")
    ds_config = {
        "train_micro_batch_size_per_gpu": batch_size,
        "fp16": {
            "enabled": True
        },
        "zero_allow_untested_optimizer": True,
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu"
            }
        }
    }
    # model.to(device) can be removed, DeepSpeed will take care for us
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters(),
        lr_scheduler=lr_scheduler,
        config=ds_config
    )
    logger.info("DeepSpeed Engine Created")

    # Train!
    start_step = 1
    epoch_trained = 0
    ####### Load Checkpoint ############
    if load_checkpoint_dir is not None:
        _, client_state = model.load_checkpoint(load_dir=load_checkpoint_dir)
        checkpoint_step = client_state["checkpoint_step"]
        start_step = checkpoint_step + 1
        epoch_trained = checkpoint_step // num_update_steps_per_epoch

    progress_bar = tqdm(range(max_train_steps), disable=(local_rank != 0))
    losses = []
    for e in range(epoch_trained, epoch):
        model.train()
        for step, batch in enumerate(train_dataloader):
            curr_step = e * num_update_steps_per_epoch + step
            if curr_step >= start_step:
                # Forward pass
                for k, v in batch.items():
                    batch[k] = v.to(device)  # Move device
                outputs = model(**batch)
                loss = outputs.loss
                # Backward pass
                model.backward(loss)
                if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    # Optimizer Step
                    model.step()
                    progress_bar.update(1)

                losses.append(loss.item())
                if step % log_every == 0:
                    logger.info("Loss: {0:.4f}".format(np.mean(losses)))
                    summary_writer.add_scalar(f"Train/loss", np.mean(losses), step)
                if step % checkpoint_every == 0:
                    model.save_checkpoint(save_dir=exp_dir, client_state={"checkpoint_step": step})
                    logger.info(f"Saved model to {exp_dir}")

    # Save the last checkpoint if not saved yet
    if step % checkpoint_every != 0:
        model.save_checkpoint(save_dir=exp_dir, client_state={"checkpoint_step": step})
        logger.info(f"Saved model to {exp_dir}")


if __name__ == "__main__":
    fire.Fire(main)

"""
deepspeed --include localhost:1,2 bert_train.py --checkpoint_dir bert_pretrain --model_name_or_path bert-base-uncased --train_file train.json --validation_file val.json --batch_size 128
"""