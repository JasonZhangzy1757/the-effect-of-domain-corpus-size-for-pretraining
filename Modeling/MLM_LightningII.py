#!/usr/bin/env python
# coding: utf-8


import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import sys

# #standard imports
import time, datetime, pytz
from typing import List, Union, Dict
from pathlib import Path
import pathlib
import os, json, sys
import numpy as np
from math import floor
import loguru

# #distributed imports
import nvgpu 

# #tokenizers and datasets
from tokenizers import BertWordPieceTokenizer 
from tokenizers.processors import TemplateProcessing
import tokenizers

# #transformer imports
import transformers
# from transformers import BertTokenizer
# from transformers import BertForMaskedLM, BertConfig, AdamW
from pytorch_lightning.loggers import CSVLogger

##########################################################
# Get Info & Define logging functions
##########################################################

model_logger = CSVLogger("mlm_logsII", name='my_exp')
logger = loguru.logger

for gpu in nvgpu.gpu_info():
    logger.info(gpu)
    
local_rank = 0
device = (
        torch.device("cuda", local_rank)
        if (local_rank > -1) and torch.cuda.is_available()
        else torch.device("cpu")
    )
##########################################################
# Configure Model 
##########################################################
bert_config = transformers.BertConfig(vocab_size=30500)

class MLMBert(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.bert = transformers.BertForMaskedLM(bert_config)
    
    def forward(self, ids, mask, labels):
        return self.bert(ids, mask, labels)
    
    def training_step(self, batch):
  
        # pull all tensor batches required for training
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        # process
        outputs = self.forward(ids=input_ids, 
                               mask=attention_mask,
                               labels=labels)
        # extract loss
        loss = outputs.loss
        logger.info(loss)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        return optimizer

logger.info('Creating model')
model = MLMBert()
logger.info('Model creation completed')

##########################################################
# Get Tokenizer & Create Dataset & DataLoader
##########################################################

# Instantiate pretrained tokenizer from file
tokenizer_path = '/home/americanthinker/notebooks/pytorch/NationalSecurityBERT/Preprocessing/Tokenization/wp-vocab-30500-vocab.txt'
tokenizer = transformers.BertTokenizer.from_pretrained(tokenizer_path, model_max_length=512)

# Get encodings from disk
data_path = '/home/americanthinker/notebooks/pytorch/NationalSecurityBERT/Preprocessing/Encodings/encodings_0_5000.pt'
encodings = torch.load(data_path)
# check to ensure we've got 15% of tokens masked
percent = sum(sum(encodings['input_ids'].detach().numpy() == 4)) / sum(sum(encodings['labels'].detach().numpy() != 4))
logger.info(f'Total of {round(percent * 100,2)}% of tokens are masked.')


class MLMDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        
    def __len__(self):
        return self.encodings['input_ids'].shape[0]
    
    def __getitem__(self, i):
        return {key : tensor[i] for key, tensor in self.encodings.items()}

logger.info('Loading dataset...')
dataset = MLMDataset(encodings)
logger.info('Dataset completed.')

BATCH_SIZE = 32
EPOCHS = 2
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=12)

#remove encodings from memory
del encodings

logger.info(f'Total of {len(train_loader)} batches.')

##########################################################
# Instantiate Trainer
##########################################################
trainer = pl.Trainer(logger=model_logger, 
                     default_root_dir='./checkpoints/',
                     accelerator="gpu",
                     devices=2, 
                     strategy=DDPStrategy(find_unused_parameters=False))
                     
trainer.fit(model=model, train_dataloaders=train_loader)
