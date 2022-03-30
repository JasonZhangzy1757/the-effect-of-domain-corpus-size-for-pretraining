#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertForMaskedLM, BertConfig, AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
import tokenizers
from tokenizers import BertWordPieceTokenizer
from tokenizers.processors import TemplateProcessing
from typing import List
import deepspeed
import loguru
import nvgpu
import os, time
from tqdm import tqdm
import json

logger = loguru.logger

for gpu in nvgpu.gpu_info():
    logger.info(gpu)
    
local_rank = 0
device = (
        torch.device("cuda", local_rank)
        if (local_rank > -1) and torch.cuda.is_available()
        else torch.device("cpu")
    )
tokenizer_path = '/home/americanthinker/notebooks/pytorch/NationalSecurityBERT/Preprocessing/Tokenization/wp-vocab-30500-vocab.txt'
text_data_path = '/home/americanthinker/notebooks/pytorch/NationalSecurityBERT/Data/full_docs/english_docs_aa.txt'

#files = [f for f in os.listdir(text_data_path) if os.path.isfile(os.path.join(text_data_path, f))]
logger.info(text_data_path)

# #### Instantiate pretrained tokenizer from file
alternative_tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

tokenizer = BertWordPieceTokenizer(tokenizer_path, strip_accents=True, lowercase=True)
tokenizer.enable_truncation(max_length=512)
tokenizer.enable_padding()
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ("[MASK]", tokenizer.token_to_id("[MASK]"))
    ],
)



def load_data_seq_512(path: str, sample_size:int=None) -> List[str]:
    with open(path) as f:
        if sample_size:
            lines = [line.strip() for line in f.readlines()[:sample_size]]
        else:
            lines = [line.strip() for line in f.readlines()]
    
    return lines

def mlm_pipe(batch: List[tokenizers.Encoding], mlm_prob=0.15) -> dict:
    '''
    Given a single instance from a batch of encodings, return masked inputs and associated arrays.
    Converts tokenizer.Encoding into a pytorch tensor.
    '''
    
    labels = torch.tensor([x.ids for x in tqdm(batch, 'Labels')])
    mask = torch.tensor([x.attention_mask for x in tqdm(batch, 'Attention Mask')])
    input_ids = labels.detach().clone()
    
    #default masking prob = 15%, don't mask special tokens 
    rand = torch.rand(input_ids.shape)
    mask_arr = (rand < mlm_prob) * (input_ids > 4)
    for i in tqdm(range(input_ids.shape[0]), 'Masking Words'):
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        input_ids[i, selection] = 4
        
    # temp = input_ids.flatten()
    # percent = sum(temp == 4)/sum(labels.flatten() != 4)
    # print(percent)
    encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}
    return encodings


logger.info('Loading data from disk into memory...')
start = time.perf_counter()
results = load_data_seq_512(text_data_path)
end = time.perf_counter() - start
logger.info(f'Loading completed: {round(end, 2)} seconds to load {len(results)} lines/documents.')

logger.info('Batch encoding data...')
s = time.perf_counter()
batch = tokenizer.encode_batch(results)
e = time.perf_counter() - s
logger.info(f'Batch encoding completed, took {round(e,2)} seconds to complete')

del results

logger.info('Masking tokens')
encodings = mlm_pipe(batch)
logger.info('Masking tokens completed')
#encodings = torch.load(encodings_data_path)

del batch

percent = sum(sum(encodings['input_ids'].detach().numpy() == 4)) / sum(sum(encodings['labels'].detach().numpy() != 4))
logger.info(f'Total of {round(percent * 100,2)}% of tokens are masked.')


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        
    def __len__(self):
        return self.encodings['input_ids'].shape[0]
    
    def __getitem__(self, i):
        return {key : tensor[i] for key, tensor in self.encodings.items()}


d = Dataset(encodings)
BATCH_SIZE = 24
loader = torch.utils.data.DataLoader(d, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True)


with open('zero2_config.json') as f:
    ds_config = json.loads(f.read())

bert_config = BertConfig(vocab_size=30500)
model = BertForMaskedLM(config=bert_config)


# Define Trainer
args = TrainingArguments(
    output_dir="/home/americanthinker/notebooks/pytorch/NationalSecurityBERT/Modeling/checkpoints/",
    do_train=True,
    do_eval=False,
    num_train_epochs=36,
    per_device_train_batch_size=12,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="epoch",
    seed=0,
    local_rank=0,
    dataloader_num_workers=0,
    load_best_model_at_end=False,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=d,
)

trainer.train()
