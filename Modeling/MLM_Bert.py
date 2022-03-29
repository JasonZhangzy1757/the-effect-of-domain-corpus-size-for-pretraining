#!/usr/bin/env python
# coding: utf-8

#standard imports
import time, datetime, pytz
from tqdm.auto import tqdm
from typing import List, Union, Dict
from re_sent_splitter import split_into_sentences
from pathlib import Path
import pathlib
import os, json, sys
import numpy as np
from math import floor
import loguru
from IPython import get_ipython

#distributed imports
import torch
from torch.nn.parallel import DataParallel
import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig
import nvgpu 

#tokenizers and datasets
from tokenizers import BertWordPieceTokenizer 
from tokenizers.processors import TemplateProcessing
import tokenizers

#transformer imports
from transformers import BertTokenizer
from transformers import BertForMaskedLM, BertConfig, AdamW
from transformers import pipeline

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
text_data_path = '/home/americanthinker/notebooks/pytorch/NationalSecurityBERT/Data/'

files = [f for f in os.listdir(text_data_path) if os.path.isfile(os.path.join(text_data_path, f))]
logger.info(f'{files}')

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


#load data from disk
def load_data_seq_512(path: str, sample_size:int=None) -> List[str]:
    with open(path) as f:
        if sample_size:
            lines = [line.strip() for line in f.readlines()[:sample_size]]
        else:
            lines = [line.strip() for line in f.readlines()]
    
    return lines

#create masked tokens
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
        
    encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}
    return encodings



logger.info('Loading data from disk into memory...')
start = time.perf_counter()
results = load_data_seq_512(os.path.join(text_data_path, files[0]))
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
BATCH_SIZE = 16
loader = torch.utils.data.DataLoader(d, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True)

del encodings

logger.info(f'Total of {len(loader)} batches.')

logger.info('Creating model')
config = BertConfig(vocab_size=30500,num_hidden_layers=12)
model = BertForMaskedLM(config)
logger.info('Model creation completed')


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

def run():
    num_batches = len(loader)
    epochs = 50
    step = 0
    
    model.train()
    
    for epoch in range(epochs):

        loop = tqdm(loader, leave=True)
        for batch in loop:
            step += 1
            # initialize calculated gradients (from prev step)
            optimizer.zero_grad()
            # pull all tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # process
            outputs = model(input_ids, attention_mask=attention_mask,
                            labels=labels)
            # extract loss
            loss = outputs.loss
            
            # calculate loss for every parameter that needs grad update
            loss.sum().backward()
            
            # update parameters
            optimizer.step()
            
            # print relevant info to progress barI 
            loop.set_description(f'Epoch {epoch}')
            
            loss_check = floor(num_batches/10)
            
            if step % loss_check == 0:
                logger.info(f'Loss: {loss.sum()}')
        
        #save checkpoint model after every epoch
        model.module.save_pretrained(f'checkpoints/run_4GB_Mar28/model-trained-{epoch}-{step}.pt') 

run()