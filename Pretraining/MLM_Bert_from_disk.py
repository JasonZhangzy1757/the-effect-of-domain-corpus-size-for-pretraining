#!/usr/bin/env python
# coding: utf-8

#standard imports
import time, datetime, pytz
import os, sys

from typing import List, Union, Dict
from tqdm.auto import tqdm
from math import floor
import numpy as np
import loguru

#distributed imports
import torch
from torch.nn.parallel import DataParallel

#transformer imports
from transformers import BertForMaskedLM, BertConfig, AdamW


############################################################
# Configure Logger + Device + Data path
############################################################

logger = loguru.logger
    
local_rank = 0
device = (
        torch.device("cuda", local_rank)
        if (local_rank > -1) and torch.cuda.is_available()
        else torch.device("cpu")
    )

encodings_path = '/home/americanthinker/notebooks/pytorch/NationalSecurityBERT/Data/encodings/whole_word_masking/'
files = sorted([f for f in os.listdir(encodings_path) if f.endswith('pt')])[:1]
# encodings_path = '/home/americanthinker/notebooks/pytorch/NationalSecurityBERT/Data/encodings/test/'
file = 'encodings_395390_combined4Gb_1.pt'
logger.info(files)

############################################################
# Assemble Dataset and Dataloader
############################################################
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        
    def __len__(self):
        return self.encodings['input_ids'].shape[0]
    
    def __getitem__(self, i):
        return {key : tensor[i] for key, tensor in self.encodings.items()}

BATCH_SIZE = 112

def assemble(file_path: str):
    encodings = torch.load(file_path)
    dataset = Dataset(encodings)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=20, pin_memory=True, shuffle=True)
    del encodings
    return loader

############################################################
# Instantiate model
############################################################
logger.info('Creating model')
config = BertConfig(vocab_size=30500)
model = BertForMaskedLM(config)
model = DataParallel(model)
model.to(device)
logger.info('Model creation completed')
logger.info(model.device_ids)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

############################################################
# Train model 
############################################################
model.train()

epochs = 100
overall_steps = 0
mean_losses = []

for epoch in range(epochs):    
    
    for file in files:
        
        data_loader = assemble(os.path.join(encodings_path, file))
        num_batches = len(data_loader)
        logger.info(f'Total batches for this load: {num_batches}')
        
        loss_check = floor(num_batches/10)
        steps = 0
        for batch in tqdm(data_loader, f'Epoch: {epoch}'):
            steps += 1           
            
            # initialize calculated gradients (from prev step)
            optimizer.zero_grad()
            
            # pull all tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # process
            outputs = model(input_ids, 
                            attention_mask=attention_mask,
                            labels=labels)
            # extract loss
            loss = outputs.loss
            # calculate loss for every parameter that needs grad update
            loss.sum().backward()
            # update parameters
            optimizer.step()

            if steps % loss_check == 0:
                logger.info(f'Loss: {loss.sum()}')
                steps_loss = loss.sum().detach().cpu()
                mean_losses.append(steps_loss.numpy())
                with open('./checkpoints/run_4GB_losses.txt', 'a') as f:
                    f.write(f'{steps_loss}')
                    f.write('\n')
                    
        overall_steps += steps
        logger.info('Deleting dataloader from memory')
        del data_loader
        
    logger.info(f'Average Loss for Epoch: {np.round(np.mean(mean_losses), 3)}')
    
    with open('./checkpoints/run_4GB_epoch_losses.txt', 'a') as f:
        f.write(f'{np.round(np.mean(mean_losses), 3)}')
        f.write('\n')
                
    mean_losses = []
    timestamp = str(datetime.datetime.now()).split('.')[0].replace(' ','_')
    model.module.save_pretrained(f'checkpoints/run_4GB_model-trained-{epoch}-{overall_steps}') 
