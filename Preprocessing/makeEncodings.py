#standard imports
import torch
import os
from typing import List
import time
from tqdm.auto import tqdm
import sys

#tokenizers and datasets
import tokenizers
from tokenizers import BertWordPieceTokenizer 
from tokenizers.processors import TemplateProcessing
from transformers import BertTokenizer

vm_tok_path = '/home/americanthinker/notebooks/pytorch/NationalSecurityBERT/Preprocessing/Tokenization/wp-vocab-30500-vocab.txt'
text_data_path = '/home/americanthinker/notebooks/pytorch/NationalSecurityBERT/Data/text/partials/'
out_path = '/home/americanthinker/notebooks/pytorch/NationalSecurityBERT/Data/encodings/test/'
files = [file for file in os.listdir(text_data_path) if not file.endswith('test')]
print(files)

def load_tokenizer_from_file(vocab_path: str) -> BertWordPieceTokenizer:
    tokenizer = BertWordPieceTokenizer(vocab_path, strip_accents=True, lowercase=True)
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
    return tokenizer


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


tokenizer = load_tokenizer_from_file(vm_tok_path)

for i, file in enumerate(files):
    print(f'Loading data from {file} file....')
    data = load_data_seq_512(os.path.join(text_data_path, file))
    print('Data in memory...')
    
    print('Starting batch encoding...this will take a few minutes...')
    s = time.perf_counter()
    batch = tokenizer.encode_batch(data)
    e = time.perf_counter() - s
    total = len(batch)
    del data
    print(f'Completed batch {i} of {len(files)} files...')
    print(f'Total time: {round(e,2)} seconds to encode {total} samples...')
    time.sleep(2)
    print(f'Starting masked token creation at 15%.')
    encodings = mlm_pipe(batch)
    del batch
    masked_percent = sum(sum(encodings['input_ids'].detach().numpy() == 4)) / sum(sum(encodings['labels'].detach().numpy() != 4))
    print(f'Masked token encodings completed, {round(masked_percent * 100,2)}% tokens are masked...')
    print(f'Serializing tensors...')
    
    torch.save(encodings, f'{out_path}encodings_{total}_{file}.pt') 
    print(f'Serialization completed, moving on to next file')
    del encodings
    