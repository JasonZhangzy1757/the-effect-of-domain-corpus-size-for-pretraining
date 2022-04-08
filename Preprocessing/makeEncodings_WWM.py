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

from whole_word_masking_ids import create_masked_lm_ids

vm_tok_path = '/home/americanthinker/notebooks/pytorch/NationalSecurityBERT/Preprocessing/Tokenization/wp-vocab-30500-vocab.txt'

test_files = ['/home/americanthinker/notebooks/pytorch/NationalSecurityBERT/Data/text/partials/xaatest',
              '/home/americanthinker/notebooks/pytorch/NationalSecurityBERT/Data/text/partials/xabtest']
files = ['/home/americanthinker/notebooks/pytorch/NationalSecurityBERT/Data/text/combined4Gb_1.txt',
            '/home/americanthinker/notebooks/pytorch/NationalSecurityBERT/Data/text/combined4Gb_2.txt']

out_path = '/home/americanthinker/notebooks/pytorch/NationalSecurityBERT/Data/encodings/whole_word_masking/'

#files = [file for file in os.listdir(text_data_path) if not file.endswith('test')]
print(test_files)

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

def create_source_ids(tokenizer):
    id_list = [tokenizer.token_to_id(word) for word in tokenizer.get_vocab()]
    return id_list

def load_data_seq_512(path: str, sample_size:int=None) -> List[str]:
    with open(path) as f:
        if sample_size:
            lines = [line.strip() for line in f.readlines()[:sample_size]]
        else:
            lines = [line.strip() for line in f.readlines()]
    
    return lines


def mlm_pipe(batch: List[tokenizers.Encoding], source_ids: list, tokenizer, mlm_prob=0.15) -> dict:
    '''
    Given a single instance from a batch of encodings, return masked inputs and associated arrays.
    Converts tokenizer.Encoding into a pytorch tensor.
    '''
    
    # default masking prob = 15%, don't mask special tokens     
    labels = torch.tensor([x.ids for x in tqdm(batch, 'Labels')])
    mask = torch.tensor([x.attention_mask for x in tqdm(batch, 'Attention Mask')])
    input_ids = torch.tensor([create_masked_lm_ids(x.ids, source_ids, tokenizer, masked_lm_prob=mlm_prob) for x in tqdm(batch, 'Input Ids')])
    
    encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}
    return encodings


tokenizer = load_tokenizer_from_file(vm_tok_path)
id_list = create_source_ids(tokenizer)

for i, file in enumerate(test_files):
    
    #load file into memory
    filename = file.split('/')[-1].split('.')[0]
    print(f'Loading data from {filename} file....')
    data = load_data_seq_512(file)
    print('Data in memory...')
    
    #batch encode data
    print('Starting batch encoding...this will take a few minutes...')
    s = time.perf_counter()
    batch = tokenizer.encode_batch(data)
    e = time.perf_counter() - s
    total = len(batch)
    del data
    print(f'Total time: {round(e,2)} seconds to encode {total} samples...')
    time.sleep(2)
    
    #create masked ids
    print(f'Starting masked token creation at 15%.')
    encodings = mlm_pipe(batch, id_list, tokenizer)
    del batch
    masked_percent = sum(sum(encodings['input_ids'].detach().numpy() == 4)) / sum(sum(encodings['labels'].detach().numpy() != 4))
    print(f'Masked token encodings completed, {round(masked_percent * 100,2)}% tokens are masked...')
    print(f'Serializing tensors...')
    
    #serialize encodings to disk
    print(f'{filename} is being serialized.')
    torch.save(encodings, f'{out_path}encodings_{total}_{filename}.pt') 
    print(f'Serialization completed, moving on to next file')
    del encodings
    