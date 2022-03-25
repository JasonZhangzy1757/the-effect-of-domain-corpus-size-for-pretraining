#standard imports
import os, time
from tqdm.auto import tqdm
from typing import List, Union
from re_sent_splitter import split_into_sentences
from pathlib import Path
import pathlib

#distributed imports
import torch
from torch.nn.parallel import DistributedDataParallel as DDP, DataParallel
from torch.utils.data import DistributedSampler, DataLoader

#tokenizers and datasets
from datasets import load_dataset
from tokenizers import BertWordPieceTokenizer
from tokenizers.processors import TemplateProcessing
import tokenizers

#transformer imports
from transformers import BertTokenizer, DataCollatorForWholeWordMask, DataCollatorForLanguageModeling
from transformers import BertForMaskedLM, BertConfig, AdamW, TrainingArguments, Trainer
from transformers import pipeline


for d in range(4):
    print(torch.cuda.get_device_properties(d))
torch.cuda.device_count()

# !nvidia-smi
# Set Tokenizer and Data paths

vm_tok_path = '/home/americanthinker/notebooks/pytorch/NationalSecurityBERT/Preprocessing/Tokenization/wp-vocab-30500-vocab.txt'
vm_data = '/home/americanthinker/notebooks/pytorch/NationalSecurityBERT/Data/subsets/'
checkpoint_path = '/home/americanthinker/notebooks/pytorch/NationalSecurityBERT/Modeling/checkpoints/'
files = [f for f in os.listdir(vm_data) if f.endswith('25K')]
files

#local paths

# local_tok_path = '/Users/americanthinker1/NationalSecurityBERT/Preprocessing/Tokenization/wp-vocab-30500-vocab.txt'
# local_data = '/Users/americanthinker1/aws_data/processed_data/processed_chunks/english_docs_aa.txt'

# Instantiate pretrained tokenizer from file
alternative_tokenizer = BertTokenizer.from_pretrained('../Preprocessing/Tokenization/wp-vocab-30500-vocab.txt')

tokenizer = BertWordPieceTokenizer('../Preprocessing/Tokenization/wp-vocab-30500-vocab.txt', strip_accents=True, lowercase=True)
tokenizer.enable_truncation(max_length=50)
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


#tokenizer.save_model('../Preprocessing/Tokenization/', prefix='BWPTokenizer')
test = files[:1]
test[0]



# Load data from local
# Data is a 98,000 line file with each line representing one document of length ~12,000 characters from PubMed articles

#load data from disk
def load_data_from_disk(path: str, sample_size=None, min_tokens_per_sent: int=4) -> List[str]:
    '''
    Utility data loading function that performs the following operations:
       1. Loads data from disk into a list. Assumes each doc is one line.
       2. Performs sentence splitting on each document.
       3. Removes all sentences with tokens < 4 (default).
       4. Returns a list of sentences
    '''
    #load data
    with open(path) as f:
        if sample_size:
            lines = [line.strip() for line in f.readlines()[:sample_size]]
        else:
            lines = [line.strip() for line in f.readlines()]

    #split data into sentences
    sentences = [split_into_sentences(i) for i in tqdm(lines, 'Sentence Splitter')]

    #remove all sentences with less than 5 tokens
    all_sentences = []
    for doc in tqdm(sentences, 'Filter Senteces'):
        for sentence in doc:
            if len(sentence.split()) > 4:
                all_sentences.append(sentence)
    print(f'Return a list of {len(all_sentences)} sentences')

    return all_sentences
results = load_data_from_disk(os.path.join(vm_data, test[0]), sample_size=10000)


# Batch encode a chunk of data
s = time.perf_counter()
batch = tokenizer.encode_batch(results)
e = time.perf_counter() - s
print(round(e,2), 'seconds')
#decrease load on memory
del results


encodings = mlm_pipe(batch)


sum(sum(encodings['input_ids'] == 4)) / sum(sum(encodings['labels'] != 4))
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, i):
        return {key : tensor[i] for key, tensor in self.encodings.items()}

d = Dataset(encodings)
del batch
loader = torch.utils.data.DataLoader(d, batch_size=384, pin_memory=True, shuffle=True)
del encodings
config = BertConfig(vocab_size=30500,num_hidden_layers=12)
model = BertForMaskedLM(config)
device = 'cuda:0'
if torch.cuda.device_count() > 1:
    model = DataParallel(model)
    model.to(device)
model.device_ids


# model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

def save_model(path: './', multiple_gpu: bool=True):
    if multiple_gpu:
        torch.save({'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss},
              f'{path}model_{step}.pt')
    else:
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss},
              f'{path}model_{step}.pt')


from math import floor

def run():
    num_batches = len(loader)
    epochs = 1
    step = 0

    for epoch in range(epochs):
        # for file in files[:2]:
        #     # setup loop with TQDM and dataloader
        #     results = load_data_from_disk(os.path.join(vm_data, file))
        #     batch = tokenizer.encode_batch(results)
        #     del results
        #     encodings = mlm_pipe(batch)
        #     del batch
        #     d = Dataset(encodings)
        #     del encodings
        #     loader = torch.utils.data.DataLoader(d, batch_size=384, pin_memory=True, shuffle=True)

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

            if step % floor(num_batches/10) == 0:
                model.module.save_pretrained(f'checkpoints/test_save_pretrained/model-trained-{step}.pt')

        #loop.set_postfix(loss=loss.item())
run()


!nvidia-smi
torch.cuda.empty_cache()
#OOP way to train model

training_args = TrainingArguments(
    output_dir=model_path,          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,
    num_train_epochs=2,            # number of training epochs, feel free to tweak
    per_device_train_batch_size=8, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=1,  # accumulating the gradients before updating the weight
    logging_steps=500,             # evaluate, log and save model checkpoints every 1000 step
    save_steps=500,
    # load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
)

trainer = Trainer(model=model,
                  args=training_args,
                  data_collator=data_collator,
                  train_dataset=dataset)
def load_checkpoint(model, checkpoint_: Union[str, pathlib.Path], parallel=False):
    checkpoint = torch.load(checkpoint_)
    model_state = checkpoint['model_state_dict']
    opt_state = checkpoint['optimizer_state_dict']
    model_loss = checkpoint['loss']
    model.load_state_dict(model_state)
    if parallel:
        if torch.cuda.device_count() > 1:
            model = DataParallel(model)
    return model, opt_state, model_loss
new_model, opt, model_loss = load_checkpoint(model, 'checkpoints/final_model_18710.pt')
mask = alternative_tokenizer.mask_token


def show_results(text: str):
    config = BertConfig(vocab_size=30500, max_position_embeddings=514, num_hidden_layers=12)
    model = BertForMaskedLM(config)
    untrained_pipe = pipeline('fill-mask', model=model, tokenizer=alternative_tokenizer)
    utresult = untrained_pipe(text)

    print()
    print("Untrained Results")
    print("*" * 150)
    for result in utresult:
        print(result)

    trained_model, opt, model_loss = load_checkpoint(model, 'checkpoints/final_model_18710.pt')
    trained_pipe = pipeline('fill-mask', model=trained_model, tokenizer=alternative_tokenizer)

    tresult = trained_pipe(text)

    print()
    print("Trained Results")
    print("*" * 150)
    for result in tresult:
        print(result)

show_results(f'Introduction Under normal physiological conditions, all cells in the body are exposed chronically to oxidants from both {mask} and exogenous sources;')
with open('../Data/subsets/xaasplit_25K') as f:
    data = [line.strip() for line in f.readlines()]
test = tokenizer.encode_batch(['Introduction Under normal physiological conditions, all cells in the body are exposed chronically to oxidants from both endogenous and exogenous sources;', 'This is another sentence.'])
input_ids = torch.tensor([x.ids for x in test])
mask = torch.tensor([x.attention_mask for x in test])
outputs = new_model(input_ids, mask)
outputs['logits'].shape


bbut = BertTokenizer.from_pretrained('bert-base-uncased')
new_model.save_pretrained('./')
new_model.get_output_embeddings()
new_model
test = BertForMaskedLM.from_pretrained('bert-base-uncased')
# Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertForMaskedLM: ['cls.seq_relationship.weight', 'cls.seq_relationship.bias']
# - This IS expected if you are initializing BertForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing BertForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
test


from transformers import BertModel, BertForMaskedLM, BertForPreTraining
bert = BertModel.from_pretrained('bert-base-uncased')
mlm = BertForMaskedLM.from_pretrained('bert-base-uncased')
