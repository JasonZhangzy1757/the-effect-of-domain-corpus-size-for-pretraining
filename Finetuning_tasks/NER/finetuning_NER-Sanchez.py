#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import transformers
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForTokenClassification, BertTokenizer, BertConfig, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from torch.nn.parallel import DataParallel
from sklearn.metrics import f1_score
from collections import defaultdict
from torch import cuda
from pytorch_lightning import Trainer
import loguru
import os

logger = loguru.logger

model_path = '../../Modeling/checkpoints/model-trained-36-130647.pt/'
token_path = '../../Preprocessing/Tokenization/wp-vocab-30500-vocab.txt'
#model_path = '../../Modeling/checkpoints/blank_bert/'
#token_path = 'bert-base-uncased'

logger.info(f'Using {model_path} model and {token_path} tokenizer')

def def_value():
    return 'O'


def read_data(path):
    with open(path) as f:
        sent_dict = {}
        label_dict = {}
        count = 0
        for line in f:
            if line.isspace():
                continue
            if '|' in line and len(line.split('|')) == 3 and (line.split('|')[1] == 'a' or line.split('|')[1] == 't'):
                idx, _, sentence = line.split('|')
                sent_dict[idx] = sent_dict.get(idx, '') + ' ' + sentence
            else:
                idx, start_pos, end_pos, word, label, _ = line.split('\t')
                if idx not in label_dict:
                    label_dict[idx] = defaultdict(def_value)
                    for i in range(int(start_pos), int(end_pos)):
                        label_dict[idx][i] = label  
                else:
                    for i in range(int(start_pos), int(end_pos)):
                        label_dict[idx][i] = label
                        
    idx_col, word_col, label_col = [], [], []
    for idx in sent_dict:
        sentence = sent_dict[idx].replace('\n', '')
        
        char_seq = 0
        for word in sentence.split(' ')[1:]:
            label = label_dict[idx][char_seq]
            if word and word[0] == '(':
                label = label_dict[idx][char_seq + 1]
            char_seq += len(word) + 1
            
            idx_col.append(idx)
            word_col.append(word)
            label_col.append(label)
    
    df = pd.DataFrame(list(zip(idx_col, word_col, label_col)),
               columns =['sentence_id', 'word', 'label'])
    return df


class SentenceGetter(object):
    
    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda x: [(w, t) for w, t in zip(x["word"].values.tolist(),
                                                        x["label"].values.tolist())]
        self.grouped = self.dataset.groupby("sentence_id").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            sentence = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return sentence
        except:
            return None
        
        
# Creating new lists and dicts that will be used at a later stage for reference and processing
def get_data(df, label_vals):
    getter = SentenceGetter(df)
    label2idx = {value: key for key, value in enumerate(label_vals)}
    sentences = [' '.join([s[0] for s in sentence]) for sentence in getter.sentences]
    labels = [[s[1] for s in sentence] for sentence in getter.sentences]
    labels = [[label2idx.get(l) for l in label] for label in labels]
    return sentences, labels


# In[9]:


class CustomDataset(Dataset):
    def __init__(self, tokenizer, sentences, labels, max_len):
        self.len = len(sentences)
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        sentence = str(self.sentences[index])
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        label = self.labels[index]
        label.extend([4]*200)
        label=label[:200]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'tags': torch.tensor(label, dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len
    
class BERTClass(torch.nn.Module):
    def __init__(self, model_path):
        super(BERTClass, self).__init__()
        self.bert = transformers.BertForTokenClassification.from_pretrained(model_path, num_labels=18)

    
    def forward(self, ids, mask, labels):
        output = self.bert(ids, mask, labels = labels)

        return output

def train(epoch):
    model.train()
    for step, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['tags'].to(device, dtype = torch.long)

        loss = model(ids, mask, labels = targets)[0]
        
        optimizer.zero_grad()
        
        loss.sum().backward()
        optimizer.step()
        
        if step % 5==0:
            print(f'Epoch: {epoch}  Step: {step}  Loss: {loss.sum()}')
            
def valid(model, testing_loader, label_vals):
    model.eval()
    eval_loss = 0
    predictions , true_labels = [], []
    nb_eval_steps = 0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['tags'].to(device, dtype = torch.long)

            output = model(ids, mask, labels=targets)
            loss, logits = output[:2]
            logits = logits.detach().cpu().numpy()
            label_ids = targets.to('cpu').numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.append(label_ids)
            eval_loss += loss.mean().item()
            nb_eval_steps += 1
        eval_loss = eval_loss/nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        pred_tags = [label_vals[p_i] for p in predictions for p_i in p]
        valid_tags = [label_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
        average = 'micro'
        print(f'Using {average} average')
        score = f1_score(pred_tags, valid_tags, average=average)
        print(f"F1-Score: {round(score,3)}") 
        
    return pred_tags, valid_tags, score


device = 'cuda' if cuda.is_available() else 'cpu'


df_train = read_data('./NCBI-disease/NCBItrainset_corpus.txt')
df_valid = read_data('./NCBI-disease/NCBIdevelopset_corpus.txt')

label_vals = list(df_train["label"].value_counts().keys())
label2idx = {value: key for key, value in enumerate(label_vals)}

train_sentences, train_labels = get_data(df_train, label_vals)
valid_sentences, valid_labels = get_data(df_valid, label_vals)

run_metric = []

for run in range(5):
    logger.info(f'Initiating model {model_path} for training')
    model = BERTClass(model_path)
    model = DataParallel(model)
    model.to(device)
    logger.info(f'Model device ids: {model.device_ids}')
    tokenizer = BertTokenizer.from_pretrained(token_path)

    #Defining some key variables that will be used later on in the training
    MAX_LEN = 200
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 8
    EPOCHS = 6
    LEARNING_RATE = 5e-05
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    training_set = CustomDataset(tokenizer, train_sentences, train_labels, MAX_LEN)
    valid_set = CustomDataset(tokenizer, valid_sentences, valid_labels, MAX_LEN)

    training_loader = DataLoader(training_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=VALID_BATCH_SIZE, shuffle=True)


    for epoch in range(EPOCHS):
        train(epoch)
    pred_tags, valid_tags, score = valid(model, valid_loader, label_vals)
    run_metric.append(score)
logger.info(f"Mean f1-score: {np.round(np.mean(run_metric), 4)}")



