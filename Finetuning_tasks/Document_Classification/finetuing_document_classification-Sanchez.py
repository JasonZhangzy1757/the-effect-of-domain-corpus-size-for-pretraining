#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
from torch.nn.parallel import DataParallel

import transformers
from transformers import AdamW, get_linear_schedule_with_warmup
import os
import loguru

logger = loguru.logger

dirpath = '/home/americanthinker/notebooks/pytorch/NationalSecurityBERT/Finetuning_tasks/Document_Classification/DC_data/text/'
files = [file for file in os.listdir(dirpath) if file.endswith('txt')]

idx2text = {}


for file in files:
    idx = file.split('.')[0]
    with open(dirpath + file) as f:
        abstract = ''
        for sentence in f:
            abstract += sentence
        abstract = abstract.replace('\n', ' ')
        idx2text[idx] = abstract

for dirpath, dirnames, filenames in os.walk('DC_data/labels/'):
    idx2label = [] 
    for filename in filenames:
        idx = filename.split('.')[0]
        with open(dirpath + filename) as f:
            labels = f.readline()
            for label in labels.split('<'):
                if not label or label.isspace():
                    continue
                key_label = label.split('--')[0].strip()
                if key_label == 'NULL':
                    continue
                idx2label.append((idx, key_label))


idx2label = list(set(idx2label))
df = pd.DataFrame(idx2label, columns=['idx', 'label'])
df['text'] = df['idx'].map(idx2text)

label_vals = list(set(df['label'].tolist()))
df = df.groupby(['idx', 'text']).agg({'label': lambda x: list(x)}).reset_index()
mlb = MultiLabelBinarizer()
df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('label')),
                          columns=mlb.classes_,
                          index=df.index))


logger.info(df.head())

del idx2label, idx2text


class HoCDataset:
    def __init__(self, tokenizer, sentences, labels, max_len):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sentences)
        
    def __getitem__(self, item):
        sentence = str(self.sentences[item])
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding = 'max_length',
            truncation=True
        )
        ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'label': torch.tensor(self.labels[item], dtype=torch.float)
        } 

    
class BERTClass(nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('../../Modeling/checkpoints/model-trained-36-130647.pt/')
        self.out = nn.Linear(768, 10)

    
    def forward(self, ids, mask, token_type_ids):
        _, output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        
        return self.out(output)
    
    
def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)


def train_loop_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    for bi, d in enumerate(data_loader):
        ids = d['ids']
        mask = d['mask']
        token_type_ids = d['token_type_ids']
        labels = d['label']
        
        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        labels = labels.to(device, dtype=torch.float)
        
        optimizer.zero_grad()
        outputs = model(ids, mask, token_type_ids)

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if bi % 50 == 0:
            print(f'bi={bi}, loss={loss}')


def eval_loop_fn(data_loader, model, device):
    model.eval()
    fin_labels = []
    fin_outputs = []
    for bi, d in enumerate(data_loader):
        with torch.no_grad():
            ids = d['ids'].to(device, dtype=torch.long)
            mask = d['mask'].to(device, dtype=torch.long)
            token_type_ids = d['token_type_ids'].to(device, dtype=torch.long)
            labels = d['label'].to(device, dtype=torch.long)
          
            outputs = model(ids, mask, token_type_ids)
          
            fin_labels.append(labels.cpu().detach().numpy())
            fin_outputs.append(torch.sigmoid(outputs).cpu().detach().numpy())

    return np.vstack(fin_outputs), np.vstack(fin_labels)


MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
EPOCHS = 4
SEED = 42
LEARNING_RATE = 3e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


label_cols = list(df.drop(['idx', 'text'], axis=1).columns)
labels = df[label_cols].values

df_train, df_test, labels_train, labels_test = train_test_split(
    df, labels, test_size=0.2, random_state=SEED)


tokenizer = transformers.BertTokenizer.from_pretrained('../../Preprocessing/Tokenization/wp-vocab-30500-vocab.txt')
model = DataParallel(BERTClass())
model.to(device)
print(model.device_ids)

train_dataset = HoCDataset(
    sentences=df_train.text.values,
    labels=labels_train,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)
train_data_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True
)

test_dataset = HoCDataset(
    sentences=df_test.text.values,
    labels=labels_test,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)
test_data_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=True,
    drop_last=True
)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
num_training_steps = int(len(df_train) / TRAIN_BATCH_SIZE * EPOCHS)
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

for epoch in range(EPOCHS):
    train_loop_fn(train_data_loader, model, optimizer, device, scheduler)
output, target = eval_loop_fn(test_data_loader, model, device)


from sklearn import metrics
output = np.array(output) >= 0.5
f1_score_micro = metrics.f1_score(target, output, average='micro')
print(f"F1 Score (Micro) = {f1_score_micro} with a Batch Size of: {TRAIN_BATCH_SIZE}")
print(f'Using model 36-130647')



