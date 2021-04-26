import os as os
import pandas as pd
from sklearn import svm, model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import re
import spacy
from string import punctuation
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time
from torch.utils.data.dataset import random_split

regex = re.compile("^J|^R|^N|^V.*")
nlp = spacy.load("en_core_web_lg")

class dataLoader:
    def __init__(self, fileName: str):
        fileDir = os.path.dirname(os.path.realpath('__file__'))
        filePath = os.path.join(fileDir, fileName)
        self.filePath = filePath
    def getData(self) -> str:
        columns = ["title","label"]
        data =  pd.read_csv(self.filePath, header = None, sep='######', names=columns,engine='python')
        return data
    
def clean_text(text):
    result = []
    doc = nlp(text.lower())
    for token in doc:      
      if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
        continue
      if(regex.match(token.tag_)):
          if token.has_vector:
              result.append(token.vector_norm)
    return result

class CustomDataset(Dataset):
    def __init__(self, df):
        self.data = df
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data.iloc[idx,2], self.data.iloc[idx,3]
    
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(_text, dtype=torch.float64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)   

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
    
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)
    

def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 50
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        
        predited_label = model(text, offsets)
        loss = criterion(predited_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predited_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(10, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predited_label = model(text, offsets)
            loss = criterion(predited_label, label)
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count    

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(_text, dtype=torch.float64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

dataLoader_q = dataLoader("tmn_data_dev.txt")
df = dataLoader_q.getData()
Encoder = LabelEncoder()
df["encoded_label"] = Encoder.fit_transform(df['label'])
df["tokens"] = df["title"].apply(lambda x: clean_text(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
myDataset = CustomDataset(df)
dataloader = DataLoader(myDataset, batch_size=8, shuffle=False, collate_fn=collate_batch)
vocab_size = nlp.vocab.vectors.n_keys
emsize = 300
num_class = len(set([label for label in df["encoded_label"] ]))
LR=5
BATCH_SIZE = 8
'''
for idx, (label, text, offsets) in enumerate(dataloader):
    print(label)
    print(text)

print(vocab_size)
'''
EPOCHS = 10 
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None
num_test = int(len(myDataset) * 0.95)
split_train_, test_dataset = random_split(myDataset, [num_test, len(myDataset) - num_test])
num_train = int(len(split_train_) * 0.95)
split_train_, split_valid_ = random_split(split_train_, [num_train, len(split_train_) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
      scheduler.step()
    else:
       total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 59)

print('Checking the results of test dataset.')
accu_test = evaluate(test_dataloader)
print('test accuracy {:8.3f}'.format(accu_test))