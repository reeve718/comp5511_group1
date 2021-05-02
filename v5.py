import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import torch
import transformers as ppb
from sklearn import svm
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import cross_val_score, ShuffleSplit
import warnings
warnings.filterwarnings('ignore')

def clean_text(text):
    return text.lower().strip(' ')


df = pd.read_csv("https://raw.githubusercontent.com/reeve718/comp5511_group1/main/tmn_data.txt",sep = "######", names=["Text", "Label"],engine='python')
df['Clean_text'] = df['Text'].map(lambda x : clean_text(x))

# For DistilBERT:
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

tokenized = df['Clean_text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

attention_mask = np.where(padded != 0, 1, 0)

input_ids = torch.tensor(padded)  
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)
    
features = last_hidden_states[0][:,0,:].numpy()

Encoder = LabelEncoder()
labels = Encoder.fit_transform(df['Label'])

lr_clf = LogisticRegression()
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=None)
scores = cross_val_score(lr_clf, features, labels, cv=cv)
print(scores)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

