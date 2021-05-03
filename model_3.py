import numpy as np
import pandas as pd
import torch
import transformers as ppb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.metrics import f1_score, classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def clean_text(text):
    return text.lower().strip(' ')

def train(clf):
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=None)
    scores = cross_val_score(clf, Train_X, Train_Y, cv=cv)
    print("Result on 5-fold validation:" + str(scores))
    print("%0.4f accuracy with a standard deviation of %0.4f" % (scores.mean(), scores.std()))
  
def evaluate(clf):
    clf.fit(Train_X, Train_Y) 
    y_predict_test = clf.predict(Test_X)
    print("Result on test data:")
    print (classification_report(Test_Y, y_predict_test))
    print("F1 micro: %0.4f" % f1_score(Test_Y, y_predict_test, average='micro'))
    print("F1 macro: %0.4f" % f1_score(Test_Y, y_predict_test, average='macro'))
    print("Accuracy: %0.4f" % (accuracy_score(Test_Y, y_predict_test)))
    
#load pre-trained bert
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

df = pd.read_csv("tmn_data.txt",sep = "######", names=["Text", "Label"],engine='python')
tokenized = df['Text'].apply(lambda x: tokenizer.encode(clean_text(x), add_special_tokens=True))

max_len = max(tokenized.apply(lambda x: len(x)))
padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

attention_mask = np.where(padded != 0, 1, 0)
input_ids = torch.tensor(padded)  
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)
    
features = last_hidden_states[0][:,0,:].numpy()
features_martix = m = np.asmatrix(features)
labels = df['Label']

Train_X, Test_X, Train_Y, Test_Y = train_test_split(features_martix, labels ,test_size=0.2)
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=None)
clf = LogisticRegression(max_iter=1000)
train(clf)
evaluate(clf)

