# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 01:04:27 2021

@author: Reeve
"""
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score 
from sklearn.base import TransformerMixin 
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from string import punctuation
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
from spacy.lang.en import English
parser = English()
columns = ["title","label"]
df =  pd.read_csv("tmn_data.txt", header = None, sep='######', names=columns,engine='python')
stopwords = list(STOP_WORDS)

def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuation ]
    print(mytokens)
    return mytokens

class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params(self, deep=True):
        return {}
    
def clean_text(text):  
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

# Vectorization
vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,2)) 
# classifier = LinearSVC()
classifier = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
#classifier = SVC(C=150, gamma=2e-2, probability=True)
pipe = Pipeline([("cleaner", predictors()),
                # ('vectorizer', vectorizer),
                 ('vectorizer', TfidfVectorizer(stop_words=STOP_WORDS,ngram_range=(1,1))),
                 ('classifier', classifier)])

X = df['title']
ylabels = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2, random_state=42)


pipe.fit(X_train, y_train)
prediction=pipe.predict(X_test)
print('Test accuracy is {}'.format(accuracy_score(y_test, prediction)))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipe, X,ylabels, cv=5)
print(scores)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))