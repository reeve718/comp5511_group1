# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 00:20:17 2021

@author: Reeve
"""
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import numpy as np
import spacy
from sklearn.base import BaseEstimator, TransformerMixin

class WordVectorTransformer(TransformerMixin,BaseEstimator):
    def __init__(self, model="en_core_web_sm"):
        self.model = model

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        nlp = spacy.load(self.model)
        return np.concatenate([nlp(doc).vector.reshape(1,-1) for doc in X])
    
corpus = [
"I went outside yesterday and picked some flowers.",
"She wore a red hat with a dress to the party.", 
"I think he was wearing athletic clothes and sneakers of some sort.", 
"I took my dog for a walk at the park.", 
"I found a hot pink hat on sale over the weekend.",
"The dog has brown fur with white spots."
]   

labels = [0,1,1,0,1,0]

transformer = WordVectorTransformer()
transformer.fit_transform(corpus)

text_clf = Pipeline([
            ('vect', WordVectorTransformer()),
            ('clf', SGDClassifier()),
            ])

text_clf.fit(corpus,labels)