import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, ShuffleSplit
import numpy as np
import spacy
from sklearn.base import BaseEstimator, TransformerMixin

def clean_text(text):
    return text.lower().strip(' ')
    
df = pd.read_csv("https://raw.githubusercontent.com/reeve718/comp5511_group1/main/tmn_data.txt",sep = "######", names=["Text", "Label"],engine='python')
df['Clean_text'] = df['Text'].map(lambda x : clean_text(x))

SVC_pipeline = Pipeline([
  ('tfidf', TfidfVectorizer(stop_words=stop_words, ngram_range=(1,2))),
  ('clf', ComplementNB(alpha=.1))
])

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=None)
scores = cross_val_score(SVC_pipeline, df['Clean_text'], df['Label'], cv=cv)

print(scores)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))