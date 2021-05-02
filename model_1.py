import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string
from nltk import pos_tag
from nltk.stem import PorterStemmer
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit
from sklearn.metrics import f1_score, classification_report, accuracy_score

def sentenceToWords(data):
  result = [word_tokenize(entry.lower()) for entry in data] 
  return result
def filter_words(words):
  stopWords = stopwords.words('english')
  return [word for word in words if ((word not in stopWords) and (word not in string.punctuation))]
def pos(words):
  regex = re.compile("^J|^R|^N|^V.*")
  return [word[0] for word in pos_tag(words) if regex.match(word[1])]
def stemming(words):
  ps = PorterStemmer()
  return [ps.stem(word) for word in words]
def lemmatiztion(words):
  wnl = WordNetLemmatizer()
  return [wnl.lemmatize(word) for word in words]

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
  print("F1 weighted: %0.4f" % f1_score(Test_Y, y_predict_test, average='weighted'))
  print("Accuracy: %0.4f" % (accuracy_score(Test_Y, y_predict_test)))
    
data = pd.read_csv("tmn_data.txt",sep = "######", names=["Text", "Label"],engine='python')
sentence = data["Text"]
tokenLists = sentenceToWords(sentence)
filerted = [filter_words(token) for token in tokenLists]
posed = [pos(token) for token in filerted]
stemmed = [stemming(token) for token in posed]
data["Processed_Text"] = [lemmatiztion(token) for token in stemmed]

#Data setup
labels = data['Label']
features = [str(entry) for entry in data['Processed_Text']]
Tfidf_vect = TfidfVectorizer()
Tfidf_vect.fit(features)
features = Tfidf_vect.transform(features)

#Split for train / test
Train_X, Test_X, Train_Y, Test_Y = train_test_split(features, labels,test_size=0.2)

clf = ComplementNB()
evaluate(clf)
