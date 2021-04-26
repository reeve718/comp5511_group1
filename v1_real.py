import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from nltk import pos_tag
from nltk.stem import PorterStemmer
from nltk import WordNetLemmatizer
import string
from sklearn import svm, model_selection, naive_bayes
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score

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

data = pd.read_csv("tmn_data.txt",sep = "######", names=["Text", "Label"],engine='python')
sentence = data["Text"]
tokenLists = sentenceToWords(sentence)
filerted = [filter_words(token) for token in tokenLists]
posed = [pos(token) for token in filerted]
stemmed = [stemming(token) for token in posed]
data["Processed_Text"] = [lemmatiztion(token) for token in stemmed]
#print(data.info())
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(data['Label'])
Train_X = [str(entry) for entry in data['Processed_Text']]
Tfidf_vect = TfidfVectorizer()
Tfidf_vect.fit(Train_X)
Train_X = Tfidf_vect.transform(Train_X)

'''
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(data['Processed_Text'],data['Label'],test_size=0.2)
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Train_X = [str(entry) for entry in Train_X]
Tfidf_vect = TfidfVectorizer()
Tfidf_vect.fit(Train_X)
Train_X = Tfidf_vect.transform(Train_X)
Test_Y = Encoder.fit_transform(Test_Y)
Test_X = [str(entry) for entry in Test_X] 
Test_X = Tfidf_vect.transform(Test_X)
'''
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
'''
SVM.fit(Train_X,Train_Y)
predictions_SVM = SVM.predict(Test_X)
result_SVM_accuracy = accuracy_score(predictions_SVM, Test_Y)*100
result_SVM = Encoder.inverse_transform(predictions_SVM)
print(result_SVM_accuracy)
#print(result_SVM)
'''
from sklearn.model_selection import cross_val_score
scores = cross_val_score(SVM, Train_X,Train_Y, cv=5)
print(scores)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
