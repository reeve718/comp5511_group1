import os as os
import pandas as pd
from sklearn import svm, model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import spacy
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

dataLoader_q = dataLoader("tmn_data.txt")
df = dataLoader_q.getData()
print(df["label"])
df["vector"] = df["title"].apply(lambda x: nlp(x))

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['vector'],df['label'],test_size=0.1)
print(Train_X)
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X,Train_Y)
predictions_SVM = SVM.predict(Test_X)
result_SVM_accuracy = accuracy_score(predictions_SVM, Test_Y)*100
result_SVM = Encoder.inverse_transform(predictions_SVM)
print(result_SVM_accuracy)
#print(result_SVM)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(SVM, Train_X,Train_Y, cv=5)
print(scores)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
