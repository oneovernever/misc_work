from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree 
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import confusion_matrix
from sklearn import neighbors
import pandas as pd
import numpy as np
import loaddata
import dataprep
import test1,test2,test3

data=pd.read_csv("capture20110816-3.binetflow")
Xdata,Ydata,XdataT,YdataT=loaddata.loaddata('capture20110816-3.binetflow')

#print(Xdata)
print(Ydata)
#print(XdataT)
print(YdataT)

# Running Naives-Bayes
clf = GaussianNB()
clf.fit(Xdata,Ydata)
Prediction = clf.predict(XdataT)
Score = clf.score(XdataT,YdataT)
print("The Score of the Gaussian Naive Bayes classifier is ", Score * 100)
print("[True Negatives, False Positives], [False Negatives, True Positives]: ")
print(confusion_matrix(YdataT, Prediction))

#It performs very low, so we will run a Decision Tree classifier.

clf = tree.DecisionTreeClassifier()
clf.fit(Xdata,Ydata)
Prediction=clf.predict(XdataT)
Score = clf.score(XdataT,YdataT)
print("The Score of the Gaussian Decision Tree classifier is ",Score*100)

#Feature importance
print("The feature importance is as follow: ")
print(dict(zip(data.columns, clf.feature_importances_)))