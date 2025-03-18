# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 21:40:25 2025

@author: ADMIN
"""

import numpy as np 
from sklearn import datasets 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score,classification_report 
 
data=datasets.load_iris() 
print(data) 
 
x=data.data 
y=data.target 
 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42) 
svm_classifier=SVC(kernel='linear') 
 
svm_classifier.fit(x_train,y_train) 
 
#make prediction of the test set 
y_pred=svm_classifier.predict(x_test) 
 
#evaluate the model 
accuracy=accuracy_score(y_test, y_pred) 
print(f'Accuracy:{accuracy*100:.2f}%') 
 
print('Classification report') 
print(classification_report(y_test, y_pred))