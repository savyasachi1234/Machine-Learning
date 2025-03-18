# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 20:10:11 2025

@author: ADMIN
"""

import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score,classification_report 
from sklearn.preprocessing import StandardScaler 
 
data = pd.read_csv('program7.csv') 
x = data.drop('Outcome',axis=1) 
y = data['Outcome'] 
y = np.array(data.iloc[:,-1]) 
 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=30,random_state=42) 
 
scaler = StandardScaler() 
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test) 
 
nb_classifier = GaussianNB() 
nb_classifier.fit(x_train,y_train) 
 
y_pred = nb_classifier.predict(x_test) 
 
accuracy = accuracy_score(y_test, y_pred) 
print(f"Accuracy: {accuracy*100:.2f}% ") 
#print(accuracy*100) 
 
print("\n Classification Report") 
print(classification_report(y_test, y_pred))