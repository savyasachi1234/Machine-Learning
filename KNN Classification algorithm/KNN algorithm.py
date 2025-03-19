# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 07:53:11 2025

@author: ADMIN
"""

import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score 
data = pd.read_csv("C:/Users/ADMIN/OneDrive/Documents/GitHub/Machine-Learning/KNN Classification algorithm/Iris.csv") 
print(data) 
data.head() 
x=data.drop('species',axis=1) 
y=data['species'] 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2) 
scaler = StandardScaler() 
x_train_scaled = scaler.fit_transform(x_train) 
x_test_scaled = scaler.transform(x_test) 
 
#Initialize the KNN classifier with k=5 
classifier = KNeighborsClassifier(n_neighbors=5) 
 
#Train the KNN classifier 
classifier.fit(x_train_scaled,y_train) 
 
y_pred = classifier.predict(x_test_scaled) 
 
accuracy = accuracy_score(y_test,y_pred) 
print(f"Accuracy of the KNN model:{accuracy * 100: .2f}%") 
 
print(f"Predicted labels: {y_pred}") 
print(f"Actual labels:{y_test.values}")
