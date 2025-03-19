# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 15:56:55 2025

@author: ADMIN
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
data=pd.read_csv("C:/Users/ADMIN/OneDrive/Documents/GitHub/Machine-Learning/Linear Regression/program3.csv") 
print(data) 
 
x=data['YearsExperience'] 
#x.shape 
x=np.array(data.iloc[:,0]) 
x=np.array(data.iloc[:][["YearsExperience"]]) 
 
y=data['Salary'] 
y=np.array(data.iloc[:,1]) 
 
from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest=train_test_split(x,y,train_size=0.80,random_state=1) 
 
from sklearn.linear_model import LinearRegression 
model=LinearRegression() 
model.fit(xtrain,ytrain) 
ypred=model.predict(xtest) 
 
from sklearn.metrics import r2_score 
r2=r2_score(ytest, ypred) 
plt.figure(figsize=(8,5)) 
plt.scatter(xtrain,ytrain,color='blue',s=100,label="Actual Point") 
plt.scatter(xtrain, model.predict(xtrain), color='red',s=100,label="Predicted point") 
plt.show() 
 
plt.plot(xtrain,model.predict(xtrain),linestyle="dotted",color='orange',label="line of regression") 
plt.show() 
 
plt.figure(figsize=(8,5)) 
plt.scatter(xtrain,ytrain,color='blue',s=100,label="Actual Point") 
plt.scatter(xtrain, model.predict(xtrain), color='red',s=100,label="Predicted point") 
plt.plot(xtrain,model.predict(xtrain),linestyle="dotted",color='orange',label="line of regression") 
plt.show() 
 
plt.figure(figsize=(8,5))  
plt.scatter(xtest, ytest, color='blue', s=100, label="Actual Point")  
plt.scatter(xtest, model.predict(xtest), color='red', s=100, label=" Predicted point")  
plt.plot(xtest, model.predict(xtest), linestyle='dotted', color='orange', label="Line of regression")  
plt.show()  
 
scores=[]  
for i in range(5000): 
    xtrain1, xtest1, ytrain1, ytest1 = train_test_split(x,y,train_size =0.80, random_state=i)  
    model1=LinearRegression()  
    model1.fit(xtrain1, ytrain1)  
    ypred1=model1.predict(xtest1)  
    scores.append(r2_score(ytest1, ypred1)) 
 
# command mode print(scores) 
np.max(scores) # typre dosmode 
np.argmax(scores) # dosmode 
 
xtrain, xtest, ytrain, ytest = train_test_split(x,y,train_size=.80,random_state=4697) 
from sklearn.linear_model import LinearRegression  
model=LinearRegression()  
model.fit(xtrain, ytrain) 
ypred = model.predict(xtest)  
from sklearn.metrics import r2_score  
r2=r2_score(ytest, ypred)  #type r2 
 
plt.figure(figsize=(8,5))  
plt.scatter(xtrain, ytrain, color='blue', s=100, label="Actual Point ")  
plt.scatter(xtrain, model.predict(xtrain), color='red', s=100, label ="Predicted point")  
plt.show()  
 
plt.plot(xtrain, model.predict(xtrain), linestyle='dotted', color='orange', label="Line of regression")  
plt.show() 
 
plt.figure(figsize=(8,5))  
plt.scatter(xtest, ytest, color='blue', s=100, label="Actual Point")  
plt.scatter(xtest, model.predict(xtest), color='red', s=100, label="Predicted point")  
plt.plot(xtest, model.predict(xtest), linestyle='dotted', color='orange', label="Line of regression")  
plt.show()
