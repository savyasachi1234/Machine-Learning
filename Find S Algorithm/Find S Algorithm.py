# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 08:17:19 2025

@author: ADMIN
"""
import pandas as pd 
import numpy as np 
data = pd.read_csv("program4_data.csv") 
print(data) 
concepts = np.array(data)[:,:-1] 
print('The target column') 
target = np.array(data)[:,-1] 
print(target) 
 
def train(con,tar): 
    for i,val in enumerate(tar): 
        if val == 'yes': 
            specific_h = con[i].copy() 
            break 
    for i,val in enumerate((con)): 
        if tar[i] == 'yes': 
            for x in range((len(specific_h))): 
                if val[x] != specific_h[x]: 
                    specific_h[x] = '?' 
                else : 
                    pass 
    return specific_h 
print('The final output') 
print(train(concepts,target))