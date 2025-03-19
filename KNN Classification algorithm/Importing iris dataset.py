# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 07:56:15 2025

@author: ADMIN
"""

from sklearn import datasets
import pandas as pd

# Load dataset
iris = datasets.load_iris()

# Convert to DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Map species numbers to names
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Save as CSV
df.to_csv("Iris.csv", index=False)

print("Iris dataset saved as Iris.csv")
