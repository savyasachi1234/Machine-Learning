# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 21:41:16 2025

@author: ADMIN
"""

import pandas as pd
import numpy as np

# Load the dataset
dataset = pd.read_csv("student.csv")

# Display dataset info
print("Dataset Info:")
print(dataset.info())

# Print dataset statistics
print("\nDataset Statistics:")
print(dataset.describe())

# Display first 5 rows
print("\nFirst 5 Rows:")
print(dataset.head())

# Display last 3 rows
print("\nLast 3 Rows:")
print(dataset.tail(3))

# Creating independent (X) and dependent (Y) datasets
X = dataset.iloc[:, :-1].values  # All columns except last
Y = dataset.iloc[:, -1].values   # Only last column

print("\nIndependent Variables (X):")
print(X)

print("\nDependent Variable (Y):")
print(Y)

# Handling missing data
print("\nMissing Data Count:")
print(dataset.isnull().sum())

# Dropping rows with missing values
dataset_cleaned = dataset.dropna()

print("\nDataset After Dropping Missing Values:")
print(dataset_cleaned)

# Filling missing CGPA values with the mean
mean_cgpa = dataset['CGPA'].mean()
dataset['CGPA'] = dataset['CGPA'].fillna(mean_cgpa)

print("\nDataset After Filling Missing Values:")
print(dataset)

# Checking again for missing values
print("\nFinal Missing Data Count:")
print(dataset.isnull().sum())
