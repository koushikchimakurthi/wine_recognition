import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.datasets import load_wine
wine = load_wine()
# wine object is a sklearn.utils.bunch object, need to convert in pandas dataframe.

# This next 2 lines of code is taken from
# "https://stackoverflow.com/questions/38105539/how-to-convert-a-scikit-learn-dataset-to-a-pandas-dataset"
data = pd.DataFrame(wine.data,columns=wine.feature_names)
data['class'] = pd.Series(wine.target)

print("----------------------------------------------")
print("------Loading and printing first 5 rows-------")
print("----------------------------------------------")
print(data.head())

print("----------------------------------------------")
print("---------Shape of data (Rows*Columns)---------")
print("----------------------------------------------")
print(data.shape)

print("----------------------------------------------")
print("----------Number of Samples per class---------")
print("----------------------------------------------")
print(data['class'].value_counts())

print("----------------------------------------------")
print("------Describing data like mean,std etc-------")
print("----------------------------------------------")
print(data.describe())

print("----------------------------------------------")
print("------------Features/Columns names------------")
print("----------------------------------------------")
print(data.columns)
