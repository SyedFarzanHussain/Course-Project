#importing libraries
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
from matplotlib.cm import get_cmap
'''
unzipping dataset
with zipfile.ZipFile("Course-Project\Dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("Course-Project")
'''
#reading the dataset
features_data=pd.read_csv("features.csv")
stores_data=pd.read_csv("stores.csv")
train_data=pd.read_csv("train.csv")

#checking the value in the data

features_data.head(10)
stores_data.head(10)
train_data.head(10)

#Data Exploration

features_data.describe().T
features_data.info()
train_data.describe().T
train_data.info()
stores_data.describe().T
stores_data.info()

#checkin Null Values

features_data.isnull().sum()
stores_data.isnull().sum()
train_data.isnull().sum()







