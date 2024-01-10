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

features_data=pd.read_csv("features.csv")
stores_data=pd.read_csv("stores.csv")
train_data=pd.read_csv("train.csv")

print(features_data.head(10))
