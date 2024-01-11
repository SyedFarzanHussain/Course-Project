#importing libraries
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
from matplotlib.cm import get_cmap
import streamlit as st

st.title("WALMART SALES DATA ANALYSIS")

#unzipping dataset
comment2='''
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

#Initial Data Exploration

comment1= '''
features_data.describe().T
features_data.info()
train_data.describe().T
train_data.info()
stores_data.describe().T
stores_data.info()
'''

#checkin Null Values

features_data.isnull().sum()
stores_data.isnull().sum()
train_data.isnull().sum()

#Data merging (train,features and store data)
new_train_df=train_data.merge(features_data,how='left',on=["Store","Date"],indicator=True).merge(stores_data,how='left').copy()
new_train_df.isnull().sum()

#lambda function to check the percentage of null values
null_percent_check=lambda x:print(f"The null percentage of {x} is {(new_train_df[x].isnull().sum()/new_train_df.shape[0])*100}")
null_percent_check("MarkDown1")
null_percent_check("MarkDown2")
null_percent_check("MarkDown3")
null_percent_check("MarkDown4")

#The null percentages are more than 60% and these are promotional offers so we can drop these columns.
new_train_df.drop(new_train_df.columns[new_train_df.columns.get_loc("MarkDown1"):new_train_df.columns.get_loc("MarkDown5")+1],axis=1,inplace=True)
print(new_train_df.columns)

#Dropping extra columns after merge
new_train_df.drop(['IsHoliday_y','_merge'], axis=1,inplace=True)
print(new_train_df.columns) 

#renaming the Holiday column
new_train_df.rename(columns={"IsHoliday_x":"IsHoliday"},inplace=True)
print(new_train_df.columns) 

#checking the count of negative values in sales
print(new_train_df[new_train_df['Weekly_Sales']<=0].Weekly_Sales.shape)

#excluding these negative values from data
new_df=new_train_df[new_train_df['Weekly_Sales']>0].copy()





