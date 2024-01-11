#importing libraries
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
from matplotlib.cm import get_cmap
import streamlit as st

#st.title("WALMART SALES DATA ANALYSIS")

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
comment3='''
null_percent_check=lambda x:print(f"The null percentage of {x} is {(new_train_df[x].isnull().sum()/new_train_df.shape[0])*100}")
null_percent_check("MarkDown1")
null_percent_check("MarkDown2")
null_percent_check("MarkDown3")
null_percent_check("MarkDown4")
'''

#The null percentages are more than 60% and these are promotional offers so we can drop these columns.
new_train_df.drop(new_train_df.columns[new_train_df.columns.get_loc("MarkDown1"):new_train_df.columns.get_loc("MarkDown5")+1],axis=1,inplace=True)
#print(new_train_df.columns)

#Dropping extra columns after merge
new_train_df.drop(['IsHoliday_y','_merge'], axis=1,inplace=True)
#print(new_train_df.columns) 

#renaming the Holiday column
new_train_df.rename(columns={"IsHoliday_x":"IsHoliday"},inplace=True)
#print(new_train_df.columns) 

#checking the count of negative values in sales
#print(new_train_df[new_train_df['Weekly_Sales']<=0].Weekly_Sales.shape)

#excluding these negative values from data
new_df=new_train_df[new_train_df['Weekly_Sales']>0].copy()

new_df["Date"]=pd.to_datetime(new_df['Date']) #convert date column into proper date time column

#separting date columns into week, month and year 
new_df["Week"] = new_df["Date"].dt.isocalendar().week
new_df["Month"] = new_df["Date"].dt.month
new_df["Year"] = new_df["Date"].dt.year

#feature understanding 

#ploting pie chart for distribution of store type
plt.pie(new_df["Type"].value_counts().tolist(),autopct="%1.1f%%",) 
plt.legend(new_df["Type"].value_counts().keys()) #extracting keys from type column as legends
plt.title('Distribution of Store Type')
plt.show()


#ploting bar chart for count of store type

Store_chart=new_df["Type"].value_counts()\
    .plot(kind='bar',title="Store Types",rot=0)
Store_chart.set_xlabel("TYPE")
Store_chart.set_ylabel("COUNT")
plt.show()

#ploting bar chart for size of store according to their type

Size_plot=stores_data.plot.bar(x="Store",y="Size",
                               color=['green' if data_value=="A" else 'red' if data_value=='B' else 'blue' for data_value in stores_data['Type']],
                               rot=0,figsize=(13, 5))

legend_labels = {'green': 'A', 'red': 'B', 'blue': 'C'} #dictionary for creating colored legends
legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in legend_labels.keys()] #making rectangle for store stype according to color

plt.legend(legend_handles, legend_labels.values(), title='Store Type', loc='upper right')
plt.title("Size of Stores")
plt.show()

#Box Plot for distribution of size of stores

sns.boxplot(x="Type",y="Size",data=stores_data)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.title("Box Plot for size of stores")
plt.show()

#ploting Correlation Matrix

#dropping date and type column to have only numeric values
corr_data=new_df.drop(columns=["Date","Type"])\
    .corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_data,annot=True,fmt=".2f")
plt.title("Correlation Matrix")
plt.show() 






