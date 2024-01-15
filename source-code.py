#importing libraries
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
from matplotlib.cm import get_cmap
import streamlit as st
import plotly.express as px
from sklearn import preprocessing
from sklearn.model_selection import train_test_split  
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

st.set_option('deprecation.showPyplotGlobalUse', False) #do not show warning when plotting on streamlit


st.title("WALMART SALES DATA ANALYSIS")

st.subheader("ABOUT PROJECT")
st.write("In this project, an analysis was conducted on the weekly sales data of 45 Walmart stores spanning the years 2010 to 2013. The initial phase involved exploratory data analysis to discern relationships among various features. Subsequently, sales forecasting techniques were employed to predict future values based on the insights gained from the exploratory analysis")



#unzipping dataset

# with zipfile.ZipFile("Course-Project\Dataset.zip", 'r') as zip_ref:
#     zip_ref.extractall("Course-Project")

#reading the dataset
features_data=pd.read_csv("features.csv")
stores_data=pd.read_csv("stores.csv")
train_data=pd.read_csv("train.csv")
test_data=pd.read_csv("test.csv")


#checking the value in the data

# features_data.head(10)
# stores_data.head(10)
# train_data.head(10)

#Initial Data Exploration


# features_data.describe().T
# features_data.info()
# train_data.describe().T
# train_data.info()
# stores_data.describe().T
# stores_data.info()


#checkin Null Values

# features_data.isnull().sum()
# stores_data.isnull().sum()
# train_data.isnull().sum()

#Data merging (train,features and store data)
new_train_df=train_data.merge(features_data,how='left',on=["Store","Date"],indicator=True).merge(stores_data,how='left').copy()
new_train_df.isnull().sum()
new_test_df=test_data.merge(features_data,how='left',on=["Store","Date"],indicator=True).merge(stores_data,how='left').copy()
new_test_df.isnull().sum()

#lambda function to check the percentage of null values

# null_percent_check=lambda x:print(f"The null percentage of {x} is {(new_train_df[x].isnull().sum()/new_train_df.shape[0])*100}")
# null_percent_check("MarkDown1")
# null_percent_check("MarkDown2")
# null_percent_check("MarkDown3")
# null_percent_check("MarkDown4")


#The null percentages are more than 60% and these are promotional offers so we can drop these columns.
new_train_df.drop(new_train_df.columns[new_train_df.columns.get_loc("MarkDown1"):new_train_df.columns.get_loc("MarkDown5")+1],axis=1,inplace=True)
#print(new_train_df.columns)

# dropping from test set as well
new_test_df.drop(new_test_df.columns[new_test_df.columns.get_loc("MarkDown1"):new_test_df.columns.get_loc("MarkDown5")+1],axis=1,inplace=True)


#Dropping extra columns after merge
new_train_df.drop(['IsHoliday_y','_merge'], axis=1,inplace=True)
#print(new_train_df.columns) 

new_test_df.drop(['IsHoliday_y','_merge'], axis=1,inplace=True)
#new_test_df.columns

#renaming the Holiday column
new_train_df.rename(columns={"IsHoliday_x":"IsHoliday"},inplace=True)
new_test_df.rename(columns={"IsHoliday_x":"IsHoliday"},inplace=True)
#print(new_test_df.columns)
#print(new_train_df.columns) 

#checking the count of negative values in sales
#print(new_train_df[new_train_df['Weekly_Sales']<=0].Weekly_Sales.shape)

#excluding these negative values from data
new_df=new_train_df[new_train_df['Weekly_Sales']>0].copy()

new_df["Date"]=pd.to_datetime(new_df['Date']) #convert date column into proper date time column
new_test_df["Date"]=pd.to_datetime(new_test_df['Date']) #convert date column into proper date time column


#separting date columns into week, month and year 
new_df["Week"] = new_df["Date"].dt.isocalendar().week
new_df["Month"] = new_df["Date"].dt.month
new_df["Year"] = new_df["Date"].dt.year

new_test_df["Week"] = new_test_df["Date"].dt.isocalendar().week
new_test_df["Month"] = new_test_df["Date"].dt.month
new_test_df["Year"] = new_test_df["Date"].dt.year

#filling null values with mean of that column
new_test_df['CPI'].fillna(new_test_df['CPI'].mean(),inplace=True)
new_test_df['Unemployment'].fillna(new_test_df['Unemployment'].mean(),inplace=True)

#initiating streamlit session

if 'show_data' not in st.session_state:
    st.session_state.show_data = False

# Create a button to toggle the sidebar
toggle_sidebar = st.sidebar.button('Data Head')

# Use the button state to toggle the sidebar
if toggle_sidebar:
    # Toggle the show_matrix variable
    st.session_state.show_data = not st.session_state.show_data
    # Display the correlation matrix plot if the button is pressed
    if st.session_state.show_data:
        st.subheader("Data Head")
        st.table(new_df.head())
      
else:
        # If button is not pressed, clear the sidebar content
        st.empty()

st.subheader("FEATURE UNDERSTANDING")
st.write("Several features need to be plotted to comprehend their interrelationships.")

#feature understanding 

# Initialize session state
if 'show_matrix' not in st.session_state:
    st.session_state.show_matrix = False

# Create a button to toggle the sidebar
toggle_sidebar = st.sidebar.button('Feature Correlation')

# Use the button state to toggle the sidebar
if toggle_sidebar:
    # Toggle the show_matrix variable
    st.session_state.show_matrix = not st.session_state.show_matrix
    # Display the correlation matrix plot if the button is pressed
    if st.session_state.show_matrix:
        # Plotting Correlation Matrix
        corr_data = new_df.drop(columns=["Date", "Type"]).corr()

        # Create a Matplotlib plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_data, annot=True, fmt=".2f")
        plt.title("Correlation Matrix")

        # Display the plot in the sidebar
        st.pyplot()
else:
        # If button is not pressed, clear the sidebar content
        st.empty()
   
    
def distribution_store_type():
     
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plotting pie chart for distribution of store type
    axs[0].pie(new_df["Type"].value_counts().tolist(), autopct="%1.1f%%", labels=new_df["Type"].value_counts().keys())
    axs[0].set_title('Distribution of Store Type')

    # Box Plot for distribution of size of stores
    sns.boxplot(x="Type", y="Size", data=new_df, ax=axs[1])
    axs[1].grid(axis="y", linestyle="--", alpha=0.7)
    axs[1].set_title("Box Plot for Size of Stores")

    # Adjust layout to prevent clipping of titles
    plt.tight_layout()
    st.pyplot()

def plot_store_count():
     
    #plotting bar chart for count of store type
    Store_chart=new_df["Type"].value_counts().plot(kind='bar',title="Store Types",rot=0)
    Store_chart.set_xlabel("TYPE")
    Store_chart.set_ylabel("COUNT")
    st.pyplot()


def plot_store_size():

    #plotting bar chart for size of store according to their type

    Size_plot=stores_data.plot.bar(x="Store",y="Size",
                                color=['green' if data_value=="A" else 'red' if data_value=='B' else 'blue' for data_value in stores_data['Type']],
                                rot=0,figsize=(13, 5))

    legend_labels = {'green': 'A', 'red': 'B', 'blue': 'C'} #dictionary for creating colored legends
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in legend_labels.keys()] #making rectangle for store stype according to color

    plt.legend(legend_handles, legend_labels.values(), title='Store Type', loc='upper right')
    plt.title("Size of Stores")
    st.pyplot()

def plot_yearly_fuel():

    #bar plot to check fuel prices over the years
    sns.barplot(x="Year", y="Fuel_Price",data=new_df)
    plt.title("Yearly Fuel Prices")
    st.pyplot()


def plot_weeklysales():
    fig2, axs1 = plt.subplots(1, 2, figsize=(18, 6))

    # Plotting bar plot for sales on each store
    sns.barplot(x="Store", y="Weekly_Sales", data=new_df, hue="Type",
                order=new_df.groupby("Store")["Weekly_Sales"].mean().sort_values(ascending=False).index,
                ax=axs[0])
    axs1[0].set_title("Sales on Each Store")
    axs1[0].grid(axis="y", linestyle="--", alpha=0.7)
    axs1[0].set_ylabel("Average Weekly Sales")
    axs1[0].tick_params(axis='x', rotation=45)

    # Plotting line plot for sales by each department
    sns.lineplot(x="Dept", y="Weekly_Sales", data=new_df, ax=axs[1])
    axs1[1].set_title("Sales by Departments")
    axs1[1].set_xticks(range(min(new_df['Dept']), max(new_df['Dept']) + 1, 4))
    axs1[1].set_xticklabels(range(min(new_df['Dept']), max(new_df['Dept']) + 1, 4), rotation=45)
    axs1[1].set_ylabel("Average Weekly Sales")
    axs1[1].grid(axis="x", linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    st.pyplot()

def plot_unemployement():

    #plotting un-employment rate in each store

    plt.figure(figsize=(12, 6))
    sns.lineplot(x="Store",y="Unemployment",data=new_df)
    plt.title("Unemployment Rate")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.xticks(new_df['Store'].unique())
    st.pyplot()

def plot_year_sales():

    #line plot for year wise sales
    plt.figure(figsize=(12, 8))
    sns.lineplot(x="Month",y="Weekly_Sales",data=new_df,hue="Year",palette="deep")
    plt.title("Year wise sales",fontsize=20)
    plt.ylabel("Average Weekly Sales")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    st.pyplot()

def plot_holiday_sales():

    #box plot for distribution of sales on holdidays and non-holiddays
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="IsHoliday",y="Weekly_Sales",data=new_df)
    plt.title("Holidays vs Non-Holidays",fontsize=20)
    st.pyplot()
    
    #sunburtst plot for distribution of sales on holdidays and non-holiddays
    sunburst_plot= px.sunburst(new_df, path=['IsHoliday', 'Type'], values='Weekly_Sales')
    sunburst_plot.update_layout(title_text="Holidays vs Non-Holidays: Sales by Store Type")
    st.plotly_chart(sunburst_plot)

 

# Create a dropdown to select the plot
selected_plot = st.selectbox('Select Plot', ['Distribution of Store Type', 'Store Count', 'Store Size', 'Yearly Fuel Prices',
                                             'Weekly Sales', 'Unemployment Rate', 'Year Wise Sales', 'Holiday Sales'])

plot_button1 = st.button('Plot')

if plot_button1:
     
    if selected_plot == 'Distribution of Store Type':
        distribution_store_type()
    elif selected_plot == 'Store Count':
        plot_store_count()
    elif selected_plot == 'Store Size':
        plot_store_size()
    elif selected_plot == 'Yearly Fuel Prices':
        plot_yearly_fuel()
    elif selected_plot == 'Weekly Sales':
        plot_weeklysales()
    elif selected_plot == 'Unemployment Rate':
        plot_unemployement()
    elif selected_plot == 'Year Wise Sales':
        plot_year_sales()
    elif selected_plot == 'Holiday Sales':
        plot_holiday_sales()


st.subheader("MODEL TRAINING")





#Machine Learning part


new_df2=new_df.copy()#making a copy of dataset for machine learning models

#encoding the columns that has string values 
label_encoder=preprocessing.LabelEncoder()
new_df2["IsHoliday"]=label_encoder.fit_transform(new_df2["IsHoliday"])
new_df2["Type"]=label_encoder.fit_transform(new_df2["Type"])

new_test_df["IsHoliday"]=label_encoder.fit_transform(new_test_df["IsHoliday"])
new_test_df["Type"]=label_encoder.fit_transform(new_test_df["Type"])

#dropping Date column from test data
new_test_df.drop(columns="Date",inplace=True)

#making correlation matrix again
# corr_=new_df2.columns.tolist()
# corr_.remove("Date")
# corr_data2=new_df2[corr_].corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_data2,annot=True,fmt=".2f")
# plt.title("Correlation Matrix")
# plt.show()

#separating features and Target Values
Features=new_df2.drop(['Weekly_Sales','Date'],axis=1)
Target=new_df2["Weekly_Sales"]

#separting training and testing set
x_train, x_test, y_train, y_test= train_test_split(Features, Target, test_size= 0.2, random_state=2)

 #using random forest algorithm
rf_model=RandomForestRegressor(n_estimators=50, #no of tress
    random_state=0,
    n_jobs=-1,#utilizaing available resources
    max_depth=50, #maximum dept of each tree
    max_features='sqrt', #maximum number of features each tree is allowed to use 
    min_samples_split=5 #minimum samples required for a split      
)

#applying XGBoost Algorithm
model = XGBRegressor(early_stopping_rounds=50)

# Create a dropdown to select the plot
selected_model = st.selectbox('Select Model', ["Random Forest Algorithm","XGBoost Algorithm"])

model_button = st.button('Train')

if model_button:
    if selected_model == 'Random Forest Algorithm':
       
        rf_model.fit(x_train,y_train)

        rf_pred=rf_model.predict(x_test)

        st.write("R2 score  :",round(r2_score(y_test, rf_pred),2))
        st.write("RMSE:",round(math.sqrt(mean_squared_error(y_test, rf_pred)),2))
        st.write("mean_absolute_error:",round(mean_absolute_error(y_test, rf_pred),2))
        
        #feature importance bar chart
        plt.barh(Features.columns, rf_model.feature_importances_)
        plt.title("Feautre Importance")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        st.pyplot()

    elif selected_model == 'XGBoost Algorithm':
        
    
        model.fit(x_train,y_train,eval_set=[(x_train,y_train),(x_test,y_test)],verbose=0)
        xgb_pred1=model.predict(x_test)
        xgb_pred2 = model.predict(new_test_df) #predicting weekly sales for 2013

        st.write("R2 score  :",round(r2_score(y_test, xgb_pred1),2))
        st.write("RMSE:",round(math.sqrt(mean_squared_error(y_test, xgb_pred1)),2))
        st.write("mean_absolute_error:",round(mean_absolute_error(y_test, xgb_pred1),2))

        #feature importance bar chart
        plt.barh(Features.columns, model.feature_importances_)
        plt.title("Feautre Importance")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        st.pyplot()

        st.subheader("MODEL PREDICTION")

        st.write("Prediction has been done with XGBOOST Algorithm since data has been trained better with it.")
    
        new_test_df["Weekly_Sales"]=xgb_pred2 #adding weekly sales to new_test_df

        #making extra column on train and test data to generate prediction plots
        new_df3=new_df.copy()
        new_df3['YearMonth'] = new_df3['Date'].dt.to_period('M') #making year-month column
        monthly_sales_train = new_df3.groupby('YearMonth')['Weekly_Sales'].mean() #grouping Year-month with weekly sales

        new_test_df2=new_test_df.copy()
        new_test_df2["Date"]=test_data["Date"] #adding date column back
        new_test_df2["Date"]=pd.to_datetime(new_test_df2['Date'])#converting it to date-time
        new_test_df2['YearMonth'] = new_test_df2['Date'].dt.to_period('M')#making year-month column
        monthly_sales_test = new_test_df2.groupby('YearMonth')['Weekly_Sales'].mean() #grouping Year-month with weekly sales
        plt.figure(figsize=(12, 8))
        train_palette = {2010: 'blue', 2011: 'green', 2012: 'red'}
        test_palette = {2012:'red',2013: 'purple'}
        sns.lineplot(x="Month", y="Weekly_Sales", data=new_df2, hue="Year", 
                        palette=train_palette, linewidth=2, errorbar=None)
        sns.lineplot(x="Month", y="Weekly_Sales", data=new_test_df, hue="Year", 
                        palette=test_palette, linewidth=3,linestyle='--', errorbar=None)
        plt.plot([new_df2['Month'].iloc[-1], new_test_df['Month'].iloc[0]], 
                    [monthly_sales_train["2012-10"], monthly_sales_test["2012-11"]], 
                    color='red', linestyle='--', linewidth=3)
        plt.title("Year wise sales(Actual and Prediction)",fontsize=20)
        plt.ylabel("Average Weekly Sales")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        st.pyplot()
        st.write("Prediction has been made from Nov-2012 till July-2013, which has been showed by dashed lines")






















