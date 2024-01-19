# Walmart Sales Data Project

## Introduction

Welcome to the Walmart Sales Data Project, a comprehensive exploration of Walmart's sales dynamics using various datasets. This repository contains four key datasets, each shedding light on different aspects of the intricate factors influencing weekly sales at Walmart stores.

### Datasets Overview

1. **Features Dataset:**
   - Columns: Store, Date, Temperature, Fuel_Price, MarkDown1, MarkDown2, MarkDown3, MarkDown4, MarkDown5, CPI, Unemployment, IsHoliday.
   - Detailed information on store-specific details, environmental factors, and promotional markdowns.

2. **Stores Dataset:**
   - Columns: Store, Type, Size.
   - Characteristics of each Walmart store, categorized by type, and details about their sizes.

3. **Train Dataset:**
   - Columns: Store, Dept, Date, Weekly_Sales, IsHoliday.
   - Historical records of weekly sales, including department-wise breakdowns, dates, and holiday indicators.

4. **Test Dataset:**
   - Columns: Store, Dept, Date, IsHoliday.
   - Similar to the train dataset structure, excluding the Weekly_Sales column, intended for sales predictions.

## Project Objective

The primary goal of this project is to utilize data science methodologies to gain insights into Walmart's sales patterns and develop a predictive model for weekly sales. The project unfolds in three main phases:

### 1. Data Cleaning
Identifying and handling missing values, outliers, and inconsistencies within the datasets to ensure a clean and reliable foundation for analysis and modeling.

### 2. Exploratory Data Analysis (EDA)
Statistical and visual exploration techniques to understand relationships between different features, identify trends, and extract meaningful patterns contributing to sales predictions.

### 3. Model Training for Weekly Sales Prediction
Leveraging machine learning algorithms to build a predictive model using historical data from the train dataset. The model will be validated for accuracy and reliability.

**Note:** The dataset for this project has been sourced from [Kaggle](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast).