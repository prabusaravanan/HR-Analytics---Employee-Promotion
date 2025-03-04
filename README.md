# Project Documentation: HR Analytics - Employee Promotion Prediction
# Introduction:
    In this project, we analyze employee data to predict the likelihood of promotion using machine learning techniques in Python. This involves data preprocessing,
    exploratory      data analysis (EDA), and visualization to gain insights before applying predictive models.
# Dataset Overview:
  The dataset consists of two files:
    •	train.csv: Contains historical employee records, including whether they were promoted (is_promoted).
    •	test.csv: Contains employee records for which promotions need to be predicted.
# Loading the Dataset:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings('ignore')
# Load datasets:
    test_df = pd.read_csv("test.csv")
    train_df = pd.read_csv("train.csv")
# Data Exploration
    Viewing Dataset Samples
# Display first five rows
    train_df.head()
# Display last five rows
    train_df.tail()
    Dataset Shape
    print("Train dataset shape:", train_df.shape)
    print("Test dataset shape:", test_df.shape)
    Data Types and Summary Statistics
# Checking data types
    train_df.info()
# Statistical summary of numerical features
    train_df.describe()
# Statistical summary of categorical features
    train_df.describe(include=['object']).T
    Exploratory Data Analysis (EDA)
    Correlation Heatmap
# Select numeric columns
    numeric_data = train_df.select_dtypes(include=[np.number])
# Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_data.corr(), annot=True, linewidth=0.8)
    plt.show()
# Visualizing Employee Promotions
    sns.set_style('darkgrid')
    sns.catplot(x='is_promoted', kind='count', data=train_df)
    plt.show()
# Department-wise Promotion Analysis
    plt.figure(figsize=(12, 10))
    sns.catplot(x='department', hue='is_promoted', kind='count', data=train_df, palette='Set3')
    plt.xticks(rotation=45, horizontalalignment='right')
    plt.show()
# Education Level vs. Promotion
    plt.figure(figsize=(5,5))
    plt.xlabel("Education")
    plt.ylabel("Number of Promotions")
    promotion_counts = train_df.groupby('education')['is_promoted'].sum()
    plt.bar(promotion_counts.index, promotion_counts.values, color=['tab:orange', 'tab:red', 'tab:green'])
    plt.show()
# Gender-wise Promotion Analysis
    train_df['gender'] = train_df['gender'].astype(str)
    train_df['is_promoted'] = train_df['is_promoted'].astype(str)
    plt.figure(figsize=(12, 10))
    sns.catplot(x='gender', hue='is_promoted', kind='count', data=train_df, palette='Set2')
    plt.show()
# Recruitment Channel Analysis
    plt.figure(figsize=(12, 10))
    sns.catplot(x='recruitment_channel', hue='is_promoted', kind='count', data=train_df, palette='dark')
    plt.show()
# Training and Promotion Relationship
    plt.figure(figsize=(12, 10))
    sns.catplot(x='no_of_trainings', hue='is_promoted', kind='count', data=train_df, palette='dark')
    plt.show()
# Previous Year Rating and Promotion
    plt.figure(figsize=(12, 10))
    sns.catplot(x='previous_year_rating', hue='is_promoted', kind='count', data=train_df, palette='Set2')
    plt.show()
# Awards and Promotion Relationship
    plt.figure(figsize=(12, 10))
    sns.catplot(x='awards_won?', hue='is_promoted', kind='count', data=train_df, palette='dark')
    plt.show()
# Data Cleaning
# Handling Missing Values
# Check missing values
    print("Missing values in train data:")
    print(train_df.isnull().sum())
    print("Missing values in test data:")
    print(test_df.isnull().sum())
# Imputing missing values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='most_frequent')
    train_df.fillna(imputer.fit_transform(train_df), inplace=True)
    test_df.fillna(imputer.transform(test_df), inplace=True)
# Renaming Columns
    train_df.rename(columns={'awards_won?': 'awards_won'}, inplace=True)
    test_df.rename(columns={'awards_won?': 'awards_won'}, inplace=True)
# Data Encoding
    One-Hot Encoding Categorical Variables
    train_cat = train_df.select_dtypes(include=['object']).columns
    test_cat = test_df.select_dtypes(include=['object']).columns
    train_df = pd.get_dummies(train_df, drop_first=True, columns=train_cat)
    test_df = pd.get_dummies(test_df, drop_first=True, columns=test_cat)
# Conclusion
    This project provides insights into employee promotion patterns by analyzing various factors such as department, education, training, awards, and recruitment channels.      The cleaned and preprocessed data can now be used for machine learning modeling to predict future promotions.







